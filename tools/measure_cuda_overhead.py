#!/usr/bin/env python3
"""
tools/measure_cuda_overhead.py — Đo VRAM thật sự dùng được trên máy này.

Trả lời câu hỏi bị treo suốt PLAN v1→v4: "4096 MiB trên giấy, nhưng còn bao nhiêu để chứa Gaussian?"
Không ai công bố con số này cho WSL2 + Turing 4 GB, và mọi bảng ngân sách trong PLAN.md §3 đang
dựa trên ước tính ~350 MB cho CUDA context. File này thay ước tính bằng số đo.

Đo bốn thứ:
  1. CUDA context  — VRAM mất trắng chỉ để khởi tạo CUDA, trước khi cấp phát gì
  2. Trần thật     — cấp phát tăng dần tới khi OOM; đây là số quyết định ngân sách
  3. Khoảng lệch   — NVML `used` vs `torch.memory_allocated`: phần allocator giữ + fragmentation
  4. cuDNN/cuBLAS  — workspace nạp lazy khi lần đầu chạy matmul/conv (nhiều người quên khoản này)

Dùng:
    python tools/measure_cuda_overhead.py --out experiments/p0_setup/cuda_overhead.json
    python tools/measure_cuda_overhead.py --quick        # bỏ qua bước dò trần (chậm nhất)

Chạy khi GPU rảnh (`nvidia-smi` = 0 MiB) và đã đóng trình duyệt, nếu không số sẽ sai.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import subprocess
import sys
import time

MB = 1024.0 ** 2


def nvml_used_mb(h) -> float:
    import pynvml
    return pynvml.nvmlDeviceGetMemoryInfo(h).used / MB


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out")
    ap.add_argument("--quick", action="store_true", help="bỏ bước dò trần")
    ap.add_argument("--step-mb", type=int, default=64, help="bước cấp phát khi dò trần")
    args = ap.parse_args()

    import pynvml
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    total_mb = pynvml.nvmlDeviceGetMemoryInfo(h).total / MB
    name = pynvml.nvmlDeviceGetName(h)
    name = name.decode() if isinstance(name, bytes) else name
    drv = pynvml.nvmlSystemGetDriverVersion()
    drv = drv.decode() if isinstance(drv, bytes) else drv

    R: dict = {
        "gpu": name, "driver": drv, "vram_total_mb": round(total_mb, 1),
        "platform": platform.platform(),
        "in_wsl": "microsoft" in platform.release().lower(),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
    }

    # ---- 0. đường nền: TRƯỚC khi import torch ------------------------------
    base = nvml_used_mb(h)
    R["baseline_used_mb"] = round(base, 1)
    print(f"GPU: {name} | {total_mb:.0f} MiB | driver {drv}")
    print(f"[0] Đường nền trước khi import torch : {base:8.1f} MiB")
    if base > 50:
        print(f"    ⚠ GPU KHÔNG rảnh ({base:.0f} MiB đang dùng) — đóng ứng dụng rồi đo lại")

    # ---- 1. CUDA context ---------------------------------------------------
    import torch
    R["torch"] = torch.__version__
    R["torch_cuda"] = torch.version.cuda
    if not torch.cuda.is_available():
        print("LỖI: torch.cuda.is_available() = False"); return 1
    p = torch.cuda.get_device_properties(0)
    R["sm"] = f"{p.major}.{p.minor}"
    R["torch_total_mb"] = round(p.total_memory / MB, 1)

    t0 = time.time()
    probe = torch.zeros(1, device="cuda")       # ép khởi tạo context
    torch.cuda.synchronize()
    ctx_t = time.time() - t0
    after_ctx = nvml_used_mb(h)
    ctx = after_ctx - base
    R["cuda_context_mb"] = round(ctx, 1)
    R["cuda_context_init_s"] = round(ctx_t, 2)
    print(f"[1] Sau khi khởi tạo CUDA context     : {after_ctx:8.1f} MiB  "
          f"(context = {ctx:.1f} MiB, mất {ctx_t:.1f}s)")

    # ---- 2. cuDNN / cuBLAS workspace nạp lazy -------------------------------
    a = torch.randn(512, 512, device="cuda")
    (a @ a).sum().item()                        # kích hoạt cuBLAS
    torch.cuda.synchronize()
    after_blas = nvml_used_mb(h)
    R["cublas_extra_mb"] = round(after_blas - after_ctx, 1)
    print(f"[2] Sau matmul đầu tiên (cuBLAS)      : {after_blas:8.1f} MiB  "
          f"(+{after_blas - after_ctx:.1f} MiB)")
    del a
    torch.cuda.empty_cache()
    gc.collect()

    fixed = nvml_used_mb(h) - base
    R["fixed_overhead_mb"] = round(fixed, 1)
    R["usable_estimate_mb"] = round(total_mb - nvml_used_mb(h), 1)
    print(f"    → Overhead cố định: {fixed:.1f} MiB · Ước tính còn dùng được: {R['usable_estimate_mb']:.1f} MiB")

    # ---- 3. dò trần thật ----------------------------------------------------
    # ⚠ KHÔNG được dò bằng cách chờ OOM. Trên WSL2/WDDM, driver NVIDIA cho phép
    # cấp phát TRÀN sang host RAM qua PCIe thay vì báo lỗi — lần đo đầu "thành công"
    # khi giữ 10816 MiB trên card 4096 MiB. Trần thật phải phát hiện bằng:
    #   (a) NVML `used` ngừng tăng dù cấp phát vẫn thành công, VÀ
    #   (b) băng thông sụp đổ (VRAM ~100+ GB/s vs PCIe ~5-10 GB/s)
    def bandwidth_gbs(x, reps: int = 20) -> float:
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(reps):
            x.mul_(1.0001)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t) / reps
        return x.numel() * 4 * 2 / dt / 1e9          # đọc + ghi

    if not args.quick:
        print(f"[3] Dò trần (bước {args.step_mb} MiB) — phát hiện plateau NVML + sụp băng thông")
        probe = torch.empty(32 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")  # 32 MiB
        bw0 = bandwidth_gbs(probe)
        R["bandwidth_vram_gbs"] = round(bw0, 1)
        print(f"    băng thông VRAM tham chiếu: {bw0:.1f} GB/s")

        blocks, held_mb = [], 0.0
        step_elems = args.step_mb * 1024 * 1024 // 4
        nvml_hist, plateau_at, spill_at = [], None, None
        for _ in range(400):
            try:
                blocks.append(torch.empty(step_elems, dtype=torch.float32, device="cuda"))
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if "out of memory" not in str(e).lower():
                    raise
                print(f"    OOM thật ở {held_mb:.0f} MiB"); break
            held_mb += args.step_mb
            u, bw = nvml_used_mb(h), bandwidth_gbs(probe, reps=5)
            nvml_hist.append(u)
            if plateau_at is None and len(nvml_hist) >= 4 and (nvml_hist[-1] - nvml_hist[-4]) < args.step_mb * 0.25:
                plateau_at = held_mb
                print(f"    ⚠ NVML PLATEAU ở {held_mb:.0f} MiB (NVML {u:.1f} MiB) — cấp phát vẫn 'thành công'")
            if spill_at is None and bw < bw0 * 0.5:
                spill_at = held_mb
                print(f"    ⚠ BĂNG THÔNG SỤP ở {held_mb:.0f} MiB: {bw:.1f} GB/s (còn {bw/bw0*100:.0f}%) — TRÀN sang host RAM")
            if held_mb % 512 == 0:
                print(f"    giữ {held_mb:6.0f} MiB | NVML {u:7.1f} | {bw:6.1f} GB/s", flush=True)
            if plateau_at and spill_at and held_mb > max(plateau_at, spill_at) + 512:
                print("    → đã xác nhận cả hai dấu hiệu, dừng"); break

        peak_nvml = max(nvml_hist) if nvml_hist else nvml_used_mb(h)
        R["nvml_peak_mb"] = round(peak_nvml, 1)
        R["nvml_plateau_at_alloc_mb"] = plateau_at
        R["bandwidth_collapse_at_alloc_mb"] = spill_at
        R["host_spill_observed"] = bool(plateau_at or spill_at)
        # Trần THẬT = phần NVML tối đa trừ overhead cố định và đường nền
        R["true_vram_ceiling_mb"] = round(peak_nvml - base - fixed, 1)
        print(f"    NVML đỉnh {peak_nvml:.1f} MiB / {total_mb:.0f} MiB tổng")
        print(f"    → TRẦN THẬT cho tensor: {R['true_vram_ceiling_mb']:.1f} MiB")
        del blocks, probe
        torch.cuda.empty_cache(); gc.collect()

    # ---- 4. ngân sách Gaussian ----------------------------------------------
    usable = R.get("true_vram_ceiling_mb", R["usable_estimate_mb"])
    R["gaussian_budget"] = {}
    print(f"\n=== NGÂN SÁCH GAUSSIAN với {usable:.0f} MiB dùng được ===")
    print(f"{'kịch bản':<38}{'B/Gaussian':>12}{'N tối đa':>14}")
    for label, bpg in (("inference (params, SH3)", 236),
                       ("fine-tune SH3 (p+g+2·Adam+stats)", 956),
                       ("fine-tune SH2", 620),
                       ("fine-tune SH1", 380),
                       ("prune TRƯỚC training_setup (A12)", 240)):
        n = usable * MB / bpg
        R["gaussian_budget"][label] = int(n)
        print(f"{label:<38}{bpg:>12}{n/1e6:>13.2f}M")

    print("\n(chưa trừ rasterizer workspace và ảnh — xem PLAN.md §3)")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        json.dump(R, open(args.out, "w"), indent=2)
        print(f"\n-> {args.out}")
    pynvml.nvmlShutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
