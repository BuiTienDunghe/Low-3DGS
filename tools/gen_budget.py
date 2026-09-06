#!/usr/bin/env python3
"""
tools/gen_budget.py — SINH bảng ngân sách bộ nhớ từ dữ liệu ĐO. Không viết tay số nào.

VÌ SAO: audit độc lập tìm 25 lỗi trong tài liệu, sinh ra bởi đúng 3 cơ chế:
  1. copy-paste không tính lại  (0.82 GB của truck bị dán vào ô drjohnson, lan ra 4 file)
  2. MiB viết thành MB          (vi phạm chính quy tắc L3 của dự án)
  3. sửa lỗi chỉ tại chỗ phát hiện, không lan ra các file khác
Cả ba đều là bệnh của bảng-viết-tay. File này là thuốc: một nguồn, sinh ra bảng, không sửa tay.

QUY ƯỚC ĐƠN VỊ — cố định, in ra mọi bảng:
  MiB/GiB = 2^20 / 2^30  dùng cho BỘ NHỚ (khớp NVML, /proc, cgroup, torch)
  MB      = 10^6         dùng cho KÍCH THƯỚC FILE (khớp quy ước 3DGS.zip, quy tắc L3)
Mọi biến đặt tên rõ đơn vị: `_mib`, `_gib`, `_mb`.

XUẤT XỨ: mỗi hệ số phải khai báo nguồn (file:hàm hoặc experiment id). Hệ số không có xuất
xứ khớp code sắp chạy thì in `?`, KHÔNG in số. Đây là quy tắc chặn lỗi "đo code A, áp cho code B".
"""
from __future__ import annotations

import json
import os
import sys

MIB = 1024.0 ** 2
GIB = 1024.0 ** 3
MB = 1e6

# =============================================================================
# HẰNG SỐ ĐO ĐƯỢC — mỗi cái kèm xuất xứ. Sửa ở ĐÂY, không sửa trong tài liệu.
# =============================================================================
MEASURED = {
    "vram_ceiling_mib": {
        "value": 3440.0,
        "src": "EXP-008 · cuda_overhead_expandable.json · true_vram_ceiling_mb",
        "note": "với PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (bắt buộc, EXP-006)",
    },
    "vram_ceiling_mib_no_expandable": {
        "value": 3808.0,
        "src": "EXP-007 · cuda_overhead.json",
        "note": "KHÔNG dùng — allocator tràn im lặng sang host RAM",
    },
    # Chi phí GPU KHÔNG phải tham số: rasterizer workspace + activations + context + WSL baseline.
    # Đo = vram_device_peak - N*236B. Đây là thứ mọi bảng cũ bỏ sót (audit defect #1).
    # ĐỈNH VRAM CẢ PIPELINE — ĐO THẬT (EXP-013), gồm forward + backward + prune_list.
    # KHÔNG ghép từ hệ số per-stage: phép ghép đó chính là lỗi đã sinh ra 8 lần sai.
    # (Ví dụ: "non-param overhead" tính từ render ra 1106 MiB, nhưng cùng đại lượng đó
    #  trong pipeline đầy đủ là 1510 MiB — nó phụ thuộc trạng thái allocator, không phải hằng số.)
    "pipeline_peak_vram_mib": {
        "train": 1505.4, "truck": 2981.4, "playroom": 3061.4, "drjohnson": 3761.4,
        "src": "EXP-013 · measure_pipeline.py --stages all · đỉnh @ s8_prune_list (model đầy đủ)",
        "note": "drjohnson 3761 > trần 3440 -> OOM THẬT (any_oom=True)",
    },
    "pipeline_peak_rss_gib": {
        "train": 4.332, "truck": 4.884, "playroom": 6.616, "drjohnson": 8.122,
        "src": "EXP-013 · VmHWM · đỉnh @ s4_load_ply",
        "note": "drjohnson chạy dưới cap 8500 MiB, dùng 8.122 GiB = 96% cap",
    },
}

# Hồi quy tuyến tính trên 4 điểm đo (EXP-013). Dùng để NỘI SUY trong dải đã đo,
# KHÔNG dùng để ngoại suy ra ngoài — đó là cách 8 lỗi trước sinh ra.
VRAM_FIT = {"slope_mib_per_million": 948.7, "intercept_mib": 531.0,
            "fit_range_n": (1_026_508, 3_405_153),
            "residual": "truck +1.3% · playroom +3.8% (ảnh lớn hơn -> rasterizer lớn hơn)"}

# -----------------------------------------------------------------------------
# PLY I/O — ĐO NGÀY 2026-09-04 bằng tools/bench_plyio.py trên ĐÚNG hàm của repo PUP,
# có torch đã import + CUDA đã init, trong WSL, đo bằng VmHWM (bộ đếm kernel, không lấy mẫu),
# mỗi phép đo một tiến trình riêng dưới cgroup MemoryMax=8G + MemorySwapMax=0.
#
# ⚠ Hai hệ số cũ ĐỀU SAI và đã bị thay:
#     4.58× (agent đo trên INRIA main, Windows Python, không có torch)  -> thật 3.56×
#     1.95× (agent, gsply)                                              -> thật 2.03–2.21×
#   Đây đúng là lỗi "đo code A áp cho code B" — lần này đo chính code sẽ chạy.
# -----------------------------------------------------------------------------
PLY_IO = {
    "load_inria":  {"x": (3.56, 3.57), "vram": True,
                    "note": "ổn định qua 4 scene; đỉnh VRAM = 1.63× params do .transpose(1,2).contiguous()"},
    "load_gsply":  {"x": (2.03, 2.21), "vram": False,
                    "note": "trả numpy trên CPU, 0 MiB VRAM; nhanh hơn ~2×; BITWISE KHỚP INRIA (4 scene × 6 tensor)"},
    "save_inria":  {"x": (13.35, 13.35), "vram": True,
                    "note": "CHỈ đo được ở train. truck/playroom/drjohnson đều OOM-KILL ở cap 8G. BLOCKER."},
    "save_gsply":  {"x": (2.98, 3.88), "vram": True,
                    "note": "rẻ hơn INRIA 3.4–4.5×; file ra = 95% file vào (bỏ nx,ny,nz)"},
    "src": "EXP-012 · experiments/plyio/results.jsonl · tools/bench_plyio.py",
}

# Byte mỗi Gaussian, fp32. 3 xyz + 3 scale + 4 rot + 1 opacity + SH.
SH_COEFFS = {0: 3, 1: 12, 2: 27, 3: 48}


def bytes_per_gaussian(sh: int, mode: str) -> int:
    floats = 3 + 3 + 4 + 1 + SH_COEFFS[sh]
    params = floats * 4
    if mode == "params":
        return params
    if mode == "finetune":            # params + grads + 2 moment Adam + ~12B densify stats
        return params * 4 + 12
    raise ValueError(mode)


SCENES = {
    #                N@30k     n_img   W     H    file_bytes (đo bằng stat, KHÔNG làm tròn)
    "train":     dict(n=1_026_508, imgs=301, w=980,  h=545,  file_bytes=254_575_516),
    "truck":     dict(n=2_541_226, imgs=251, w=979,  h=546,  file_bytes=630_225_580),
    "playroom":  dict(n=2_546_116, imgs=225, w=1264, h=832,  file_bytes=631_438_300),
    "drjohnson": dict(n=3_405_153, imgs=263, w=1332, h=876,  file_bytes=844_477_944),
}

# B/pixel — ĐO TRỰC TIẾP từ tensor trong repo đang chạy, không đọc từ source của repo khác.
IMG_BYTES_PER_PIXEL = {
    "pup":   {"value": 12, "src": "measure_pipeline.py s3 · tensor [3,H,W] float32 · alpha_mask=False"},
    "inria": {"value": 16, "src": "đọc cameras.py:48 self.alpha_mask=ones_like — CHƯA đo, chỉ áp cho repo INRIA"},
}

PRUNE_RATIO = 0.66      # LightGaussian mặc định -> N' = 0.34 N


def fmt(v, unit="GiB", width=8):
    return f"{'?':>{width}}" if v is None else f"{v:>{width}.2f}"


def main() -> int:
    repo = sys.argv[1] if len(sys.argv) > 1 else "pup"
    bpp = IMG_BYTES_PER_PIXEL[repo]
    ceil_mib = MEASURED["vram_ceiling_mib"]["value"]

    print("=" * 96)
    print(f"BẢNG NGÂN SÁCH — sinh tự động từ số đo | repo={repo}")
    print("ĐƠN VỊ: bộ nhớ = MiB/GiB (2^20/2^30) · kích thước file = MB (10^6)  [quy tắc L3]")
    print("=" * 96)
    print(f"\nTrần VRAM: {ceil_mib:.1f} MiB = {ceil_mib/1024:.3f} GiB")
    print(f"  nguồn: {MEASURED['vram_ceiling_mib']['src']}")
    print(f"  {MEASURED['vram_ceiling_mib']['note']}")
    print(f"\nẢnh: {bpp['value']} B/pixel — {bpp['src']}")

    pv = MEASURED["pipeline_peak_vram_mib"]
    pr = MEASURED["pipeline_peak_rss_gib"]
    print(f"\n{'='*96}\nĐỈNH CẢ PIPELINE — ĐO THẬT, KHÔNG GHÉP TỪ HỆ SỐ\n{'='*96}")
    print(f"  nguồn: {pv['src']}")
    print(f"{'scene':<12}{'N':>11}{'VRAM đỉnh':>12}{'còn lại':>10}{'RSS đỉnh':>11}   kết luận")
    print("-" * 72)
    for name, s in SCENES.items():
        v = pv.get(name); r = pr.get(name)
        if v is None:
            continue
        left = ceil_mib - v
        verdict = "✅ chạy được" if left > 0 else f"❌ OOM (vượt {-left:.0f} MiB)"
        print(f"{name:<12}{s['n']:>11,}{v:>10.0f} MiB{left:>9.0f}{r:>10.2f} GiB   {verdict}")
    f = VRAM_FIT
    nmax = (ceil_mib - f["intercept_mib"]) / f["slope_mib_per_million"]
    print(f"\n  Hồi quy 4 điểm: VRAM ≈ {f['slope_mib_per_million']:.0f} MiB/triệu Gaussian "
          f"+ {f['intercept_mib']:.0f} MiB   ({f['residual']})")
    print(f"  => TRẦN THỰC TẾ ≈ **{nmax:.2f} triệu Gaussian** cho pipeline đầy đủ (model chưa prune)")
    print(f"  ⚠ chỉ nội suy trong {f['fit_range_n'][0]:,}–{f['fit_range_n'][1]:,}; KHÔNG ngoại suy")

    # ---- bytes/Gaussian ----
    print(f"\n{'-'*96}\nBYTE / GAUSSIAN\n{'-'*96}")
    print(f"{'SH':>3}{'floats':>8}{'params B':>10}{'finetune B':>12}{'N tối đa (params)':>20}{'N tối đa (FT)':>16}")
    for sh in (3, 2, 1, 0):
        p, f = bytes_per_gaussian(sh, "params"), bytes_per_gaussian(sh, "finetune")
        print(f"{sh:>3}{3+3+4+1+SH_COEFFS[sh]:>8}{p:>10}{f:>12}"
              f"{ceil_mib*MIB/p/1e6:>19.2f}M{ceil_mib*MIB/f/1e6:>15.2f}M")

    # ---- khả thi theo scene ----
    print(f"\n{'-'*96}\nKHẢ THI THEO SCENE  (tensor + GPU không-phải-tham-số vs trần {ceil_mib:.0f} MiB)\n{'-'*96}")
    hdr = (f"{'scene':<11}{'N':>10}{'file MB':>9}{'ảnh GiB':>9}"
           f"{'FT SH3':>9}{'+ovh':>8}{'còn':>8}  {'A12':>8}{'+ovh':>8}{'còn':>8}")
    print(hdr); print("-" * len(hdr))
    for name, s in SCENES.items():
        n = s["n"]
        img_gib = s["imgs"] * s["w"] * s["h"] * bpp["value"] / GIB
        file_mb = s["file_bytes"] / MB if s["file_bytes"] else None
        ft_mib = n * bytes_per_gaussian(3, "finetune") / MIB
        # A12 = prune TRƯỚC training_setup -> đỉnh = max(N*params, N'*finetune)
        n_pruned = n * (1 - PRUNE_RATIO)
        a12_mib = max(n * bytes_per_gaussian(3, "params"), n_pruned * bytes_per_gaussian(3, "finetune")) / MIB
        o = None
        ft_tot = ft_mib + o if o else None
        a12_tot = a12_mib + o if o else None
        print(f"{name:<11}{n:>10,}{fmt(file_mb,'MB',9)}{img_gib:>9.2f}"
              f"{ft_mib/1024:>9.2f}{fmt(ft_tot/1024 if ft_tot else None,'',8)}"
              f"{fmt((ceil_mib-ft_tot)/1024 if ft_tot else None,'',8)}"
              f"  {a12_mib/1024:>8.2f}{fmt(a12_tot/1024 if a12_tot else None,'',8)}"
              f"{fmt((ceil_mib-a12_tot)/1024 if a12_tot else None,'',8)}")

    print("\n'?' = chưa đo GPU không-phải-tham-số cho scene đó. KHÔNG ngoại suy từ scene khác —")
    print("     đó chính là lỗi đã sinh ra 25 defect (áp hệ số của code/scene này cho cái khác).")

    # ---- A12: sửa lỗi số học ----
    print(f"\n{'-'*96}\nA12 — SỬA LỖI (tài liệu cũ ghi 240 B/Gaussian, SAI)\n{'-'*96}")
    p3, f3 = bytes_per_gaussian(3, "params"), bytes_per_gaussian(3, "finetune")
    eff = (1 - PRUNE_RATIO) * f3
    print(f"  Đỉnh A12 = max(N × {p3} B [scoring], N' × {f3} B [Adam sau prune])")
    print(f"  N' = {1-PRUNE_RATIO:.2f} N  ->  {1-PRUNE_RATIO:.2f} × {f3} = {eff:.0f} B mỗi Gaussian GỐC")
    print(f"  => hệ số chi phối là {max(p3, eff):.0f} B, KHÔNG phải 240 B")
    print(f"  => H8: giảm {f3}/{max(p3,eff):.0f} = {f3/max(p3,eff):.2f}×, KHÔNG phải 4×")
    print(f"     ({f3/max(p3,eff):.2f} = 1/(1-{PRUNE_RATIO}) = đúng bằng tỉ lệ prune)")
    print(f"  => N tối đa dưới A12 = {ceil_mib*MIB/max(p3,eff)/1e6:.2f} M, KHÔNG phải 14.33 M")

    # ---- PLY I/O: RAM đỉnh theo scene ----
    print(f"\n{'-'*96}\nPLY I/O — RAM ĐỈNH (đo trên code thật, {PLY_IO['src']})\n{'-'*96}")
    hdr2 = (f"{'scene':<11}{'file MB':>9}{'load INRIA':>12}{'load gsply':>12}"
            f"{'save INRIA':>12}{'save gsply':>12}")
    print(hdr2); print("-" * len(hdr2))
    for name, s in SCENES.items():
        if not s["file_bytes"]:
            continue
        fmb = s["file_bytes"] / MB
        fg = s["file_bytes"] / GIB
        li = fg * PLY_IO["load_inria"]["x"][1]
        lg = fg * PLY_IO["load_gsply"]["x"][1]
        si = fg * PLY_IO["save_inria"]["x"][0]
        sg = fg * PLY_IO["save_gsply"]["x"][1]
        si_s = f"{si:>10.2f}❌" if name != "train" else f"{si:>10.2f} "
        print(f"{name:<11}{fmb:>9.1f}{li:>12.2f}{lg:>12.2f}{si_s:>12}{sg:>12.2f}")
    print("  (GiB) ❌ = đã OOM-KILL thực tế ở cap 8G, con số chỉ là ngoại suy từ train")
    print(f"  · {PLY_IO['save_inria']['note']}")
    print(f"  · {PLY_IO['load_gsply']['note']}")

    print(f"\n{'-'*96}\nCẦN ĐO TIẾP (ô '?' ở trên)\n{'-'*96}")
    for name in SCENES:
        if name not in ov:
            print(f"  · GPU không-phải-tham-số cho {name} (render + fine-tune)")
    print("  · GPU không-phải-tham-số trên đường FINE-TUNE (backward) — mọi số hiện có là forward-only")
    print("  · file_bytes cho playroom")
    return 0


if __name__ == "__main__":
    sys.exit(main())
