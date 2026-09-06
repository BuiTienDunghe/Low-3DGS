"""
src/memory/counters.py — Bộ đếm bộ nhớ CHÍNH XÁC do kernel giữ, thay cho lấy mẫu.

VÌ SAO: lấy mẫu RSS ở 2–5 Hz KHÔNG bắt được đỉnh. Bằng chứng từ chính dữ liệu của dự án
(`p1_smoke_truck_.../memory_render.jsonl`): đỉnh 5.264 GB xuất hiện ở **đúng 1 trong 92 mẫu**,
ngay sau đó tụt 3330 MB trong một khoảng lấy mẫu. Tốc độ thay đổi đo được tới 6.26 GB/s —
ở tốc độ đó toàn bộ 3440 MiB VRAM khả dụng bị quét qua trong ~1 giây, tức 2 mẫu ở 2 Hz.
Không thể chặn trên một đỉnh bằng cách đó.

Kernel đã giữ sẵn các mốc cao nhất, chính xác tuyệt đối:
  /proc/<pid>/status : VmHWM (đỉnh RSS), RssAnon, RssFile, RssShmem, VmSwap
  /sys/fs/cgroup/... : memory.peak, memory.swap.peak (cả cây tiến trình)

PHÂN BIỆT QUAN TRỌNG — RssAnon vs RssFile:
  `plyfile` memory-map file .ply. Đọc hết thuộc tính làm toàn bộ 601 MB bị fault vào RSS dưới dạng
  **page file-backed, có thể bị evict**. Phần đó KHÔNG phải nhu cầu bộ nhớ cứng — kernel sẽ thu hồi
  khi thiếu. Chỉ **RssAnon** mới là nhu cầu thật. Báo cáo "peak RSS" gộp cả hai là báo cáo quá tay.
"""
from __future__ import annotations

import os
from typing import Any

KB = 1024
GB = float(2 ** 30)

_STATUS_KEYS = ("VmHWM", "VmRSS", "RssAnon", "RssFile", "RssShmem", "VmSwap", "VmPeak")


def proc_status(pid: int | None = None) -> dict[str, float]:
    """Đọc /proc/<pid>/status → dict {key: GB}. VmHWM là đỉnh RSS chính xác từ lúc process sinh ra."""
    p = f"/proc/{pid or 'self'}/status"
    out: dict[str, float] = {}
    try:
        with open(p) as f:
            for line in f:
                k = line.split(":", 1)[0]
                if k in _STATUS_KEYS:
                    out[k] = int(line.split()[1]) * KB / GB
    except (OSError, ValueError, IndexError):
        pass
    return out


def vm_hwm_gb(pid: int | None = None) -> float | None:
    """Đỉnh RSS CHÍNH XÁC (GB). Không bao giờ bỏ sót đỉnh thoáng qua."""
    return proc_status(pid).get("VmHWM")


def reset_vm_hwm() -> bool:
    """Đặt lại mốc VmHWM về RSS hiện tại, để đo đỉnh THEO TỪNG STAGE.

    Ghi '5' vào /proc/self/clear_refs. Trả True nếu thực sự reset được.
    Nếu kernel không hỗ trợ → phải chạy mỗi stage một tiến trình riêng.
    """
    before = vm_hwm_gb()
    try:
        with open("/proc/self/clear_refs", "w") as f:
            f.write("5")
    except OSError:
        return False
    after = vm_hwm_gb()
    if before is None or after is None:
        return False
    return after < before - 0.01 or before < 0.02      # đã tụt, hoặc vốn đã rất thấp


def _cgroup_path() -> str | None:
    try:
        with open("/proc/self/cgroup") as f:
            rel = f.readline().strip().split(":")[-1]
        base = f"/sys/fs/cgroup{rel}"
        return base if os.path.isdir(base) else None
    except OSError:
        return None


def cgroup_mem(field: str = "memory.peak") -> float | None:
    """Đọc bộ đếm cgroup v2 (GB). `memory.peak` phủ CẢ CÂY tiến trình, khác VmHWM chỉ self."""
    base = _cgroup_path()
    if not base:
        return None
    try:
        with open(os.path.join(base, field)) as f:
            return int(f.read().strip()) / GB
    except (OSError, ValueError):
        return None


def cgroup_reset_peak() -> bool:
    """cgroup v2 cho reset memory.peak bằng cách ghi vào nó (kernel >= 6.6)."""
    base = _cgroup_path()
    if not base:
        return False
    try:
        with open(os.path.join(base, "memory.peak"), "w") as f:
            f.write("0")
        return True
    except OSError:
        return False


def snapshot(pid: int | None = None) -> dict[str, Any]:
    """Ảnh chụp đầy đủ. Dùng ở đầu/cuối mỗi stage."""
    s = proc_status(pid)
    return {
        "vm_hwm_gb": s.get("VmHWM"),
        "vm_rss_gb": s.get("VmRSS"),
        "rss_anon_gb": s.get("RssAnon"),      # <- nhu cầu bộ nhớ THẬT
        "rss_file_gb": s.get("RssFile"),      # <- page cache, có thể evict
        "rss_shmem_gb": s.get("RssShmem"),
        "vm_swap_gb": s.get("VmSwap"),
        "cgroup_peak_gb": cgroup_mem("memory.peak"),
        "cgroup_current_gb": cgroup_mem("memory.current"),
        "cgroup_swap_peak_gb": cgroup_mem("memory.swap.peak"),
    }


def capabilities() -> dict[str, bool]:
    """Cơ chế nào thực sự dùng được trên máy này. Chạy một lần lúc khởi động, ghi vào header."""
    return {
        "proc_status": bool(proc_status()),
        "vm_hwm": vm_hwm_gb() is not None,
        "rss_anon_split": "RssAnon" in proc_status(),
        "clear_refs_resets_hwm": reset_vm_hwm(),
        "cgroup_peak": cgroup_mem("memory.peak") is not None,
        "cgroup_peak_resettable": cgroup_reset_peak(),
        "cgroup_swap_peak": cgroup_mem("memory.swap.peak") is not None,
    }


if __name__ == "__main__":
    import json
    print("=== Cơ chế khả dụng ===")
    caps = capabilities()
    for k, v in caps.items():
        print(f"  {k:<26} {'CÓ' if v else 'KHÔNG'}")

    print("\n=== Kiểm chứng: lấy mẫu có bỏ sót đỉnh không? ===")
    import threading, time
    sampled_peak = 0.0
    stop = threading.Event()

    def sampler(hz):
        nonlocal_peak = 0.0
        while not stop.is_set():
            v = proc_status().get("VmRSS") or 0.0
            nonlocal_peak = max(nonlocal_peak, v)
            stop.wait(1.0 / hz)
        globals()["_sp"] = nonlocal_peak

    for hz in (2.0, 5.0, 50.0):
        stop.clear()
        reset_vm_hwm()
        base = vm_hwm_gb() or 0.0
        t = threading.Thread(target=sampler, args=(hz,), daemon=True); t.start()
        blob = bytearray(400 * 2**20)      # 400 MB thoáng qua
        blob[::4096] = b"\x01" * (len(blob) // 4096)
        del blob
        stop.set(); t.join(timeout=2)
        exact = (vm_hwm_gb() or 0.0) - base
        samp = globals().get("_sp", 0.0) - base
        miss = (1 - samp / exact) * 100 if exact > 0 else 0
        print(f"  {hz:>5.0f} Hz: lấy mẫu {samp:6.3f} GB | VmHWM chính xác {exact:6.3f} GB "
              f"| BỎ SÓT {miss:5.1f}%")

    print("\n=== snapshot() ===")
    print(json.dumps(snapshot(), indent=2))
