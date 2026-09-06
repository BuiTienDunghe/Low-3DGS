"""
src/memory/killswitch.py — Chặn "chết âm thầm" do swap.

Vấn đề (PLAN.md §2, §8): máy có pagefile 13.5 GB (Windows) + 4 GB swap (WSL). Khi tiến trình
vượt RAM, nó KHÔNG bị OOM mà bị đẩy sang swap và chậm đi 10–50×. Run không chết, chỉ kéo dài
từ 30 phút thành 10 tiếng — và ta không biết vì sao.

Giải pháp: một thread nền kiểm tra RSS (và tuỳ chọn VRAM device) mỗi giây. Vượt ngưỡng →
gọi hook lưu trạng thái → thoát với mã 3 (= "RAM-bound", phân biệt với crash thường).
Mỗi lần trip là MỘT ĐIỂM DỮ LIỆU cho cost model, ghi vào 05_EXPERIMENTS.md bảng OOM.

Dùng:
    from src.memory.killswitch import KillSwitch
    ks = KillSwitch(rss_limit_gb=9.5, on_trip=lambda why: save_ckpt("emergency"))
    ks.start()
    ... train ...
    ks.stop()

Hoặc:  with KillSwitch(rss_limit_gb=9.5): ...

Ngưỡng mặc định 9.5 GB = WSL cap 10 GB trừ headroom. Chỉnh theo `free -g` trước run.
Mã thoát:  3 = RSS vượt ngưỡng · 4 = VRAM device vượt ngưỡng · 5 = swap bắt đầu bị dùng (nếu bật)
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from typing import Callable

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None

try:
    import pynvml
    pynvml.nvmlInit()
    _H = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception:  # pragma: no cover
    pynvml = None
    _H = None

try:
    import torch
    _TORCH_CUDA = torch.cuda.is_available()
except Exception:  # pragma: no cover
    torch = None
    _TORCH_CUDA = False

GB = float(2 ** 30)
MB = float(2 ** 20)

EXIT_RSS = 3
EXIT_VRAM = 4
EXIT_SWAP = 5
EXIT_VRAM_SPILL = 6      # EXP-006: VRAM tràn sang host RAM qua PCIe (không ném OOM)


class KillSwitch:
    def __init__(self, rss_limit_gb: float = 9.5, vram_limit_mb: float | None = 3600,
                 swap_limit_gb: float | None = 0.5, check_hz: float = 1.0,
                 spill_margin_mb: float | None = 256, on_trip: Callable[[dict], None] | None = None,
                 log_path: str | None = None, hard_exit: bool = True):
        self.rss_limit = rss_limit_gb
        self.vram_limit = vram_limit_mb          # mặc định 3600 < trần thật 3808 (EXP-007)
        self.spill_margin_mb = spill_margin_mb
        self.swap_limit = swap_limit_gb
        # đường nền VRAM lúc khởi động (gồm WSL2 GPU-PV ~161 MiB + context)
        self._vram_base_mb = 0.0
        self.period = 1.0 / check_hz
        self.on_trip = on_trip
        self.log_path = log_path
        self.hard_exit = hard_exit
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.tripped: dict | None = None
        self._proc = psutil.Process() if psutil else None
        self._swap0 = self._swap_used() or 0.0  # swap đã dùng sẵn trước run không tính
        self._vram_base_mb = self._vram_dev() or 0.0

    # ----- readers ---------------------------------------------------
    def _rss(self) -> float | None:
        try:
            return self._proc.memory_info().rss / GB if self._proc else None
        except Exception:
            return None

    def _swap_used(self) -> float | None:
        try:
            return psutil.swap_memory().used / GB if psutil else None
        except Exception:
            return None

    def _vram_dev(self) -> float | None:
        try:
            return pynvml.nvmlDeviceGetMemoryInfo(_H).used / MB if _H is not None else None
        except Exception:
            return None

    # ------------------------------------------------------------------
    # 🔴 EXP-006: trên WSL2/WDDM, vượt VRAM KHÔNG ném OutOfMemoryError.
    # Driver tràn im lặng sang host RAM qua PCIe: NVML `used` đứng yên (plateau)
    # trong khi torch vẫn cấp phát "thành công", và tốc độ tụt 3.6×.
    # Bắt exception là vô dụng — phải phát hiện bằng phân kỳ giữa hai con số:
    #     torch.memory_reserved  (torch nghĩ nó đang giữ bao nhiêu)
    #     NVML used              (thực tế nằm trên card bao nhiêu)
    # Khi reserved vượt NVML quá `spill_margin_mb`, phần chênh đang ở host RAM.
    def _vram_spill_mb(self) -> float | None:
        if _H is None or torch is None or not _TORCH_CUDA:
            return None
        try:
            reserved = torch.cuda.memory_reserved() / MB
            dev = pynvml.nvmlDeviceGetMemoryInfo(_H).used / MB
            return reserved - (dev - self._vram_base_mb)   # >0 nghĩa là đã tràn
        except Exception:
            return None

    # ----- core ------------------------------------------------------
    def _check(self) -> dict | None:
        rss = self._rss()
        if rss is not None and rss > self.rss_limit:
            return {"why": "rss", "code": EXIT_RSS, "rss_gb": round(rss, 3), "limit_gb": self.rss_limit}
        if self.vram_limit is not None:
            v = self._vram_dev()
            if v is not None and v > self.vram_limit:
                return {"why": "vram_device", "code": EXIT_VRAM, "vram_mb": round(v, 1), "limit_mb": self.vram_limit}
        if self.spill_margin_mb is not None:
            sp = self._vram_spill_mb()
            if sp is not None and sp > self.spill_margin_mb:
                return {"why": "vram_spill_to_host", "code": EXIT_VRAM_SPILL,
                        "spill_mb": round(sp, 1), "margin_mb": self.spill_margin_mb,
                        "note": "torch reserved vuot NVML used -> dang chay tren PCIe, cham ~3.6x (EXP-006)"}
        if self.swap_limit is not None:
            s = self._swap_used()
            if s is not None and (s - self._swap0) > self.swap_limit:
                return {"why": "swap", "code": EXIT_SWAP, "swap_delta_gb": round(s - self._swap0, 3),
                        "limit_gb": self.swap_limit, "rss_gb": round(rss or 0, 3)}
        return None

    def _loop(self) -> None:
        while not self._stop.is_set():
            hit = self._check()
            if hit:
                hit["t"] = time.time()
                hit["pid"] = os.getpid()
                self.tripped = hit
                msg = f"[killswitch] TRIP {hit['why']}: {json.dumps(hit)}"
                print(msg, file=sys.stderr, flush=True)
                if self.log_path:
                    try:
                        with open(self.log_path, "a") as f:
                            f.write(json.dumps({"type": "killswitch", **hit}) + "\n")
                    except Exception:
                        pass
                if self.on_trip:
                    try:
                        self.on_trip(hit)
                    except Exception as e:  # hook lỗi cũng không được chặn việc thoát
                        print(f"[killswitch] on_trip raised: {e!r}", file=sys.stderr, flush=True)
                if self.hard_exit:
                    sys.stderr.flush()
                    os._exit(hit["code"])  # không chờ finalizers — chúng có thể chính là thứ đang swap
                return
            self._stop.wait(self.period)

    # ----- lifecycle -------------------------------------------------
    def start(self) -> "KillSwitch":
        if psutil is None:
            print("[killswitch] psutil không có — KHÔNG có bảo vệ RAM", file=sys.stderr)
            return self
        self._thread = threading.Thread(target=self._loop, daemon=True, name="killswitch")
        self._thread.start()
        print(f"[killswitch] armed: rss>{self.rss_limit} GB"
              + (f", vram_device>{self.vram_limit} MB" if self.vram_limit else "")
              + (f", swap_delta>{self.swap_limit} GB" if self.swap_limit is not None else ""),
              flush=True)
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def __enter__(self) -> "KillSwitch":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


if __name__ == "__main__":  # demo: python -m src.memory.killswitch  → tự trip ở 0.3 GB
    ks = KillSwitch(rss_limit_gb=0.3, swap_limit_gb=None, hard_exit=False).start()
    blob = bytearray(400 * 2**20)
    time.sleep(2.5)
    ks.stop()
    print("tripped:", ks.tripped)
