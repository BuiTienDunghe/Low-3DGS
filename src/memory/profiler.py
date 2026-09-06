"""
src/memory/profiler.py — Đo RAM + VRAM đỉnh cho TỪNG STAGE của pipeline fine-tune.

Đây là công cụ thu dữ liệu thô cho cost model C2 (PLAN.md §7). Mỗi stage ghi một dòng JSONL:

    {"stage": "load_ply", "wall_s": 12.3,
     "rss_peak_gb": 5.81,  "rss_start_gb": 0.42,  "rss_end_gb": 2.10,
     "vram_alloc_peak_mb": 1420.5, "vram_reserved_peak_mb": 1536.0,   # torch (tensor thật / allocator giữ)
     "vram_device_peak_mb": 1890.0,                                    # NVML (cái quyết định OOM)
     "ram_avail_min_gb": 3.9, "swap_used_max_gb": 0.0,
     "n_gaussians": 5987516, "sh_degree": 3, "extra": {...}}

Ba con số VRAM khác nhau và KHÔNG được lẫn (02_METHODOLOGY.md §2):
  alloc    = torch.cuda.max_memory_allocated  — tensor đang sống, dùng để fit cost model
  reserved = torch.cuda.max_memory_reserved   — allocator đã giữ (>= alloc)
  device   = NVML used                        — gồm CUDA context + mọi process; đây là số so với "4 GB"

RSS đỉnh KHÔNG lấy được bằng một lần đọc (RSS lên rồi xuống trong stage), nên profiler chạy
một thread lấy mẫu nền (mặc định 5 Hz) trong suốt stage. torch peak thì reset ở đầu stage.

Dùng:
    from src.memory.profiler import StageProfiler
    prof = StageProfiler("experiments/<run>/stages.jsonl", run_id="...")
    with prof.stage("load_ply", n_gaussians=None):
        gaussians.load_ply(path)
    with prof.stage("score", n_gaussians=gaussians.get_xyz.shape[0], sh_degree=3):
        ...
    prof.close()      # ghi footer với tổng và đỉnh toàn run

Không có torch / pynvml / psutil → trường tương ứng = null, không crash. Profiler không được
phép làm hỏng run.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None

try:
    import pynvml
    pynvml.nvmlInit()
    _NVML_H = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception:  # pragma: no cover
    pynvml = None
    _NVML_H = None

try:
    import torch
    _TORCH_CUDA = torch.cuda.is_available()
except Exception:  # pragma: no cover
    torch = None
    _TORCH_CUDA = False


GB = float(2 ** 30)
MB = float(2 ** 20)


def _rss_gb() -> float | None:
    if psutil is None:
        return None
    try:
        return psutil.Process().memory_info().rss / GB
    except Exception:
        return None


def _ram_avail_gb() -> float | None:
    if psutil is None:
        return None
    try:
        return psutil.virtual_memory().available / GB
    except Exception:
        return None


def _swap_used_gb() -> float | None:
    if psutil is None:
        return None
    try:
        return psutil.swap_memory().used / GB
    except Exception:
        return None


def _vram_device_mb() -> float | None:
    if _NVML_H is None:
        return None
    try:
        return pynvml.nvmlDeviceGetMemoryInfo(_NVML_H).used / MB
    except Exception:
        return None


def _git_hash(repo_dir: str | None = None) -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo_dir,
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


def env_snapshot() -> dict[str, Any]:
    """Một lần / run. Ghi vào header — mọi bảng kết quả phải truy được về đây."""
    snap: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "in_wsl": "microsoft" in platform.release().lower(),
        "git": _git_hash(),
        "ram_total_gb": round(psutil.virtual_memory().total / GB, 2) if psutil else None,
        "ram_avail_gb_at_start": round(_ram_avail_gb() or 0, 2) if psutil else None,
        "cpu_count": os.cpu_count(),
        "env": {k: os.environ.get(k) for k in ("PYTORCH_CUDA_ALLOC_CONF", "CUDA_VISIBLE_DEVICES")},
    }
    if torch is not None:
        snap["torch"] = torch.__version__
        snap["torch_cuda"] = torch.version.cuda
        if _TORCH_CUDA:
            p = torch.cuda.get_device_properties(0)
            snap["gpu"] = p.name
            snap["gpu_total_mb"] = round(p.total_memory / MB, 1)
            snap["gpu_sm"] = f"{p.major}.{p.minor}"
    if _NVML_H is not None:
        try:
            drv = pynvml.nvmlSystemGetDriverVersion()
            snap["driver"] = drv.decode() if isinstance(drv, bytes) else drv
            snap["vram_device_mb_at_start"] = round(_vram_device_mb() or 0, 1)
        except Exception:
            pass
    return snap


class _Sampler(threading.Thread):
    """Thread nền: theo dõi RSS đỉnh, RAM available tối thiểu, swap tối đa, VRAM device đỉnh."""

    def __init__(self, hz: float):
        super().__init__(daemon=True)
        self.period = 1.0 / hz
        # KHÔNG đặt tên `_stop`: threading.Thread._stop là method nội bộ mà join() gọi;
        # gán Event() vào đó làm join() ném "TypeError: 'Event' object is not callable".
        # (Cùng lỗi đã gặp ở tools/measure_ply_load.py — sửa một chỗ, quên chỗ này.)
        self._stop_evt = threading.Event()
        self.rss_peak = _rss_gb() or 0.0
        self.ram_avail_min = _ram_avail_gb() or float("inf")
        self.swap_max = _swap_used_gb() or 0.0
        self.vram_dev_peak = _vram_device_mb() or 0.0
        self.samples = 0

    def run(self) -> None:
        while not self._stop_evt.is_set():
            r = _rss_gb()
            if r is not None:
                self.rss_peak = max(self.rss_peak, r)
            a = _ram_avail_gb()
            if a is not None:
                self.ram_avail_min = min(self.ram_avail_min, a)
            s = _swap_used_gb()
            if s is not None:
                self.swap_max = max(self.swap_max, s)
            v = _vram_device_mb()
            if v is not None:
                self.vram_dev_peak = max(self.vram_dev_peak, v)
            self.samples += 1
            self._stop_evt.wait(self.period)

    def stop(self) -> None:
        self._stop_evt.set()
        self.join(timeout=2.0)


class StageProfiler:
    def __init__(self, out_path: str, run_id: str | None = None, hz: float = 5.0,
                 sync_cuda: bool = True, verbose: bool = True):
        self.out_path = out_path
        self.hz = hz
        self.sync_cuda = sync_cuda and _TORCH_CUDA
        self.verbose = verbose
        self.stages: list[dict[str, Any]] = []
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
        self._f = open(out_path, "a", buffering=1)  # line-buffered: mất điện vẫn còn dòng đã ghi
        header = {"type": "header", "run_id": run_id, **env_snapshot()}
        self._f.write(json.dumps(header) + "\n")
        self._t0 = time.time()

    # ------------------------------------------------------------------
    @contextmanager
    def stage(self, name: str, **meta: Any) -> Iterator[dict[str, Any]]:
        """Ghi lại một stage. `meta` (n_gaussians, sh_degree, resolution, ...) đi thẳng vào dòng JSONL.
        Yield một dict — bên trong stage có thể `rec["extra"]["foo"] = ...` để thêm số liệu."""
        rec: dict[str, Any] = {"stage": name, "extra": {}}
        rec.update({k: v for k, v in meta.items() if v is not None})

        if self.sync_cuda:
            torch.cuda.synchronize()
        if _TORCH_CUDA:
            torch.cuda.reset_peak_memory_stats()
            rec["vram_alloc_start_mb"] = round(torch.cuda.memory_allocated() / MB, 1)
        try:                                  # reset mốc VmHWM để đo đỉnh THEO STAGE
            from src.memory import counters as _c
            _c.reset_vm_hwm()
        except Exception:
            pass
        rec["rss_start_gb"] = round(_rss_gb() or 0, 3)
        rec["vram_device_start_mb"] = round(_vram_device_mb() or 0, 1)

        sampler = _Sampler(self.hz)
        sampler.start()
        t = time.monotonic()      # phải khớp với monotonic ở cuối stage, nếu không wall_s là rác
        err: BaseException | None = None
        try:
            yield rec
        except BaseException as e:  # ghi lại rồi ném tiếp — OOM cũng là data
            err = e
            raise
        finally:
            if self.sync_cuda:
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
            sampler.stop()
            rec["wall_s"] = round(time.monotonic() - t, 3)   # monotonic: đồng hồ tường nhảy lùi
                                                             # tới 1.8 s trong run truck (WSL2 resync)
            rec["rss_end_gb"] = round(_rss_gb() or 0, 3)
            # ĐỈNH CHÍNH XÁC từ bộ đếm kernel. Lấy mẫu bỏ sót 99.9% một đỉnh 400 MB ở 2 Hz
            # trên chính máy này (src/memory/counters.py). Sampler chỉ còn để lấy HÌNH DẠNG.
            try:
                from src.memory import counters as _c
                exact = _c.vm_hwm_gb()
                if exact is not None:
                    rec["rss_peak_gb"] = round(exact, 3)
                    rec["rss_peak_src"] = "VmHWM"
                    rec["rss_peak_sampled_gb"] = round(sampler.rss_peak, 3)
                    snap = _c.snapshot()
                    rec["rss_anon_end_gb"] = snap.get("rss_anon_gb")   # nhu cầu THẬT
                    rec["rss_file_end_gb"] = snap.get("rss_file_gb")   # page cache, evict được
                    rec["cgroup_peak_gb"] = snap.get("cgroup_peak_gb")
                else:
                    raise RuntimeError
            except Exception:
                rec["rss_peak_gb"] = round(max(sampler.rss_peak, rec["rss_end_gb"], rec["rss_start_gb"]), 3)
                rec["rss_peak_src"] = "sampled (KHÔNG tin cậy cho đỉnh)"
            rec["ram_avail_min_gb"] = round(sampler.ram_avail_min, 3) if sampler.ram_avail_min != float("inf") else None
            rec["swap_used_max_gb"] = round(sampler.swap_max, 3)
            rec["vram_device_peak_mb"] = round(max(sampler.vram_dev_peak, rec["vram_device_start_mb"]), 1)
            rec["sampler_n"] = sampler.samples
            if _TORCH_CUDA:
                rec["vram_alloc_peak_mb"] = round(torch.cuda.max_memory_allocated() / MB, 1)
                rec["vram_reserved_peak_mb"] = round(torch.cuda.max_memory_reserved() / MB, 1)
                rec["vram_alloc_end_mb"] = round(torch.cuda.memory_allocated() / MB, 1)
            if err is not None:
                rec["error"] = f"{type(err).__name__}: {str(err)[:200]}"
                rec["oom"] = "out of memory" in str(err).lower() or isinstance(err, MemoryError)
            self.stages.append(rec)
            self._f.write(json.dumps(rec) + "\n")
            if self.verbose:
                print(f"[prof] {name:<16} {rec['wall_s']:7.1f}s | "
                      f"RSS peak {rec['rss_peak_gb']:.2f} GB | "
                      f"VRAM alloc {rec.get('vram_alloc_peak_mb', '-')} / device {rec['vram_device_peak_mb']} MB"
                      + (f" | ERROR {rec['error']}" if err else ""), flush=True)

    # ------------------------------------------------------------------
    def close(self) -> dict[str, Any]:
        footer = {
            "type": "footer",
            "wall_total_s": round(time.time() - self._t0, 2),
            "n_stages": len(self.stages),
            "rss_peak_gb": max((s.get("rss_peak_gb", 0) for s in self.stages), default=None),
            "rss_peak_stage": max(self.stages, key=lambda s: s.get("rss_peak_gb", 0))["stage"] if self.stages else None,
            "vram_device_peak_mb": max((s.get("vram_device_peak_mb", 0) for s in self.stages), default=None),
            "vram_device_peak_stage": max(self.stages, key=lambda s: s.get("vram_device_peak_mb", 0))["stage"] if self.stages else None,
            "vram_alloc_peak_mb": max((s.get("vram_alloc_peak_mb") or 0 for s in self.stages), default=None),
            "any_oom": any(s.get("oom") for s in self.stages),
            "swap_touched": any((s.get("swap_used_max_gb") or 0) > 0.05 for s in self.stages),
        }
        self._f.write(json.dumps(footer) + "\n")
        self._f.close()
        if self.verbose:
            print(f"[prof] DONE  RSS peak {footer['rss_peak_gb']} GB @ {footer['rss_peak_stage']} | "
                  f"VRAM device peak {footer['vram_device_peak_mb']} MB @ {footer['vram_device_peak_stage']}"
                  + ("  ⚠ SWAP TOUCHED" if footer["swap_touched"] else ""), flush=True)
        return footer


# ----------------------------------------------------------------------
def read_stages(path: str) -> tuple[dict, list[dict], dict | None]:
    """Đọc lại một file stages.jsonl → (header, stages, footer)."""
    header, stages, footer = {}, [], None
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        if d.get("type") == "header":
            header = d
        elif d.get("type") == "footer":
            footer = d
        else:
            stages.append(d)
    return header, stages, footer


if __name__ == "__main__":  # smoke test: python -m src.memory.profiler
    import tempfile
    out = os.path.join(tempfile.gettempdir(), "prof_smoke.jsonl")
    p = StageProfiler(out, run_id="smoke")
    with p.stage("alloc_cpu", n=1):
        x = bytearray(200 * 2**20)  # 200 MB
        time.sleep(0.5)
    del x
    if _TORCH_CUDA:
        with p.stage("alloc_gpu", n=2) as rec:
            y = torch.empty(64 * 2**20, dtype=torch.float32, device="cuda")  # 256 MB
            rec["extra"]["elems"] = y.numel()
            torch.cuda.synchronize()
            time.sleep(0.5)
        del y
    print(json.dumps(p.close(), indent=2))
    print("->", out)
