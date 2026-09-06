#!/usr/bin/env python3
"""
tools/profile_gpu.py — Lấy mẫu GPU (NVML) + RAM (psutil) theo thời gian, ghi JSONL.

Đây là nguồn dữ liệu thô cho cost model C2: mỗi dòng là một snapshot đồng thời
của VRAM (device-level, cái quyết định OOM) và RAM (RSS của tiến trình + available toàn hệ).

Dùng:
    python tools/profile_gpu.py --out experiments/<run>/memory.jsonl            # đến khi Ctrl+C
    python tools/profile_gpu.py --out ... --pid 12345                          # đến khi PID kết thúc
    python tools/profile_gpu.py --out ... --duration 30                        # 30 giây
    python tools/profile_gpu.py --summary experiments/<run>/memory.jsonl       # tóm tắt đỉnh

Chạy được cả trên Windows (nvml.dll) lẫn WSL2 (/usr/lib/wsl/lib/libnvidia-ml.so.1).
Mọi trường không đọc được ghi null thay vì crash — profiler không được phép làm hỏng run.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None

try:
    import pynvml
except ImportError:  # pragma: no cover
    pynvml = None


# ---------------------------------------------------------------------------
# NVML helpers — mỗi cái tự nuốt lỗi và trả None
# ---------------------------------------------------------------------------
def _safe(fn, *a, default=None):
    try:
        return fn(*a)
    except Exception:
        return default


# NVML báo "GpuIdle" và "ApplicationsClocksSetting" như throttle reason, nhưng chúng KHÔNG phải
# throttle theo nghĩa ta quan tâm (nhiệt/nguồn). Lần đo gate đầu tiên báo "throttled 87-93%" trong khi
# nhiệt chỉ 63-67°C — toàn bộ là GpuIdle giữa các frame. Lọc ra, nếu không mọi bảng timing sẽ bị loại nhầm.
_BENIGN_THROTTLE = {"GpuIdle", "ApplicationsClocksSetting", "DisplayClockSetting"}


def _throttle_reasons(h) -> list[str] | None:
    """Lý do throttle THẬT (nhiệt/nguồn/HW), đã loại các lý do lành tính như GpuIdle."""
    getter = getattr(pynvml, "nvmlDeviceGetCurrentClocksEventReasons", None) or \
             getattr(pynvml, "nvmlDeviceGetCurrentClocksThrottleReasons", None)
    if getter is None:
        return None
    mask = _safe(getter, h)
    if mask is None:
        return None
    names = []
    for attr in dir(pynvml):
        if attr.startswith(("nvmlClocksEventReason", "nvmlClocksThrottleReason")) and \
           not attr.endswith(("None", "All")):
            bit = getattr(pynvml, attr)
            if isinstance(bit, int) and bit and (mask & bit):
                name = attr.split("Reason", 1)[-1]
                if name not in _BENIGN_THROTTLE:
                    names.append(name)
    return sorted(set(names))


def sample(h, proc: "psutil.Process | None") -> dict:
    mem = _safe(pynvml.nvmlDeviceGetMemoryInfo, h)
    util = _safe(pynvml.nvmlDeviceGetUtilizationRates, h)
    row = {
        "t": time.time(),
        "vram_used_mb": round(mem.used / 2**20, 1) if mem else None,
        "vram_total_mb": round(mem.total / 2**20, 1) if mem else None,
        "util_gpu": util.gpu if util else None,
        "util_mem": util.memory if util else None,
        "temp_c": _safe(pynvml.nvmlDeviceGetTemperature, h, pynvml.NVML_TEMPERATURE_GPU),
        "clock_sm_mhz": _safe(pynvml.nvmlDeviceGetClockInfo, h, pynvml.NVML_CLOCK_SM),
        "clock_max_sm_mhz": _safe(pynvml.nvmlDeviceGetMaxClockInfo, h, pynvml.NVML_CLOCK_SM),
        "power_w": (lambda p: round(p / 1000, 2) if p is not None else None)(
            _safe(pynvml.nvmlDeviceGetPowerUsage, h)),
        "throttle": _throttle_reasons(h),
    }
    if psutil:
        vm = psutil.virtual_memory()
        row["ram_avail_gb"] = round(vm.available / 2**30, 3)
        row["ram_used_pct"] = vm.percent
        sw = psutil.swap_memory()
        row["swap_used_gb"] = round(sw.used / 2**30, 3)
        if proc is not None:
            try:
                row["proc_rss_gb"] = round(proc.memory_info().rss / 2**30, 3)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                row["proc_rss_gb"] = None
    return row


# ---------------------------------------------------------------------------
def run(args) -> int:
    if pynvml is None:
        print("ERROR: pip install nvidia-ml-py", file=sys.stderr)
        return 1
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(args.gpu)
    name = _safe(pynvml.nvmlDeviceGetName, h)
    if isinstance(name, bytes):
        name = name.decode()
    drv = _safe(pynvml.nvmlSystemGetDriverVersion)
    if isinstance(drv, bytes):
        drv = drv.decode()

    proc = None
    if args.pid and psutil:
        try:
            proc = psutil.Process(args.pid)
        except psutil.NoSuchProcess:
            print(f"WARN: pid {args.pid} không tồn tại — chỉ lấy mẫu hệ thống", file=sys.stderr)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    period = 1.0 / args.hz
    t0 = time.time()
    n = 0
    peak = {"vram_used_mb": 0, "proc_rss_gb": 0.0, "temp_c": 0, "min_clock_sm_mhz": None}

    header = {"type": "header", "ts": datetime.now(timezone.utc).isoformat(), "gpu": name,
              "driver": drv, "hz": args.hz, "pid": args.pid, "platform": sys.platform}
    with open(args.out, "a", buffering=1) as f:          # line-buffered: ghi ngay, không mất khi crash
        f.write(json.dumps(header) + "\n")
        try:
            while True:
                row = sample(h, proc)
                f.write(json.dumps(row) + "\n")
                n += 1
                if row["vram_used_mb"] is not None:
                    peak["vram_used_mb"] = max(peak["vram_used_mb"], row["vram_used_mb"])
                if row.get("proc_rss_gb"):
                    peak["proc_rss_gb"] = max(peak["proc_rss_gb"], row["proc_rss_gb"])
                if row["temp_c"] is not None:
                    peak["temp_c"] = max(peak["temp_c"], row["temp_c"])
                c = row["clock_sm_mhz"]
                if c and (peak["min_clock_sm_mhz"] is None or c < peak["min_clock_sm_mhz"]):
                    peak["min_clock_sm_mhz"] = c
                if not args.quiet and n % max(1, int(args.hz * 5)) == 0:
                    print(f"[{time.time()-t0:7.1f}s] vram {row['vram_used_mb']} MB | "
                          f"rss {row.get('proc_rss_gb')} GB | avail {row.get('ram_avail_gb')} GB | "
                          f"{row['temp_c']}°C {row['clock_sm_mhz']} MHz | thr {row['throttle']}",
                          flush=True)
                if proc is not None and not proc.is_running():
                    break
                if args.duration and time.time() - t0 >= args.duration:
                    break
                time.sleep(period)
        except KeyboardInterrupt:
            pass
        finally:
            footer = {"type": "footer", "samples": n, "wall_s": round(time.time() - t0, 2), "peak": peak}
            f.write(json.dumps(footer) + "\n")
    pynvml.nvmlShutdown()
    print(json.dumps(footer, indent=2))
    return 0


def summarize(path: str) -> int:
    rows = [json.loads(l) for l in open(path) if l.strip()]
    data = [r for r in rows if "type" not in r]
    if not data:
        print("no samples")
        return 1
    def col(k):
        v = [r[k] for r in data if r.get(k) is not None]
        return v
    vram, rss, temp, clk = col("vram_used_mb"), col("proc_rss_gb"), col("temp_c"), col("clock_sm_mhz")
    maxclk = col("clock_max_sm_mhz")
    out = {
        "samples": len(data),
        "wall_s": round(data[-1]["t"] - data[0]["t"], 1),
        "vram_peak_mb": max(vram) if vram else None,
        "vram_mean_mb": round(sum(vram) / len(vram), 1) if vram else None,
        "rss_peak_gb": max(rss) if rss else None,
        "temp_peak_c": max(temp) if temp else None,
        "clock_min_mhz": min(clk) if clk else None,
        "clock_max_mhz": maxclk[0] if maxclk else None,
        "throttled_frac": round(sum(1 for r in data if r.get("throttle")) / len(data), 3),
    }
    if out["clock_min_mhz"] and out["clock_max_mhz"]:
        out["clock_min_frac_of_max"] = round(out["clock_min_mhz"] / out["clock_max_mhz"], 3)
    print(json.dumps(out, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", help="JSONL đầu ra (append)")
    ap.add_argument("--hz", type=float, default=2.0, help="tần số lấy mẫu (mặc định 2)")
    ap.add_argument("--pid", type=int, help="dừng khi PID này kết thúc; ghi RSS của nó")
    ap.add_argument("--duration", type=float, help="giây; dừng sau chừng này")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--summary", metavar="JSONL", help="chỉ tóm tắt file đã ghi rồi thoát")
    args = ap.parse_args()
    if args.summary:
        return summarize(args.summary)
    if not args.out:
        ap.error("--out là bắt buộc (hoặc dùng --summary)")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
