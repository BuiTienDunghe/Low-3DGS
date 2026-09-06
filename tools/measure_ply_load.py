#!/usr/bin/env python3
"""
tools/measure_ply_load.py — Đo chi phí RAM THẬT khi nạp checkpoint 3DGS `.ply`.

Trả lời câu hỏi trung tâm của PLAN.md §2: khi fine-tune, RAM có bind trước VRAM không?
Giả thuyết trong plan là `plyfile` + `GaussianModel.load_ply()` tạo ~3 bản sao → đỉnh RAM
gấp ~3× kích thước file. Chưa ai đo. File này đo.

Bốn đường nạp, cùng một file, mỗi cái một tiến trình con sạch (để đỉnh RSS không lẫn nhau):

  A. plyfile_full   — `PlyData.read()` rồi rút từng thuộc tính thành mảng float32 riêng
                      (đúng cách `GaussianModel.load_ply` của INRIA làm)
  B. plyfile_only   — chỉ `PlyData.read()`, không rút gì (tách chi phí của thư viện)
  C. numpy_direct   — tự đọc header rồi `np.fromfile` một phát vào structured array
  D. numpy_mmap     — `np.memmap` phần thân, chỉ copy các cột cần

Đo: đỉnh RSS (lấy mẫu nền 20 Hz + tracemalloc), thời gian, RSS cuối.
KHÔNG cần torch/CUDA — chạy được trước khi toolchain xong, trên Windows lẫn WSL.

Dùng:
    python tools/measure_ply_load.py <file.ply> [--methods A,B,C,D] [--out results.json]
    python tools/measure_ply_load.py --scan datasets/pretrained/models   # mọi point_cloud.ply
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import threading
import time

GB = float(2**30)
MB = float(2**20)

# 3DGS .ply: x,y,z, nx,ny,nz (bỏ), f_dc_0..2, f_rest_0..44, opacity, scale_0..2, rot_0..3
ATTR_GROUPS = {
    "xyz":        ["x", "y", "z"],
    "opacity":    ["opacity"],
    "scaling":    [f"scale_{i}" for i in range(3)],
    "rotation":   [f"rot_{i}" for i in range(4)],
    "f_dc":       [f"f_dc_{i}" for i in range(3)],
    "f_rest":     [f"f_rest_{i}" for i in range(45)],
}


def read_header(path: str) -> tuple[int, list[str], int, str]:
    """(n_vertex, property_names, header_bytes, format) — không nạp thân file."""
    with open(path, "rb") as f:
        raw = f.read(16384)
    end = raw.find(b"end_header")
    if end < 0:
        raise ValueError("không thấy end_header trong 16 KB đầu")
    nl = raw.find(b"\n", end)
    header_bytes = nl + 1
    txt = raw[:end].decode("ascii", "replace")
    n = int(re.search(r"element vertex (\d+)", txt).group(1))
    props = re.findall(r"property float (\S+)", txt)
    fmt = re.search(r"format (\S+)", txt).group(1)
    return n, props, header_bytes, fmt


# ---------------------------------------------------------------- samplers
class PeakRSS(threading.Thread):
    def __init__(self, hz: float = 20.0):
        super().__init__(daemon=True)
        import psutil
        self.p = psutil.Process()
        self.period = 1.0 / hz
        self.peak = self.p.memory_info().rss
        # KHÔNG đặt tên `_stop`: threading.Thread._stop là method nội bộ mà join() gọi;
        # gán Event() vào đó làm join() ném "TypeError: 'Event' object is not callable".
        self._stop_evt = threading.Event()
        self.n = 0

    def run(self):
        while not self._stop_evt.is_set():
            try:
                self.peak = max(self.peak, self.p.memory_info().rss)
            except Exception:
                pass
            self.n += 1
            self._stop_evt.wait(self.period)

    def stop(self):
        self._stop_evt.set()
        self.join(timeout=2)
        return self.peak


# ---------------------------------------------------------------- methods
def m_plyfile_full(path):
    """Bắt chước GaussianModel.load_ply của INRIA."""
    import numpy as np
    from plyfile import PlyData
    ply = PlyData.read(path)
    el = ply.elements[0]
    out = {}
    out["xyz"] = np.stack([np.asarray(el["x"]), np.asarray(el["y"]), np.asarray(el["z"])], axis=1)
    out["opacity"] = np.asarray(el["opacity"])[..., np.newaxis]
    names = {p.name for p in el.properties}
    for key in ("scaling", "rotation", "f_dc", "f_rest"):
        cols = [c for c in ATTR_GROUPS[key] if c in names]
        if not cols:
            continue
        arr = np.zeros((el.count, len(cols)), dtype=np.float32)
        for i, c in enumerate(cols):
            arr[:, i] = np.asarray(el[c])
        out[key] = arr
    return {k: v.shape for k, v in out.items()}, sum(v.nbytes for v in out.values())


def m_plyfile_only(path):
    from plyfile import PlyData
    ply = PlyData.read(path)
    el = ply.elements[0]
    return {"count": el.count, "props": len(el.properties)}, el.data.nbytes


def m_numpy_direct(path):
    import numpy as np
    n, props, hb, fmt = read_header(path)
    if "little_endian" not in fmt:
        raise NotImplementedError(f"format {fmt} chưa hỗ trợ")
    dt = np.dtype([(p, "<f4") for p in props])
    with open(path, "rb") as f:
        f.seek(hb)
        data = np.fromfile(f, dtype=dt, count=n)
    out = {}
    names = set(props)
    for key, cols in ATTR_GROUPS.items():
        cols = [c for c in cols if c in names]
        if cols:
            out[key] = np.stack([data[c] for c in cols], axis=1)
    del data
    return {k: v.shape for k, v in out.items()}, sum(v.nbytes for v in out.values())


def m_numpy_mmap(path):
    import numpy as np
    n, props, hb, fmt = read_header(path)
    dt = np.dtype([(p, "<f4") for p in props])
    mm = np.memmap(path, dtype=dt, mode="r", offset=hb, shape=(n,))
    out = {}
    names = set(props)
    for key, cols in ATTR_GROUPS.items():
        cols = [c for c in cols if c in names]
        if cols:
            a = np.empty((n, len(cols)), dtype=np.float32)
            for i, c in enumerate(cols):
                a[:, i] = mm[c]          # copy đúng cột cần, không giữ cả file
            out[key] = a
    del mm
    return {k: v.shape for k, v in out.items()}, sum(v.nbytes for v in out.values())


def m_chunked(path, chunk=500_000):
    """E. Đọc theo KHỐI HÀNG: mỗi khối rút đủ mọi thuộc tính rồi 'bàn giao' và giải phóng.

    Mô phỏng `src/loading/ply_stream.py`: trong thực tế mỗi khối được `.cuda()` ngay rồi xoá,
    nên đỉnh RAM ≈ một khối chứ không phải cả model. Ở đây chỉ cộng dồn checksum + đếm byte
    để chứng minh giới hạn dưới của đỉnh RAM mà không giữ output.
    """
    import numpy as np
    n, props, hb, fmt = read_header(path)
    dt = np.dtype([(p, "<f4") for p in props])
    names = set(props)
    groups = {k: [c for c in v if c in names] for k, v in ATTR_GROUPS.items()}
    groups = {k: v for k, v in groups.items() if v}
    total, acc = 0, 0.0
    itemsize = dt.itemsize
    with open(path, "rb") as f:
        f.seek(hb)
        done = 0
        while done < n:
            m = min(chunk, n - done)
            block = np.fromfile(f, dtype=dt, count=m)       # chỉ khối này nằm trong RAM
            for key, cols in groups.items():
                a = np.stack([block[c] for c in cols], axis=1)
                total += a.nbytes
                acc += float(a[0, 0])                        # chạm dữ liệu để không bị tối ưu đi
                del a
            del block
            done += m
    return {"n": n, "chunks": (n + chunk - 1) // chunk, "chunk": chunk, "checksum": round(acc, 3)}, total


METHODS = {"A": ("plyfile_full", m_plyfile_full), "B": ("plyfile_only", m_plyfile_only),
           "C": ("numpy_direct", m_numpy_direct), "D": ("numpy_mmap", m_numpy_mmap),
           "E": ("chunked_500k", m_chunked)}


# ---------------------------------------------------------------- child
def run_one(path: str, key: str) -> dict:
    import tracemalloc
    import psutil
    name, fn = METHODS[key]
    proc = psutil.Process()
    rss0 = proc.memory_info().rss
    sampler = PeakRSS()
    sampler.start()
    tracemalloc.start()
    t = time.time()
    err = None
    shapes, useful = None, None
    try:
        shapes, useful = fn(path)
    except BaseException as e:
        err = f"{type(e).__name__}: {e}"
    wall = time.time() - t
    _, tm_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak = sampler.stop()
    rss1 = proc.memory_info().rss
    fsz = os.path.getsize(path)
    return {
        "method": name, "key": key, "file_mb": round(fsz / MB, 1),
        "wall_s": round(wall, 2),
        "rss_start_gb": round(rss0 / GB, 3),
        "rss_peak_gb": round(peak / GB, 3),
        "rss_end_gb": round(rss1 / GB, 3),
        "peak_over_file": round((peak - rss0) / fsz, 2),
        "tracemalloc_peak_gb": round(tm_peak / GB, 3),
        "useful_bytes_gb": round(useful / GB, 3) if useful else None,
        "shapes": {k: list(v) for k, v in shapes.items()} if isinstance(shapes, dict) and shapes and isinstance(next(iter(shapes.values())), tuple) else shapes,
        "error": err,
    }


# ---------------------------------------------------------------- parent
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ply", nargs="?", help="đường dẫn point_cloud.ply")
    ap.add_argument("--scan", help="thư mục: đo mọi point_cloud.ply tìm được")
    ap.add_argument("--methods", default="A,B,C,D")
    ap.add_argument("--out", help="ghi JSON")
    ap.add_argument("--_child", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args._child:                                  # tiến trình con: đo 1 method rồi in JSON
        print(json.dumps(run_one(args.ply, args._child)))
        return 0

    files = []
    if args.scan:
        files = sorted(glob.glob(os.path.join(args.scan, "**", "point_cloud.ply"), recursive=True))
    elif args.ply:
        files = [args.ply]
    if not files:
        ap.error("cần <file.ply> hoặc --scan <dir>")

    keys = [k.strip().upper() for k in args.methods.split(",")]
    results = []
    print(f"{'file':<34} {'N':>10} {'MB':>7} {'method':<14} {'peak GB':>8} {'×file':>6} {'useful':>7} {'s':>6}")
    print("-" * 100)
    for f in files:
        n, props, _, _ = read_header(f)
        label = "/".join(f.replace("\\", "/").split("/")[-4:-2])
        for k in keys:
            r = subprocess.run([sys.executable, __file__, f, "--_child", k],
                               capture_output=True, text=True)
            if r.returncode != 0:
                print(f"{label:<34} {n:>10,} {'':>7} {METHODS[k][0]:<14}  CHILD FAILED: {r.stderr.strip()[:60]}")
                continue
            d = json.loads(r.stdout.strip().splitlines()[-1])
            d["file"] = f
            d["n_gaussians"] = n
            d["n_props"] = len(props)
            results.append(d)
            if d["error"]:
                print(f"{label:<34} {n:>10,} {d['file_mb']:>7.0f} {d['method']:<14}  ERROR: {d['error'][:50]}")
            else:
                print(f"{label:<34} {n:>10,} {d['file_mb']:>7.0f} {d['method']:<14} "
                      f"{d['rss_peak_gb']:>8.2f} {d['peak_over_file']:>6.2f} "
                      f"{d['useful_bytes_gb']:>7.2f} {d['wall_s']:>6.1f}")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        json.dump(results, open(args.out, "w"), indent=2)
        print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
