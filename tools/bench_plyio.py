#!/usr/bin/env python3
"""
tools/bench_plyio.py — Đo đọc/ghi .ply trên ĐÚNG code sẽ chạy, và VERIFY gsply có tương đương không.

Thay thế `measure_ply_load.py`, vốn đo một bản float32 tự viết rồi bị báo cáo nhầm như code INRIA
(EXP-003 đã rút lại). Ở đây gọi thẳng `GaussianModel.load_ply` / `.save_ply` của repo.

Ba khác biệt so với công cụ cũ:
  1. ĐO BẰNG BỘ ĐẾM KERNEL (VmHWM), không lấy mẫu. Lấy mẫu 2 Hz bỏ sót 99.9% một đỉnh 400 MB
     trên chính máy này (src/memory/counters.py).
  2. CÓ TORCH đã import và CUDA đã khởi tạo — `load_ply` đẩy tensor lên GPU, bản đo cũ không có.
  3. Tách RssAnon (nhu cầu thật) khỏi RssFile (page cache của mmap, kernel evict được).
     "Peak RSS" gộp cả hai là báo quá tay.

VERIFY (claim đang bị nghi ngờ): tài liệu ghi "gsply verify torch.equal bitwise trên cả 6 tensor"
nhưng KHÔNG script nào làm việc đó. Ở đây làm thật, và in kết quả — kể cả khi nó bác bỏ claim.

ĐƠN VỊ: bộ nhớ MiB/GiB (2^20/2^30) · file MB (10^6). Không trộn.

Dùng:
    python tools/bench_plyio.py --scene truck --op load_inria
    python tools/bench_plyio.py --all                      # mọi scene × mọi op
    python tools/bench_plyio.py --scene drjohnson --op save_inria --cap 8G
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

MIB = 1024.0 ** 2
GIB = 1024.0 ** 3
MB = 1e6
PROJ = "/mnt/d/low-3DGS"
REPO = os.path.expanduser("~/l3dgs/third_party/gaussian-splatting-pup")
MODELS = f"{PROJ}/datasets/pretrained/models"
SCENES = ["train", "truck", "playroom", "drjohnson"]
OPS = ["load_inria", "load_gsply", "save_inria", "save_gsply", "verify_gsply"]


def ply_path(scene: str, it: int = 30000) -> str:
    return f"{MODELS}/{scene}/point_cloud/iteration_{it}/point_cloud.ply"


# =============================================================================
# CHILD
# =============================================================================
def child(args) -> int:
    sys.path.insert(0, PROJ)
    from src.memory import counters

    import torch                                   # phải import TRƯỚC extension
    torch.zeros(1, device="cuda"); torch.cuda.synchronize()

    sys.path.insert(0, REPO)
    os.chdir(REPO)

    src = ply_path(args.scene)
    file_bytes = os.path.getsize(src)
    tmpdir = tempfile.mkdtemp(prefix="plyio_")
    dst = os.path.join(tmpdir, "out.ply")

    counters.reset_vm_hwm()
    base = counters.snapshot()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.monotonic()
    extra: dict = {}
    err = None

    try:
        if args.op == "load_inria":
            from scene import GaussianModel
            g = GaussianModel(3)
            g.load_ply(src)
            torch.cuda.synchronize()
            extra["n"] = int(g.get_xyz.shape[0])
            extra["device"] = str(g.get_xyz.device)

        elif args.op == "load_gsply":
            import gsply
            d = gsply.plyread(src)
            extra["n"] = int(d.means.shape[0])
            extra["fields"] = [k for k in ("means", "scales", "quats", "opacities", "sh0", "shN")
                               if getattr(d, k, None) is not None]

        elif args.op in ("save_inria", "save_gsply"):
            from scene import GaussianModel
            g = GaussianModel(3)
            g.load_ply(src)
            torch.cuda.synchronize()
            n = int(g.get_xyz.shape[0])
            counters.reset_vm_hwm()                  # chỉ đo phần GHI
            after_load = counters.snapshot()
            t0 = time.monotonic()
            if args.op == "save_inria":
                g.save_ply(dst)
            else:
                import gsply
                import numpy as np
                # Shape do verify_gsply xác nhận (KHÔNG transpose — đó là lỗi ở bản trước):
                #   INRIA _features_dc   (N,1,3)  -> gsply sh0 (N,3)   : squeeze(1)
                #   INRIA _features_rest (N,15,3) -> gsply shN (N,15,3): giữ nguyên
                gsply.plywrite(
                    dst,
                    g._xyz.detach().cpu().numpy(),
                    scales=g._scaling.detach().cpu().numpy(),
                    quats=g._rotation.detach().cpu().numpy(),
                    opacities=g._opacity.detach().cpu().numpy().squeeze(-1),
                    sh0=g._features_dc.detach().squeeze(1).cpu().numpy(),
                    shN=g._features_rest.detach().cpu().numpy(),
                )
            extra["n"] = n
            extra["out_bytes"] = os.path.getsize(dst)
            extra["rss_anon_after_load_gb"] = after_load.get("rss_anon_gb")

        elif args.op == "verify_gsply":
            # CLAIM ĐANG KIỂM: gsply cho tensor bitwise-identical với load_ply của INRIA?
            import gsply
            import numpy as np
            from scene import GaussianModel
            g = GaussianModel(3)
            g.load_ply(src)
            torch.cuda.synchronize()
            d = gsply.plyread(src)
            checks = {}
            pairs = {
                "xyz":          (g._xyz,          d.means),
                "scaling":      (g._scaling,      d.scales),
                "rotation":     (g._rotation,     d.quats),
                "opacity":      (g._opacity,      d.opacities),
                "features_dc":  (g._features_dc,  d.sh0),
                "features_rest": (g._features_rest, d.shN),
            }
            for name, (a, b) in pairs.items():
                A = a.detach().cpu().numpy()
                B = np.asarray(b)
                rec = {"inria_shape": list(A.shape), "gsply_shape": list(B.shape)}
                Af, Bf = A.reshape(A.shape[0], -1), B.reshape(B.shape[0], -1)
                if Af.shape != Bf.shape:
                    rec["equal"] = False
                    rec["why"] = f"shape khac sau khi flatten: {Af.shape} vs {Bf.shape}"
                else:
                    eq = bool(np.array_equal(Af, Bf))
                    rec["equal"] = eq
                    if not eq:
                        # thử hoán vị (INRIA giu (N,3,K), gsply co the (N,K,3))
                        if A.ndim == 3 and B.ndim == 3 and A.shape[1] == B.shape[2]:
                            eq2 = bool(np.array_equal(A.transpose(0, 2, 1), B))
                            rec["equal_after_transpose"] = eq2
                        diff = np.abs(Af - Bf)
                        rec["max_abs_diff"] = float(diff.max())
                        rec["mean_abs_diff"] = float(diff.mean())
                        rec["n_differing"] = int((diff > 0).sum())
                checks[name] = rec
            extra["checks"] = checks
            extra["all_equal"] = all(
                c.get("equal") or c.get("equal_after_transpose") for c in checks.values())
            extra["n"] = int(g.get_xyz.shape[0])

    except BaseException as e:
        err = f"{type(e).__name__}: {str(e)[:300]}"

    wall = time.monotonic() - t0
    torch.cuda.synchronize()
    end = counters.snapshot()
    shutil.rmtree(tmpdir, ignore_errors=True)

    peak_hwm = end.get("vm_hwm_gb")
    start_rss = base.get("vm_rss_gb") or 0.0
    delta = (peak_hwm - start_rss) if peak_hwm else None

    out = {
        "scene": args.scene, "op": args.op,
        "file_bytes": file_bytes, "file_mb": round(file_bytes / MB, 2),
        "wall_s": round(wall, 2),
        "vm_hwm_gib": peak_hwm, "delta_gib": round(delta, 3) if delta else None,
        "x_file": round(delta * GIB / file_bytes, 2) if delta else None,
        "rss_anon_end_gib": end.get("rss_anon_gb"),
        "rss_file_end_gib": end.get("rss_file_gb"),
        "cgroup_peak_gib": end.get("cgroup_peak_gb"),
        "swap_gib": end.get("vm_swap_gb"),
        "vram_alloc_peak_mib": round(torch.cuda.max_memory_allocated() / MIB, 1),
        "vram_reserved_peak_mib": round(torch.cuda.max_memory_reserved() / MIB, 1),
        "error": err, **extra,
    }
    print("###JSON###" + json.dumps(out))
    return 1 if err else 0


# =============================================================================
# PARENT
# =============================================================================
def run(scene: str, op: str, cap: str | None) -> dict:
    cmd = [sys.executable, os.path.abspath(__file__), "--_child", "--scene", scene, "--op", op]
    if cap:
        cmd = ["systemd-run", "--user", "--scope", "-q",
               f"-pMemoryMax={cap}", "-pMemorySwapMax=0", "-pMemoryAccounting=yes"] + cmd
    t = time.monotonic()
    r = subprocess.run(cmd, capture_output=True, text=True)
    dur = time.monotonic() - t
    line = next((l for l in r.stdout.splitlines() if l.startswith("###JSON###")), None)
    if line:
        d = json.loads(line[len("###JSON###"):]); d["wall_total_s"] = round(dur, 1); return d
    tail = (r.stderr or r.stdout).strip().splitlines()[-4:]
    killed = r.returncode in (137, -9) or any("oom" in l.lower() or "Killed" in l for l in tail)
    return {"scene": scene, "op": op, "failed": True, "oom_killed": killed,
            "returncode": r.returncode, "error": "\n".join(tail)[:400], "wall_total_s": round(dur, 1)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", choices=SCENES)
    ap.add_argument("--op", choices=OPS)
    ap.add_argument("--cap", help="giới hạn cgroup, vd 8G — biến swap âm thầm thành OOM rõ ràng")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", default=f"{PROJ}/experiments/plyio/results.jsonl")
    ap.add_argument("--_child", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args._child:
        return child(args)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    jobs = ([(s, o) for s in SCENES for o in OPS] if args.all
            else [(args.scene, args.op)])
    if not args.all and (not args.scene or not args.op):
        ap.error("cần --scene và --op, hoặc --all")

    print("ĐƠN VỊ: bộ nhớ GiB (2^30) · file MB (10^6)")
    print(f"{'scene':<11}{'op':<14}{'file MB':>9}{'ΔRSS GiB':>10}{'×file':>7}"
          f"{'anon':>7}{'cache':>7}{'VRAM MiB':>10}{'giây':>7}  ghi chú")
    print("-" * 100)
    results = []
    for scene, op in jobs:
        d = run(scene, op, args.cap)
        results.append(d)
        with open(args.out, "a") as f:
            f.write(json.dumps(d) + "\n")
        if d.get("failed"):
            tag = "OOM-KILL" if d.get("oom_killed") else f"FAIL rc={d['returncode']}"
            print(f"{scene:<11}{op:<14}{'':>9}{tag:>10}  {d['error'].splitlines()[-1][:44] if d.get('error') else ''}")
            continue
        note = ""
        if op == "verify_gsply":
            note = "BITWISE KHỚP" if d.get("all_equal") else "KHÁC NHAU — xem JSON"
        elif d.get("error"):
            note = d["error"][:44]
        elif op.startswith("save") and d.get("out_bytes"):
            note = f"ra {d['out_bytes']/MB:.1f} MB"
        print(f"{scene:<11}{op:<14}{d['file_mb']:>9.1f}"
              f"{(d['delta_gib'] or 0):>10.3f}{(d['x_file'] or 0):>7.2f}"
              f"{(d.get('rss_anon_end_gib') or 0):>7.2f}{(d.get('rss_file_end_gib') or 0):>7.2f}"
              f"{d['vram_alloc_peak_mib']:>10.1f}{d['wall_s']:>7.1f}  {note}")
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
