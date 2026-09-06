#!/usr/bin/env python3
"""tools/show_plyio.py — In kết quả bench_plyio.py. Tách khỏi heredoc để khỏi lỗi escaping."""
from __future__ import annotations
import json, sys, os

MB = 1e6
path = sys.argv[1] if len(sys.argv) > 1 else "experiments/plyio/results.jsonl"
rows = [json.loads(l) for l in open(path) if l.strip()]

# giữ bản mới nhất cho mỗi (scene, op)
latest: dict[tuple, dict] = {}
for r in rows:
    latest[(r.get("scene"), r.get("op"))] = r

print("ĐƠN VỊ: bộ nhớ GiB (2^30) · file MB (10^6)\n")
print(f"{'scene':<11}{'op':<14}{'file MB':>9}{'ΔRSS':>8}{'×file':>7}"
      f"{'anon':>7}{'cache':>7}{'VRAM MiB':>10}{'giây':>7}  ghi chú")
print("-" * 98)
order = ["train", "truck", "playroom", "drjohnson"]
ops = ["load_inria", "load_gsply", "save_inria", "save_gsply", "verify_gsply"]
for s in order:
    for o in ops:
        d = latest.get((s, o))
        if not d:
            continue
        if d.get("failed"):
            tag = "OOM-KILL" if d.get("oom_killed") else f"FAIL rc={d.get('returncode')}"
            print(f"{s:<11}{o:<14}{'':>9}{tag:>8}   {(d.get('error') or '').splitlines()[-1][:44] if d.get('error') else ''}")
            continue
        note = ""
        if o == "verify_gsply":
            note = "BITWISE KHỚP" if d.get("all_equal") else "KHÁC — xem chi tiết"
        elif d.get("error"):
            note = "ERR " + d["error"][:40]
        elif o.startswith("save") and d.get("out_bytes"):
            note = f"ra {d['out_bytes']/MB:.1f} MB ({d['out_bytes']/d['file_bytes']*100:.0f}% vào)"
        print(f"{s:<11}{o:<14}{d.get('file_mb') or 0:>9.1f}"
              f"{(d.get('delta_gib') or 0):>8.3f}{(d.get('x_file') or 0):>7.2f}"
              f"{(d.get('rss_anon_end_gib') or 0):>7.2f}{(d.get('rss_file_end_gib') or 0):>7.2f}"
              f"{(d.get('vram_alloc_peak_mib') or 0):>10.1f}{(d.get('wall_s') or 0):>7.1f}  {note}")

# --- chi tiết verify ---
for s in order:
    d = latest.get((s, "verify_gsply"))
    if not d or d.get("failed") or not d.get("checks"):
        continue
    print(f"\n=== verify gsply vs INRIA — {s} (N={d.get('n',0):,}) — all_equal={d.get('all_equal')} ===")
    for k, v in d["checks"].items():
        eq, tr = v.get("equal"), v.get("equal_after_transpose")
        st = "KHỚP" if eq else ("KHỚP sau transpose" if tr else "KHÁC")
        print(f"  {k:<15} INRIA {str(v['inria_shape']):<16} gsply {str(v['gsply_shape']):<16} {st}")
        if not eq and not tr:
            print(f"      max_diff={v.get('max_abs_diff')} n_khác={v.get('n_differing')} {v.get('why','')}")

# --- tổng hợp hệ số, kèm xuất xứ ---
print("\n=== HỆ SỐ ĐO ĐƯỢC (dùng cho gen_budget.py — KHÔNG ngoại suy sang code khác) ===")
for o in ("load_inria", "load_gsply", "save_inria", "save_gsply"):
    xs = [(s, latest[(s, o)]["x_file"]) for s in order
          if (s, o) in latest and latest[(s, o)].get("x_file")]
    if xs:
        vals = [x for _, x in xs]
        print(f"  {o:<12} {min(vals):.2f}–{max(vals):.2f}× file   "
              f"({', '.join(f'{s}:{x:.2f}' for s, x in xs)})")
