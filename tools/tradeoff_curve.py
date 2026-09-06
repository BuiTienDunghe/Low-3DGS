#!/usr/bin/env python3
"""
Dựng ĐƯỜNG CONG ĐÁNH ĐỔI và vẽ HAI đường cạnh nhau:

  đường "ngây thơ"   : chất lượng so với checkpoint gốc (21.8157)
                       <- đây là cách gần như mọi bài nén 3DGS báo cáo
  đường "trung thực" : chất lượng so với ĐỐI CHỨNG (không nén, cùng 5000 bước)
                       <- đã trừ phần "luyện thêm"

Khoảng cách giữa hai đường là hằng số +0.3587 dB (EXP-018) — và đó chính là
lượng công của "luyện thêm" bị tính nhầm thành công của "nén".

Điểm đáng tìm: tỉ lệ nén mà đường NGÂY THƠ còn nói "miễn phí" (>= 0) trong khi
đường TRUNG THỰC đã âm. Trong dải đó, kết luận "nén không mất gì" là ảo giác.

Dùng: python tools/tradeoff_curve.py [--json out.json]
"""
import csv, json, os, statistics as st, sys

PROJ = "/mnt/d/low-3DGS"
SWEEP = os.path.join(PROJ, "experiments", "sweep_train.csv")
REPEATS = os.path.join(PROJ, "experiments", "repeats_train.csv")
N_BASE = 1026508
METRICS = [("psnr", "PSNR", 4, +1), ("ssim", "SSIM", 5, +1), ("lpips", "LPIPS", 5, -1)]
# dấu +1: cao hơn = tốt hơn.  -1: thấp hơn = tốt hơn.


def ply_bytes(n):
    """Kích thước .ply INRIA: 62 thuộc tính float32 + header.
    header = 1525 + số chữ số của N (đã kiểm chứng: N=1026508 -> 1532, N=349013 -> 1531)."""
    return n * 248 + 1525 + len(str(n))


def load():
    pts, ctrl, base = {}, {m: [] for m, _, _, _ in METRICS}, {}
    if os.path.isfile(REPEATS):
        for r in csv.DictReader(open(REPEATS)):
            it, occ = int(r["iteration"]), int(r["occurrence"])
            if r["arm"] == "control" and it == 35000:
                for m, _, _, _ in METRICS:
                    ctrl[m].append(float(r[m]))
            if it == 30001 and occ == 0:
                for m, _, _, _ in METRICS:
                    base[m] = float(r[m])
            # điểm 0.66 lấy từ loạt lặp, seed 0, cùng mọi cờ
            if r["arm"] == "prune" and int(r["seed"]) == 0:
                d = pts.setdefault(0.66, {"n": int(r["n_gauss"]), "seeds": {0}})
                if it == 35000:
                    d["final"] = {m: float(r[m]) for m, _, _, _ in METRICS}
                elif it == 30001 and occ == 1:
                    d["damaged"] = {m: float(r[m]) for m, _, _, _ in METRICS}
    if os.path.isfile(SWEEP):
        for r in csv.DictReader(open(SWEEP)):
            # Đường cong là seed 0. EXP-021 ghi thêm seed 1/2 ở mức 50%/60% vào cùng file;
            # không lọc thì dòng sau ghi đè dòng trước và hình lặng lẽ đổi sang seed khác.
            if int(r["seed"]) != 0:
                continue
            p = float(r["pct"]); it, occ = int(r["iteration"]), int(r["occurrence"])
            d = pts.setdefault(p, {"n": int(r["n_gauss"]), "seeds": {int(r["seed"])}})
            if it == 35000:
                d["final"] = {m: float(r[m]) for m, _, _, _ in METRICS}
            elif it == 30001 and occ == 1:
                d["damaged"] = {m: float(r[m]) for m, _, _, _ in METRICS}
    return pts, {m: st.mean(v) for m, v in ctrl.items() if v}, base


def main():
    pts, ctrl, base = load()
    pts = {p: d for p, d in sorted(pts.items()) if "final" in d}
    if not pts:
        print("chưa có dữ liệu quét — loạt chạy chưa xong."); return
    if not ctrl:
        print("thiếu dữ liệu đối chứng (repeats_train.csv)"); return

    n_ctrl = len([1 for _ in open(REPEATS)])
    print("=" * 100)
    print(f"ĐƯỜNG CONG ĐÁNH ĐỔI — scene train, seed 0, {len(pts)} điểm")
    print(f"  gốc     : {base['psnr']:.4f} dB · {N_BASE:,} Gaussian · {ply_bytes(N_BASE)/1e6:.2f} MB")
    print(f"  đối chứng (không nén, +5000 bước, trung bình 3 seed): {ctrl['psnr']:.4f} dB")
    print(f"  => riêng 'luyện thêm' đã cho {ctrl['psnr']-base['psnr']:+.4f} dB")
    print("=" * 100)

    hdr = (f"{'cắt':>5s} {'N còn lại':>11s} {'MB':>7s} {'nén':>6s} │ "
           f"{'sau cắt':>8s} {'phục hồi':>9s} │ {'NGÂY THƠ':>9s} {'TRUNG THỰC':>11s} │ ảo giác?")
    print("\n--- PSNR ---")
    print(hdr); print("-" * len(hdr))
    illusion = []
    for p, d in pts.items():
        n = d["n"]; b = ply_bytes(n)
        fin = d["final"]["psnr"]; dam = d.get("damaged", {}).get("psnr")
        naive = fin - base["psnr"]; honest = fin - ctrl["psnr"]
        flag = ""
        if naive >= 0 > honest:
            flag = "◄ CÓ"; illusion.append(p)
        print(f"{p*100:4.0f}% {n:11,d} {b/1e6:7.2f} {ply_bytes(N_BASE)/b:5.2f}x │ "
              f"{dam:8.4f} {fin:9.4f} │ {naive:+9.4f} {honest:+11.4f} │ {flag}")

    for m, lab, nd, sign in METRICS[1:]:
        print(f"\n--- {lab} ---")
        h2 = f"{'cắt':>5s} {'sau cắt':>9s} {'phục hồi':>9s} │ {'NGÂY THƠ':>10s} {'TRUNG THỰC':>11s}"
        print(h2); print("-" * len(h2))
        for p, d in pts.items():
            fin = d["final"][m]; dam = d.get("damaged", {}).get(m)
            print(f"{p*100:4.0f}% {dam:9.{nd}f} {fin:9.{nd}f} │ "
                  f"{fin-base[m]:+10.{nd}f} {fin-ctrl[m]:+11.{nd}f}")

    print("\n" + "=" * 100)
    print("CÁCH ĐỌC")
    print("=" * 100)
    if illusion:
        lo, hi = min(illusion) * 100, max(illusion) * 100
        print(f"  ► VÙNG ẢO GIÁC: cắt {lo:.0f}%–{hi:.0f}%.")
        print("    Ở đây cách so ngây thơ (với checkpoint gốc) cho ra số DƯƠNG — tức là")
        print("    'nén còn làm model tốt hơn' — trong khi so với đối chứng thì đã ÂM.")
        print("    Toàn bộ 'cải thiện' đó là công của 5000 bước luyện thêm, không phải của nén.")
    else:
        print("  Không có điểm nào mà hai cách so cho kết luận trái dấu.")
    ok = [p for p, d in pts.items() if d["final"]["psnr"] - ctrl["psnr"] >= -0.05]
    print(f"  ► Cắt được mà gần như không mất gì (trong 0.05 dB so với đối chứng): "
          + (", ".join(f"{p*100:.0f}%" for p in ok) if ok else "không có điểm nào"))

    if "--json" in sys.argv:
        out = sys.argv[sys.argv.index("--json") + 1]
        json.dump({"baseline": base, "control": ctrl, "n_base": N_BASE,
                   "points": [{"pct": p, "n": d["n"], "bytes": ply_bytes(d["n"]),
                               "final": d["final"], "damaged": d.get("damaged")}
                              for p, d in pts.items()]},
                  open(out, "w"), indent=1)
        print(f"\nđã ghi {out}")


if __name__ == "__main__":
    main()
