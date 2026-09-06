#!/usr/bin/env python3
"""
EXP-021 — Chốt ĐẦU GỐI: chi phí nén có thanh sai số ở ba mức 50% / 60% / 66%.

Ghép cặp THEO SEED (control seed k trừ prune seed k), vì EXP-018 cho thấy
cùng một seed tác động lên cả hai nhánh như nhau, nên lấy hiệu trong cùng seed
triệt tiêu ~2/3 nhiễu (0.055 -> 0.018 dB).

Nguồn: 50%/60% từ sweep_train.csv · 66% từ repeats_train.csv (arm=prune)
       đối chứng từ repeats_train.csv (arm=control)

Dùng: python tools/knee_errorbars.py
"""
import csv, math, os, statistics as st

PROJ = "/mnt/d/low-3DGS"
SWEEP = os.path.join(PROJ, "experiments", "sweep_train.csv")
REPEATS = os.path.join(PROJ, "experiments", "repeats_train.csv")
TCRIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}
METRICS = [("psnr", "PSNR", 4), ("ssim", "SSIM", 5), ("lpips", "LPIPS", 5)]
NOISE = {"psnr": 0.0181, "ssim": 0.00006, "lpips": 0.00015}   # EXP-018


def load():
    ctrl, prune = {}, {}          # {seed: {metric: v}} ; {pct: {seed: {...}}}
    if os.path.isfile(REPEATS):
        for r in csv.DictReader(open(REPEATS)):
            if int(r["iteration"]) != 35000 or int(r["occurrence"]) != 0:
                continue
            s = int(r["seed"]); vals = {m: float(r[m]) for m, _, _ in METRICS}
            if r["arm"] == "control":
                ctrl[s] = vals
            elif r["arm"] == "prune":
                prune.setdefault(0.66, {})[s] = vals
    if os.path.isfile(SWEEP):
        for r in csv.DictReader(open(SWEEP)):
            if int(r["iteration"]) != 35000 or int(r["occurrence"]) != 0:
                continue
            prune.setdefault(float(r["pct"]), {})[int(r["seed"])] = \
                {m: float(r[m]) for m, _, _ in METRICS}
    return ctrl, prune


def main():
    ctrl, prune = load()
    if not ctrl:
        print("thiếu dữ liệu đối chứng"); return

    print("=" * 92)
    print("ĐẦU GỐI CÓ THANH SAI SỐ — chi phí nén thật (ghép cặp theo seed)")
    print(f"  đối chứng: {len(ctrl)} seed · trung bình PSNR "
          f"{st.mean(v['psnr'] for v in ctrl.values()):.4f}")
    print("=" * 92)

    for m, lab, nd in METRICS:
        print(f"\n--- {lab} ---   (nhiễu ghép cặp EXP-018: {NOISE[m]})")
        hdr = f"  {'cắt':>5s} {'n':>3s} " + " ".join(f"{'s'+str(k):>10s}" for k in (0, 1, 2)) \
              + f" {'trung bình':>13s} {'KTC 95%':>22s}  kết luận"
        print(hdr); print("  " + "-" * (len(hdr) - 2))
        for pct in sorted(prune):
            if pct not in (0.50, 0.60, 0.66):
                continue
            seeds = sorted(set(prune[pct]) & set(ctrl))
            costs = [prune[pct][s][m] - ctrl[s][m] for s in seeds]
            cells = []
            for k in (0, 1, 2):
                cells.append(f"{prune[pct][k][m]-ctrl[k][m]:+10.{nd}f}"
                             if k in prune[pct] and k in ctrl else f"{'—':>10s}")
            if len(costs) >= 2:
                mu, sd = st.mean(costs), st.stdev(costs)
                h = TCRIT.get(len(costs) - 1, 2.0) * sd / math.sqrt(len(costs))
                ci = f"[{mu-h:+.{nd}f}, {mu+h:+.{nd}f}]"
                if mu - h <= 0 <= mu + h:
                    verdict = "MIỄN PHÍ (KTC chứa 0)"
                else:
                    verdict = "TỐN thật"
                avg = f"{mu:+.{nd}f}±{sd:.{nd}f}"
            else:
                ci, verdict, avg = "—", "chưa đủ seed", (f"{costs[0]:+.{nd}f}" if costs else "—")
            print(f"  {pct*100:4.0f}% {len(costs):>3d} " + " ".join(cells)
                  + f" {avg:>13s} {ci:>22s}  {verdict}")

    print("\n" + "=" * 92)
    print("CÁCH ĐỌC")
    print("=" * 92)
    free, costly, pending = [], [], []
    for pct in sorted(prune):
        if pct not in (0.50, 0.60, 0.66):
            continue
        seeds = sorted(set(prune[pct]) & set(ctrl))
        costs = [prune[pct][s]["psnr"] - ctrl[s]["psnr"] for s in seeds]
        if len(costs) < 2:
            pending.append(pct); continue
        mu, sd = st.mean(costs), st.stdev(costs)
        h = TCRIT.get(len(costs) - 1, 2.0) * sd / math.sqrt(len(costs))
        (free if mu - h <= 0 <= mu + h else costly).append(pct)
    fmt = lambda xs: ", ".join(f"{p*100:.0f}%" for p in xs) if xs else "không có"
    print(f"  Miễn phí (chưa phân biệt được với 0): {fmt(free)}")
    print(f"  Tốn thật                            : {fmt(costly)}")
    if pending:
        print(f"  Chưa đủ seed                        : {fmt(pending)}")
    if free and costly:
        print(f"\n  => ĐẦU GỐI nằm giữa {max(free)*100:.0f}% và {min(costly)*100:.0f}%.")
        print(f"     Khuyến nghị thực dụng: cắt tới {max(free)*100:.0f}%, đừng quá.")


if __name__ == "__main__":
    main()
