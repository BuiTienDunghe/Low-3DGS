#!/usr/bin/env python3
"""
Gộp các lần chạy lặp thành THANH SAI SỐ, và tách hai loại biến thiên:

  (1) NHÂN CUDA KHÔNG TẤT ĐỊNH — so seed 0 chạy lại với seed 0 lần đầu.
      Cùng seed, cùng mã, cùng dữ liệu. Lệch bao nhiêu thì đó là sàn nhiễu phần cứng.
  (2) BIẾN THIÊN THEO SEED — độ lệch chuẩn qua các seed 0/1/2.
      Đây mới là thanh sai số dùng để nói "-0.106 dB có khác 0 không".

Câu hỏi cuối: khoảng tin cậy của CHI PHÍ NÉN (prune - control) có chứa 0 không?

Dùng: python tools/aggregate_repeats.py
"""
import csv, math, os, statistics as st

PROJ = "/mnt/d/low-3DGS"
CSV = os.path.join(PROJ, "experiments", "repeats_train.csv")
ORIG = {  # EXP-016/017 — cũng là seed 0, chạy trước khi patch seed (mặc định = 0)
    "prune":   os.path.join(PROJ, "experiments", "exp016_train_v66_ft5k", "metric.csv"),
    "control": os.path.join(PROJ, "experiments", "exp017_train_noprune", "metric.csv"),
}
# t hai phía 95%, theo bậc tự do
TCRIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}
METRICS = [("psnr", "PSNR", 4), ("ssim", "SSIM", 5), ("lpips", "LPIPS", 5)]


def load_repeats():
    runs = {}
    if not os.path.isfile(CSV):
        return runs
    with open(CSV, newline="") as f:
        for r in csv.DictReader(f):
            k = (int(r["seed"]), r["arm"])
            slot = runs.setdefault(k, {})
            it, occ = int(r["iteration"]), int(r["occurrence"])
            slot[(it, occ)] = {m: float(r[m]) for m, _, _ in METRICS}
            slot["n"] = r["n_gauss"]
            slot["vram"] = r["vram_mib"]
            slot["wall"] = r["wall_s"]
    return runs


def load_orig():
    out = {}
    for arm, path in ORIG.items():
        if not os.path.isfile(path):
            continue
        slot, seen = {}, {}
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                it = int(r["iteration"])
                occ = seen.get(it, 0); seen[it] = occ + 1
                slot[(it, occ)] = {m: float(r[m]) for m, _, _ in METRICS}
        out[arm] = slot
    return out


def fmt(vals, nd):
    if len(vals) == 1:
        return f"{vals[0]:.{nd}f} (n=1)"
    m, sd = st.mean(vals), st.stdev(vals)
    return f"{m:.{nd}f} ± {sd:.{nd}f}"


def ci(vals, nd):
    n = len(vals)
    if n < 2:
        return "—"
    m, sd = st.mean(vals), st.stdev(vals)
    t = TCRIT.get(n - 1, 2.0)
    h = t * sd / math.sqrt(n)
    return f"[{m-h:+.{nd}f}, {m+h:+.{nd}f}]"


def main():
    runs, orig = load_repeats(), load_orig()
    if not runs:
        print(f"chưa có {CSV} — loạt chạy lặp chưa xong."); return

    seeds = sorted({s for s, _ in runs})
    done = [s for s in seeds if (s, "prune") in runs and (s, "control") in runs]
    print("=" * 84)
    print(f"CÁC CẶP ĐÃ HOÀN THÀNH: seed {done}   (mỗi cặp = control + prune)")
    print("=" * 84)

    # ---------- (1) sàn nhiễu phần cứng ----------
    if orig and (0, "prune") in runs and (0, "control") in runs:
        print("\n(1) NHÂN CUDA KHÔNG TẤT ĐỊNH — cùng seed 0, chạy hai lần")
        print("    (EXP-016/017 lần đầu  vs  rep_*_s0 lần lại)")
        for arm in ("control", "prune"):
            if arm not in orig:
                continue
            a = orig[arm].get((35000, 0)); b = runs[(0, arm)].get((35000, 0))
            if not a or not b:
                continue
            parts = [f"{lab} {a[m]:.{nd}f} vs {b[m]:.{nd}f} (lệch {b[m]-a[m]:+.{nd}f})"
                     for m, lab, nd in METRICS]
            print(f"    {arm:8s}  " + " · ".join(parts))
        print("    -> đây là SÀN NHIỄU. Mọi khác biệt nhỏ hơn mức này là vô nghĩa.")

    # ---------- (2) biến thiên theo seed ----------
    print("\n(2) THEO SEED — mỗi seed một cặp độc lập")
    hdr = f"    {'seed':>4s} {'baseline':>9s} {'control':>9s} {'prune':>9s} " \
          f"{'luyện thêm':>11s} {'CHI PHÍ NÉN':>12s}"
    for m, lab, nd in METRICS:
        print(f"\n  --- {lab} ---")
        print(hdr)
        gains, costs = [], []
        for s in done:
            b = runs[(s, "control")][(30001, 0)][m]
            c = runs[(s, "control")][(35000, 0)][m]
            p = runs[(s, "prune")][(35000, 0)][m]
            gains.append(c - b); costs.append(p - c)
            print(f"    {s:>4d} {b:9.{nd}f} {c:9.{nd}f} {p:9.{nd}f} "
                  f"{c-b:+11.{nd}f} {p-c:+12.{nd}f}")
        if len(done) >= 2:
            print(f"    {'':>4s} {'':>9s} {'':>9s} {'':>9s} "
                  f"{fmt(gains,nd):>11s} {fmt(costs,nd):>12s}")
            print(f"    KTC 95% chi phí nén: {ci(costs, nd)}"
                  f"    (n={len(costs)}, thanh sai số rất rộng)")
            lo_hi = [st.mean(costs) - TCRIT.get(len(costs)-1,2)*st.stdev(costs)/math.sqrt(len(costs)),
                     st.mean(costs) + TCRIT.get(len(costs)-1,2)*st.stdev(costs)/math.sqrt(len(costs))]
            if lo_hi[0] <= 0 <= lo_hi[1]:
                print(f"    -> KTC CHỨA 0: chưa chứng minh được nén làm {lab} thay đổi.")
            else:
                sign = "TỆ ĐI" if (st.mean(costs) < 0) == (m != "lpips") else "TỐT LÊN"
                print(f"    -> KTC KHÔNG chứa 0: nén làm {lab} {sign} thật.")

    # ---------- kiểm tra vệ sinh ----------
    print("\n(3) KIỂM TRA VỆ SINH")
    bases = [runs[(s, a)][(30001, 0)]["psnr"] for s in done for a in ("control", "prune")]
    print(f"    baseline@30001 qua {len(bases)} lần chạy: "
          f"min {min(bases):.6f}  max {max(bases):.6f}  biên độ {max(bases)-min(bases):.2e}")
    print("    (baseline chưa qua bước tối ưu nào -> phải giống hệt nhau)")
    ns = {(s, a): runs[(s, a)].get("n") for s in done for a in ("control", "prune")}
    print(f"    số Gaussian: " + " · ".join(f"s{s}/{a}={ns[(s,a)]}" for s, a in sorted(ns)))


if __name__ == "__main__":
    main()
