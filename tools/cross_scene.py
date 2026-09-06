#!/usr/bin/env python3
"""
So sánh chéo HAI SCENE để kiểm chứng ba phát biểu của dự án:

  C1  "luyện thêm" một mình cho một khoản dương đáng kể
  C2  so với checkpoint gốc mà không có đối chứng thì THỔI PHỒNG lợi ích của nén
  C3  PSNR ồn hơn SSIM/LPIPS nhiều lần

Cả hai scene đều so ở mức cắt 50%, ghép cặp theo seed.

  train        checkpoint gốc INRIA, 1.026.508 hạt, CHƯA từng tinh chỉnh thêm
  truck-864k   đầu ra EXP-014, 864.017 hạt, ĐÃ tinh chỉnh 5000 bước

Khác biệt then chốt giữa hai scene không chỉ là nội dung ảnh, mà là
model nền còn cách hội tụ bao xa. Đó chính là biến ta muốn dò.

Dùng: python tools/cross_scene.py
"""
import csv, math, os, statistics as st

E = "/mnt/d/low-3DGS/experiments"
TCRIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}
METRICS = [("psnr", "PSNR", 4), ("ssim", "SSIM", 5), ("lpips", "LPIPS", 5)]


def rows(path):
    p = os.path.join(E, path)
    return list(csv.DictReader(open(p))) if os.path.isfile(p) else []


def pick(rs, it, occ=0, **eq):
    out = {}
    for r in rs:
        if int(r["iteration"]) != it or int(r["occurrence"]) != occ:
            continue
        if any(r.get(k) != v for k, v in eq.items()):
            continue
        out[int(r["seed"])] = {m: float(r[m]) for m, _, _ in METRICS}
    return out


def load_train():
    rep, sw = rows("repeats_train.csv"), rows("sweep_train.csv")
    return {
        "ten": "train (checkpoint gốc INRIA)",
        "N_nen": 1026508, "N_nen_da_tinh_chinh": False,
        "base": pick(rep, 30001, 0, arm="control"),
        "ctrl": pick(rep, 35000, 0, arm="control"),
        "prune": pick(sw, 35000, 0, pct="0.50"),
        "damaged": pick(sw, 30001, 1, pct="0.50"),
    }


def load_scene2():
    s2 = rows("scene2_truck864.csv")
    return {
        "ten": "truck-864k (đã tinh chỉnh sẵn)",
        "N_nen": 864017, "N_nen_da_tinh_chinh": True,
        "base": pick(s2, 30001, 0, arm="control"),
        "ctrl": pick(s2, 35000, 0, arm="control"),
        "prune": pick(s2, 35000, 0, arm="prune"),
        "damaged": pick(s2, 30001, 1, arm="prune"),
    }


def stat(vals, nd):
    if not vals:
        return None
    if len(vals) == 1:
        return (vals[0], None, None, 1)
    mu, sd = st.mean(vals), st.stdev(vals)
    h = TCRIT.get(len(vals) - 1, 2.0) * sd / math.sqrt(len(vals))
    return (mu, sd, h, len(vals))


def fmt(s, nd):
    if s is None:
        return "—"
    mu, sd, h, n = s
    return f"{mu:+.{nd}f}" + (f" ± {sd:.{nd}f}" if sd is not None else f" (n=1)")


def main():
    scenes = [load_train(), load_scene2()]
    for sc in scenes:
        if not sc["ctrl"]:
            print(f"!! thiếu dữ liệu đối chứng cho {sc['ten']}")
    print("=" * 96)
    print("SO SÁNH CHÉO HAI SCENE — mức cắt 50%, ghép cặp theo seed")
    print("=" * 96)
    for sc in scenes:
        b = list(sc["base"].values())
        bl = b[0]["psnr"] if b else float("nan")
        print(f"  {sc['ten']:<34} N nền {sc['N_nen']:>9,}  baseline {bl:8.4f} dB"
              f"  ({'ĐÃ' if sc['N_nen_da_tinh_chinh'] else 'CHƯA'} tinh chỉnh trước)")
        print(f"  {'':34} seed có: đối chứng {sorted(sc['ctrl'])} · nén {sorted(sc['prune'])}")

    for key, tieu_de in (("C1", "C1 — RIÊNG 'LUYỆN THÊM' (đối chứng − baseline)"),
                         ("C2", "C2 — NGÂY THƠ vs TRUNG THỰC (nén − baseline  vs  nén − đối chứng)"),
                         ("C3", "C3 — TỈ LỆ TÍN HIỆU/NHIỄU CỦA TỪNG THƯỚC ĐO")):
        print("\n" + "=" * 96)
        print(tieu_de)
        print("=" * 96)
        for m, lab, nd in METRICS:
            cells = []
            for sc in scenes:
                seeds_c = sorted(set(sc["base"]) & set(sc["ctrl"]))
                seeds_p = sorted(set(sc["ctrl"]) & set(sc["prune"]))
                gain = [sc["ctrl"][s][m] - sc["base"][s][m] for s in seeds_c]
                cost = [sc["prune"][s][m] - sc["ctrl"][s][m] for s in seeds_p]
                naive = [sc["prune"][s][m] - sc["base"][s][m] for s in seeds_p]
                if key == "C1":
                    cells.append(fmt(stat(gain, nd), nd))
                elif key == "C2":
                    a, b_ = stat(naive, nd), stat(cost, nd)
                    if a and b_:
                        cells.append(f"ngây thơ {a[0]:+.{nd}f} → trung thực {b_[0]:+.{nd}f}"
                                     f"  (chênh {a[0]-b_[0]:+.{nd}f})")
                    else:
                        cells.append("—")
                else:
                    s_ = stat(cost, nd)
                    if s_ and s_[1] and s_[1] > 0:
                        cells.append(f"|{s_[0]:+.{nd}f}| / {s_[1]:.{nd}f} = {abs(s_[0])/s_[1]:6.1f}×")
                    else:
                        cells.append("—")
            print(f"  {lab:6s}  " + "   │   ".join(f"{c:<44}" for c in cells))
        if key == "C1":
            g = []
            for sc in scenes:
                seeds_c = sorted(set(sc["base"]) & set(sc["ctrl"]))
                v = [sc["ctrl"][s]["psnr"] - sc["base"][s]["psnr"] for s in seeds_c]
                g.append(st.mean(v) if v else float("nan"))
            if all(x == x for x in g) and g[1] != 0:
                print(f"\n  → 'luyện thêm' trên model ĐÃ hội tụ nhỏ hơn {g[0]/g[1]:.1f} LẦN "
                      f"({g[0]:+.4f} → {g[1]:+.4f} dB)")
                print("    ⇒ confound KHÔNG phải hằng số; nó là hàm của khoảng cách tới hội tụ.")

    print("\n" + "=" * 96)
    print("KIỂM TRA VỆ SINH")
    print("=" * 96)
    for sc in scenes:
        vals = [v["psnr"] for v in sc["base"].values()]
        if vals:
            print(f"  {sc['ten']:<34} baseline qua {len(vals)} lần chạy: "
                  f"biên độ {max(vals)-min(vals):.2e}")


if __name__ == "__main__":
    main()
