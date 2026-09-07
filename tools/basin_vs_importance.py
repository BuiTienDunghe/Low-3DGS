#!/usr/bin/env python3
"""
EXP-027 — kiểm giả thuyết: biên miền hút nằm ở một TỈ LỆ KHỐI LƯỢNG QUAN TRỌNG cố định?

Nếu đúng, hai model rất khác nhau sẽ trùng nhau trên trục đó, và ta dự đoán được
mức nén miễn phí mà KHÔNG cần chạy fine-tune.

Tiêu chí biên miền hút: điểm dừng lệch khỏi mốc đối chứng đúng 2 lần nhiễu giữa các seed.
Nội suy tuyến tính giữa hai mức cắt kẹp lấy ngưỡng đó.
"""
import csv, os, statistics as st
E = "/mnt/d/low-3DGS/experiments"

def read(f): return list(csv.DictReader(open(os.path.join(E, f))))

prof = {}
for r in read("importance_profile.csv"):
    prof.setdefault(r["label"], []).append((float(r["pct"]), float(r["imp_removed_frac"])))
for k in prof: prof[k].sort()

def interp(xs_ys, x):
    for (x0,y0),(x1,y1) in zip(xs_ys, xs_ys[1:]):
        if x0 <= x <= x1:
            return y0 + (x-x0)/(x1-x0)*(y1-y0)
    return None

# --- train: điểm dừng theo mức cắt, mốc đối chứng, nhiễu seed ---
rep = read("repeats_train.csv"); sw = read("sweep_train.csv")
ctrl_tr = [float(r["psnr"]) for r in rep if r["arm"]=="control" and r["iteration"]=="35000"]
CE_tr, SD_tr = st.mean(ctrl_tr), st.stdev(ctrl_tr)
end_tr = {}
for p in ["0.20","0.40","0.50","0.60","0.70","0.80"]:
    v=[float(r["psnr"]) for r in sw if r["pct"]==p and r["iteration"]=="35000" and r["seed"]=="0"]
    if v: end_tr[float(p)] = v[0]
end_tr[0.66] = st.mean(float(r["psnr"]) for r in rep if r["arm"]=="prune" and r["iteration"]=="35000")

# --- truck864 ---
s2 = read("scene2_truck864.csv"); a2 = read("exp026_scene2_attractor.csv")
ctrl_t8 = [float(r["psnr"]) for r in s2 if r["arm"]=="control" and r["iteration"]=="35000" and r["occurrence"]=="0"]
CE_t8, SD_t8 = st.mean(ctrl_t8), st.stdev(ctrl_t8)
end_t8 = {float(r["pct"]): float(r["psnr"]) for r in a2 if r["iteration"]=="35000" and r["occurrence"]=="0"}
end_t8[0.50] = st.mean(float(r["psnr"]) for r in s2 if r["arm"]=="prune" and r["iteration"]=="35000" and r["occurrence"]=="0")

def boundary(end, CE, SD, label):
    pts = sorted((p, abs(e-CE)/SD) for p,e in end.items())
    print(f"\n  {label}: mốc {CE:.4f}, nhiễu seed SD {SD:.4f}")
    for p,s in pts: print(f"    cắt {p*100:5.1f}%  lệch {s:6.2f}× nhiễu")
    for (p0,s0),(p1,s1) in zip(pts, pts[1:]):
        if s0 < 2.0 <= s1:
            pb = p0 + (2.0-s0)/(s1-s0)*(p1-p0)
            print(f"    -> BIÊN (2× nhiễu) ở cắt {pb*100:.2f}%")
            return pb
    return None

print("="*70); print("BƯỚC 1 — xác định biên miền hút theo tiêu chí 2× nhiễu seed"); print("="*70)
b_tr = boundary(end_tr, CE_tr, SD_tr, "train")
b_t8 = boundary(end_t8, CE_t8, SD_t8, "truck-864k")

print()
print("="*70); print("BƯỚC 2 — quy đổi sang trục KHỐI LƯỢNG QUAN TRỌNG BỊ CẮT"); print("="*70)
i_tr = interp(prof["train"], b_tr); i_t8 = interp(prof["truck864"], b_t8)
print(f"  train      : biên ở cắt {b_tr*100:5.2f}% số hạt  ->  {i_tr*100:6.3f}% khối lượng quan trọng")
print(f"  truck-864k : biên ở cắt {b_t8*100:5.2f}% số hạt  ->  {i_t8*100:6.3f}% khối lượng quan trọng")
print(f"\n  tỉ số trên trục SỐ HẠT        : {b_tr/b_t8:.2f}×")
print(f"  tỉ số trên trục KHỐI LƯỢNG QT : {i_tr/i_t8:.2f}×")
print()
if abs(i_tr/i_t8 - 1) < 0.25:
    print("  => GIẢ THUYẾT ĐƯỢC ỦNG HỘ: hai model trùng nhau trên trục khối lượng quan trọng.")
else:
    print("  => GIẢ THUYẾT BỊ BÁC BỎ: đổi trục KHÔNG làm hai model trùng nhau.")
    print(f"     Chúng vẫn lệch {i_tr/i_t8:.2f} lần. Không dự đoán được mức nén miễn phí")
    print("     chỉ từ phân bố điểm quan trọng.")

print()
print("="*70); print("BƯỚC 3 — nhưng phân bố điểm quan trọng ĐO ĐƯỢC lượng dư thừa"); print("="*70)
print(f"  {'cắt':>6} {'train':>10} {'truck-864k':>12}   (khối lượng quan trọng nằm ở nhóm hạt yếu nhất)")
for p in [0.20, 0.50, 0.66]:
    a = interp(prof["train"], p); b = interp(prof["truck864"], p)
    print(f"  {p*100:5.0f}% {a*100:9.3f}% {b*100:11.3f}%   -> truck864 gấp {b/a:.2f}×")
print()
print("  Ở truck-864k, nhóm hạt yếu nhất mang NHIỀU khối lượng quan trọng hơn hẳn")
print("  => phân bố ÍT LỆCH hơn => còn ít dư thừa để cắt. Đúng chiều với miền hút hẹp hơn,")
print("     nhưng quan hệ định lượng KHÔNG đơn giản.")
