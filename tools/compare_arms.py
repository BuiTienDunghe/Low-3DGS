#!/usr/bin/env python3
"""
So sánh hai nhánh của cặp đối chứng và TÁCH hai hiệu ứng đang bị trộn vào nhau:

  hiệu ứng "luyện thêm"  = control@35000 - baseline@30001
  hiệu ứng "prune+phục hồi" = prune@35000  - baseline@30001
  CHI PHÍ THẬT CỦA NÉN   = prune@35000 - control@35000

Ý nghĩa: EXP-014 báo +0.1123 dB cho model đã nén so với baseline. Nhưng model đó
được luyện thêm 5000 bước mà baseline không có. Chỉ khi trừ đi phần "luyện thêm"
mới biết nén thực sự lấy đi hay cho thêm bao nhiêu.

Dùng: python tools/compare_arms.py <thư_mục_prune> <thư_mục_control>
"""
import csv, os, sys

FIELDS = [("psnr", "PSNR", "+", 4), ("ssim", "SSIM", "+", 5), ("lpips", "LPIPS", "-", 5)]


def read_metrics(d):
    """Trả về {iteration: {field: value}}. Với iter 30001 có thể có 2 dòng
    (trước prune / sau prune) -> giữ cả hai theo thứ tự xuất hiện."""
    path = os.path.join(d, "metric.csv")
    if not os.path.isfile(path):
        sys.exit(f"thiếu {path}")
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append({k: r[k] for k in r})
    return rows


def fnum(rows, it, occurrence=0):
    sel = [r for r in rows if int(r["iteration"]) == it]
    if len(sel) <= occurrence:
        return None
    return sel[occurrence]


def gaussians(d, it):
    p = os.path.join(d, "point_cloud", f"iteration_{it}", "point_cloud.ply")
    if not os.path.isfile(p):
        return None, None
    n = None
    with open(p, "rb") as f:
        head = f.read(4000)
    for line in head.split(b"\n"):
        if line.startswith(b"element vertex"):
            n = int(line.split()[2])
            break
    return n, os.path.getsize(p)


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    dp, dc = sys.argv[1], sys.argv[2]
    rp, rc = read_metrics(dp), read_metrics(dc)

    base_p = fnum(rp, 30001, 0)   # nhánh prune: dòng đầu ở 30001 = TRƯỚC khi prune
    afterprune = fnum(rp, 30001, 1)
    fin_p = fnum(rp, 35000, 0)
    base_c = fnum(rc, 30001, 0)
    fin_c = fnum(rc, 35000, 0)

    for nm, v in (("prune@30001(trước)", base_p), ("prune@35000", fin_p),
                  ("control@30001", base_c), ("control@35000", fin_c)):
        if v is None:
            sys.exit(f"thiếu số liệu: {nm} — nhánh chưa chạy xong?")

    print("=" * 78)
    print("KIỂM TRA NHẤT QUÁN: hai nhánh phải có CÙNG baseline ở iter 30001")
    print("=" * 78)
    ok = True
    for k, label, _, nd in FIELDS:
        a, b = float(base_p[k]), float(base_c[k])
        same = abs(a - b) < 1e-9
        ok &= same
        print(f"  {label:6s} prune={a:.{nd}f}  control={b:.{nd}f}  "
              f"lệch={a-b:+.2e}  {'✓' if same else '✗ KHÁC NHAU'}")
    print("  -> " + ("cùng điểm xuất phát, so sánh hợp lệ."
                     if ok else "KHÁC NHAU: có yếu tố không xác định, so sánh KHÔNG hợp lệ."))

    np_, bp = gaussians(dp, 35000)
    nc, bc = gaussians(dc, 35000)
    print()
    print("=" * 78)
    print("KẾT QUẢ")
    print("=" * 78)
    if np_ and nc:
        print(f"  số Gaussian : control {nc:,}  ->  prune {np_:,}   ({nc/np_:.2f}x ít hơn)")
        print(f"  kích thước  : control {bc/1e6:.2f} MB  ->  prune {bp/1e6:.2f} MB   ({bc/bp:.2f}x nhỏ hơn)")
        print()
    hdr = f"  {'':8s} {'baseline':>10s} {'control':>10s} {'prune':>10s} " \
          f"{'luyện thêm':>12s} {'prune+phục hồi':>15s} {'CHI PHÍ NÉN':>13s}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for k, label, better, nd in FIELDS:
        b = float(base_c[k]); c = float(fin_c[k]); p = float(fin_p[k])
        g_train = c - b        # chỉ do luyện thêm
        g_total = p - b        # prune + phục hồi so với gốc
        cost = p - c           # nén thực sự đáng giá bao nhiêu
        print(f"  {label:8s} {b:10.{nd}f} {c:10.{nd}f} {p:10.{nd}f} "
              f"{g_train:+12.{nd}f} {g_total:+15.{nd}f} {cost:+13.{nd}f}")
    print()
    if afterprune is not None:
        print(f"  (ngay sau khi prune, chưa phục hồi: PSNR {float(afterprune['psnr']):.4f}"
              f" = {float(afterprune['psnr'])-float(base_p['psnr']):+.4f} dB)")

    print()
    print("=" * 78)
    print("CÁCH ĐỌC")
    print("=" * 78)
    dp_ = float(fin_p["psnr"]) - float(base_c["psnr"])
    dc_ = float(fin_c["psnr"]) - float(base_c["psnr"])
    cost = dp_ - dc_
    print(f"  So với model gốc, model đã nén {'hơn' if dp_>0 else 'kém'} {abs(dp_):.4f} dB.")
    print(f"  Nhưng model KHÔNG nén, luyện thêm đúng bằng đó bước, {'hơn' if dc_>0 else 'kém'} {abs(dc_):.4f} dB.")
    print(f"  => Phần do 'luyện thêm' : {dc_:+.4f} dB")
    print(f"  => Phần do 'nén'        : {cost:+.4f} dB   <-- ĐÂY MỚI LÀ CHI PHÍ CỦA NÉN")
    print()
    if cost >= -0.05:
        print("  KẾT LUẬN: nén gần như không mất gì (trong khoảng 0.05 dB).")
    elif cost >= -0.2:
        print("  KẾT LUẬN: nén có mất, nhưng nhỏ (< 0.2 dB).")
    else:
        print("  KẾT LUẬN: nén mất đáng kể. Phát biểu 'nén miễn phí' KHÔNG đứng vững.")
    if dc_ > 0.05:
        print(f"  CẢNH BÁO: 'luyện thêm' một mình đã cho {dc_:+.4f} dB. Mọi bài báo so model")
        print("            đã fine-tune với checkpoint gốc mà không có đối chứng này đều")
        print("            đang tính công của 'luyện thêm' vào cho 'nén'.")


if __name__ == "__main__":
    main()
