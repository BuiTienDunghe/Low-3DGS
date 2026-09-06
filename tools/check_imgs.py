import os, sys
from PIL import Image
root = "/mnt/d/low-3DGS/datasets/tandt"
for s in ("train", "truck"):
    d = os.path.join(root, s, "images")
    fs = sorted(os.listdir(d))
    im = Image.open(os.path.join(d, fs[0]))
    w, h = im.size
    # 3DGS -r 1: nếu chiều rộng > 1600 thì thu nhỏ về 1600
    if w > 1600:
        sc = w / 1600.0
        rw, rh = 1600, int(h / sc)
    else:
        rw, rh = w, h
    px = rw * rh
    tot = len(fs) * px * 12 / 2**30
    print(f"{s}: {len(fs)} ảnh, gốc {w}x{h} -> dùng {rw}x{rh} = {px/1e6:.2f} Mpx"
          f" | 12 B/px x {len(fs)} = {tot:.2f} GiB RAM ảnh")
