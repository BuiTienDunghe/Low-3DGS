#!/usr/bin/env python3
"""
tools/ply_to_checkpoint.py — Chuyển `point_cloud.ply` (bundle pretrained INRIA) thành `chkpntNNNNN.pth`
để LightGaussian dùng được qua `--start_checkpoint` (prune_finetune.py) và `--teacher_model` (distill_train.py).

Vì sao cần (paper_notes/lightgaussian.md D1/D2): bundle INRIA chỉ có .ply; script chính thức nạp .pth
có Adam state. .pth tạo ở đây có optimizer state RỖNG (vừa training_setup) → khác official. Ghi rõ.

⚠️ ĐÍNH CHÍNH (2026-09-02): giả thuyết ban đầu của tôi — rằng `load_ply` để `spatial_lr_scale = 0`
làm đóng băng vị trí Gaussian — là **SAI**. Luồng thật: `Scene(...)` gọi `create_from_pcd(pcd, cameras_extent)`
trước, **đã set** `spatial_lr_scale`; `load_ply()` chỉ thay tensor, không đụng thuộc tính đó; `training_setup`
chạy sau nên learning rate đúng. Script này vẫn set lại tường minh + assert như một **lưới an toàn**, không
phải sửa bug.

⚠️ CÓ THỂ KHÔNG CẦN SCRIPT NÀY. Có hai đường rẻ hơn:
  1. Fine-tune bằng INRIA gốc: sửa `train.py:51` thành `Scene(dataset, gaussians, load_iteration=30000)`
     (đúng cách `render.py:51` đã làm). `training_setup(opt)` chạy sau nên thứ tự vẫn đúng.
  2. Dùng repo PUP 3D-GS — đã có sẵn `--start_pointcloud` và đã vá lỗi iteration của LightGaussian.
Chỉ dùng script này khi thật sự cần một `.pth` (ví dụ `distill_train.py --teacher_model`, xem
paper_notes/lightgaussian.md T9).

Hệ quả vẫn đúng và quan trọng: vì `Scene()` phải dựng point cloud COLMAP trước, **không chạy được
chỉ với `.ply`** — bắt buộc có đủ dataset COLMAP (`sparse/` + `images/`) ở `-s`.

Dùng (trong WSL, venv đã kích hoạt):
    python tools/ply_to_checkpoint.py \
        --repo ~/l3dgs/third_party/LightGaussian \
        -m /mnt/d/low-3DGS/datasets/pretrained/models/truck \
        -s /mnt/d/low-3DGS/datasets/tandt/truck \
        --iteration 30000 \
        --out /mnt/d/low-3DGS/datasets/pretrained/models/truck/chkpnt30000.pth
"""
from __future__ import annotations

import argparse
import os
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, help="thư mục LightGaussian hoặc gaussian-splatting (để import scene/arguments)")
    ap.add_argument("-m", "--model_path", required=True, help="thư mục model INRIA (chứa point_cloud/iteration_N/)")
    ap.add_argument("-s", "--source_path", required=True, help="dataset COLMAP (để tính cameras_extent)")
    ap.add_argument("--iteration", type=int, default=30000)
    ap.add_argument("--sh_degree", type=int, default=3)
    ap.add_argument("--out", help="mặc định <model_path>/chkpnt<iteration>.pth")
    ap.add_argument("--data_device", default="cpu", help="cpu để không tốn VRAM cho ảnh (ta chỉ cần extent)")
    ap.add_argument("--resolution", "-r", type=int, default=8, help="tải ảnh ở res thấp — chỉ cần camera, không cần pixel")
    args = ap.parse_args()

    repo = os.path.abspath(os.path.expanduser(args.repo))
    sys.path.insert(0, repo)
    os.chdir(repo)  # vài import trong repo dùng đường dẫn tương đối

    import torch
    from argparse import ArgumentParser as _AP
    from arguments import ModelParams, OptimizationParams
    from scene import GaussianModel, Scene

    out = args.out or os.path.join(args.model_path, f"chkpnt{args.iteration}.pth")
    t0 = time.time()

    # --- dựng args giả cho ModelParams/OptimizationParams (dùng default của repo) ---
    p = _AP()
    lp = ModelParams(p)
    op = OptimizationParams(p)
    ns = p.parse_args([
        "-m", args.model_path, "-s", args.source_path,
        "--sh_degree", str(args.sh_degree),
        "--data_device", args.data_device,
        "-r", str(args.resolution),
        "--eval",                      # để split train/test giống lúc đánh giá (không ảnh hưởng extent)
    ])
    dataset = lp.extract(ns)
    opt = op.extract(ns)

    # --- nạp .ply qua Scene → có cameras_extent ---
    gaussians = GaussianModel(args.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    n = gaussians.get_xyz.shape[0]
    extent = float(scene.cameras_extent)
    print(f"[ply2ckpt] loaded {n:,} gaussians  | cameras_extent={extent:.4f}  | {time.time()-t0:.1f}s")

    # --- BẪY: spatial_lr_scale ---
    before = float(getattr(gaussians, "spatial_lr_scale", 0.0))
    gaussians.spatial_lr_scale = extent
    print(f"[ply2ckpt] spatial_lr_scale: {before} -> {extent}  (0 = xyz lr bằng 0, vị trí đóng băng)")

    # --- optimizer rỗng + capture ---
    gaussians.training_setup(opt)
    if hasattr(gaussians, "active_sh_degree"):
        gaussians.active_sh_degree = gaussians.max_sh_degree  # pretrained đã full SH
    payload = (gaussians.capture(), args.iteration)
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    torch.save(payload, out)
    sz = os.path.getsize(out) / 2**20
    print(f"[ply2ckpt] wrote {out}  ({sz:.1f} MB)  | total {time.time()-t0:.1f}s")

    # --- kiểm tra lại: restore được không ---
    g2 = GaussianModel(args.sh_degree)
    (mp, it) = torch.load(out, map_location="cuda" if torch.cuda.is_available() else "cpu", weights_only=False)
    g2.restore(mp, opt)
    assert g2.get_xyz.shape[0] == n, "restore mismatch"
    assert float(g2.spatial_lr_scale) == extent, "spatial_lr_scale không được lưu — kiểm tra capture()"
    print(f"[ply2ckpt] restore OK: {g2.get_xyz.shape[0]:,} gaussians, iter={it}, spatial_lr_scale={g2.spatial_lr_scale:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
