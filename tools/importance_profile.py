#!/usr/bin/env python3
"""
Đo PHÂN BỐ KHỐI LƯỢNG QUAN TRỌNG của một model — chỉ forward pass, không huấn luyện.

Ý tưởng đang kiểm (EXP-027): biên miền hút của công thức tinh chỉnh có thể nằm ở một
**tỉ lệ khối lượng quan trọng bị cắt** cố định, chứ không phải ở một tỉ lệ *số hạt* cố định.

Nếu đúng thì hai model rất khác nhau — `train` (miền hút tới ~68% số hạt) và `truck-864k`
(chỉ tới ~21%) — sẽ TRÙNG NHAU trên trục "khối lượng quan trọng bị cắt". Khi đó ta dự đoán
được mức nén miễn phí của một model MÀ KHÔNG CẦN chạy fine-tune.

Dùng đúng điểm quan trọng mà pipeline dùng để prune (`v_important_score`, v_pow=0.1),
và đúng quy tắc chọn của `prune_gaussians`: cắt các hạt có điểm <= giá trị ở vị trí
int(p*(N-1)) sau khi sắp xếp tăng dần.
"""
import os, sys, csv

REPO = os.path.expanduser("~/l3dgs/third_party/gaussian-splatting-pup")
sys.path.insert(0, REPO)
os.chdir(REPO)

import torch
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from prune import prune_list, calculate_v_imp_score

GRID = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.66, 0.68, 0.70, 0.80, 0.90, 0.95]


def main():
    parser = ArgumentParser()
    lp = ModelParams(parser); pp = PipelineParams(parser)
    parser.add_argument("--start_pointcloud", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--v_pow", type=float, default=0.1)
    args = parser.parse_args(sys.argv[1:])

    safe_state(False)
    dataset, pipe = lp.extract(args), pp.extract(args)
    os.makedirs(dataset.model_path, exist_ok=True)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.load_ply(args.start_pointcloud)
    bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0],
                      dtype=torch.float32, device="cuda")
    N = gaussians.get_xyz.shape[0]
    print(f"\nN = {N:,} hạt · {len(scene.getTrainCameras())} view train\n", flush=True)

    with torch.no_grad():
        _, imp = prune_list(gaussians, scene, pipe, bg)
        v = calculate_v_imp_score(gaussians, imp, args.v_pow).squeeze()

    v = v.float()
    total = v.sum().item()
    sv, _ = torch.sort(v)                       # tăng dần — đúng như prune_gaussians
    csum = torch.cumsum(sv, 0)

    print(f"tổng khối lượng quan trọng = {total:.6g}")
    print(f"{'cắt':>6} {'hạt bị cắt':>12} {'hạt còn lại':>12} {'khối lượng bị cắt':>19}")
    print("-" * 54)
    rows = []
    for p in GRID:
        # prune_gaussians: idx = int(p*(N-1)); cắt mọi hạt có điểm <= sorted[idx]
        idx = int(p * (N - 1))
        nrem = idx + 1
        frac = (csum[idx].item() / total) if total > 0 else float("nan")
        rows.append(dict(label=args.label, n_total=N, pct=p, n_removed=nrem,
                         n_kept=N - nrem, imp_removed_frac=round(frac, 8)))
        print(f"{p*100:5.0f}% {nrem:>12,} {N-nrem:>12,} {frac*100:>18.4f}%")

    hdr = list(rows[0].keys())
    new = not os.path.isfile(args.out)
    with open(args.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=hdr)
        if new:
            w.writeheader()
        w.writerows(rows)
    print(f"\nđã ghi {args.out}")


if __name__ == "__main__":
    main()
