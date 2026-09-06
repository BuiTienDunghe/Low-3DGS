#!/usr/bin/env python3
"""
BƯỚC 1 của hướng B — đo riêng THIỆT HẠI của việc cắt bậc màu, chưa phục hồi gì.

45 trong 59 số của mỗi hạt (76%) là hệ số màu phụ thuộc góc nhìn. Câu hỏi:
bỏ chúng đi thì mất bao nhiêu dB, TRƯỚC khi tinh chỉnh?

Cách đo: render() truyền sh_degree=pc.active_sh_degree xuống rasterizer, nên
đặt active_sh_degree = d chính là mô phỏng chính xác việc chỉ lưu tới bậc d —
các hệ số cao hơn không bao giờ được đánh giá.

KHÔNG huấn luyện, KHÔNG lưu file. Dùng đúng bộ hàm đo của repo
(loss_utils.ssim · lpipsPyTorch VGG · image_utils.psnr) nên số so được thẳng
với mọi kết quả trước: train baseline = 21.815698 ở bậc 3.
"""
import os, sys, csv, time

REPO = os.path.expanduser("~/l3dgs/third_party/gaussian-splatting-pup")
sys.path.insert(0, REPO)
os.chdir(REPO)

import torch
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams
from scene import Scene, GaussianModel
from gaussian_renderer import render
from utils.general_utils import safe_state
from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr
from lpipsPyTorch import lpips

# Số float mỗi hạt: 11 hình học (xyz 3 · scale 3 · quay 4 · độ đục 1) + màu.
# Bậc d có (d+1)^2 hệ số mỗi kênh × 3 kênh.
GEOM = 11
def floats(d):  return GEOM + 3 * (d + 1) ** 2
def onDisk(d):  return floats(d) + 3          # .ply còn thêm 3 pháp tuyến (luôn = 0)


def main():
    parser = ArgumentParser()
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--start_pointcloud", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(sys.argv[1:])

    safe_state(False)
    dataset, pipe = lp.extract(args), pp.extract(args)

    # Scene ghi input.ply + cameras.json vào model_path và KHÔNG tự tạo thư mục
    # (prune_finetune.py tạo sẵn qua prepare_output_and_logger).
    os.makedirs(dataset.model_path, exist_ok=True)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.load_ply(args.start_pointcloud)
    N = gaussians.get_xyz.shape[0]
    bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0],
                      dtype=torch.float32, device="cuda")
    cams = scene.getTestCameras()
    print(f"\nN = {N:,} hạt · {len(cams)} view test\n", flush=True)

    rows = []
    for d in (3, 2, 1, 0):
        gaussians.active_sh_degree = d
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        t0 = time.monotonic()
        acc = dict(l1=0.0, psnr=0.0, ssim=0.0, lpips=0.0)
        with torch.no_grad():
            for cam in cams:
                img = torch.clamp(render(cam, gaussians, pipe, bg)["render"], 0.0, 1.0)
                gt = torch.clamp(cam.original_image.to("cuda"), 0.0, 1.0)
                acc["l1"]    += l1_loss(img, gt).mean().double().item()
                acc["psnr"]  += psnr(img, gt).mean().double().item()
                acc["ssim"]  += ssim(img, gt).mean().double().item()
                acc["lpips"] += lpips(img, gt, net_type="vgg").mean().double().item()
        for k in acc:
            acc[k] /= len(cams)
        # ĐỈNH VRAM ĐO TỪ TRONG TIẾN TRÌNH (EXP-015 kết luận 3), không phải NVML lấy mẫu.
        peak = torch.cuda.max_memory_allocated() / 2 ** 20
        f, b = floats(d), onDisk(d)
        rows.append(dict(degree=d, floats=f, disk_b=b, n=N,
                         mb=N * b * 4 / 1e6, shrink=onDisk(3) / b,
                         vram_peak_mib=round(peak, 1),
                         wall_s=round(time.monotonic() - t0, 1), **acc))
        print(f"bậc {d}: {f:2d} float/hạt · {N*b*4/1e6:7.2f} MB · "
              f"PSNR {acc['psnr']:.4f} · SSIM {acc['ssim']:.5f} · LPIPS {acc['lpips']:.5f} "
              f"· đỉnh VRAM {peak:.0f} MiB · {rows[-1]['wall_s']:.0f}s", flush=True)

    base = rows[0]
    print("\n" + "=" * 88)
    print("THIỆT HẠI CỦA VIỆC CẮT BẬC MÀU (chưa phục hồi gì)")
    print("=" * 88)
    print(f"{'bậc':>4s} {'float/hạt':>10s} {'MB':>8s} {'nhỏ hơn':>8s} │ "
          f"{'PSNR':>9s} {'ΔPSNR':>9s} │ {'ΔSSIM':>10s} {'ΔLPIPS':>10s}")
    print("-" * 88)
    for r in rows:
        print(f"{r['degree']:>4d} {r['floats']:>10d} {r['mb']:>8.2f} {r['shrink']:>7.2f}× │ "
              f"{r['psnr']:>9.4f} {r['psnr']-base['psnr']:>+9.4f} │ "
              f"{r['ssim']-base['ssim']:>+10.5f} {r['lpips']-base['lpips']:>+10.5f}")

    print("\nĐỐI CHIẾU với trục CẮT HẠT (EXP-019, cùng scene, cùng bộ đo):")
    print("  cắt 50%  -> 2.00× nhỏ hơn, thiệt hại trước phục hồi  -0.246 dB")
    print("  cắt 66%  -> 2.94× nhỏ hơn, thiệt hại trước phục hồi  -1.148 dB")
    print("  cắt 80%  -> 5.00× nhỏ hơn, thiệt hại trước phục hồi  -2.972 dB")

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nđã ghi {args.out}")


if __name__ == "__main__":
    main()
