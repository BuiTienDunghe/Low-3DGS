#!/usr/bin/env python3
"""
tools/measure_pipeline.py — Đo RAM/VRAM từng stage của ĐÚNG code sẽ chạy. Không mô phỏng, không ước lượng.

VÌ SAO TỒN TẠI: dự án này đã 8 lần ước lượng sai rồi phải lập lại kế hoạch. Mọi lần đều cùng một
nguyên nhân: **đo (hoặc đọc) một đoạn code, rồi áp hệ số đó cho một đoạn code khác.**
  - đo `load_ply` bản float32 tự viết (1.96x) rồi áp cho `load_ply` thật của INRIA (4.58x)
  - đọc `alpha_mask` của INRIA main (16 B/px) rồi áp cho PUP (fork cũ, 12 B/px)
  - dự đoán RSS của truck bằng hệ số gsply (1.95x) trong khi repo dùng loader gốc

File này gọi **chính các hàm trong repo** (`sceneLoadTypeCallbacks`, `cameraList_from_camInfos`,
`GaussianModel.load_ply`, `training_setup`, `prune_list`) và đo quanh từng cái. Số nó in ra là số
của code sẽ chạy thật, không phải của một bản tương đương.

TÁCH ĐƯỢC (đây là điểm mấu chốt — `Scene.__init__` gộp hết vào một chỗ nên không tách được):
  s2_scene_info   COLMAP metadata, CHƯA nạp ảnh
  s3_cameras      nạp ảnh  -> chi phí ẢNH, riêng biệt
  s4_load_ply     nạp model -> chi phí PLY, riêng biệt
  s5_render_one / s6_render_all
  s7_training_setup   Adam cấp phát  -> chi phí OPTIMIZER, riêng biệt
  s8_prune_list       importance score -> chi phí SCORING, riêng biệt

Mỗi config chạy trong TIẾN TRÌNH RIÊNG (allocator sạch, không nhiễu chéo).

Dùng:
  python tools/measure_pipeline.py --scene train --repo pup --stages all --out results.jsonl
  python tools/measure_pipeline.py --matrix --out experiments/pipeline_matrix/     # quét cả ma trận
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

GB = float(2**30)

SCENES = {
    "train":     ("tandt/train",  "train"),
    "truck":     ("tandt/truck",  "truck"),
    "playroom":  ("db/playroom",  "playroom"),
    "drjohnson": ("db/drjohnson", "drjohnson"),
}
REPOS = {
    "pup":   "~/l3dgs/third_party/gaussian-splatting-pup",
    "inria": "~/l3dgs/third_party/gaussian-splatting",
}
PROJ = "/mnt/d/low-3DGS"


# =============================================================================
# CHILD: chạy một config, in JSON một dòng
# =============================================================================
def child(args) -> int:
    # BẮT BUỘC tuyệt đối hoá TRƯỚC os.chdir: repo cần cwd của nó, nhưng --out là
    # đường dẫn tương đối so với thư mục dự án. Không làm bước này thì JSONL rơi
    # vào ~/l3dgs/third_party/<repo>/experiments/... và biến mất.
    if args.out:
        args.out = os.path.abspath(args.out)
    repo = os.path.expanduser(REPOS[args.repo])
    sys.path.insert(0, repo)
    os.chdir(repo)
    sys.path.insert(0, PROJ)

    from src.memory.profiler import StageProfiler

    prof = StageProfiler(args.out or "/dev/null",
                         run_id=f"{args.scene}_{args.repo}_{args.data_device}_r{args.resolution}",
                         hz=args.hz, verbose=not args.quiet)

    # ---- s1: import torch + khởi tạo CUDA ----------------------------------
    with prof.stage("s1_cuda_init") as rec:
        import torch
        torch.zeros(1, device="cuda")
        torch.cuda.synchronize()
        rec["extra"]["torch"] = torch.__version__
        rec["extra"]["alloc_conf"] = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")

    from argparse import ArgumentParser as _AP
    from arguments import ModelParams, OptimizationParams, PipelineParams
    from scene import GaussianModel
    from scene.dataset_readers import sceneLoadTypeCallbacks
    from utils.camera_utils import cameraList_from_camInfos

    src_rel, model_name = SCENES[args.scene]
    source = f"{PROJ}/datasets/{src_rel}"
    model = f"{PROJ}/datasets/pretrained/models/{model_name}"

    p = _AP()
    lp, op, pp = ModelParams(p), OptimizationParams(p), PipelineParams(p)
    ns = p.parse_args(["-s", source, "-m", model, "--eval",
                       "-r", str(args.resolution), "--data_device", args.data_device,
                       "--sh_degree", "3"])
    dataset, opt, pipe = lp.extract(ns), op.extract(ns), pp.extract(ns)

    # ---- s2: metadata COLMAP (CHƯA nạp pixel) ------------------------------
    with prof.stage("s2_scene_info") as rec:
        scene_info = sceneLoadTypeCallbacks["Colmap"](dataset.source_path, dataset.images, dataset.eval)
        rec["extra"]["n_train_cams"] = len(scene_info.train_cameras)
        rec["extra"]["n_test_cams"] = len(scene_info.test_cameras)
        rec["extra"]["cameras_extent"] = float(scene_info.nerf_normalization["radius"])

    # ---- s3: NẠP ẢNH (chi phí ảnh, tách riêng) -----------------------------
    with prof.stage("s3_cameras_train", n_images=len(scene_info.train_cameras)) as rec:
        train_cams = cameraList_from_camInfos(scene_info.train_cameras, 1.0, dataset)
        if train_cams:
            c = train_cams[0]
            im = c.original_image
            rec["extra"]["img_shape"] = list(im.shape)          # (C,H,W) -> C=3 hay 4?
            rec["extra"]["img_dtype"] = str(im.dtype)
            rec["extra"]["img_device"] = str(im.device)
            rec["extra"]["bytes_per_pixel"] = im.element_size() * im.shape[0]
            rec["extra"]["one_img_mb"] = im.numel() * im.element_size() / 2**20
            rec["extra"]["all_train_imgs_gb"] = (im.numel() * im.element_size() * len(train_cams)) / GB
            rec["extra"]["has_alpha_mask_attr"] = hasattr(c, "alpha_mask") and getattr(c, "alpha_mask") is not None

    with prof.stage("s3b_cameras_test", n_images=len(scene_info.test_cameras)):
        test_cams = cameraList_from_camInfos(scene_info.test_cameras, 1.0, dataset)

    # ---- s4: NẠP MODEL (chi phí PLY, tách riêng) ---------------------------
    ply = f"{model}/point_cloud/iteration_{args.iteration}/point_cloud.ply"
    gaussians = GaussianModel(3)
    with prof.stage("s4_load_ply") as rec:
        rec["extra"]["ply_mb"] = os.path.getsize(ply) / 2**20
        gaussians.load_ply(ply)
        rec["extra"]["n_gaussians"] = int(gaussians.get_xyz.shape[0])
        rec["extra"]["xyz_device"] = str(gaussians.get_xyz.device)
    gaussians.spatial_lr_scale = float(scene_info.nerf_normalization["radius"])
    N = int(gaussians.get_xyz.shape[0])
    ply_mb = os.path.getsize(ply) / 2**20

    import torch
    from gaussian_renderer import render
    bg = torch.zeros(3, device="cuda")

    # ---- s5/s6: render -----------------------------------------------------
    if "render" in args.stages or args.stages == "all":
        with prof.stage("s5_render_one", n_gaussians=N):
            with torch.no_grad():
                _ = render(test_cams[0], gaussians, pipe, bg)
                torch.cuda.synchronize()
        with prof.stage("s6_render_all_test", n_gaussians=N, n_views=len(test_cams)):
            with torch.no_grad():
                for c in test_cams:
                    _ = render(c, gaussians, pipe, bg)
                torch.cuda.synchronize()

    # ---- s7: Adam cấp phát (chi phí optimizer, tách riêng) -----------------
    if "train" in args.stages or args.stages == "all":
        with prof.stage("s7_training_setup", n_gaussians=N) as rec:
            gaussians.training_setup(opt)
            torch.cuda.synchronize()
            rec["extra"]["n_param_groups"] = len(gaussians.optimizer.param_groups)

        # ---- s9: MỘT BƯỚC FINE-TUNE THẬT (forward + loss + backward + step) ----
        # Đây là thứ mọi bảng ngân sách cũ THIẾU: activations của backward. Số render
        # (forward-only) là CẬN DƯỚI, không dùng thay được.
        try:
            from utils.loss_utils import l1_loss, ssim
            import random
            with prof.stage("s9_finetune_step", n_gaussians=N) as rec:
                cam = train_cams[random.Random(0).randrange(len(train_cams))]
                pkg = render(cam, gaussians, pipe, bg)
                image, vsp, vis, radii = (pkg["render"], pkg["viewspace_points"],
                                          pkg["visibility_filter"], pkg["radii"])
                gt = cam.original_image.cuda()
                loss = (1.0 - opt.lambda_dssim) * l1_loss(image, gt) \
                       + opt.lambda_dssim * (1.0 - ssim(image, gt))
                loss.backward()
                torch.cuda.synchronize()
                rec["extra"]["loss"] = float(loss.detach())
                rec["extra"]["vram_after_backward_mb"] = round(torch.cuda.memory_allocated() / MB, 1)
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                rec["extra"]["vram_after_step_mb"] = round(torch.cuda.memory_allocated() / MB, 1)
        except Exception as e:
            print(f"[s9 bỏ qua] {type(e).__name__}: {str(e)[:140]}", file=sys.stderr)

        # ---- s8: importance scoring (chi phí scoring, tách riêng) ----------
        try:
            from prune import prune_list
            class _S:  # Scene giả, chỉ cần getTrainCameras
                def __init__(self, cams): self._c = cams
                def getTrainCameras(self): return self._c
            with prof.stage("s8_prune_list", n_gaussians=N, n_views=len(train_cams)) as rec:
                gl, il = prune_list(gaussians, _S(list(train_cams)), pipe, bg)
                torch.cuda.synchronize()
                rec["extra"]["imp_nonzero"] = int((il != 0).sum())
                rec["extra"]["imp_sum"] = float(il.sum())
        except Exception as e:
            print(f"[s8 bỏ qua] {type(e).__name__}: {str(e)[:100]}", file=sys.stderr)

    footer = prof.close()
    summary = {
        "scene": args.scene, "repo": args.repo, "data_device": args.data_device,
        "resolution": args.resolution, "iteration": args.iteration,
        "n_gaussians": N, "ply_mb": round(ply_mb, 1),
        "stages": {s["stage"]: {"rss_peak_gb": s.get("rss_peak_gb"),
                                "vram_device_peak_mb": s.get("vram_device_peak_mb"),
                                "vram_alloc_peak_mb": s.get("vram_alloc_peak_mb"),
                                "wall_s": s.get("wall_s"),
                                **({k: v for k, v in s.get("extra", {}).items()})}
                   for s in prof.stages},
        "footer": footer,
    }
    print("###JSON###" + json.dumps(summary))
    return 0


# =============================================================================
# PARENT
# =============================================================================
def run_config(scene, repo, data_device, resolution, stages, iteration, out_dir, quiet=True):
    """out_dir LUÔN là THƯ MỤC; tên file do hàm này sinh ra.
    (Trước đây `--out` bị dùng vừa như file ở tiến trình con vừa như thư mục ở tiến trình cha
    -> os.makedirs tạo chính file .jsonl thành directory. Giờ tách bạch: cha nhận thư mục.)"""
    os.makedirs(out_dir, exist_ok=True)
    out_jsonl = os.path.join(out_dir, f"{scene}_{repo}_{data_device}_r{resolution}_stages.jsonl")
    if os.path.isdir(out_jsonl):          # dọn rác của bug cũ
        import shutil; shutil.rmtree(out_jsonl)
    if os.path.exists(out_jsonl):
        os.remove(out_jsonl)              # không append chồng lên run trước
    cmd = [sys.executable, os.path.abspath(__file__), "--_child",
           "--scene", scene, "--repo", repo, "--data_device", data_device,
           "--resolution", str(resolution), "--stages", stages,
           "--iteration", str(iteration), "--out", out_jsonl]
    if quiet:
        cmd.append("--quiet")
    t = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    dur = time.time() - t
    line = next((l for l in r.stdout.splitlines() if l.startswith("###JSON###")), None)
    if line:
        d = json.loads(line[len("###JSON###"):]); d["wall_total_s"] = round(dur, 1); d["ok"] = True
        return d
    err = (r.stderr or r.stdout).strip().splitlines()
    oom = any("out of memory" in l.lower() or "MemoryError" in l for l in err)
    return {"scene": scene, "repo": repo, "data_device": data_device, "resolution": resolution,
            "ok": False, "oom": oom, "returncode": r.returncode,
            "error": "\n".join(err[-6:])[:600], "wall_total_s": round(dur, 1)}


def print_stage_table(d: dict) -> None:
    """In bảng từng stage + các hệ số suy ra TỪ SỐ ĐO (không phải từ giả định)."""
    st = d["stages"]
    print(f"\n=== {d['scene']} | {d['repo']} | data_device={d['data_device']} | "
          f"N={d['n_gaussians']:,} | ply={d['ply_mb']:.0f} MB ===")
    print(f"{'stage':<22}{'RSS đỉnh':>10}{'ΔRSS':>9}{'VRAMdev':>10}{'VRAMalloc':>11}{'giây':>8}")
    print("-" * 70)
    prev = 0.0
    for name, s in st.items():
        r = s.get("rss_peak_gb") or 0.0
        print(f"{name:<22}{r:>10.3f}{r - prev:>+9.3f}"
              f"{s.get('vram_device_peak_mb') or 0:>10.1f}{s.get('vram_alloc_peak_mb') or 0:>11.1f}"
              f"{s.get('wall_s') or 0:>8.2f}")
        prev = max(prev, r)
    ft = d["footer"]
    print(f"\nĐỉnh RSS  {ft['rss_peak_gb']:.3f} GB @ {ft['rss_peak_stage']}")
    print(f"Đỉnh VRAM {ft['vram_device_peak_mb']:.0f} MB @ {ft['vram_device_peak_stage']}")
    print(f"OOM: {ft['any_oom']} · chạm swap: {ft['swap_touched']}")

    img = st.get("s3_cameras_train", {})
    if img.get("bytes_per_pixel"):
        print(f"\nẢNH (đo trực tiếp từ tensor): {img.get('img_shape')} {img.get('img_dtype')} "
              f"trên {img.get('img_device')} -> **{img['bytes_per_pixel']} B/pixel**, "
              f"alpha_mask lưu lại: {img.get('has_alpha_mask_attr')}")
        print(f"  tổng ảnh train theo lý thuyết {img.get('all_train_imgs_gb', 0):.3f} GB")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", choices=list(SCENES))
    ap.add_argument("--repo", choices=list(REPOS), default="pup")
    ap.add_argument("--data_device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--resolution", "-r", type=int, default=1)
    ap.add_argument("--iteration", type=int, default=30000)
    ap.add_argument("--stages", default="all", help="all | render | train")
    ap.add_argument("--hz", type=float, default=20.0, help="tần số lấy mẫu RSS (cao để bắt đỉnh thoáng qua)")
    ap.add_argument("--out", help="THU MUC dau ra (ten file tu sinh). O che do --_child day la FILE.")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--matrix", action="store_true", help="quét ma trận cấu hình")
    ap.add_argument("--_child", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args._child:
        return child(args)

    if not args.matrix:
        if not args.scene:
            ap.error("cần --scene hoặc --matrix")
        out_dir = args.out or "experiments/pipeline_matrix"
        d = run_config(args.scene, args.repo, args.data_device, args.resolution,
                       args.stages, args.iteration, out_dir, quiet=args.quiet)
        if not d.get("ok"):
            print(json.dumps(d, indent=2)); return 1
        print_stage_table(d)
        return 0

    # --- ma trận: nhỏ trước, dừng nhánh khi OOM -----------------------------
    out_dir = args.out or "experiments/pipeline_matrix"
    os.makedirs(out_dir, exist_ok=True)
    order = ["train", "truck", "playroom", "drjohnson"]          # tăng dần theo N
    configs = [(s, "pup", dd, 1) for s in order for dd in ("cpu", "cuda")]
    results = []
    print(f"{'scene':<11}{'dev':<6}{'N':>10}{'RSS đỉnh':>10}{'VRAM đỉnh':>11}{'stage đỉnh RSS':>20}{'s':>7}")
    print("-" * 78)
    for scene, repo, dd, res in configs:
        d = run_config(scene, repo, dd, res, args.stages, args.iteration, out_dir)
        results.append(d)
        with open(os.path.join(out_dir, "matrix.jsonl"), "a") as f:
            f.write(json.dumps(d) + "\n")
        if d.get("ok"):
            ft = d["footer"]
            print(f"{scene:<11}{dd:<6}{d['n_gaussians']:>10,}{ft['rss_peak_gb']:>10.2f}"
                  f"{ft['vram_device_peak_mb']:>11.0f}{ft['rss_peak_stage']:>20}{d['wall_total_s']:>7.0f}")
        else:
            tag = "OOM" if d.get("oom") else f"FAIL rc={d['returncode']}"
            print(f"{scene:<11}{dd:<6}{'':>10}{tag:>10}  {d['error'].splitlines()[-1][:40] if d.get('error') else ''}")
    print(f"\n-> {out_dir}/matrix.jsonl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
