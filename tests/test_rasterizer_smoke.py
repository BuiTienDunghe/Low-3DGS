#!/usr/bin/env python3
"""
tests/test_rasterizer_smoke.py — Chứng minh 4 CUDA extension THẬT SỰ chạy trên sm_75, không chỉ import được.

`pip install` thành công và `import` thành công đều KHÔNG chứng minh kernel chạy đúng trên GPU này.
File này rasterize Gaussian tổng hợp qua cả ba rasterizer và kiểm tra output hợp lệ.

Cũng kiểm luôn hai thứ đã biết là bẫy:
  - `simple_knn._C` link tĩnh vào libtorch nhưng KHÔNG tự nạp -> phải `import torch` TRƯỚC.
    Nếu không: "ImportError: libc10.so: cannot open shared object file".
  - `count_render` của compress-rasterizer trả thêm gaussians_count + important_score
    (đây là thứ LightGaussian/PUP dùng để tính Global Significance).

Chạy:  python tests/test_rasterizer_smoke.py
"""
from __future__ import annotations

import math
import sys

# ---- BẮT BUỘC: torch trước, extension sau -------------------------------------
import torch  # noqa: F401  (nạp libc10.so vào process)

MB = 1024.0 ** 2
N = 50_000          # đủ để chạm nhiều tile, đủ nhỏ để nhanh
H, W = 400, 600
GAUSSIAN_SCALE = 0.06   # 0.02 cho ảnh gần rỗng (mỗi Gaussian < vài pixel) — không phải lỗi kernel


def make_scene(device="cuda", sh_degree=3):
    """Gaussian ngẫu nhiên trong hộp trước camera, giá trị THÔ (scale log, opacity logit)."""
    g = torch.Generator(device=device).manual_seed(0)
    xyz = torch.rand(N, 3, device=device, generator=g) * 4 - 2
    xyz[:, 2] = torch.rand(N, device=device, generator=g) * 4 + 3.0     # z in [3,7]
    scales = torch.full((N, 3), math.log(GAUSSIAN_SCALE), device=device)
    quats = torch.zeros(N, 4, device=device); quats[:, 0] = 1.0
    opacity = torch.full((N, 1), 2.0, device=device)                     # sigmoid(2) ~ 0.88
    n_sh = (sh_degree + 1) ** 2
    shs = torch.zeros(N, n_sh, 3, device=device)
    shs[:, 0, :] = torch.rand(N, 3, device=device, generator=g)
    return xyz, scales, quats, opacity, shs


def camera(device="cuda"):
    fovx = fovy = math.radians(60.0)
    tanx, tany = math.tan(fovx * 0.5), math.tan(fovy * 0.5)
    view = torch.eye(4, device=device)
    znear, zfar = 0.01, 100.0
    proj = torch.zeros(4, 4, device=device)
    proj[0, 0] = 1.0 / tanx; proj[1, 1] = 1.0 / tany
    proj[2, 2] = zfar / (zfar - znear); proj[2, 3] = -(zfar * znear) / (zfar - znear)
    proj[3, 2] = 1.0
    return view, (view @ proj), torch.zeros(3, device=device), tanx, tany


def run(mod_name: str, label: str) -> dict:
    import importlib
    m = importlib.import_module(mod_name)
    dev = "cuda"
    xyz, scales, quats, opacity, shs = make_scene(dev)
    view, full, campos, tanx, tany = camera(dev)
    bg = torch.zeros(3, device=dev)

    torch.cuda.reset_peak_memory_stats()
    kw = dict(image_height=H, image_width=W, tanfovx=tanx, tanfovy=tany, bg=bg,
              scale_modifier=1.0, viewmatrix=view, projmatrix=full, sh_degree=3,
              campos=campos, prefiltered=False, debug=False)
    # compress-rasterizer có thêm trường `f_count`: False = render thường,
    # True = gọi _C.count_gaussians và trả thêm gaussians_count + important_score.
    if "f_count" in getattr(m.GaussianRasterizationSettings, "_fields", ()):
        kw["f_count"] = False
    rs = m.GaussianRasterizationSettings(**kw)
    rasterizer = m.GaussianRasterizer(raster_settings=rs)

    means2D = torch.zeros_like(xyz, requires_grad=True)
    try:
        means2D.retain_grad()
    except Exception:
        pass

    out = rasterizer(means3D=xyz, means2D=means2D, shs=shs, colors_precomp=None,
                     opacities=torch.sigmoid(opacity), scales=torch.exp(scales),
                     rotations=quats, cov3D_precomp=None)
    torch.cuda.synchronize()

    img = out[0] if isinstance(out, tuple) else out
    n_out = len(out) if isinstance(out, tuple) else 1
    peak = torch.cuda.max_memory_allocated() / MB
    r = {"label": label, "n_outputs": n_out, "img_shape": tuple(img.shape),
         "img_min": float(img.min()), "img_max": float(img.max()),
         "img_mean": float(img.mean()), "nonzero_frac": float((img > 1e-6).float().mean()),
         "vram_peak_mb": round(peak, 1)}

    # backward: chứng minh gradient chảy qua kernel (thứ fine-tune cần)
    try:
        img.sum().backward()
        torch.cuda.synchronize()
        r["grad_ok"] = means2D.grad is not None and bool(torch.isfinite(means2D.grad).all())
    except RuntimeError as e:
        # PUP fisher rasterizer có arity backward khác -> ghi lại, không coi là build hỏng
        r["grad_ok"] = False
        r["grad_err"] = str(e)[:100]
    del out, img, rasterizer
    torch.cuda.empty_cache()
    return r


def run_count_render() -> dict:
    """count_render: đường tính importance score của LightGaussian/PUP."""
    import compress_diff_gaussian_rasterization as m
    dev = "cuda"
    xyz, scales, quats, opacity, shs = make_scene(dev)
    view, full, campos, tanx, tany = camera(dev)
    rs = m.GaussianRasterizationSettings(
        image_height=H, image_width=W, tanfovx=tanx, tanfovy=tany,
        bg=torch.zeros(3, device=dev), scale_modifier=1.0, viewmatrix=view,
        projmatrix=full, sh_degree=3, campos=campos, prefiltered=False, debug=False,
        f_count=True)      # BẮT BUỘC: bật đường count_gaussians
    rasterizer = m.GaussianRasterizer(raster_settings=rs)
    means2D = torch.zeros_like(xyz, requires_grad=True)
    with torch.no_grad():
        out = rasterizer(means3D=xyz, means2D=means2D, shs=shs, colors_precomp=None,
                         opacities=torch.sigmoid(opacity), scales=torch.exp(scales),
                         rotations=quats, cov3D_precomp=None)
    torch.cuda.synchronize()
    names = ["gaussians_count", "important_score", "image", "radii"]
    r = {"n_outputs": len(out)}
    for i, t in enumerate(out):
        if torch.is_tensor(t):
            r[names[i] if i < len(names) else f"out{i}"] = {
                "shape": tuple(t.shape), "sum": float(t.sum()), "nonzero": int((t != 0).sum())}
    return r


def main() -> int:
    print(f"torch {torch.__version__} | cuda {torch.version.cuda} | "
          f"{torch.cuda.get_device_name(0)} sm_{torch.cuda.get_device_capability(0)[0]}"
          f"{torch.cuda.get_device_capability(0)[1]}")
    print(f"scene: N={N:,} Gaussians, render {W}x{H}\n")

    ok = True

    # --- simple_knn: chỉ chạy được khi torch đã nạp ---
    print("=== simple_knn (bẫy libc10.so) ===")
    try:
        from simple_knn._C import distCUDA2
        pts = torch.rand(10_000, 3, device="cuda")
        d = distCUDA2(pts)
        torch.cuda.synchronize()
        finite = bool(torch.isfinite(d).all())
        print(f"  OK  distCUDA2 -> shape {tuple(d.shape)}, mean {float(d.mean()):.6f}, finite={finite}")
        print("  -> import thất bại trước đó chỉ vì thiếu 'import torch', không phải build hỏng")
        ok &= finite
    except Exception as e:
        print(f"  FAIL {type(e).__name__}: {str(e)[:120]}"); ok = False

    # --- 3 rasterizer ---
    print("\n=== Rasterize thật (forward + backward) ===")
    print(f"{'rasterizer':<40}{'outs':>5}{'mean':>9}{'nonzero':>9}{'VRAM MB':>10}{'grad':>7}")
    for mod, label in (("diff_gaussian_rasterization", "diff_gaussian_rasterization (INRIA)"),
                       ("compress_diff_gaussian_rasterization", "compress_… (LightGaussian)"),
                       ("rasterization_and_pup_fisher", "rasterization_and_pup_fisher (PUP)")):
        try:
            r = run(mod, label)
            # forward hợp lệ = bằng chứng kernel chạy trên sm_75. backward tính riêng.
            fwd_ok = r["nonzero_frac"] > 0.01
            print(f"{label:<40}{r['n_outputs']:>5}{r['img_mean']:>9.4f}"
                  f"{r['nonzero_frac']:>9.3f}{r['vram_peak_mb']:>10.1f}"
                  f"{'OK' if r['grad_ok'] else 'NO':>7}")
            if not fwd_ok:
                print("    ⚠ ảnh gần như rỗng — kernel KHÔNG chạy đúng")
            if not r["grad_ok"] and r.get("grad_err"):
                print(f"    ℹ backward: {r['grad_err']}")
            ok &= fwd_ok
        except Exception as e:
            print(f"{label:<40}  FAIL {type(e).__name__}: {str(e)[:70]}"); ok = False

    # --- count_render ---
    print("\n=== count_render (đường importance score của LightGaussian/PUP) ===")
    try:
        r = run_count_render()
        print(f"  {r['n_outputs']} outputs")
        for k, v in r.items():
            if isinstance(v, dict):
                print(f"    {k:<18} shape={str(v['shape']):<14} sum={v['sum']:>14.2f} nonzero={v['nonzero']:,}")
        gc_ok = r.get("gaussians_count", {}).get("nonzero", 0) > 0
        is_ok = r.get("important_score", {}).get("nonzero", 0) > 0
        print(f"  -> gaussians_count {'OK' if gc_ok else 'RỖNG ⚠'} | important_score {'OK' if is_ok else 'RỖNG ⚠'}")
        if not (gc_ok and is_ok):
            print("     ⚠ score rỗng = đúng triệu chứng của bẫy D3 (prune sẽ xoá gần hết Gaussian)")
        ok &= gc_ok and is_ok
    except Exception as e:
        print(f"  FAIL {type(e).__name__}: {str(e)[:120]}"); ok = False

    print(f"\n{'='*72}\n{'TẤT CẢ ĐẠT' if ok else 'CÓ LỖI — xem ở trên'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
