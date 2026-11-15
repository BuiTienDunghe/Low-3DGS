import os, sys, time, torch, numpy as np
sys.path.insert(0, os.path.expanduser("~/l3dgs/third_party/gaussian-splatting-pup"))
os.chdir(os.path.expanduser("~/l3dgs/third_party/gaussian-splatting-pup"))

from scene.dataset_readers import readColmapSceneInfo
from scene.cameras import Camera
from scene.gaussian_model import GaussianModel
from prune import prune_list, calculate_v_imp_score

SRC   = "/mnt/d/low-3DGS/datasets/tandt/truck"
PLY   = "/mnt/d/low-3DGS/datasets/pretrained/models/truck/point_cloud/iteration_30000/point_cloud.ply"
V_POW = 0.1
PCT   = 0.66

class Pipe:  # matches PipelineParams fields used by count_render
    convert_SHs_python = False
    compute_cov3D_python = False
    debug = False

print("[1] reading colmap ...", flush=True)
info = readColmapSceneInfo(SRC, "images", True)
train_infos = info.train_cameras
print("    train cams:", len(train_infos), " test cams:", len(info.test_cameras), flush=True)

print("[2] building geometry-only cameras (dummy pixels; count_render never reads original_image) ...", flush=True)
cams = []
for i, c in enumerate(train_infos):
    w, h = c.image.size
    dummy = torch.zeros((3, h, w), dtype=torch.float32)
    cams.append(Camera(colmap_id=c.uid, R=c.R, T=c.T, FoVx=c.FovX, FoVy=c.FovY,
                       image=dummy, gt_alpha_mask=None, image_name=c.image_name,
                       uid=i, data_device="cpu"))
    del dummy
print("    built", len(cams), "cams @", cams[0].image_width, "x", cams[0].image_height, flush=True)

print("[3] load_ply ...", flush=True)
g = GaussianModel(3)
g.load_ply(PLY)
g.active_sh_degree = g.max_sh_degree
N = g.get_xyz.shape[0]
print("    N =", N, flush=True)

class FakeScene:
    def __init__(self, cams): self._c = cams
    def getTrainCameras(self): return self._c

scene = FakeScene(cams)
bg = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")

runs = []
for r in range(3):
    torch.cuda.synchronize(); t0 = time.time()
    gl, il = prune_list(g, scene, Pipe(), bg)
    torch.cuda.synchronize()
    runs.append((gl.detach().clone().float().cpu(), il.detach().clone().float().cpu()))
    print(f"    run {r}: {time.time()-t0:.1f}s  peak={torch.cuda.max_memory_allocated()/2**20:.0f}MiB "
          f"count_sum={runs[-1][0].sum().item():.0f} imp_sum={runs[-1][1].sum().item():.4f}", flush=True)

vol = torch.prod(g.get_scaling, dim=1).detach().cpu()

def mask_of(il):
    v = calculate_v_imp_score(g, il.cuda(), V_POW).detach().cpu()
    s, _ = torch.sort(v, dim=0)
    idx = int(PCT * (s.shape[0] - 1))
    thr = s[idx]
    return (v <= thr).squeeze(), v, thr

res = []
for (gl, il) in runs:
    m, v, thr = mask_of(il)
    res.append((m, v, thr))
    print(f"    prune_mask pruned={int(m.sum())} ({100.0*m.float().mean():.3f}%) thr={thr.item():.6g}", flush=True)

print("\n=== RAW COUNT DETERMINISM ===", flush=True)
for a, b in [(0, 1), (0, 2), (1, 2)]:
    d = (runs[a][0] - runs[b][0])
    nz = int((d != 0).sum())
    print(f" run{a} vs run{b}: exact-equal={bool(nz==0)} differing={nz}/{N} ({100.0*nz/N:.4f}%) "
          f"maxabs={d.abs().max().item():.0f} mean_signed={d.mean().item():+.6f} "
          f"sum_a={runs[a][0].sum().item():.0f} sum_b={runs[b][0].sum().item():.0f}", flush=True)
    di = (runs[a][1] - runs[b][1])
    print(f"           imp: differing={int((di!=0).sum())} maxabs={di.abs().max().item():.6g} "
          f"relsum={(runs[a][1].sum()-runs[b][1].sum()).item()/runs[a][1].sum().item():+.3e}", flush=True)

print("\n=== important_score == opacity * count ? ===", flush=True)
op = g.get_opacity.detach().squeeze().cpu()
pred = op * runs[0][0]
err = (pred - runs[0][1]).abs()
den = runs[0][1].abs().clamp_min(1e-12)
print(f" max abs err={err.max().item():.6g}  max rel err={(err/den).max().item():.6g}", flush=True)

print("\n=== PRUNE DECISION STABILITY (percent=0.66, v_pow=0.1) ===", flush=True)
for a, b in [(0, 1), (0, 2), (1, 2)]:
    ma, mb = res[a][0], res[b][0]
    sa, sb = ~ma, ~mb
    inter = int((sa & sb).sum()); union = int((sa | sb).sum())
    flip = int((ma ^ mb).sum())
    print(f" run{a} vs run{b}: survivors={int(sa.sum())}/{int(sb.sum())} "
          f"flipped={flip} ({100.0*flip/N:.4f}% of all, {100.0*flip/max(int(sa.sum()),1):.4f}% of survivors) "
          f"Jaccard={inter/union:.8f}", flush=True)

print("\n=== how tight is the decision boundary? ===", flush=True)
v = res[0][1]; thr = res[0][2]
sv, _ = torch.sort(v)
i = int(PCT * (N - 1))
for k in (1, 10, 100, 1000, 10000):
    lo = sv[max(i-k, 0)].item(); hi = sv[min(i+k, N-1)].item()
    print(f"  +-{k:>6} ranks around cut: v in [{lo:.6g}, {hi:.6g}]  rel width={(hi-lo)/max(thr.item(),1e-12):.3e}", flush=True)

print("\n=== per-camera count magnitude sanity (max possible hits) ===", flush=True)
c0 = runs[0][0]
print(f" count: min={c0.min().item():.0f} median={c0.median().item():.0f} mean={c0.mean().item():.1f} "
      f"max={c0.max().item():.0f} over {len(cams)} cams; zero-count={int((c0==0).sum())}", flush=True)
print(f" theoretical max pixel-hits per gaussian over all cams = {len(cams)*cams[0].image_width*cams[0].image_height}", flush=True)
print("DONE", flush=True)
