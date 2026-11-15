#!/usr/bin/env python3
"""Do he so bo nho workspace theo (N, so pixel). Chay vai buoc that:
render -> l1+ssim -> backward -> Adam.step, roi in dinh torch.cuda.max_memory_allocated()."""
import os, sys, json, time
REPO = os.path.expanduser('~/l3dgs/third_party/gaussian-splatting-pup')
sys.path.insert(0, REPO); os.chdir(REPO)
import torch
from random import randint
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene import Scene, GaussianModel
from gaussian_renderer import render
from utils.general_utils import safe_state
from utils.loss_utils import l1_loss, ssim

p = ArgumentParser()
lp = ModelParams(p); op = OptimizationParams(p); pp = PipelineParams(p)
p.add_argument('--start_pointcloud', required=True)
p.add_argument('--nsub', type=int, default=0)
p.add_argument('--iters', type=int, default=4)
p.add_argument('--tag', default='probe')
args = p.parse_args(sys.argv[1:])
safe_state(False)
dataset, opt, pipe = lp.extract(args), op.extract(args), pp.extract(args)
os.makedirs(dataset.model_path, exist_ok=True)

g = GaussianModel(dataset.sh_degree)
scene = Scene(dataset, g)
g.load_ply(args.start_pointcloud)
if args.nsub and args.nsub < g.get_xyz.shape[0]:
    idx = torch.randperm(g.get_xyz.shape[0], device='cuda')[:args.nsub]
    import torch.nn as nn
    g._xyz = nn.Parameter(g._xyz[idx].detach().requires_grad_(True))
    g._features_dc = nn.Parameter(g._features_dc[idx].detach().requires_grad_(True))
    g._features_rest = nn.Parameter(g._features_rest[idx].detach().requires_grad_(True))
    g._opacity = nn.Parameter(g._opacity[idx].detach().requires_grad_(True))
    g._scaling = nn.Parameter(g._scaling[idx].detach().requires_grad_(True))
    g._rotation = nn.Parameter(g._rotation[idx].detach().requires_grad_(True))
    del idx
g.active_sh_degree = g.max_sh_degree
g.training_setup(opt)
g.max_radii2D = torch.zeros((g.get_xyz.shape[0]), device='cuda')
N = g.get_xyz.shape[0]
bg = torch.tensor([1,1,1] if dataset.white_background else [0,0,0], dtype=torch.float32, device='cuda')
cams = scene.getTrainCameras().copy()
H, W = cams[0].image_height, cams[0].image_width
torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
resident = torch.cuda.memory_allocated()/2**20
t_iters=[]
import time as _t
for i in range(args.iters):
    _t0=_t.monotonic()
    cam = cams[randint(0, len(cams)-1)]
    out = render(cam, g, pipe, bg)
    gt = cam.original_image.cuda()
    L1 = l1_loss(out['render'], gt)
    loss = 0.8*L1 + 0.2*(1.0 - ssim(out['render'], gt))
    loss.backward()
    g.optimizer.step(); g.optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize(); t_iters.append(_t.monotonic()-_t0)
    del out, gt, L1, loss
torch.cuda.synchronize()
peak_alloc = torch.cuda.max_memory_allocated()/2**20
peak_res = torch.cuda.max_memory_reserved()/2**20
rec = dict(tag=args.tag, N=N, W=W, H=H, px=W*H, resident_after_setup_MiB=round(resident,1),
           peak_alloc_MiB=round(peak_alloc,1), peak_reserved_MiB=round(peak_res,1), s_per_iter=round(sum(t_iters[max(2,len(t_iters)//4):])/max(1,len(t_iters)-max(2,len(t_iters)//4)),4))
print('RESULT ' + json.dumps(rec), flush=True)
with open(os.path.expanduser('~/l3dgs/out/memprobe/results.jsonl'),'a') as f:
    f.write(json.dumps(rec)+chr(10))
