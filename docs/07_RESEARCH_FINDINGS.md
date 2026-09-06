# Kết quả khảo sát cộng đồng — sửa plan theo thực tế

Nguồn: 4 agent khảo sát repo/paper/issue trên mạng (2026-09-02). Mỗi claim có URL.
Mục "⚠️ chưa verify" là agent tự đánh dấu — **không được cite những mục đó**.

> Trạng thái: **Cả 4 agent đã xong.** Mọi claim then chốt đã được tôi verify lại trong source đã clone.

---

## PHẦN 0 — BỐN QUYẾT ĐỊNH LỚN (đọc phần này trước)

| # | Quyết định | Thay cho | Vì sao |
|---|---|---|---|
| **Q1** | **Không tự viết PLY loader/writer — dùng [`gsply`](https://github.com/OpsiClear/gsply)** | `src/loading/ply_stream.py` | MIT, wheel **cp312 manylinux, không cần build CUDA**. Đọc 1.95× file / **nhanh hơn 6×**; ghi 1.30× thay vì **12.08×**. Đã verify `torch.equal` bitwise trên cả 6 tensor |
| **Q2** | **`save_ply` mới là blocker, không phải `load_ply`** | Toàn bộ trọng tâm cũ | INRIA `save_ply` ≈ **10.2 GB** cho drjohnson — vượt cả RAM. Xem EXP-004 |
| **Q3** | ✅ **CHỐT: KHÔNG dùng repo LightGaussian gốc.** Chạy tiêu chí của nó bên trong [PUP 3D-GS](https://github.com/j-alex-hanson/gaussian-splatting-pup) (torch 2.4.1, đã vá lỗi iteration) hoặc [gaussian-splatting-lightning](https://github.com/yzslab/gaussian-splatting-lightning) (trên gsplat, không cần build rasterizer riêng) | Repo LightGaussian gốc | **Agent 1 xác nhận và làm mạnh thêm:** repo gốc có **5 bẫy im lặng** đã verify, rasterizer pin **không implement công thức trong paper**, và `distill_train.py` **không dùng được `.ply`**. Xem P8 |
| **Q4** | **Bỏ MipNeRF360 khỏi phạm vi, chỉ làm T&T + Deep Blending** | Kế hoạch stress `bicycle` | Fine-tune 5.7M Gaussian ≈ 5.5 GB VRAM — bất khả thi. Và **LightGaussian không có số Deep Blending nào** → đó là khoảng trống công bố được, không phải reproduction thất bại |

---

## PHẦN 6 — Tooling: cái gì đã có sẵn (Agent 2)

### Đã verify trong source đã clone (2026-09-02)

| Claim | Verify | Vị trí |
|---|---|---|
| `np.zeros((N,45))` thiếu dtype → **float64** | ✅ đúng | `gaussian_model.py:load_ply`, cả `features_dc`, `features_extra`, `scales`, `rots` |
| `elements[:] = list(map(tuple, attributes))` | ✅ đúng | `gaussian_model.py:save_ply` |
| `alpha_mask` luôn `ones_like` → 16 B/px | ✅ đúng | `cameras.py:48` |
| `data_device` mặc định `"cuda"` | ✅ đúng | `arguments/__init__.py:57` |
| `train.py` gọi `Scene(dataset, gaussians)` **không** `load_iteration` | ✅ đúng | `train.py:51` vs `render.py:51` |
| Patch CUDA 12 **chưa có** trong upstream | ✅ đúng | `grep -c cstdint` = 0 · `grep -c float.h` = 0 |

### Thay thế và bản vá cụ thể

| Việc | Dùng gì | Ghi chú |
|---|---|---|
| Đọc/ghi `.ply` | [`gsply`](https://github.com/OpsiClear/gsply) v0.4.6, MIT, 59★ | Trả **giá trị thô trước activation** (scale log, opacity logit) — đúng cái 3DGS cần. ⚠️ `plywrite` bỏ `nx,ny,nz` |
| Cache ảnh uint8 (giảm 4×) | [`gaussian-splatting-lightning`](https://github.com/yzslab/gaussian-splatting-lightning) 1.1k★, push 25/05/2026 — có `--data.image_uint8`, `image_on_cpu` (mặc định True), `train_max_num_images_to_cache`, `async_caching`, sẵn `configs/image_on_gpu-uint8.yaml` | **Giải trực tiếp EXP-005** |
| Vá uint8 cho chính INRIA | [PR #816](https://github.com/graphdeco-inria/gaussian-splatting/pull/816) — `--store_images_as_uint8`, `original_image` thành property tự convert | ~20 dòng, PR còn mở |
| Lazy load ảnh từ đĩa | nerfstudio `full_images_datamanager.py`: `cache_images: "cpu"/"gpu"/"disk"` + `cache_images_type: "uint8"/"float32"`, tự hạ cấp khi > 500 ảnh | Tham khảo thiết kế |
| Fine-tune từ `.ply` bằng INRIA gốc | Sửa `train.py:51` thành `Scene(dataset, gaussians, load_iteration=30000)` | **Đơn giản hơn `ply_to_checkpoint.py` của tôi.** `training_setup(opt)` chạy sau nên thứ tự vẫn đúng; lưu ý `spatial_lr_scale` vẫn phải set |
| Build trên Ubuntu 24.04 + CUDA 12 | [issue #923](https://github.com/graphdeco-inria/gaussian-splatting/issues/923): `#include <cstdint>` vào `rasterizer_impl.h`, `#include <float.h>` vào `simple_knn.cu`, `export TORCH_CUDA_ARCH_LIST="7.5"` | **Đã tự động hoá trong `setup_wsl_sudo.sh` bước 5/6** |
| `cfg_args` bundle cũ | [issue #198](https://github.com/graphdeco-inria/gaussian-splatting/issues/198): thiếu `data_device` → `AttributeError`. **Phải truyền `--data_device` tường minh** | Đã sửa trong `p1_smoke.sh` |
| Tăng tốc miễn phí | Branch `3dgs_accel` của diff-gaussian-rasterization (đã tích hợp Taming-3DGS + fused-ssim) + `--optimizer_type sparse_adam` | README INRIA §494-512 |

### gsplat — tách làm hai quyết định

- **Nén: dùng ngay.** `from gsplat.exporter import load_ply_to_splats` đọc thẳng PLY định dạng INRIA;
  output khớp đúng cái `PngCompression().compress()` cần. SH layout, activation log/logit, quy ước quaternion wxyz
  **khớp INRIA từng byte** — không cần math chuyển đổi. Công bố: 236 → **16.5 MB, −0.5 dB**.
  ⚠️ `load_ply_to_splats` **chỉ có trên `main`/1.6.0**, không có ở 1.5.3; `main` đang có
  [build break #1028](https://github.com/nerfstudio-project/gsplat/issues/1028); không có wheel cp312+cu126 → build từ nguồn (`MAX_JOBS=1-2`).
- **Fine-tune: KHÔNG port.** Con số thật single-GPU là **9.0 → 5.7 GB ≈ 1.6×**, không phải 4×
  (4× là 4 GPU, hoặc MCMC train model *nhỏ hơn*). [Migration guide của chính gsplat](https://docs.gsplat.studio/main/migration/migration_inria.html)
  đo việc thay rasterizer là **9.11 → 8.78 GB = 3.6%**. `simple_trainer.py --ckpt` **chỉ eval, không resume train được**.
  `sparse_grad` chỉ phủ `{means,quats,scales}` = 10/59 float (17%) trong khi checkpoint của ta **81% là SH**.

### Nén rẻ, không cần build

| Repo | Vì sao |
|---|---|
| [FCGS](https://github.com/YihangChen-ee/FCGS) | `--ply_path_from`, feed-forward, **không tối ưu theo scene, không cần ảnh**. Chỉnh `chunk_size_list` ở `encode_single_scene.py:34` cho 4 GB |
| [mini-splatting `ms_c/`](https://github.com/fatPeter/mini-splatting) | Pass nén hoàn toàn `torch.no_grad()`, **không cần CUDA extension**. Tác giả đo RTX 3090: train 7.45 → **2.61 GB**, render 2.79 → **0.40 GB**; và có chạy trên **GTX 1060 6 GB** |
| [c3dgs](https://github.com/KeKsBoTer/c3dgs) | 795 → 28.8 MB (26×), PSNR 27.21 → 26.98. Bỏ finetune mất ~1 dB |
| [splat-transform](https://github.com/playcanvas/splat-transform) · [3dgsconverter](https://github.com/francescofugazzi/3dgsconverter) | Zero-GPU |

### ✅ Cái duy nhất của tôi mà agent KHÔNG tìm thấy bản có sẵn

> **Per-stage RAM+VRAM profiler.** `src/memory/profiler.py` (tách 3 chiều alloc/reserved/NVML +
> thread lấy mẫu + gắn theo stage) **chi tiết hơn mọi thứ đã công bố.** → Tiếp tục xây. Đây là phần thật sự của mình.

---

## PHẦN 7 — Nên reproduce method nào (Agent 3)

### Bảng xếp hạng (số lấy từ CSV của benchmark 3DGS.zip; T&T baseline 23.14/411 MB · DB 29.41/676 MB)

| # | Method | Repo | Commit cuối | Stack hiện đại? | Post-hoc `.ply`? | T&T | DB | Verdict |
|---|---|---|---|---|---|---|---|---|
| **1** | **PUP 3D-GS** (CVPR'25) | [j-alex-hanson/…-pup](https://github.com/j-alex-hanson/gaussian-splatting-pup) 152★ | **22/11/2025** | ✅ **torch 2.4.1** | ✅ `--start_pointcloud` | 22.72 @90% | **28.85** @90% | **Cao** — repo post-hoc **duy nhất** trên torch 2.x, 0 issue mở |
| 2 | **LightGaussian** | [VITA-Group](https://github.com/VITA-Group/LightGaussian) 819★ | 30/12/2024 | Pin py3.9/torch1.12 | ✅ | 23.11 / 22 MB | **không có** | **TB-Cao** — đúng pipeline của ta nhưng **chết** |
| 3 | gsplat MCMC+PNG | [gsplat](https://github.com/nerfstudio-project/gsplat) 5.6k★ | 01/09/2026 | ✅ | một phần | **23.54 / 6.88 MB** | — | Cao (build) / TB (làm mục tiêu reproduce) |
| 4 | reduced-3dgs | [graphdeco-inria](https://github.com/graphdeco-inria/reduced-3dgs) 239★ | 22/09/2025 | py3.7 pin | **⅓** — prune+SH nằm trong `train.py` | **23.57 / 14 MB** | 29.63 / 18 MB | TB — **không tránh được train from scratch** |
| 5 | Self-Organizing Gaussians | [fraunhoferhhi](https://github.com/fraunhoferhhi/Self-Organizing-Gaussians) 394★ | gần đây | ✅ py3.10/torch2.4 | ❌ | 23.56 / 22.8 MB | 29.26 / 17.7 MB | TB — **code sạch nhất ngành**, sai workflow |
| 6 | Mini-Splatting | [fatPeter](https://github.com/fatPeter/mini-splatting) 227★ | 10/2024 | py3.7 | `ms_c/run.py` ✅ | 23.18 | **29.98** | TB — paper **duy nhất** có run GTX 1060 6 GB |
| 7 | Compact-3DGS | [maincold2](https://github.com/maincold2/Compact-3DGS) 500★ | 09/2024 | py3.7.13 | ❌ | 23.32 / 20.9 MB | 29.73 / 23.8 MB | Thấp-TB — cần **tiny-cuda-nn**, rủi ro ≥ rasterizer |
| 8 | HAC / HAC++ | [YihangChen-ee](https://github.com/YihangChen-ee/HAC-plus) 209★ | 11/2025 | py3.7.13 | ❌ Scaffold anchor | **24.33 / 7.26 MB** | **30.34 / 5.5 MB** | Thấp — số SOTA nhưng **biểu diễn không tương thích** |
| 11 | Trimming the Fat | [salmanali96](https://github.com/salmanali96/Trimming-the-Fat) 15★ | 2 commit | py3.8 | ❌ cần `.pth` mà INRIA không phát hành | 23.96 / 336 MB | 29.50 / 534 MB | Thấp |
| 13 | MEGS² | [IGL-HKUST](https://github.com/IGL-HKUST/MEGS-2) 60★ | 02/2026 | py3.7.13 | ❌ densify tới 4.5M trước | 23.45 / 51 MB | 30.17 / 54 MB | Thấp — tối ưu VRAM *render*, không phải storage |

**Ba repo rẻ đáng biết thêm:**
[**REFINE**](https://github.com/ZhangChen2022/REFINE) (ECCV'26) — `pip install torch numpy plyfile`, **một file, không rasterizer**,
README **khuyến nghị đúng pretrained models của INRIA**, chunked sẵn (`--camera_chunk`, `--point_chunk`), score Hessian không cần render.
Zero-shot @50%: T&T 22.97 · DB 29.28. **Tốn một buổi chiều thay vì một tháng.**
[**NanoGS**](https://github.com/saliteta/NanoGS) (ECCV'26) — pip, **NumPy thuần, zero VRAM**, `.ply` vào → `.ply` ra.
[**GSCodec Studio**](https://github.com/JasonLSC/GSCodec_Studio) 139★ Apache-2.0 — fork gsplat có package `post_training/` thật.

### Rủi ro thật của LightGaussian không phải cái tôi lo

Tôi lo rasterizer fork không build. Agent diff nó: **~450 dòng cộng thêm**, `setup.py` **giống hệt từng byte**,
chỉ thêm một export `count_gaussians`. Đường rasterize/backward lõi là stock. Và hai patch CUDA 12 cần thiết
thì **stock INRIA cũng thiếu y hệt** → không phải rủi ro riêng của LightGaussian.

**Rủi ro thật, tệ hơn:**
1. **Không có số Deep Blending nào.** [Project page](https://lightgaussian.github.io/) chỉ liệt kê MipNeRF360 và T&T;
   dòng `DeepBlending.csv` của survey là `fan2024lightgaussian,,,,,,,,` — **rỗng**. → 2/4 scene của ta không có gì để so.
2. **Repo chết.** [Issue #40](https://github.com/VITA-Group/LightGaussian/issues/40) (build submodule, 10/2024) và
   [#51](https://github.com/VITA-Group/LightGaussian/issues/51) (12/2025, hỏi **đúng câu ta sẽ gặp**: `--teacher_model` lấy từ đâu) — **cả hai mở, không ai trả lời**.
3. **Trên leaderboard hiện tại nó thuộc nhóm yếu**: T&T 23.11/22 MB, **thua PSNR baseline 23.14**.
   reduced-3dgs (23.57/14 MB), SOG (23.56/22.8 MB), gsplat MCMC+PNG (23.54/**6.88 MB**) thắng **cả hai trục cùng lúc**.

### 🔧 Phát hiện kỹ thuật đáng giá nhất: đảo thứ tự prune và `training_setup`

Cả LightGaussian lẫn PUP đều gọi `load_ply()` → `training_setup(opt)` → **rồi mới** prune ở iteration 2.
**Adam cấp phát moment đủ cỡ TRƯỚC khi prune.**

Chuyển pass tính importance + prune lên **trước** `training_setup()`:

```
đỉnh: 0.96 KB/Gaussian  ->  0.24 KB/Gaussian
bicycle: 5.5 GB -> 1.35 GB
```

**~5 dòng code**, và là một đóng góp kỹ thuật báo cáo được. Đây đúng là loại phát hiện project này cần tìm.

### Ngân sách VRAM fine-tune theo scene (số học từ 944 B/Gaussian, **chưa ai chạy thật trên card 4 GB**)

| Scene | ≈N | Tensor fine-tune | Trên 4 GB |
|---|---|---|---|
| T&T **train** | 1.03M | ~1.0 GB | ✅ thoải mái |
| DB **playroom** | 2.55M | ~2.4 GB | ⚠️ chật |
| T&T **truck** | 2.54M | ~2.4 GB | ⚠️ rất chật |
| DB **drjohnson** | 3.41M | ~3.2 GB | ❌ nhiều khả năng OOM |
| M360 bicycle | ~5.7M | ~5.5 GB | ❌ không |

→ **Q4: bỏ MipNeRF360.** Và với thủ thuật đảo thứ tự ở trên, drjohnson từ 3.2 GB xuống ~0.8 GB — **cứu được scene lớn nhất**.

### Nếu đổi sang PUP thì mất gì

PUP chỉ prune — không có SH distillation, không có quantization. Nhưng:
- **VecTree quantization: port miễn phí.** Cả 3 file trong `vectree/` (`vectree.py`, `utils.py`, `vq.py`) có
  **zero** tham chiếu tới `diff_gaussian_rasterization` hay `simple_knn`. Chỉ cần torch, numpy, plyfile, einops.
  Đọc `.ply` + `imp_score.npz` → ghi model đã nén. Copy thư mục sang là chạy.
- **SH distillation: đây mới là việc port thật.** `distill_train.py` nạp teacher và student từ `.pth` (L74-77) mà
  INRIA không phát hành `.pth`. Thay bằng nhánh `load_ply` mà PUP đã có sẵn. ~200 dòng.
  → Agent nhận xét: **"không tồn tại pipeline post-hoc prune + SH-distill + quantize nào được bảo trì trên stack hiện đại"**
  — nếu ta làm, đó là deliverable thật.

⚠️ **Cảnh báo về PUP:** `--prune_type fisher` cấp phát `torch.zeros(N,6,6)` **trên GPU** + `torch.linalg.svdvals` theo batch
→ ~470 MB cho drjohnson **trước cả workspace SVD**. Tiêu chí LightGaussian chỉ 1 float/Gaussian (~13 MB).
Trên 4 GB, **tiêu chí LightGaussian mới là cái rẻ**; dùng `--fisher_resolution 4` và bắt đầu từ scene nhỏ.

### Baseline thứ hai

**So sánh chính đề xuất: Fisher score của PUP vs Global Significance của LightGaussian, trong cùng repo PUP.**
Cùng loader, cùng fine-tune, cùng split, một lần build. Paper PUP đã công bố head-to-head này
(T&T: LightGaussian 23.08 vs PUP 22.72 — **LightGaussian thắng PSNR** nhưng thua SSIM/LPIPS) → có đích để đối chiếu.

**Không chọn reduced-3dgs** dù số đẹp nhất: prune và SH-culling nằm trong vòng densification của `train.py`
from-scratch, không có `--start_pointcloud`; và `compress.py` post-hoc vẫn import `kmeans_cuda` từ rasterizer của họ
→ không tránh được build, không tránh được train.
**Không chọn HAC/Scaffold-GS**: checkpoint là `.ply` **+** `color_mlp.pt`/`cov_mlp.pt`/`opacity_mlp.pt` — biểu diễn khác,
không có đường từ `.ply` INRIA sang.

### Agent 3 chưa verify
- **Chưa ai công khai chạy bất kỳ method nào trong bảng trên card 4 GB.** Bằng chứng tốt nhất là Mini-Splatting:
  *"a low-cost graphics card (a GTX 1060 6G GPU)"* — 6 GB, không phải 4. Bảng VRAM ở trên là **số học từ cấu trúc model, không phải run thật**.
  → **Phải tự validate trên T&T `train` ngay tuần đầu.**
- **Python 3.12 + CUDA 12.6 chưa được verify cho bất kỳ repo học thuật nào** — không có issue báo thành công *lẫn* thất bại.
- Số Gaussian mỗi scene dao động ±20% giữa các lần train.
- Số size của Mini-Splatting trên survey **sai** (chữ số là số Gaussian ×1000, cột size trống) — **đừng cite**.
- Smol-GS và GSICO đứng đầu leaderboard nhưng repo ghi *"Codes to be released"*.

---

## PHẦN 1 — Lỗi trong plan sẽ làm hỏng SỐ LIỆU (sửa trước khi chạy experiment đầu tiên)

Đây là phần quan trọng nhất. Bốn lỗi dưới đây sẽ khiến mọi con số của project **không so sánh được với
literature**, và không thể sửa hồi tố.

### L1. LPIPS — phải dùng bản vendored của INRIA, KHÔNG dùng `lpips` pip

INRIA `metrics.py` gọi `lpips(..., net_type='vgg')` từ thư mục **`lpipsPyTorch/` tự viết trong repo**,
không phải package `lpips` trên PyPI. (Lưu ý bẫy: `lpipsPyTorch/__init__.py` mặc định `'alex'`, nhưng
`metrics.py` ghi đè thành `'vgg'`.)

**Và bản vendored này có LỖI CHUẨN HOÁ đã biết:**
[issue #1239](https://github.com/graphdeco-inria/gaussian-splatting/issues/1239) (mở 30/05/2025, **vẫn OPEN**) —
hằng số mean/std thiết kế cho input `[-1,1]` nhưng ảnh vào là `[0,1]` từ `torchvision to_tensor()`.
Sửa lỗi làm LPIPS **tăng**, tức mọi paper dòng 3DGS đang báo LPIPS **lạc quan một cách hệ thống**.

**Cộng đồng quyết định GIỮ lỗi để so sánh được.** KISS-GS (8/2026) nói rõ: VGG backbone, ảnh giữ `[0,1]`.
→ **Ta làm y hệt, và ghi rõ trong report.**

Ba tool ba kiểu LPIPS **không tương thích**: INRIA (vendored VGG) · Splatwizard (`lpips` pip) ·
gsplat (`torchmetrics`, [chính docs gsplat cảnh báo](https://docs.gsplat.studio/main/tests/eval.html)
*"different from what's reported in the original paper"*). **Chọn INRIA, không bao giờ trộn.**

### L2. SSIM — cửa sổ Gauss 11×11 σ=1.5 của INRIA, KHÔNG dùng `skimage`

`utils/loss_utils.py`: window 11×11, σ=1.5, `C1=0.01²`, `C2=0.03²`, `padding=5`, conv theo nhóm kênh.
`skimage.metrics.structural_similarity` cho số **khác**. Plan cũ ghi "skimage hoặc impl trong repo" — **sai**, phải chốt INRIA.

### L3. MB = 10⁶ byte, KHÔNG phải 2²⁰

Quy ước 3DGS.zip. Nhầm → sai **4.9%**, lớn hơn biên cải thiện của nhiều paper.
`src/quantization/size.py` phải dùng 10⁶.

### L4. Cách đo "model size" trong plan là SAI quy ước

Plan cũ định báo cáo 4 số: `ply_fp32` / `raw_quantized` / `npz_fp16` / `entropy`. Quy ước thật của ngành:

> **Size = tổng byte trên đĩa của MỌI THỨ cần để decode và render, sau bước coding cuối cùng của method.
> Đơn vị MB = 10⁶ byte.**

Bằng chứng: **HAC** ([2403.14530](https://arxiv.org/abs/2403.14530)) tính cả MLP + hash grid + anchor,
mã hoá arithmetic. **Compact-3DGS** ([CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Lee_Compact_3D_Gaussian_Representation_for_Radiance_Field_CVPR_2024_paper.html))
quantize 8-bit → Morton sort → **Huffman** → **DEFLATE**, báo số sau DEFLATE.

**Ba cái bẫy:**
1. **Baseline 3DGS KHÔNG BAO GIỜ được nén.** T&T 411.0 MB / DB 676.0 MB là `.ply` fp32 thô. Mọi tiêu đề
   "nén ×N" đều so bản entropy-coded với bản chưa nén. → Ta phải báo cáo **cả hai**, không thì tỉ lệ gây hiểu nhầm.
2. `npz_fp16` và "quantized nhưng chưa nén" **không phải size báo cáo được**. Giữ chúng làm **chẩn đoán trong profiler**, không bao giờ làm số headline.
3. Đơn vị (L3).

**Số tham chiếu cho đúng scene của ta** ([leaderboard 3DGS.zip](https://w-m.github.io/3dgs-compression-survey/), cập nhật 14/07/2026):
T&T vanilla 3DGS **411.0 MB / 23.14 PSNR** · DB vanilla **676.0 MB / 29.41 PSNR**.
Nhỏ nhất hiện tại: SmolGS-base 4.8 MB (T&T) / 2.9 MB (DB); HAC++-lowrate 5.4 / 3.1 MB.

---

## PHẦN 2 — XOÁ khỏi plan (đã có sẵn, viết lại là tự chuốc số không so sánh được)

| Thành phần trong plan | Xử lý | Dùng gì thay |
|---|---|---|
| Module `evaluation/metrics.py` (PSNR/SSIM/LPIPS) | **XOÁ** | Copy nguyên `metrics.py` + `utils/loss_utils.py` + `lpipsPyTorch/` của INRIA |
| `quantization/size.py` 4 cách đo | **XOÁ phần lớn** | [**ffsplat**](https://github.com/w-m/ffsplat) (Apache-2.0, nhóm KISS-GS) — nhận `--input-format=3DGS-INRIA.ply`. ⚠️ pre-alpha, 22 issue mở, CUDA 12.x, JIT-compile kernel gsplat |
| Tự implement baseline nén | **XOÁ** | [**Splatwizard**](https://github.com/splatwizard/splatwizard) (MIT, 31★) có sẵn **12 model**: 3DGS, 2DGS, Speedy-Splat, Trimming-the-Fat, **LightGaussian**, PUP-3DGS, Compressed-3DGS, **Compact-3DGS**, ControlGS, **HAC**, CAT-3DGS, MesonGS. Thêm [gsplat `PngCompression`](https://docs.gsplat.studio/main/apis/compression.html): 1M Gaussian 236 → **16.5 MB, −0.5 dB** |
| Tự thiết kế eval protocol | **XOÁ** | 3DGS.zip: 9 scene MipNeRF360, cạnh dài max 1600 px, test = `idx % 8 == 0`, PSNR ≥2 chữ số thập phân, SSIM/LPIPS ≥3. Cộng INRIA `full_eval.py` cho độ phân giải từng dataset (T&T/DB native; MipNeRF360 outdoor `-i images_4`, indoor `-i images_2`) |
| Per-stage RAM+VRAM profiler | **GIỮ** | Nhưng **không được claim "đầu tiên"** — xem P3.b |
| Cost model RAM+VRAM | **GIỮ, HẠ HẠNG** | Xem P3.a |
| Iso-memory benchmark | **GIỮ, ĐỔI KHUNG** | Xem P3.c |
| Pareto trên trục bộ nhớ | **GIỮ** | Chưa ai làm trên trục này |

**Splatwizard — cảnh báo:** đo *"peak GPU memory usage"* **chỉ khi render**, không đo host RAM, không đo
training/fine-tune, không tách theo stage. Thí nghiệm chạy trên **một RTX 3090 24 GB**. Không có issue nào
về Windows hay OOM → chưa ai thử ở cấu hình như của ta. Deps nặng (`torch_scatter`, `open3d`, `pycolmap`).
→ **Dùng làm kho implementation tham chiếu, KHÔNG dùng làm harness đo bộ nhớ.**
⚠️ README claim "CVPR'26" — agent **không verify được**.

**3DGS.zip không định nghĩa quy ước đo size.** Nguyên văn: *"the resultant size in megabytes (MB, 10⁶ bytes),
**as provided by the respective authors**."* Họ chép số, không chạy lại. Nhưng **protocol ảnh** thì đáng theo.

---

## PHẦN 3 — Kiểm tra "đã bị scoop chưa" (câu hỏi tôi cần biết ở tháng 1, không phải tháng 4)

| Claim trong plan | Phán quyết |
|---|---|
| (a) Cost model RAM+VRAM chung | **Bị scoop một phần, và giá trị thấp dù sao** |
| (b) Profiler RAM+VRAM theo stage | **Mất quyền claim "đầu tiên"** |
| (c) Iso-memory benchmark | **Khái niệm đã in; chỉ còn cái tên là trống** |
| (d) **RAM bind trước VRAM khi fine-tune trên GPU yếu** | ✅ **THỰC SỰ CÒN TRỐNG — đây là contribution** |

**(b)** *A LoD of Gaussians* (SIGGRAPH 2026, [2507.01110](https://arxiv.org/abs/2507.01110)) có **Figure 5**:
*"Peak memory consumption of CPU and GPU for a training iteration..."* và §4.5 tách CPU (*"majority of RAM
usage is consumed by per-Gaussian properties and their corresponding ADAM optimizer states"*) vs GPU.
Nhưng đó là **một hình cho một scene + vài đoạn văn**, không phải profiler hệ thống, và là paper systems cho
scene tỉ Gaussian. → **Vẫn ship được tool, không claim "first".**

**(a)** CLM đã viết nửa VRAM dạng đóng (59 attr × 4 B × {param, grad, 2 moment Adam}); *A LoD of Gaussians*
cho nửa host bằng thực nghiệm: *"roughly 1 GB of RAM per million Gaussians"*; README CLM: *"~8 GB CPU memory
per 10 million Gaussians"*. Ghép lại là **một đoạn methodology**, không phải contribution #1.

**(c)** Từ `"iso-memory"` gần như trống trên arXiv, nhưng so-ở-cùng-ngân-sách là chuẩn mực sẵn có, và
**MEGS² (ICLR 2026, [2509.07021](https://arxiv.org/abs/2509.07021)) đã in luận điểm biện minh**: model đã nén
có thể cần **nhiều** bộ nhớ render **hơn** bản chưa nén → file size là trục sai. Họ chỉ không xây benchmark.
Phần còn lại đáng bảo vệ: ngành chuẩn hoá theo **số Gaussian** hoặc **file size**, **chưa ai theo byte bộ nhớ**
(Taming 3DGS và Gaussians on a Diet đều match theo count: 0.628M vs 0.632M). → Khung lại thành đóng góp
dạng benchmark-track, cite MEGS² và Taming làm tiền lệ.

### (d) — vì sao nó còn trống, và cách bảo vệ

Mọi paper systems chạy trên workstation 32–256 GB và coi host RAM là miễn phí:

| Paper | Bằng chứng |
|---|---|
| CLM ([2511.04951](https://arxiv.org/abs/2511.04951), ASPLOS 2026) | Testbed 128–256 GB; báo 6.0–37.8 GB pinned host nhưng gọi là *"under 10% of the 256 GB RAM"* |
| GS-Scale ([2509.15645](https://arxiv.org/abs/2509.15645), ASPLOS) | Bảng phần cứng ghi 32/64/1024 GB host và **không hề đo mức tiêu thụ host**. "CPU bottleneck" của họ là FLOPS/bandwidth, **không phải dung lượng** |
| TideGS ([2605.20150](https://arxiv.org/abs/2605.20150), ICML 2026 Spotlight) | 256 GB; host RAM là cache size chỉnh được, không phải ràng buộc |
| [2603.08499](https://arxiv.org/abs/2603.08499) (03/2026) | **Câu đáng cite nhất:** *"we profile only GPU memory, since we are not constrained by the CPU one."* |

**Không paper 3DGS training nào nhắc tới GTX 1650/1660 hay bất kỳ card 4 GB nào.** "Consumer GPU" trong
literature này nghĩa là **một chiếc RTX 4090**. Bộ nhớ khi *nạp checkpoint* có **zero** coverage học thuật.

### 🔴 Đòn phản biện phải chặn trước — và ta đã có sẵn dữ liệu

Reviewer sẽ nói: *"trần RAM của anh là hệ quả của implementation tham chiếu (`plyfile` gián tiếp qua numpy,
parse cả checkpoint, default `--data_device`), không phải tính chất của 3DGS."*

Hai dữ kiện ủng hộ họ:
- *Gaussians on a Diet* ([2604.20046](https://arxiv.org/abs/2604.20046)): *"three-quarters memory is used for
  dataset storage"*, sửa bằng host-side prefetching → **8.55 → 2.98 GB peak**, khoản tiết kiệm lớn nhất của họ,
  trình bày như một chú thích kỹ thuật.
- PlayCanvas [splat-transform v3.0.0](https://github.com/playcanvas/splat-transform) (10/07/2026):
  *"~7× faster using ~one fifth the peak memory"* nhờ streaming chunked PLY loading.

**✅ Ta ĐÃ tự chứng minh điều này hôm nay — và đó là điểm mạnh, không phải điểm yếu.**
EXP-003 đo được: nạp kiểu INRIA = **2.0× file**; streaming theo khối = **0.24 GB cố định** (drjohnson giảm 6.5×).
Tức trần RAM khi *nạp model* **đúng là artifact, và ta có bản sửa kèm số đo**.

→ **Khung lại contribution (d) cho trung thực và mạnh hơn:**

> Không phải *"RAM bind trước VRAM"* (ngây thơ, dễ bị bác), mà:
> **"Phần nào của trần RAM là artifact có thể sửa, và phần nào là cấu trúc không sửa được?"**
> - **Sửa được, đã đo:** nạp checkpoint (2.0× file → 0.24 GB nhờ streaming) — độc lập trùng khớp với splat-transform v3
> - **Sửa được, người khác đã chỉ ra:** lưu dataset (Gaussians on a Diet: 8.55 → 2.98 GB)
> - **Còn lại — phải chứng minh là cấu trúc:** optimizer state khi fine-tune, teacher+student cùng lúc lúc distill,
>   ảnh training ở `data_device=cpu` (drjohnson 3.43 GB, lớn hơn cả model 1.56 GB)

Chỉ khi trần RAM **vẫn còn** sau khi đã tính hết streaming parse + mmap + `data_device=cpu` thì (d) mới sống sót.
Đó chính là thí nghiệm phải thiết kế.

**Một kẽ hở nữa:** *Gaussians on a Diet* đo trên **Jetson AGX Xavier — bộ nhớ CPU/GPU hợp nhất**, nên
"peak memory" của họ **đã trộn lẫn hai thứ mà không nói ra**. Máy của ta có RAM và VRAM tách bạch → tách được.

---

## PHẦN 4 — Công trình mới cần biết (2025 → 09/2026)

### Trực tiếp trùng chủ đề (nén post-hoc checkpoint đã train)

| Paper | Ngày / Venue | Code | Vì sao quan trọng |
|---|---|---|---|
| **KISS-GS: 3DGS Compression Kept Simple** | [2608.26948](https://arxiv.org/abs/2608.26948) · 27/08/2026 · **ECCV 2026** | [ffsplat](https://github.com/w-m/ffsplat) Apache-2.0 | **Quan trọng nhất.** Nén **tách hoàn toàn khỏi training** — đúng luận đề của ta. 85×–319× so với vanilla. **Decode peak memory 0.37–1.24 GB.** Cùng nhóm với 3DGS.zip |
| **FlexGaussian** | [2507.06671](https://arxiv.org/abs/2507.06671) · ACM MM 2025 | [repo](https://github.com/Supercomputing-System-AI-Lab/FlexGaussian) | **Training-free, <1 phút desktop / <3 phút Jetson.** bicycle 1450 → 71 MB, −0.99 dB. Nhắm *mức giảm PSNR*, không nhắm ngân sách size. Deps cũ (CUDA 11.6, torch 1.12.1) |
| **SizeGS** | [2412.05808](https://arxiv.org/abs/2412.05808) · rev 29/11/2025 | không tìm thấy | **Nén post-training tới ngân sách SIZE mục tiêu** bằng mixed-integer programming. Gần ý tưởng "budget-constrained" nhất — nhưng là *file size*, không phải *memory* |
| 3DTurboQuant | [2604.05366](https://arxiv.org/abs/2604.05366) · 07/04/2026 | [repo](https://github.com/JaeLee18/3DTurboQuant) | Quantization training-free |
| MesonGS | [2409.09756](https://arxiv.org/abs/2409.09756) · ECCV 2024 | trong Splatwizard | Codec post-training gốc |

### Hệ thống bộ nhớ / offloading — nơi ngành thực sự dịch chuyển

| Paper | Ngày / Venue | Điểm đáng chú ý |
|---|---|---|
| **GS-Scale** | [2509.15645](https://arxiv.org/abs/2509.15645) · ASPLOS | **Toàn bộ Gaussian nằm ở host memory**, nạp lên GPU theo nhu cầu. **Giảm 3.3–5.6× GPU memory**; 4M → 18M Gaussian trên **RTX 4070 Mobile** ← card laptop, gần ta nhất |
| **CLM** | [2511.04951](https://arxiv.org/abs/2511.04951) · ASPLOS 2026 · [229★](https://github.com/nyu-systems/CLM-GS) | CPU offloading, 100M+ Gaussian trên 1 GPU 24 GB. Cần 24 GB VRAM + 128 GB RAM — ta không dùng được, nhưng là tiền lệ bắt buộc phải cite |
| **TideGS** | [2605.20150](https://arxiv.org/abs/2605.20150) · **ICML 2026 Spotlight** · [166★](https://github.com/sponge-lab/TideGS) | Phân cấp SSD–CPU–GPU. **>1 tỉ Gaussian trên 1 GPU 24 GB** |
| **A LoD of Gaussians** | [2507.01110](https://arxiv.org/abs/2507.01110) · **SIGGRAPH 2026** | Out-of-core trong CPU memory. **Chính là paper làm mất quyền claim (b)** |
| **Gaussians on a Diet** | [2604.20046](https://arxiv.org/abs/2604.20046) · 21/04/2026 | Giảm tới **80% peak training memory**. Jetson AGX Xavier (unified memory — xem kẽ hở ở P3.d) |
| **MEGS²** | [2509.07021](https://arxiv.org/abs/2509.07021) · **ICLR 2026** | Giảm 50% static VRAM / 40% rendering VRAM. **Đã in luận điểm biện minh cho iso-memory** |
| Virtual Memory for 3DGS | [2506.19415](https://arxiv.org/abs/2506.19415) | Streaming just-in-time các Gaussian nhìn thấy được |
| PocketGS | [2601.17354](https://arxiv.org/abs/2601.17354) | **Training** 3DGS trên điện thoại. ⚠️ chi tiết 6 GB/iPhone 15 **chưa verify** |

### Anchor / entropy đời sau
HAC++ ([2501.12255](https://arxiv.org/abs/2501.12255), [209★](https://github.com/YihangChen-ee/HAC-plus)) — 100× vs vanilla, đang đứng đầu leaderboard của đúng dataset ta dùng ·
PCGS ([2503.08511](https://arxiv.org/abs/2503.08511), AAAI 2026 Oral) · SALVQ ([2509.13482](https://arxiv.org/abs/2509.13482), IEEE TIP) ·
SymGS ([2511.13264](https://arxiv.org/abs/2511.13264)) — cắm thêm lên compressor có sẵn, tới 108×.

### Chuẩn hoá định dạng — chưa có trong plan
**Khronos `KHR_gaussian_splatting`** công bố **03/02/2026** ([press](https://www.khronos.org/news/press/gltf-gaussian-splatting-press-release),
[spec](https://github.com/KhronosGroup/glTF/tree/main/extensions/2.0/Khronos/KHR_gaussian_splatting)),
kèm `KHR_gaussian_splatting_compression_spz` (Niantic SPZ, MIT, nhỏ hơn PLY ~90%).
Spec README vẫn ghi "Release Candidate" lúc kiểm tra. Cộng **MPEG-I Gaussian Splat Coding**
([mpeg.expert/gsc](https://mpeg.expert/gsc/index.html)) ⚠️ agent không tự fetch tài liệu MPEG.

### Survey mới hơn bản 02/2025
**SUCCESS-GS** — [2512.07197](https://arxiv.org/abs/2512.07197), 08/12/2025, static + dynamic GS.
Tracker đi kèm [Awesome-Efficient-GS](https://github.com/CMLab-Korea/Awesome-Efficient-GS) (cập nhật 09/07/2026).
⚠️ Không có v2; claim rằng nó phủ tới 04/2026 là **chưa verify**.

---

## PHẦN 5 — Cảnh báo môi trường

Trong tất cả tool ở trên, **chỉ `gsplat` có tài liệu hỗ trợ Windows** ([INSTALL_WIN.md](https://github.com/nerfstudio-project/gsplat)).
Splatwizard (`torch_scatter`, `open3d`, `pycolmap`) và ffsplat (`uv`, CUDA 12.x, JIT-compile kernel gsplat)
đều sẽ khó. → Củng cố quyết định đi WSL2.

Lưu ý agent nêu và **đúng với ta**: *"WSL2's default memory cap will interact directly with the RAM ceiling
you're trying to measure."* Ta đã đặt `memory=10GB` trong `.wslconfig` → **mọi số RAM phải ghi kèm cap này**,
và nên chạy thêm một cấu hình cap khác (ví dụ 12 GB) để chứng minh kết quả không phải artifact của cap.

---

## Agent chưa verify được (KHÔNG cite)
- Splatwizard được nhận CVPR 2026 (chỉ có claim trong README); yêu cầu Python/CUDA của nó; nó có chạy trên Windows / 4 GB không
- Backbone LPIPS của NerfBaselines (site ghi VGG, nguồn thứ cấp ghi AlexNet) — **phải tự kiểm trước khi cite**
- Chi tiết 6 GB / iPhone 15 / unified memory của PocketGS
- Code công khai của POTR, MesonGS++, SizeGS, dictionary-learning, GSICO, EntropyGS
- Tình trạng tài liệu MPEG-I GSC
- Hai mục IEEE Xplore [11189884](https://ieeexplore.ieee.org/document/11189884) và [11449264](https://ieeexplore.ieee.org/document/11449264) — không lấy được tác giả/venue/ngày
