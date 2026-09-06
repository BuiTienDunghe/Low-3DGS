# LightGaussian: Unbounded 3D Gaussian Compression with 15× Reduction and 200+ FPS

**Fan, Wang, Wang, Wang, Pan, Wang (VITA-Group)** · NeurIPS 2024 · arXiv 2311.17245
Code: https://github.com/VITA-Group/LightGaussian (commit `6676b98`, 2024-12-29, clone tại `~/l3dgs/third_party/LightGaussian`)

> Trạng thái: **§1–7, §9 hoàn chỉnh (2026-09-02)** — code đã đọc, số/equation lấy từ arXiv HTML của paper.
> PDF lưu ở scratchpad `papers/2311.17245.pdf` để đối chiếu lại khi cần.

---

## 1. Problem

3DGS train xong thường có **vài triệu Gaussian**, mỗi cái 59 float (SH deg 3 chiếm 48/59) →
checkpoint hàng trăm MB tới >1 GB, VRAM render cao, FPS hạn chế trên thiết bị yếu.
Phần lớn Gaussian đóng góp rất ít vào ảnh render cuối.

## 2. Prior work & gap

Nén NeRF (VQ, hash grid) không áp thẳng được vì 3DGS là explicit point-based.
Pruning theo opacity của vanilla 3DGS quá thô — không xét Gaussian có *được nhìn thấy* hay không.

## 3. Key idea (1 câu)

> Xếp hạng Gaussian bằng **đóng góp thực tế vào mọi training view** (nhân với volume đã chuẩn hoá),
> prune 66% thấp nhất, fine-tune ngắn để phục hồi, distill SH deg 3→2, rồi vector-quantize
> phần SH ít quan trọng.

## 4. Method — 3 module, đúng thứ tự pipeline

### 4.1 Gaussian Pruning & Recovery (`prune_finetune.py`, `prune.py`)

**Global Significance** — theo paper:

```
Eq. 3:   GS_j = Σ_{i=1}^{M·H·W}  𝟙(G(X_j), r_i) · σ_j · T · γ(Σ_j)        (M view, H×W pixel; T = transmittance)
Eq. 4:   γ(Σ) = (V_norm)^β ,   V_norm = min( V(Σ) / V_max90 , 1 ) ,   β = 0.1
```

🔴 **CODE KHÔNG IMPLEMENT Eq. 3.** Đã verify trong submodule đã clone (2026-09-02):

```
.gitmodules  ->  Kevin-2017/compress-diff-gaussian-rasterization @ 240618bb (28/11/2023)
forward.cu:474:   important_score[collected_id[j]] += con_o.w;   //opacity
                                                     ^^^^^^^^^ KHÔNG có  * T
```

Chú thích trong chính code ghi `//opacity`. Bản `main` của rasterizer đã sửa ở commit
[`9893d8c6`](https://github.com/Kevin-2017/compress-diff-gaussian-rasterization/commit/9893d8c607684bc1a432e179cc97035d72d84520)
(06/11/2024, *"importance score multiple by transmittance"*) — đúng **một dòng**: `+= con_o.w * T`.
**LightGaussian commit cuối là 30/12/2024, tức MỘT THÁNG SAU bản sửa, mà vẫn pin SHA cũ.**
`git submodule update --init --recursive` cho bạn **bản chưa sửa**. Đây là nội dung
[issue #46](https://github.com/VITA-Group/LightGaussian/issues/46).

Sắc thái: paper ra trước bản sửa, nên bản pin **có lẽ chính là thứ tạo ra số trong paper**.
Chuyển sang `main` làm code khớp *phương trình* nhưng có thể **không tái tạo được *số***.
Chưa ai đo khác biệt. → Ta chạy **cả hai**, và đó là một thí nghiệm có giá trị (A7′ trong ablation).

Và theo code thực tế (**lệch cả Eq. 3 lẫn Eq. 4** — xem §5):

```
important_score_j = Σ_{view i} Σ_{pixel p}  [contribution của j tại p trong view i]     ← count_render (CUDA)
gaussians_count_j = Σ_{view i} Σ_{pixel p}  1[j chạm p]                                 ← count_render (CUDA)

volume_j          = Π scaling_j                                                         (3 trục)
v_norm_j          = (volume_j / volume_p90)^v_pow          với v_pow = 0.1, p90 = phần tử thứ 90% khi sort giảm dần
                                                            (code: sorted_volume[int(0.9·N)] — tức phần tử NHỎ hơn 90% còn lại)

GS_j              = important_score_j · v_norm_j             ← prune_type = "v_important_score" (mặc định paper)
```

Prune: bỏ `prune_percent` (0.66) Gaussian có `GS_j` thấp nhất. Có `prune_decay` cho prune nhiều lần
(lần thứ i prune `decay^i · percent`), mặc định chỉ prune 1 lần ở iter 30 001.

**Recovery:** fine-tune tiếp tới iter 35 000 (5 000 iter) với loss 3DGS chuẩn `(1-λ)·L1 + λ·(1-SSIM)`,
`position_lr_max_steps=35000`.

**Các prune_type khác trong code** (dùng làm baseline cho ablation của ta):
`important_score` (không nhân volume) · `count` (chỉ đếm hit — rất gần *random-ish*) · `opacity` (= baseline vanilla) ·
`max_v_important_score` (nhân max-scale thay vì volume).

### 4.2 SH Distillation (`distill_train.py`)

- Teacher = model gốc SH deg 3 (nạp bằng `torch.load(--teacher_model)` → **cần `.pth`**, không phải `.ply`)
- Student = bản copy, `max_sh_degree = new_max_sh = 2`
- Loss: `(1-λ)·L1(student, teacher_render) + λ·(1-SSIM)` — **distill trên ảnh render của teacher**, không phải GT
- `--augmented_view`: 2/3 số iteration dùng **pseudo-view** (camera nội suy) thay vì train view → mở rộng phủ góc nhìn
- `--enable_covariance`: cho phép cập nhật scale/rotation trong distill (mặc định đóng băng)
- Chạy tới iter 40 000 (5 000 iter nữa sau prune-finetune)

### 4.3 VecTree Quantization (`vectree/vectree.py`)

- Input: `.ply` đã prune + distill, và `imp_score.npz` (GS_j đã tính)
- `vq_ratio = 0.6`: 60% Gaussian **kém quan trọng nhất** → SH bị thay bằng codebook index; 40% quan trọng giữ nguyên
- `codebook_size = 8192`, k-means trên SH_rest (deg 1–2)
- Sau đó entropy coding → file `extreme_saving/`

Ba module ghép lại cho ra 15× nén + 200+ FPS (theo paper — **chưa verify**).

## 5. Equation → Code mapping

| Khái niệm trong paper | File:hàm | Ghi chú |
|---|---|---|
| Eq. 3 `Σ 𝟙 · σ · **T** · γ` | `forward.cu:474` | 🔴 **THIẾU `* T`** ở SHA đã pin. Xem §4.1 |
| Cộng dồn song song | `forward.cu` | 🔴 **`+=` thường, KHÔNG atomicAdd** (`grep -c atomicAdd` = **0**). Nhiều thread ghi cùng `collected_id[j]` → **score không xác định và lệch thấp**. Báo ở [#24](https://github.com/VITA-Group/LightGaussian/issues/24) và [#46](https://github.com/VITA-Group/LightGaussian/issues/46), **tác giả chưa từng trả lời** |
| Hit count / contribution per Gaussian | `gaussian_renderer/__init__.py:count_render` → `submodules/compress-diff-gaussian-rasterization` | Kernel CUDA riêng, trả `gaussians_count, important_score, image, radii` |
| Tích luỹ qua mọi train view | `prune.py:prune_list` | Loop `scene.getTrainCameras()`, cộng dồn; `gc.collect()` mỗi view |
| Volume normalisation Eq. 4 `(min(V/V_max90, 1))^β` | `prune.py:calculate_v_imp_score` | ⚠️ **Code KHÔNG có `min(·,1)`**: `torch.pow(volume / kth, v_pow)` không clamp → Gaussian lớn hơn p90 nhận γ > 1. p90 = `sorted_desc[int(0.9N)]`. Paper↔code lệch — ta chạy **cả hai** biến thể (A7 official-code, A7′ paper-faithful) và ghi khác biệt |
| Prune theo % thấp nhất | `scene/gaussian_model.py:prune_gaussians(percent, scores)` | Sort scores, mask |
| Recovery fine-tune | `prune_finetune.py` vòng train chuẩn | 30 001 → 35 000 |
| Distill loss | `distill_train.py:~142–145` | L1 + SSIM giữa student và **teacher render** |
| Pseudo-view | `distill_train.py:~132` (`iteration % 3`) | 2/3 iter dùng augmented view |
| SH deg 3→2 | `distill_train.py:64,78` (`dataset.sh_degree = new_max_sh`) | |
| VQ | `vectree/vectree.py` | Chưa đọc chi tiết |

## 6. Experiments trong paper

| | |
|---|---|
| Datasets | MipNeRF360, Tanks&Temples (truck, train), Deep Blending (playroom, drjohnson), NeRF-Synthetic |
| Baseline | **"3D-GS\*" = họ tự train lại**, KHÔNG phải checkpoint INRIA phát hành (truck: 25.37 vs INRIA 25.187) |
| Hardware | **A6000** (mọi đánh giá hiệu năng) |
| Prune | 66% (và 80% trong ablation), β = 0.1, tại iter 30k của 3DGS |
| Co-adaptation (recovery) | **5 000 iter** sau prune |
| Distill | SH 3 → 2, pseudo-view với σ = 0.1 |
| VQ | ratio 60% (SH của 60% Gaussian kém quan trọng nhất), codebook 8192, **+5 000 iter** fine-tune sau VQ |
| Split / res | Theo 3DGS: mỗi ảnh thứ 8 làm test; native resolution |

## 7. Results đã báo cáo (chép nguyên, điều kiện như §6)

**Table 1 — Tanks&Temples trung bình:**

| | FPS | Size | PSNR | SSIM | LPIPS |
|---|---|---|---|---|---|
| 3D-GS\* | 106 | 433 MB | 23.66 | 0.845 | 0.178 |
| LightGaussian | **357** | **25 MB** (17×) | 23.44 (−0.22) | 0.832 | 0.202 |

**Per-scene (Table 8–10):** truck 25.37 → **25.40** (+0.03) · train 21.96 → **21.84** (−0.12). *(DB per-scene chưa trích — đọc PDF trang bảng khi cần.)*

**Ablation Table 2 (scene `room`, MipNeRF360):** prune-only **30.67** → + co-adaptation **31.64** = **recovery +0.97 dB** sau 5k iter.
→ Đây là số neo cho H1 của ta (fine-tune ngắn phục hồi phần lớn quality).

**NeRF-Synthetic (Table 7):** 52.38 → 7.89 MB, FPS 310 → 411.

**Tham chiếu INRIA (để so checkpoint phát hành, 3DGS paper Table 8/9, A6000, native res):**
truck 25.187 / 0.148 · train 21.097 / 0.218 · playroom 30.044 / 0.241 · drjohnson 28.766 / 0.244 (PSNR / LPIPS).

**Quy tắc so sánh cho reproduction (§10):** baseline của ta là checkpoint INRIA → báo cáo **Δ = sau − trước** trên chính
baseline đó, và so **Δ** với Δ của paper (truck +0.03, train −0.12), **không** so số tuyệt đối với 3D-GS\*.

## 7b. 🔴 BẪY IM LẶNG trong code — tất cả đã verify trong repo đã clone (2026-09-02)

Đây là phần quan trọng nhất của paper note. Năm lỗi dưới đây **không báo lỗi** — chúng cho ra kết quả sai
mà trông như đã chạy đúng.

| # | Bẫy | Verify | Hậu quả |
|---|---|---|---|
| **T1** | `--prune_iterations` mặc định `[30_001]`, nhưng vòng lặp là `range(first_iter=0, opt.iterations+1)` = `range(0, 30001)` → **giá trị lớn nhất là 30000** | `prune_finetune.py:66,99,313` | **KHÔNG BAO GIỜ PRUNE.** Chạy 30k iteration fine-tune thuần, không lỗi, không cảnh báo. Tác giả xác nhận ở [#7](https://github.com/VITA-Group/LightGaussian/issues/7). PUP 3D-GS đã vá bằng cách parse `iteration_30000` từ đường dẫn |
| **T2** | `--prune_type` mặc định `important_score`, **không phải** `v_important_score` của paper | `prune_finetune.py:218` | Chạy **sai thuật toán** |
| **T3** | `--prune_percent` mặc định **0.1**, không phải 0.66 | `prune_finetune.py:316` | Sai tỉ lệ |
| **T4** | `vectree.py --sh_degree` mặc định **2**, nhưng `.ply` của INRIA là **SH 3** (62 cột) | `vectree/vectree.py:18` | Cắt **sai cột** SH trong im lặng, chỉ nổ lúc ghi: `could not assign tuple of length 62 to structure with 41 fields` ([#15](https://github.com/VITA-Group/LightGaussian/issues/15)). **Luôn truyền `--sh_degree 3`** |
| **T5** | Race condition không atomic trong kernel (xem §5) | `grep -c atomicAdd` = 0 | Importance score **không xác định giữa các lần chạy** và **lệch thấp** |

**Lệnh đúng** (từ [#18](https://github.com/VITA-Group/LightGaussian/issues/18), có người báo chạy được):

```
--prune_iterations 2 --prune_percent 0.66 --prune_type v_important_score --prune_decay 1
--iteration 5000 --position_lr_init 0.000005 --position_lr_max_steps 5000 --v_pow 0.1
```

**Cổng kiểm tra bắt buộc:** log phải in `Before prune iteration…` / `After prune iteration…` với mức giảm ~2.9×.
Không thấy hai dòng đó → T1 đã xảy ra. Số Gaussian tụt còn một chữ số → rasterizer sai (xem T7).

### Ba khác biệt ngầm nữa

| # | Vấn đề | Verify |
|---|---|---|
| **T6** | **LightGaussian dùng `AdamW`, INRIA dùng `Adam`** — weight decay 0.01 áp lên **mọi** tham số Gaussian (xyz, opacity, scale, rotation, SH). Cộng thêm `ExponentialLR(gamma=0.95)` chồng lên. Khác biệt không ghi trong paper, không ai bàn | `LightGaussian/scene/gaussian_model.py:217` = `AdamW` vs `gaussian-splatting/…:193` = `Adam` |
| **T7** | `setup.py` cài dưới tên **`diff_gaussian_rasterization`** — **trùng tên INRIA**. Cài cái này ghi đè cái kia trong im lặng | `submodules/compress-diff-gaussian-rasterization/setup.py:18` |
| **T8** | Thiếu dependency: **`einops`** được `vectree/vq.py` import nhưng **không có trong `environment.yml`** (`grep -c einops` = 0). `vectree.py:150` gọi `os.system("zip -r …")` mà **Ubuntu 24.04 không cài sẵn `zip`** — `os.system` không raise, dòng sau `os.path.getsize` mới ném `FileNotFoundError`. `scripts/*.sh` ghi vào `logs_prune/` mà thư mục đó **không có trong repo** | verify trực tiếp |

### 🔴 T9 — `distill_train.py` KHÔNG dùng được `.ply`

```python
(teacher_model_params, _) = torch.load(args.teacher_model)   # distill_train.py:74
```

Teacher **bắt buộc là `.pth`**, không có đường `--start_pointcloud` cho teacher, và cả khối gated bằng `if checkpoint:`.
INRIA chỉ phát hành `.ply` → **giai đoạn SH distillation không chạy được nếu không viết code.**
Đúng là [#51](https://github.com/VITA-Group/LightGaussian/issues/51), mở từ 12/2025, **không một trả lời**.

*(Thêm: `torch.load` mặc định `weights_only=True` từ torch 2.6 → checkpoint kiểu 3DGS ném `UnpicklingError`.)*

### ✅ Giả thuyết của TÔI bị bác bỏ: `spatial_lr_scale` KHÔNG bằng 0

Tôi đã ghi trong `tools/ply_to_checkpoint.py` rằng `load_ply` để `spatial_lr_scale = 0` làm đóng băng vị trí.
**Sai.** Luồng thật: `Scene(dataset, gaussians)` với `load_iteration=None` → `create_from_pcd(pcd, self.cameras_extent)`
→ **có set** `spatial_lr_scale`. `load_ply()` thay tensor nhưng **không đụng** thuộc tính đó, và `training_setup(opt)`
chạy sau. Learning rate đúng. `load_ply` cũng set `active_sh_degree = max_sh_degree`.

**Nhưng hệ quả vẫn quan trọng:** vì `Scene()` phải dựng point cloud COLMAP trước,
**không chạy được chỉ với `.ply`** — phải có đủ dataset COLMAP (`sparse/` + `images/`) ở `-s`.

---

## 8. Limitations

- Cả 3 module đều cần model **đã train xong 30k** → không giảm peak VRAM lúc train (chỉ giảm lúc render/lưu)
- `prune_list` render **mọi** train view (251 với truck) → S2 là stage tốn thời gian + VRAM rasterizer
- Distill giữ **2 model cùng lúc** (teacher deg 3 + student) → đỉnh VRAM thứ 2 của pipeline
- `v_pow`, `prune_percent`, `vq_ratio` là hyperparam cố định, không adaptive theo budget
- Pin **PyTorch 1.12.1 / CUDA 11.6 / Python 3.9** — codebase 2023

## 9. Liên quan tới project này

| Câu hỏi | Trả lời |
|---|---|
| Giảm train-time VRAM? | **Không.** Post-hoc. Nhưng đây đúng là mục tiêu fine-tune của ta |
| Reproduce được trên 4 GB / 10 GB RAM? | Có khả năng — sau prune còn 34% Gaussian. **Chưa đo.** Đỉnh RAM nghi ở `load_ply` (S0), đỉnh VRAM nghi ở `prune_list` (S2) và distill (S4) |
| Vai trò | **C1 — method để reproduce**, đồng thời là **A7/A8 trong ablation** |

### Deviations đã biết trước khi chạy (cập nhật khi thực chạy)

| # | Official | Của ta | Lý do | Ảnh hưởng dự kiến |
|---|---|---|---|---|
| D1 | `--start_checkpoint chkpnt30000.pth` (có Adam state) | `--start_pointcloud point_cloud.ply` từ bundle INRIA | Bundle không có `.pth` | Optimizer khởi tạo lại → 5k iter recovery có thể chưa đủ; **so PSNR trước/sau** |
| D2 | `--teacher_model *.pth` | Tự tạo `.pth` từ `.ply` bằng `tools/ply_to_checkpoint.py` | Như trên | Không ảnh hưởng (teacher không train) |
| D3 | PyTorch 1.12 / CUDA 11.6 / py3.9 | PyTorch 2.x / CUDA 12.6 / py3.12 | Driver + toolchain hiện đại | Nếu `compress-diff-gaussian-rasterization` không build → port `count_render` sang rasterizer INRIA mới |
| D4 | **A6000** 48 GB | GTX 1650 Ti 4 GB, 10 GB RAM | Đây là chủ đề project | FPS không so được; quality so được nếu cùng res/split |
| D5 | Native res | Native res cho T&T/DB (≤ 1.6K nên không bị auto-rescale) | Protocol | Khớp — chỉ khác ở `bicycle` (P7) |
| D6 | Baseline 3D-GS\* tự train (truck 25.37) | Checkpoint INRIA (truck 25.187) | Không train from scratch | So **Δ**, không so tuyệt đối |
| D7 | Eq. 4 có `min(V/V_max90, 1)` | Code không clamp | Paper↔code lệch | Chạy cả 2 biến thể, báo cáo khác biệt |

## 9b. 🎯 MỐC REPRODUCTION THẬT — dùng CỤM re-run độc lập, KHÔNG dùng số trong paper

Paper tự báo T&T 23.44 (v6) / 23.11 (v5). Nhưng **số của paper đã đổi qua các bản arXiv**
(v1–v4 dùng **7/9** scene Mip360 → baseline 3D-GS phồng lên 29.13; v5 đổi sang 9 scene → 27.53).
Quan trọng hơn: **nhiều paper đã chạy lại LightGaussian bằng code chính thức**, và đó mới là mốc đúng.

### Mip-NeRF360 (9 scene) — cụm re-run độc lập

| Nguồn | PSNR | Size | Ghi chú |
|---|---|---|---|
| FCGS ([2410.08017](https://arxiv.org/html/2410.08017v3)) | 27.31 | 48.61 MB | tự đo cả thời gian |
| ControlGS | 27.24 | — | |
| ACE-GS | 27.03 | 41.06 MB | |
| HAC ([2403.14530](https://arxiv.org/html/2403.14530v3)) | 27.00 | 44.54 MB | |
| RadSplat | 26.99 | — | 1.046M Gaussian |
| FlexGaussian ([2507.06671](https://arxiv.org/html/2507.06671v1)) | 26.96 | 52.05 MiB | **RTX 3090, code chính thức** |
| NeuralGS | 26.95 | 48.71 MB | *"official code, default configurations"* |
| VEDAL | 26.82 ± 0.07 | 183 MB | **3 seed, có std** |
| GaussianPOP | 26.81 | — | |
| POTR | 26.75 | 54.5 MB | dùng đúng pretrained INRIA |
| **Paper tự báo (v6 / v5)** | **27.13 / 27.28** | 45 / 42 MB | |

→ **Cụm độc lập 26.75–27.31, tức ở mức hoặc THẤP HƠN số paper.** Size 39–55 MB, **cao hơn** 42–45 MB paper báo.

### Tanks & Temples — mốc cho `truck`, `train` của ta

| Nguồn | PSNR | Size | Baseline 3DGS của họ |
|---|---|---|---|
| SPARE-GS (rate cao nhất) | 23.56 | 238.77 MB | **23.83** ⚠️ đã sửa từ 23.74 |
| ACE-GS | 23.49 | 24.71 MB | 23.74 |
| FCGS | 23.62 | 28.60 MB | 23.71 |
| **FlexGaussian** | **23.14** | 27.87 MiB | ⚠️ **23.14 là CHÉP từ paper 3DGS**, không phải đo. Số đo thật của họ: **23.36** (train 21.78 / truck 24.93) |
| ~~GS²~~ | ~~23.16~~ | — | ❌ **BỎ HÀNG NÀY** — GS² đánh giá **21 scene T&T**, không có truck/train. Protocol không so được |
| VEDAL | 23.15 ± 0.06 | 165 MB | 23.68 |
| NeuralGS | 23.11 | 24.74 MB | 23.75 |
| ProtoGS | 23.09 | — | 21.99 |
| PUP 3D-GS @**90%** prune | 23.08 | 43.33 MB | 23.77 |
| POTR | 22.86 | 29.1 MB | 23.36 |
| HAC | 22.83 | 22.43 MB | ⚠️ **chép lỗi**, xem dưới |

### Deep Blending — LightGaussian **CHƯA BAO GIỜ đánh giá**, và đây là khoảng trống của ta

| Nguồn | PSNR | Baseline 3DGS của họ | Δ |
|---|---|---|---|
| SPARE-GS | **29.66** | 29.77 | −0.11 |
| LearnedPrior | 29.56 | 30.06 (Mini-Splatting) | −0.50 |
| ACE-GS | 29.48 | 29.77 | −0.29 |
| SafeguardGS `Prune_LG` | 29.07 | — | |
| ControlGS | 29.41 | 29.88 | −0.47 |
| POTR | 29.16 | 29.43 | −0.27 |
| NeuralGS | 29.12 | 29.42 | −0.30 |
| VEDAL | 28.85 ± 0.08 | 29.42 | −0.57 |
| FlexGaussian | 28.82 | 29.41 | −0.59 |
| PUP @90% | 28.51 | 28.98 | −0.47 |
| GaussianPOP | 27.57 | 29.42 | −1.85 |
| **HAC** | **27.01** | 29.42 | **−2.41** ⚠️ |

**🔴 Cảnh báo chuỗi trích dẫn sai — quan trọng cho tính trung thực của report:**
- HAC báo DB **27.01** — **outlier nặng** so với mọi phép đo độc lập khác (cụm 28.8–29.6). Con số này
  **lan nguyên văn** sang **HAC++** (Table II: 27.01/0.872/0.308/33.94) và **SymGS** (Table 1, chép nguyên cả hàng).
- Hàng **T&T của HAC là bản chép có lỗi** từ LightGaussian v1–v4: PSNR 22.83, LPIPS 0.242, size 22.43 MB khớp,
  nhưng SSIM 0.822 là ô **của 3D-GS** trong chính bảng đó (SSIM thật của LightGaussian là 0.807).
- **GaussianSpa** Table 2 hàng LightGaussian là **chép nguyên v5**, không phải đo lại.
- Survey [2502.19457](https://arxiv.org/html/2502.19457v1) ghi rõ trong caption: *"All values are sourced from their respective papers."*
- **MEGS²** chép nguyên bộ ba chất lượng của v6; **chỉ cột VRAM là của họ.**

→ **Quy tắc cho project:** khi so sánh, so với **cụm re-run độc lập** (nêu rõ nguồn nào), tuyệt đối không
lấy một số đơn lẻ từ HAC/HAC++/SymGS cho Deep Blending.

### 🎯 Số neo cho H1 (recovery sau prune)

**GaussianPOP** Table 3 — scene `bicycle`, prune 80% post-hoc từ pretrained 3DGS, PSNR **ngay sau prune → sau fine-tune**:

| Method | Sau prune | Sau fine-tune | Phục hồi |
|---|---|---|---|
| **LightGaussian** | **18.46** | **24.35** | **+5.89 dB** |
| PUP 3D-GS | 18.01 | 24.37 | +6.36 |
| C3DGS | 19.15 | 24.39 | +5.24 |

**REFINE** ([2606.09074](https://arxiv.org/html/2606.09074v4)) cho mốc **zero-shot, KHÔNG fine-tune** — chính là A3 của ta:

| ratio | Mip360 | T&T | DB |
|---|---|---|---|
| 0.1 | 27.34 | 23.39 | 29.52 |
| 0.3 | 27.33 | 23.38 | 29.51 |
| 0.5 | 26.92 | 23.24 | 29.36 |
| 0.7 | 24.73 | 21.96 | 28.38 |

→ Prune tới 30% gần như miễn phí; **sụp đổ bắt đầu giữa 0.5 và 0.7**. Đây là dự đoán tốt nhất cho collapse curve của ta.

---

## 10. Reproduction của mình `[điền sau P2]`

| Scene | Metric | Paper | Của ta | Điều kiện khác | Giải thích |
|---|---|---|---|---|---|
| truck | PSNR sau prune 66% + FT | ? | | | |
| truck | PSNR sau distill SH2 | ? | | | |
| truck | Size (entropy) | ? | | | |
| playroom | … | | | | |
