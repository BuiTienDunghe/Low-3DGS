# Low-VRAM 3DGS — Project Plan v2

> ⛔ **SUPERSEDED bởi [`PLAN.md`](PLAN.md) (v3, 2026-09-01).** File này là lịch sử. Khác biệt chính:
> mục tiêu đổi sang **fine-tune** (không train from scratch); thêm ràng buộc **RAM**;
> bỏ C4 densification controller; codebase nền đổi từ gsplat sang **INRIA + LightGaussian**.

**Tên đầy đủ:** Training 3D Gaussian Splatting Under a Hard VRAM Budget:
A Memory Cost Model, a Budgeted-Densification Controller, and an Iso-Memory Benchmark on 4GB Consumer GPU

**Tên ngắn:** Low-VRAM 3DGS

> Đọc `00_CRITIQUE.md` trước. Plan v2 này là kết quả sau khi sửa 7 vấn đề ở đó.
>
> ⚠️ **CẬP NHẬT QUAN TRỌNG:** `06_WHAT_FITS.md` đính chính V1 và chia project thành
> **2 track**. Thứ tự phase ở §6 dưới đây đã bị **thay thế** bởi §5 của `06_WHAT_FITS.md` —
> Track A (prune + fine-tune từ pretrained checkpoint) được kéo lên sớm vì nó ít rủi ro
> nhất và chạy được trên scene lớn. Đọc `06_WHAT_FITS.md` trước khi bắt đầu P1.

---

## 1. Thay đổi cốt lõi so với v1

| | Plan v1 | Plan v2 |
|---|---|---|
| Trọng tâm kỹ thuật | Post-hoc compression (prune / quantize model đã train) | **Train-time memory control** + post-hoc compression như một thành phần |
| Contribution chính | "Memory-aware adaptive pruning" | **Memory cost model + iso-memory benchmark**, controller là thành phần thứ hai |
| Novelty claim | Ngầm định là mới | **Thừa nhận công khai là reproduction + extension**, delta được nêu rõ |
| Đơn vị so sánh | Cùng compression ratio | **Cùng VRAM budget (iso-memory)** |
| SH degree | Compression trick, Phase 5 | **Memory knob lớn nhất, Phase 2** |
| Seeds | Không nhắc | **3 seeds, report mean ± std** |
| Compute | Không ước lượng | **Có budget table, có cắt giảm** |

---

## 2. Research Questions

### RQ chính (đo được, không cần novelty để có giá trị)

> **Cho một hard VRAM budget B (B < bộ nhớ mà 3DGS chuẩn cần), cấu hình 3DGS nào
> đạt rendering quality cao nhất, và tại giá trị B nào thì chất lượng sụp đổ?**

Đây là câu hỏi **benchmark/analysis**, không phải câu hỏi "method mới". Ưu điểm:
nó tạo ra giá trị kể cả khi method đề xuất thua — vì bản thân câu trả lời đã là kết quả.

### RQ phụ

- **RQ2 (cost model):** Peak training VRAM của 3DGS có dự đoán được từ
  `(N, resolution, scene statistics)` trước khi OOM xảy ra không? Sai số bao nhiêu?
- **RQ3 (allocation):** Với cùng budget B, nên tiêu vào **nhiều Gaussian + SH thấp**
  hay **ít Gaussian + SH đầy đủ**? Đây là trade-off chưa được trả lời rõ trong literature
  và là câu hỏi thú vị nhất của project.
- **RQ4 (controller):** Closed-loop controller theo *measured VRAM* có ổn định hơn
  open-loop theo *primitive count* (kiểu Taming 3DGS) không, đặc biệt khi rasterizer
  workspace biến động?

### Hypotheses

| ID | Hypothesis | Cách bác bỏ |
|---|---|---|
| H1 | Với cùng B, giảm SH degree cho phép giữ nhiều Gaussian hơn và thắng về PSNR so với giữ SH deg 3 | Đo iso-memory, nếu PSNR không khác quá `2*std` thì H1 bị bác bỏ |
| H2 | Peak VRAM dự đoán được với sai số < 10% bằng model tuyến tính theo N + số hạng tile-coverage | Fit trên 2 scene, test trên 3 scene còn lại |
| H3 | Importance-based selection thắng random selection ở cùng N | Nếu không, importance score vô dụng — **báo cáo negative result** |
| H4 | Closed-loop VRAM controller có OOM rate thấp hơn open-loop count-cap ở cùng target | Đếm OOM trên toàn bộ run |
| H5 | Fine-tuning ngắn sau prune phục hồi phần lớn quality đã mất | Đo PSNR trước/sau recovery |

**Quy tắc:** mỗi hypothesis phải nêu rõ điều kiện bị bác bỏ **trước khi** chạy experiment.

---

## 3. Contributions — xếp theo rủi ro

Nêu rõ cái nào là reproduction, cái nào là của mình. Không mập mờ.

### C1 — Empirical VRAM cost model cho 3DGS training `[rủi ro thấp, giá trị cao]`

Xây model dự đoán peak VRAM từ `(N, image resolution, tile-coverage statistics, SH degree, dtype)`.
Validate cross-scene. Ship kèm `tools/predict_vram.py`: nhập scene + config, trả lời
**"config này có OOM trên 4GB không"** trước khi tốn 2 giờ train.

Tại sao có giá trị: literature toàn báo cáo peak VRAM như một con số quan sát được,
gần như không ai model hoá nó. Đây là thứ **chắc chắn làm được** trên 1650 Ti và
là nền tảng cho mọi phần còn lại.

### C2 — Iso-memory benchmark `[rủi ro thấp, giá trị cao]`

Toàn bộ literature so sánh compression methods ở **cùng compression ratio** hoặc
**cùng quality**. Project này so ở **cùng VRAM budget** — đây là góc nhìn của người
thực sự bị giới hạn phần cứng, và là reframing hợp lệ, dễ bảo vệ.

Deliverable: bảng `method x budget {1.5, 2.0, 2.5, 3.0, 3.5} GB` -> `PSNR/SSIM/LPIPS/FPS`,
kèm Pareto front.

### C3 — Memory allocation study: Gaussian count vs SH degree `[rủi ro trung bình, thú vị nhất]`

Ở cùng B, quét grid `(N, SH degree)` trên đường iso-memory. Trả lời RQ3.
Chưa thấy ai làm rõ ràng ở dạng này. Đây là phần dễ gây ấn tượng nhất khi phỏng vấn
vì nó là một **insight**, không phải một con số.

### C4 — VRAM-budgeted densification controller `[rủi ro cao, đây là ý tưởng gốc của v1]`

Closed-loop controller: đo `torch.cuda.max_memory_allocated()` thật trong lúc train,
điều tiết densification threshold + prune-to-budget để giữ peak VRAM dưới B.

**Positioning bắt buộc và trung thực:**
- Taming 3DGS đã làm budgeted densification, target = **primitive count**, mục tiêu = **speed** trên GPU lớn.
- Gaussians on a Diet đã làm memory-bounded training.
- Delta của project này: target = **measured VRAM** (bao gồm rasterizer workspace, thứ
  không suy ra được từ N), closed-loop, và validate trên card 4GB thật.
- Trong report phải viết: *"a re-implementation of budget-constrained densification,
  extended from a primitive-count target to a measured-VRAM target"* — **không** viết "novel method".

### C5 — Reproduction của một compression method `[rủi ro trung bình, bắt buộc cho CV]`

Chọn 1 trong: LightGaussian / Compact-3DGS / reduced-3dgs. Ưu tiên **reduced-3dgs**
(INRIA, cùng codebase gốc, khả năng reproduce cao nhất) hoặc **LightGaussian**
(đúng ý định ban đầu). Quyết định sau literature review Phase 3.

---

## 4. Hardware & Stack — quyết định kỹ thuật

### 4.1 Hardware (constraint, không né tránh)

```
GPU     NVIDIA GTX 1650 Ti Mobile, 4GB GDDR6, Turing SM 7.5
        -> usable ~3.3 GB sau CUDA context
        -> KHÔNG có TF32; có FP16 tensor core nhưng rasterizer không dùng
CPU     AMD Ryzen 5 4600H (6C/12T)
RAM     16 GB   -> đủ cho COLMAP scene nhỏ/vừa, KHÔNG đủ cho scene lớn
Disk    ~90 GB
OS      Windows 11
```

**Cảnh báo laptop:** thermal throttling khi train nhiều giờ. Phải log GPU clock +
temperature trong `tools/profile_gpu.py`, và **loại bỏ throttled runs khỏi timing benchmark**
(quality metrics vẫn dùng được).

### 4.2 Quyết định: dùng `gsplat`, không dùng INRIA CUDA repo làm codebase chính

| Tiêu chí | INRIA `gaussian-splatting` | `gsplat` (nerfstudio) |
|---|---|---|
| Peak VRAM MipNeRF360 @30k | 9.0 GB | **5.6 GB** |
| Build trên Windows | MSVC + CUDA toolkit, hay fail | Dễ hơn, có wheel |
| Densification code | Trong Python, nhưng gắn chặt CUDA ext | Python, dễ hack |
| `packed=True` (sparse intermediates) | Không | **Có** |
| `sparse_grad` + SparseAdam | Không | **Có** |
| Là chuẩn so sánh trong paper | Có | Có (đã validate khớp INRIA) |

**Kết luận:** dùng gsplat làm nền cho mọi experiment. **Vẫn build INRIA repo một lần**
ở Phase 1 để (a) chứng minh làm được, (b) verify gsplat khớp số với reference.
Đây là điểm cộng lớn khi phỏng vấn: cho thấy bạn chọn tool có lý do, không chọn bừa.

### 4.3 Windows vs WSL2

Thử theo thứ tự, dừng ở cái đầu tiên chạy được:
1. **Native Windows + conda + CUDA 11.8 + MSVC Build Tools 2019/2022** — nhanh nhất, VRAM overhead thấp nhất
2. **WSL2 + Ubuntu** — build dễ hơn nhiều, nhưng GPU passthrough tốn thêm VRAM (đo cụ thể, ghi vào report)

Ghi lại chính xác cái nào dùng và VRAM overhead của nó. Đây là dữ liệu, không phải chi tiết vặt.

---

## 5. Dataset Strategy

### 5.1 Ràng buộc quyết định dataset

PSNR **phụ thuộc resolution**. Không được so số của mình với số trong paper nếu khác
resolution. Mọi bảng phải ghi resolution.

### 5.2 Chọn scene

| Tier | Dataset | Scene | Res | Vì sao |
|---|---|---|---|---|
| **Dev** | NeRF Synthetic (Blender) | `lego`, `chair` | 800x800 | Nhỏ, nhanh, có ground-truth pose (bỏ qua COLMAP). Vòng lặp dev nhanh. |
| **Main** | Tanks & Temples | `truck`, `train` | 1/2 res | Real-world, kích thước vừa, được dùng rộng rãi trong 3DGS papers |
| **Main** | Deep Blending | `playroom`, `drjohnson` | 1/2 res | Indoor, khác đặc tính với T&T |
| **Stress** | MipNeRF360 | `bicycle` hoặc `garden` | **1/4 hoặc 1/8** | Scene nặng nhất. Ở full res sẽ OOM — **và đó chính là kết quả cần báo cáo** |
| Optional | Self-captured | 1 scene | — | Chỉ làm nếu còn thời gian. COLMAP tự chụp là hố thời gian. |

**Final benchmark: 4 scene** (`lego`, `truck`, `playroom`, `bicycle@1/4`). Đủ đa dạng
(synthetic / outdoor / indoor / unbounded), không quá tốn compute.

### 5.3 Train/test split — protocol cố định

- MipNeRF360 / T&T / DB: **mỗi ảnh thứ 8 làm test** (`llffhold=8`) — đúng chuẩn 3DGS paper
- NeRF Synthetic: dùng split `train`/`test` chính thức
- **Không được đổi split giữa các experiment.** Ghi hash của danh sách test image vào mỗi run.

---

## 6. Phases (đã sửa) + Decision Gates

Mỗi phase có **exit criteria**. Không đạt thì không sang phase sau — rẽ theo fallback trong `03_RISKS.md`.

### P0 — Environment & Hardware Profiling `[3-5 ngày]`
- Cài CUDA + PyTorch + gsplat + COLMAP. Ghi lại toàn bộ version vào `docs/env.md`
- `tools/profile_gpu.py`: log VRAM allocated/reserved, GPU clock, temp, power, utilization
- Đo **CUDA context overhead** (VRAM mất trắng trước khi làm gì)
- Đo VRAM overhead của WSL2 nếu dùng
- **Exit:** chạy được một forward+backward rasterization dummy trên GPU và đo được peak VRAM

### P1 — 3DGS Fundamentals + Reproduce Baseline `[1-2 tuần]`
- Train `lego` đến hội tụ trên 1650 Ti
- Build INRIA repo một lần, verify gsplat ra PSNR tương đương (chênh < 0.3 dB)
- Đọc code: training loop, densification, split/clone, opacity reset, SH progressive activation
- `docs/paper_notes/3dgs_original.md`: map từng equation trong paper sang dòng code
- **Exit:** `lego` PSNR trong khoảng ~1 dB so với paper, VÀ giải thích được densification bằng lời

### P2 — Memory Profiling + Cost Model (C1) `[1-2 tuần]` **<- phase quan trọng nhất**
- Instrument từng thành phần: params / grads / Adam states / rasterizer buffers / activations
- Quét `N` và `resolution`, ghi peak VRAM -> fit cost model
- Đo riêng đóng góp của SH degree (0/1/2/3)
- Xác định **chính xác** cái gì OOM trước khi train xong ở mỗi scene
- **Exit:** `tools/predict_vram.py` dự đoán peak VRAM sai số < 15% trên scene chưa thấy

### P3 — Literature Review `[1 tuần, chạy song song P2]`
- Bắt đầu từ survey `3DGS.zip` (CGF 2025) và `Compression in 3DGS: A Survey` (2502.19457)
- Điền `04_LITERATURE.md`, tối thiểu 12 paper
- Chọn method để reproduce -> ghi lý do chọn
- **Exit:** bảng literature đầy đủ + 1 method được chọn có lý do viết thành văn

### P4 — Iso-Memory Baselines (C2) `[2 tuần]`
Chạy 4 trivial baselines ở từng budget:
`hard-cap-N` · `random-prune` · `opacity-prune` · `grad-threshold-tuning`
- **Exit:** có Pareto front của trivial baselines. Đây là **thanh chắn** mà C4 phải vượt.

### P5 — Allocation Study (C3) `[1-2 tuần]`
- Grid `(N, SH degree)` trên đường iso-memory, trả lời RQ3
- **Exit:** biểu đồ trả lời "nhiều Gaussian SH thấp hay ít Gaussian SH cao"

### P6 — Reproduce SOTA compression (C5) `[2-3 tuần]`
- Map paper equations -> code, document mọi khác biệt implementation
- Reproduce ở scale nhỏ; nếu không khớp paper, **giải thích tại sao** (res khác, iter khác, hardware khác)
- **Exit:** số của mình + giải thích chênh lệch so với paper

### P7 — Quantization `[1-2 tuần]`
- Per-attribute sensitivity: quantize từng nhóm (`xyz`/`scale`/`rot`/`opacity`/`SH_dc`/`SH_rest`) riêng lẻ, đo PSNR drop
- Trả lời: attribute nào chịu được INT8, cái nào bắt buộc FP16+
- **Đo model size đúng cách** — xem `02_METHODOLOGY.md`
- **Exit:** bảng sensitivity per-attribute

### P8 — VRAM Controller (C4) `[2-3 tuần]`
- Implement closed-loop controller
- **Exit:** train được cả 4 scene ở `--vram-budget 3.0GB` với **0 lần OOM**, VÀ so được với P4 baselines

### P9 — Ablation `[1-2 tuần]` → xem §8
### P10 — Multi-scene benchmark + Stress test `[2 tuần]`
- Budget sweep: 3.5 / 3.0 / 2.5 / 2.0 / 1.5 GB trên cả 4 scene -> **collapse curve**
### P11 — Pareto Analysis + Plots `[1 tuần]`
### P12 — Technical Report `[1-2 tuần]`
### P13 — README + CV framing `[3-5 ngày]`

**Tổng: ~4-6 tháng part-time.** Nếu ít thời gian hơn, xem §9 (Minimum Viable Project).

---

## 7. Compute Budget

Ước lượng thời gian train trên 1650 Ti (phải hiệu chỉnh lại sau P1):

| Scene | Res | 7k iters | 30k iters |
|---|---|---|---|
| `lego` | 800x800 | ~20 phút | ~1.5 h |
| `truck` | 1/2 | ~35 phút | ~2.5 h |
| `playroom` | 1/2 | ~30 phút | ~2.2 h |
| `bicycle` | 1/4 | ~45 phút | ~3.0 h |

**Chính sách cắt giảm compute (bắt buộc):**

| Loại run | Iters | Seeds | Scenes |
|---|---|---|---|
| Exploratory / debug | 7k | 1 | 1 (`lego`) |
| Ablation | 30k | 1 | 2 |
| **Headline comparison** | 30k | **3** | **4** |

Headline: `7 methods x 4 scenes x 3 seeds x ~2.3h` ≈ **190 GPU-hours**.
Cộng exploratory + ablation ≈ **250-300 GPU-hours** tổng.
Ở 6h/ngày (để máy còn dùng được) ≈ **7-8 tuần chạy**. Lên lịch chạy đêm.

`scripts/run_queue.py` phải có: **checkpoint-resume**, **OOM-retry với budget thấp hơn**, và
**append kết quả vào JSONL ngay sau mỗi run** (đừng để mất 40h vì crash ở run cuối).

---

## 8. Ablation Matrix (đã bổ sung trivial baselines)

| ID | Method | Mục đích |
|---|---|---|
| A0 | Vanilla 3DGS, không giới hạn | Upper bound (có thể OOM trên vài scene — **đó là kết quả**) |
| A1 | Hard cap N | Trivial baseline #1 |
| A2 | Random prune -> N | Trivial baseline #2 (kiểm tra H3) |
| A3 | Opacity-threshold prune | Trivial baseline #3 |
| A4 | Tăng `densify_grad_threshold` | Trivial baseline #4 (zero-code) |
| A5 | Importance prune, no recovery | Cô lập tác dụng của importance score |
| A6 | Importance prune + recovery FT | Kiểm tra H5 |
| A7 | SH reduction only | Cô lập knob SH (C3) |
| A8 | Quantization only | Cô lập quantization |
| A9 | Prune + SH + quantization | Kết hợp post-hoc |
| A10 | SOTA reproduction (C5) | Reference point |
| A11 | **Ours: VRAM controller (C4)** | Contribution |
| A12 | Ours + post-hoc compression | Full pipeline |

Mọi hàng phải đo ở **cùng budget**, không phải cùng compression ratio.

---

## 9. Definition of Done

### Minimum Viable Project (nếu hết thời gian, dừng ở đây vẫn có CV tốt)
- [ ] P0, P1, P2 hoàn tất — có cost model + `predict_vram.py`
- [ ] P4 — trivial baselines ở >= 3 budgets, >= 2 scenes
- [ ] P5 — allocation study
- [ ] Report + README + plots
- [ ] Seeds >= 3 cho headline, có mean ± std

MVP này **đã đủ mạnh cho CV** vì nó là analysis + engineering rigor, không phụ thuộc
vào việc method mới có thắng hay không.

### Full version
- [ ] Tất cả MVP
- [ ] P6 — reproduce 1 SOTA method, có giải thích chênh lệch
- [ ] P7 — per-attribute quantization sensitivity
- [ ] P8 — controller, 0 OOM ở 3.0GB trên 4 scene
- [ ] P9 — 13-row ablation
- [ ] P10 — collapse curve trên 5 budgets x 4 scenes
- [ ] P11 — Pareto (quality-vs-VRAM, quality-vs-size, quality-vs-FPS)
- [ ] P12 — mini-paper 6-10 trang
- [ ] Negative results được ghi lại, không xoá

---

## 10. CV framing — trung thực

**Không viết:**
> "Developed a novel memory-aware Gaussian pruning strategy."

Sai vì: không novel (V2), và interviewer biết 3DGS sẽ hỏi về Taming 3DGS.

**Viết:**
> **Low-VRAM 3D Gaussian Splatting** — Reproduced 3DGS and a SOTA compression method
> on a 4GB GTX 1650 Ti; built an empirical VRAM cost model that predicts peak training
> memory within X%, and used it to benchmark pruning, SH reduction and quantization
> **at matched memory budgets** across N scenes (PSNR/SSIM/LPIPS, VRAM, model size, FPS),
> with a budgeted-densification controller that trains without OOM at a 3GB target.

Điểm mạnh của cách viết này: mọi mệnh đề đều **verify được** và không claim novelty.
`X` và `N` chỉ điền sau khi có số thật.

**Câu chuyện kể trong phỏng vấn** (chính là thứ có giá trị nhất):
> "Tôi bắt đầu định nén model 3DGS đã train, rồi phát hiện ra đó là bài toán sai —
> peak memory nằm ở lúc train chứ không phải lúc render, nên post-hoc compression
> không giúp gì cho việc train được trên 4GB. Tôi chuyển sang model hoá memory,
> và phát hiện rasterizer workspace chứ không phải parameter mới là thủ phạm OOM.
> Từ đó tôi benchmark các method ở cùng memory budget thay vì cùng compression ratio,
> vì đó mới là câu hỏi mà người bị giới hạn phần cứng thực sự quan tâm."

Câu chuyện này mạnh hơn hẳn "tôi làm được method mới" — vì nó thể hiện
**diagnostic reasoning**, thứ mà interviewer thật sự đang test.

---

## 11. Repository Structure

```
low-3DGS/
├── README.md
├── requirements.txt / environment.yml
├── .gitignore                    # datasets/, results/checkpoints/, *.ply
│
├── configs/                      # 1 YAML / experiment, versioned
│   ├── base.yaml
│   ├── baselines/{hard_cap,random_prune,opacity_prune,grad_thresh}.yaml
│   ├── sh_study.yaml
│   ├── quantization.yaml
│   ├── sota_repro.yaml
│   └── ours_controller.yaml
│
├── src/
│   ├── memory/                   # <- MỚI, trái tim của project
│   │   ├── profiler.py           # instrument từng thành phần VRAM
│   │   ├── cost_model.py         # C1: predict peak VRAM
│   │   └── controller.py         # C4: closed-loop budget controller
│   ├── gaussian/                 # model, SH, covariance, I/O
│   ├── training/                 # loop, densification, schedules
│   ├── rendering/
│   ├── pruning/                  # importance scores + strategies
│   ├── quantization/             # per-attribute quant, VQ, entropy coding
│   ├── distillation/             # SH reduction / distillation
│   └── evaluation/               # PSNR/SSIM/LPIPS, aggregation
│
├── tools/
│   ├── profile_gpu.py
│   ├── predict_vram.py           # deliverable của C1
│   ├── benchmark.py
│   ├── aggregate_results.py      # JSONL -> bảng markdown/latex
│   ├── make_plots.py             # Pareto fronts
│   └── visualize.py
│
├── scripts/
│   ├── run_queue.py              # resume + OOM-retry + logging
│   └── {train,prune,evaluate,benchmark}.sh
│
├── tests/                        # <- MỚI
│   ├── test_cost_model.py
│   ├── test_quantization.py      # roundtrip error bounds
│   ├── test_pruning.py           # invariants: N giảm đúng, không NaN
│   └── test_metrics.py           # PSNR/SSIM vs reference impl
│
├── experiments/                  # 1 dir/run: config + log + metrics.jsonl
├── results/{metrics,plots,renders,checkpoints}/
└── docs/
    ├── 00_CRITIQUE.md            # <- bạn đang đọc series này
    ├── 01_PROJECT_PLAN.md
    ├── 02_METHODOLOGY.md         # protocol: seeds, splits, đo size, đo VRAM
    ├── 03_RISKS.md               # risk register + kill criteria
    ├── 04_LITERATURE.md
    ├── 05_EXPERIMENTS.md         # nhật ký, gồm cả thất bại
    ├── env.md
    └── paper_notes/
```

---

## 12. Quy tắc (mở rộng từ §25 của plan v1)

Giữ nguyên 12 quy tắc gốc, bổ sung:

13. **Mọi bảng phải ghi resolution, số iteration, và số seed.** Thiếu một trong ba thì bảng vô giá trị.
14. **Không so số của mình với số trong paper** trừ khi cùng dataset, cùng resolution, cùng split, cùng iteration.
15. **Rasterizer là non-deterministic.** Không hứa bit-exact reproducibility. Report `mean ± std`.
16. **Trivial baseline chạy trước method của mình**, không phải sau. Tránh tự lừa mình.
17. **Negative result được ghi vào `05_EXPERIMENTS.md` trong vòng 24h**, khi còn nhớ context.
18. **Không claim novelty.** Claim reproduction + extension + analysis. An toàn hơn và vẫn đủ mạnh.
19. **Ghi lại mọi lần OOM** kèm N, resolution, iteration. Đây là data cho C1, không phải lỗi vặt.
20. **Mỗi run ghi ra JSONL ngay lập tức.** Không tích luỹ trong RAM rồi ghi cuối.
