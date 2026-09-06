# Low-VRAM 3DGS — PLAN v4 (nguồn sự thật duy nhất)

**Tên:** *What Does It Actually Take to Reproduce 3DGS Compression on a 4 GB GPU?*
*A Reproducibility Audit and Joint RAM–VRAM Study of Post-hoc Gaussian Splatting Compression*

**Chốt:** 2026-09-02 · **Thời gian:** 4–6 tháng part-time · **Máy:** GTX 1650 Ti 4 GB / 15.4 GB RAM (WSL2 cap 10 GB)

> v4 thay thế v3. Cơ sở: 6 agent khảo sát cộng đồng (`07_RESEARCH_FINDINGS.md`) + 3 thí nghiệm đã chạy
> (`05_EXPERIMENTS.md`) + verify trực tiếp trong source đã clone. Mọi claim then chốt đã được tôi tự kiểm.

---

## 0. Mục tiêu (không đổi từ v3)

> Lấy model 3DGS **đã train sẵn**, nén (prune / giảm SH / quantize) rồi **fine-tune** phục hồi chất lượng —
> toàn bộ trên máy **RAM ≤ 10 GB, VRAM ≤ ~3.7 GB** — và đo chính xác **ràng buộc nào bind ở stage nào**.

**Trong phạm vi:** T&T (`truck`, `train`) + Deep Blending (`playroom`, `drjohnson`) · reproduce tiêu chí
LightGaussian + PUP · profiler RAM+VRAM theo stage · benchmark iso-memory · Pareto · report.
**Ngoài phạm vi:** train from scratch · MipNeRF360 (xem §4) · densification controller · scene tự chụp · dynamic/SLAM.

---

## 1. Bảy điều v4 sửa so với v3

| # | v3 nói | Thực tế đo/verify được | Nguồn |
|---|---|---|---|
| 1 | Nạp `.ply` tốn ~2× file | **4.58×** (INRIA thật). Tôi đo 1.96× vì dùng float32, INRIA dùng `np.zeros(...)` **không dtype → float64** | EXP-003 đính chính |
| 2 | `load_ply` là nút thắt | **`save_ply` mới là nút thắt: 12.08× ≈ 10.2 GB** cho drjohnson — vượt cả RAM | EXP-004 |
| 3 | Ảnh 12 B/pixel | **16 B/pixel** — `cameras.py:48` luôn tạo `alpha_mask = ones_like` | EXP-005 |
| 4 | Tự viết `ply_stream.py` | **Dùng [`gsply`](https://github.com/OpsiClear/gsply)** — 1.95× đọc / 1.30× ghi, nhanh hơn 6×, wheel cp312, không cần build CUDA | §5 |
| 5 | Reproduce trong repo LightGaussian | **Repo gốc có 5 bẫy im lặng + rasterizer pin KHÔNG implement công thức paper.** Dùng **PUP 3D-GS** | §5, paper_notes §7b |
| 6 | Mốc so sánh = số trong paper | **Số paper đổi qua các bản arXiv.** Mốc đúng = **cụm re-run độc lập** (T&T 22.83–23.62) | paper_notes §9b |
| 7 | 4 scene + bicycle stress | **Bỏ MipNeRF360.** Fine-tune 5.7M Gaussian ≈ 5.5 GB VRAM — bất khả thi | §4 |

---

## 2. Máy — số đo thật

| | Đo được | Ý nghĩa |
|---|---|---|
| GPU | GTX 1650 Ti, 4096 MiB, sm_75, driver 560.70 | CUDA ≤ 12.6. `TORCH_CUDA_ARCH_LIST="7.5"` |
| Màn hình | Trên **iGPU AMD**, dGPU idle **0 MiB** | Trọn 4 GB cho compute |
| RAM | 15.37 GB tổng · **5.03 GB** khi Chrome/Zalo mở · ~10–11 GB khi đóng | **Bắt buộc đóng app** khi chạy |
| WSL2 | `.wslconfig` → **9.7 GiB / 9.2 avail**, swap 4 GB trên D: | Mọi số RAM phải ghi kèm cap này |
| Disk | C: 41 GB · **D: 122 GB** | Đủ |
| OS | Win 11 Home + WSL2 Ubuntu 24.04.3, systemd, kernel 6.6.87 | GPU passthrough **đã verify** |
| Toolchain | ⏳ chờ `scripts/setup_wsl_sudo.sh` (cần sudo) | Chặn mọi thứ GPU |

**gcc 13 KHÔNG phải vấn đề** — guard của CUDA 12.4–12.6 là `__GNUC__ > 13`, và NVIDIA liệt kê Ubuntu 24.04
(gcc 13.2) là được hỗ trợ. Đừng hạ cấp xuống gcc-11. (gcc **14** mới hỏng, nhưng 24.04 dùng 13.)

---

## 3. Ngân sách bộ nhớ — số đã đo

### 3.0 🔴 HAI ĐƯỜNG CHẾT IM LẶNG trên máy này (đọc trước mọi thứ khác)

| | Cơ chế | Triệu chứng | Phát hiện bằng |
|---|---|---|---|
| **RAM** | pagefile 13.5 GB (Win) + swap 4 GB (WSL) | Run không crash, chậm 10–50× | `killswitch` theo RSS |
| **VRAM** | **Driver WSL2/WDDM tràn sang host RAM qua PCIe** (EXP-006) | **`OutOfMemoryError` KHÔNG kích hoạt.** NVML plateau, tốc độ tụt 3.6× | ✅ **ĐÃ DẬP bằng `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** (EXP-008) — GPU OOM sạch trở lại, giá 368 MiB. `killswitch` giữ làm lưới an toàn |

Bằng chứng EXP-006: cấp phát **10816 MiB thành công trên card 4096 MiB**. NVML đứng ở 4061.5 MiB
trong khi băng thông tụt 54 → 15 GB/s.

→ **Quy tắc:** mọi ngân sách VRAM phải **cưỡng chế tường minh**. Không bao giờ "để OOM tự báo".
Một run "chạy được" có thể đang chạy 3.6× chậm trên PCIe mà không dấu hiệu nào.

### 3.1 VRAM — đã đo (EXP-007)

```
4096.0 MiB  tổng
-  34.5 MiB  driver giữ (NVML đỉnh chỉ tới 4061.5)
- 161.0 MiB  đường nền WSL2 GPU-PV        <- câu trả lời cho "WSL2 overhead" mà plan hỏi từ v3
-  62.5 MiB  CUDA context                 <- ước tính cũ 350 MiB, SAI 5.6 lần
-  32.0 MiB  cuBLAS workspace (lazy)
- 368.0 MiB  chi phí của expandable_segments (đổi lấy OOM sạch, EXP-008)
─────────────
= 3440.0 MiB dùng được cho tensor    <- SỐ CHÍNH THỨC. (3808 nếu tắt expandable_segments, nhưng khi đó tràn im lặng)
```

Băng thông VRAM 53–57 GB/s. Lần đo đầu ra 37–38 GB/s vì clock chưa boost → **phải warm-up trước mọi phép đo tốc độ.**

| Kịch bản | B/Gaussian | N tối đa | truck 2.54M | playroom 2.55M | drjohnson 3.41M |
|---|---|---|---|---|---|
| Inference (params SH3) | 236 | 15.28 M | ✅ | ✅ | ✅ |
| **Fine-tune SH3** | 956 | **3.77 M** | ✅ 2.42 GB (dư 1.0) | ✅ 2.43 GB (dư 1.0) | ❌ 3.26 GB, chỉ còn **0.18 GB** cho rasterizer |
| Fine-tune SH2 | 620 | 5.55 M | ✅ | ✅ | ✅ 2.11 GB (dư 1.3) |
| **A12: prune trước `training_setup`** | 240 | 14.33 M | ✅ | ✅ | ✅ **0.82 GB** |

→ `truck` và `playroom` fine-tune **thoải mái**. `drjohnson` SH3 **không đủ** — bắt buộc A12 hoặc SH2.
Đây chính là lý do A12 từ "thủ thuật hay" thành **bắt buộc** cho scene lớn nhất.



### Bảng chi phí theo stage (nền của C2)

| Stage | RAM | VRAM | Nguồn |
|---|---|---|---|
| Nạp `.ply` (INRIA nguyên bản) | **4.58 × file** — drjohnson 3.86 GB | 0 | đo |
| Nạp `.ply` (**gsply**) | **1.95 ×** — drjohnson 1.65 GB, **1.5 s** | 0 | đo |
| Ảnh training, `data_device=cpu` | **16 B/px** × n × H × W | 0 | verify code |
| Ảnh training, `data_device=cuda` | 0 | **16 B/px** — playroom 3.52, drjohnson **4.57 GB** | verify code |
| Model trên GPU (params) | — | N × 236 B | tính |
| Fine-tune (params+grad+2 Adam+stats) | — | **N × ~956 B** | tính |
| Importance score (`prune_list`) | +ảnh | +rasterizer, **không có `no_grad()`** → dựng autograd graph mỗi view | verify code |
| SH distill | — | **2 model cùng lúc** (teacher+student) | verify code |
| **Ghi `.ply` (INRIA)** | **12.08 × file ≈ 10.2 GB** ❌ | 0 | ngoại suy |
| **Ghi `.ply` (gsply)** | **1.30 ×** ≈ 1.04 GB | 0 | đo |

### Ngân sách theo scene (`data_device=cpu`, dùng gsply)

| scene | N @30k | ảnh (16 B/px) | RAM đỉnh ước tính | VRAM fine-tune | Khả thi |
|---|---|---|---|---|---|
| **train** | 1.03 M | 2.40 GB | ~4.3 GB | **0.98 GB** | ✅ thoải mái |
| **truck** | 2.54 M | 2.00 GB | ~4.3 GB | **2.42 GB** | ⚠️ VRAM chật |
| **playroom** | 2.55 M | 3.52 GB | ~5.8 GB | **2.43 GB** | ⚠️ chật cả hai |
| **drjohnson** | 3.41 M | 4.57 GB | ~7.2 GB | **3.25 GB** | ❌ nhiều khả năng OOM |
| ~~bicycle~~ | ~5.7 M | — | — | ~5.5 GB | ❌ **ngoài phạm vi** |

### 🔧 Thủ thuật cứu scene lớn: đảo thứ tự prune và `training_setup`

Cả LightGaussian lẫn PUP đều `load_ply()` → `training_setup(opt)` → **rồi mới** prune.
**Adam cấp phát moment đủ cỡ TRƯỚC khi prune.** Chuyển pass importance + prune lên **trước** `training_setup()`:

```
đỉnh VRAM: 956 B/Gaussian  ->  ~240 B/Gaussian
drjohnson: 3.25 GB -> 0.82 GB
```

**~5 dòng code.** Đây vừa là thứ cứu drjohnson, vừa là một đóng góp kỹ thuật báo cáo được → **thí nghiệm A12**.

---

## 4. Dataset

| Scene | N @30k | n ảnh | dims | Vai trò |
|---|---|---|---|---|
| `train` (T&T) | 1,026,508 | 301 | 980×545 | **Dev + gate.** Scene duy nhất chắc chắn fit |
| `truck` (T&T) | 2,541,226 | 251 | 979×546 | Main. Có nhiều số so sánh nhất |
| `playroom` (DB) | 2,546,116 | 225 | 1264×832 | Main |
| `drjohnson` (DB) | 3,405,153 | 263 | 1332×876 | **Stress.** Cần thủ thuật §3 |

Bundle có sẵn cả **@7k** (55–68% N) → **8 điểm N miễn phí** cho cost model.
Split: `llffhold=8` (`cfg_args` ghi `eval=True`). Res: `-r 1` (mọi ảnh ≤ 1.6K, không bị auto-rescale).

**Vì sao bỏ MipNeRF360:** fine-tune 5.7M ≈ 5.5 GB VRAM. Và **LightGaussian không có số Deep Blending nào**
→ DB là **khoảng trống công bố được**, không phải reproduction thất bại. Đổi stress scene từ `bicycle` sang `drjohnson`.

---

## 5. Stack — quyết định và lý do

| Hạng mục | Chọn | Lý do |
|---|---|---|
| Môi trường | **WSL2 Ubuntu 24.04** | Không có MSVC; GPU passthrough đã verify; repos Linux-first |
| CUDA / PyTorch | **12.6 / cu126** (fallback cu124) | Driver 560.70 hỗ trợ 12.6 |
| **Repo nền** | 🔄 **[PUP 3D-GS](https://github.com/j-alex-hanson/gaussian-splatting-pup)** | torch **2.4.1**, commit cuối 22/11/2025, **0 issue mở**. Đã implement **đúng tiêu chí LightGaussian** (`--prune_type v_important_score`, có ghi nguồn). **Đã vá lỗi iteration** của LightGaussian. Rasterizer đổi tên → chung sống được với INRIA. **Hai method, một lần build** |
| Fallback repo | [gaussian-splatting-lightning](https://github.com/yzslab/gaussian-splatting-lightning) | Reimplement LightGaussian **trên gsplat** → khỏi build rasterizer riêng. Nạp `.ply` thẳng. Chạy scoring dưới `no_grad()`. 1.1k★, active 05/2026 |
| Đọc/ghi `.ply` | **[`gsply`](https://github.com/OpsiClear/gsply)** | Wheel cp312 manylinux, không build CUDA. Verify `torch.equal` bitwise |
| **Metrics** | **Copy nguyên `metrics.py` + `utils/loss_utils.py` + `lpipsPyTorch/` của INRIA** | Xem §9. Tự viết = số không so sánh được |
| Size đo | [`ffsplat`](https://github.com/w-m/ffsplat) (nhóm KISS-GS) | ⚠️ pre-alpha |
| Baseline nén thêm | [Splatwizard](https://github.com/splatwizard/splatwizard) (12 model) · [gsplat `PngCompression`](https://docs.gsplat.studio/main/apis/compression.html) · [REFINE](https://github.com/ZhangChen2022/REFINE) | REFINE = `pip install torch numpy plyfile`, một file, **không rasterizer** |
| Tracking | JSONL + `aggregate_results.py` | |

### Patch bắt buộc trước khi build (đã verify là **thiếu** ở mọi repo)

```bash
#include <cstdint>   ->  cuda_rasterizer/rasterizer_impl.h     # gcc 13 bỏ include bắc cầu
#include <cfloat>    ->  simple-knn/simple_knn.cu              # CUDA 12 bỏ FLT_MAX/FLT_MIN
export TORCH_CUDA_ARCH_LIST="7.5"
pip install --no-build-isolation ...     # setup.py import torch ở module level, không có pyproject.toml
```
Đã tự động hoá trong `scripts/setup_wsl_sudo.sh`. Kiểm `third_party/glm/` **không rỗng** trước khi build.

---

## 6. Research Questions & Hypotheses

| | Câu hỏi |
|---|---|
| **RQ1** | Ở từng stage của pipeline nén, **ràng buộc nào bind trước — RAM hay VRAM** — và bao nhiêu phần là *artifact sửa được* so với *cấu trúc không sửa được*? |
| **RQ2** | Code LightGaussian đã phát hành có tái tạo được kết quả trong paper không, và **sai lệch đến từ đâu**? |
| **RQ3** | Ở cùng ngân sách bộ nhớ, nên giữ **nhiều Gaussian + SH thấp** hay **ít Gaussian + SH cao**? |
| **RQ4** | Ảnh training nên nằm ở **CPU hay GPU** khi cả hai đều thiếu? |

| ID | Hypothesis | Số neo | Bác bỏ khi |
|---|---|---|---|
| **H1** | Fine-tune 5k iter sau prune 66% phục hồi ≥ 80% PSNR đã mất | GaussianPOP `bicycle` @80%: **18.46 → 24.35 = +5.89 dB** | phục hồi < 50% |
| **H2** | Đỉnh RAM ở stage **save**, đỉnh VRAM ở stage **fine-tune** hoặc **distill** | save 12.08× vs load 4.58× | profiler cho ngược lại |
| **H3** | Sau khi thay gsply + `data_device=cpu` + uint8, trần RAM **vẫn còn** và là cấu trúc (Adam state, teacher+student) | Gaussians-on-a-Diet: 8.55 → 2.98 GB | trần biến mất hoàn toàn → **RQ1 sụp, phải báo cáo** |
| **H4** | Prune theo importance thắng random ở cùng N | REFINE zero-shot: ratio 0.5 gần như miễn phí, 0.7 sụp | chênh < 2×std → importance vô dụng, **report negative** |
| **H5** | Ở cùng byte budget, SH2 + nhiều Gaussian thắng SH3 + ít Gaussian | — | chênh < 2×std |
| **H6** | Ảnh trên CPU chậm < 25% nhưng cho phép scene mà GPU-resident sẽ OOM | playroom `cuda` = 4.12 GB > 4 GB | chậm > 40% |
| **H7** | Rasterizer pin (thiếu `* T`) cho kết quả **khác** bản `main` (có `* T`) | chưa ai đo | chênh < 2×std |
| **H8** | Đảo prune trước `training_setup` giảm đỉnh VRAM ~4× mà không đổi chất lượng | 956 → 240 B/Gaussian | chất lượng lệch > 2×std |

---

## 7. Contributions — xếp theo mức độ chắc chắn

### C1 — Reproducibility audit của LightGaussian `[rủi ro thấp, giá trị CAO NHẤT]`

Đã có sẵn 8 phát hiện **đã verify trong source**, trước khi chạy một dòng nào:

| | Phát hiện | Hệ quả |
|---|---|---|
| T1 | `--prune_iterations` mặc định `[30_001]` nhưng vòng lặp `range(0, 30001)` → max 30000 | **Không bao giờ prune**, không lỗi |
| T2/T3 | `--prune_type` mặc định `important_score` (không phải `v_important_score`); `--prune_percent` = 0.1 (không phải 0.66) | Sai thuật toán, sai tỉ lệ |
| T4 | `vectree --sh_degree` mặc định 2, `.ply` INRIA là SH 3 | Cắt sai cột SH trong im lặng |
| T5 | Kernel dùng `+=` thường, **`atomicAdd` = 0** | Score **không xác định** và lệch thấp |
| T6 | LightGaussian dùng **AdamW** (weight decay 0.01 lên mọi tham số), INRIA dùng **Adam** | Khác biệt không ghi trong paper |
| T7 | `setup.py` cài dưới tên `diff_gaussian_rasterization` — **trùng INRIA** | Ghi đè im lặng |
| T9 | `distill_train.py` bắt buộc `.pth`; INRIA chỉ phát hành `.ply` | **SH distillation không chạy được** |
| **D1** | Submodule pin `240618bb`: `important_score += con_o.w;` — **KHÔNG có `* T`** | **Code không implement Eq. 3 của paper.** Bản sửa có từ 11/2024, repo cập nhật 12/2024 mà vẫn pin SHA cũ |

Cộng thêm **chuỗi trích dẫn sai** đã lần ra: HAC báo Deep Blending 27.01 (outlier, cụm độc lập 28.8–29.6),
số này **lan nguyên văn** sang HAC++ và SymGS; hàng T&T của HAC là **bản chép có lỗi** (lấy nhầm ô SSIM của 3D-GS).

> Đây là contribution mạnh nhất và **không phụ thuộc vào việc experiment có đẹp hay không**.

### C2 — Per-stage RAM+VRAM profiler `[rủi ro thấp]`
Agent tìm **không thấy** công cụ tương đương. *A LoD of Gaussians* (SIGGRAPH 2026) có một hình CPU-vs-GPU
cho một scene → **không claim "đầu tiên"**, nhưng vẫn ship được tool và bảng đầy đủ 8 stage × 4 scene.

### C3 — Phân rã trần bộ nhớ: artifact vs cấu trúc `[rủi ro TB, đây là luận đề]`

| Thành phần | Sửa được? | Bằng chứng |
|---|---|---|
| Nạp checkpoint | ✅ 4.58× → 1.95× (gsply) | đã đo |
| **Ghi checkpoint** | ✅ 12.08× → 1.30× | đã đo |
| Ảnh training | ✅ 16 → 4 B/px (uint8) | `gaussian-splatting-lightning`, PR #816 |
| **Adam state khi fine-tune** | ❓ **phải chứng minh là cấu trúc** | 4× params |
| **Teacher+student khi distill** | ❓ | 2× model |

**Đòn phản biện phải chặn:** *"trần RAM của anh là artifact của implementation."* Câu trả lời trung thực:
**đúng một phần, và đây là phần nào** — với số đo cho từng phần. Chỉ khi trần **vẫn còn** sau khi đã sửa hết
thì H3 mới sống. Nếu không → **báo cáo negative, RQ1 sụp**. Đó là kết quả hợp lệ.

### C4 — Benchmark ở cùng ngân sách bộ nhớ `[rủi ro thấp, phải nêu tiền lệ]`

⚠️ **Không được claim đặt ra thuật ngữ.** `"iso-memory"` đã tồn tại từ 2007 (nghĩa khác, HPC), và trong ML
có Basso-Bert 2025 định nghĩa rõ *"comparison also referred to as 'at iso-memory'"*, SCV-GNN 2023 *"an iso-memory comparison"*.
Nhưng: **1 abstract duy nhất trên toàn arXiv**, **0 trong NeRF/3DGS**, chưa từng được hình thức hoá thành protocol.

**Khung trung thực:** *"we adopt the `iso-X` convention already standard for iso-FLOPs and iso-accuracy, and apply
the occasionally-used but never-formalized term iso-memory as a named protocol for neural rendering, where
equal-memory evaluation has to date only been done implicitly through rate–distortion curves."*

**Bắt buộc cite tiền lệ:** Dettmers & Zettlemoyer 2022 (fixed total model bits) · TaCQ / COLM 2025 (*"at matched memory budgets"*)
· Basso-Bert 2025 (thuật ngữ) · 3DGS.zip / GSCodec Studio / Splatwizard (RD curves) · **MEGS² ICLR 2026**
(đã in luận điểm: model nén có thể cần **nhiều** VRAM render **hơn** bản chưa nén).
Và định nghĩa rõ ở lần dùng đầu — tránh nhầm với nghĩa HPC 2007.

### C5 — Deep Blending gap `[rủi ro thấp]`
LightGaussian **chưa bao giờ đánh giá DB**; các phép đo độc lập trải **27.01 → 29.66 (2.65 dB)**.
Đo lại cẩn thận trên `playroom` + `drjohnson` với protocol ghi rõ là đóng góp có ích ngay.

### ❌ Đã bỏ
Densification controller (off-mission, đã bị Taming 3DGS + Gaussians-on-a-Diet scoop) ·
cost model RAM+VRAM dạng công thức (CLM + LoD đã có; là **một đoạn methodology**, không phải contribution).

---

## 8. Pipeline + điểm đo

```
[S0] Tải checkpoint            RAM: 0            VRAM: 0
[S1] Nạp .ply (gsply)          RAM: 1.95×file    VRAM: 0          <- so với INRIA 4.58×
[S2] Lên GPU                   RAM: giảm         VRAM: N×236B
[S3] Importance score          RAM: +ảnh         VRAM: +rasterizer  <- không no_grad(), render mọi view
[S4] Prune  ** TRƯỚC training_setup **                              <- A12, cứu 4× VRAM
[S5] training_setup (Adam)     —                 VRAM: N'×956B
[S6] Fine-tune 5k iter         —                 VRAM: đỉnh nghi #1
[S7] SH distill 3->2           —                 VRAM: 2 model      <- đỉnh nghi #2
[S8] VQ + entropy coding       CPU
[S9] Ghi .ply (gsply)          RAM: 1.30×         <- so với INRIA 12.08× = 10.2 GB
[S10] Evaluate                 LPIPS-VGG process riêng
```

Kill-switch RSS > 9.5 GB (`src/memory/killswitch.py`), exit code 3. **Mỗi lần trip là một data point.**

---

## 9. Protocol đo — bốn lỗi sẽ làm hỏng mọi số liệu

| # | Quy tắc | Vì sao |
|---|---|---|
| **L1** | **LPIPS = bản vendored `lpipsPyTorch/` của INRIA, backbone VGG.** KHÔNG dùng `lpips` pip, KHÔNG dùng torchmetrics | Bản vendored có **lỗi chuẩn hoá đã biết** ([#1239](https://github.com/graphdeco-inria/gaussian-splatting/issues/1239), vẫn mở): hằng số mean/std cho `[-1,1]` nhưng ảnh vào `[0,1]`. Sửa lỗi làm LPIPS **tăng** → mọi paper dòng 3DGS báo LPIPS **lạc quan hệ thống**. **Ngành giữ lỗi để so sánh được.** KISS-GS (8/2026) nói rõ họ giữ `[0,1]`. Ta làm y hệt và **ghi rõ trong report** |
| **L2** | **SSIM = `utils/loss_utils.py` của INRIA** (Gauss 11×11, σ=1.5, C1=0.01², C2=0.03², padding=5, conv theo nhóm kênh). KHÔNG dùng `skimage` | Cho số khác |
| **L3** | **MB = 10⁶ byte**, không phải 2²⁰ | Sai 4.9%, lớn hơn biên cải thiện của nhiều paper |
| **L4** | **Size = tổng byte trên đĩa của MỌI THỨ cần để decode, sau bước coding cuối.** `npz_fp16` và "quantized chưa nén" **chỉ là chẩn đoán**, không bao giờ là số headline. Luôn báo **cả** baseline chưa nén | HAC tính cả MLP+hash grid, mã hoá arithmetic; Compact-3DGS Huffman→DEFLATE. Baseline 3DGS **không bao giờ được nén** (T&T 411.0 MB / DB 676.0 MB là `.ply` fp32 thô) |

**Seeds:** noise floor trước tiên (A7 trên `truck`, 3 seed, ghi std). Improvement chỉ tính khi `> 2×std`.
Rasterizer **non-deterministic** (thêm cả race condition T5) → `mean ± std`, không hứa bit-exact.
**Headline 3 seed, không bao giờ cắt.** Exploratory/ablation 1 seed, ghi rõ.

**Mốc so sánh:** dùng **cụm re-run độc lập** (paper_notes §9b), không dùng số đơn lẻ từ paper.
T&T cụm 22.83–23.62 · DB cụm 28.8–29.6 (bỏ outlier HAC 27.01) · FlexGaussian là gần nhất (code chính thức, RTX 3090).

---

## 10. Phases & gates (24 tuần)

| Tuần | Phase | Việc | **Gate** |
|---|---|---|---|
| 1 | **P0** Env | `setup_wsl_sudo.sh` · venv + torch cu126 · **gsply** · build PUP (4 ext, có patch) | `torch.cuda.is_available()` ✅ · đo **CUDA context + WSL overhead** |
| 1–2 | **P1** Gate sống-chết | Render `train` + `truck` pretrained, `--data_device cpu`, đo RAM+VRAM 10 stage · đo lại `load_ply`/`save_ply` **tại chỗ, có torch** | **`truck` render được, PSNR ≈ 25.187 ±0.3 dB, RSS < 9 GB.** Không đạt → hạ `-r 2`, ghi bảng OOM |
| 2–4 | **P2** Reproduce | PUP repo: `v_important_score` @66% trên `train` → `truck` → `playroom` → `drjohnson` · **kiểm 8 bẫy T1–T9** | Log in `Before/After prune` với mức giảm ~2.9×. Không thấy → T1 |
| 3–4 ‖ | **P3** Literature | `04_LITERATURE.md` ≥ 12 paper · chốt baseline thứ 2 | Bảng đầy đủ |
| 5–7 | **P4** Profiler + C3 | Instrument 10 stage × 4 scene × {gsply, INRIA} × {cpu, cuda} | Bảng phân rã artifact-vs-cấu trúc (H3) |
| 7–9 | **P5** Baselines | random · opacity · hard-cap · `important_score` vs `v_important_score` · **rasterizer pin vs main (H7)** · noise floor 3 seed **trước** | Pareto trivial baselines |
| 9–11 | **P6** Allocation + placement | Grid (N, SH) iso-memory · ảnh CPU vs GPU · **A12 đảo thứ tự prune (H8)** | Trả lời RQ3, RQ4 |
| 11–13 | **P7** Stress | `drjohnson` với và không có A12 · tìm điểm bind | Collapse curve |
| 13–16 | **P8** Quantization | Per-attribute sensitivity · VQ (`--sh_degree 3`!) · entropy coding · REFINE làm baseline rẻ | Bảng sensitivity |
| 16–19 | **P9** Benchmark | 4 scene × ~10 config × 3 seed, chạy đêm | JSONL đầy đủ |
| 19–22 | **P10** Phân tích + report | Pareto (quality–RAM, quality–VRAM, quality–size, quality–FPS) · mini-paper | Draft |
| 22–24 | **P11** Đóng gói | README số thật · configs · CV bullet · **buffer 2 tuần** | Done |

**Gate quan trọng nhất: P1.** Nếu `truck` không render nổi trong 10 GB RAM → đổi sang `train` + `-r 2`,
và **chính điều đó là kết quả C3**.

---

## 11. Ablation

| ID | Config | Mục đích |
|---|---|---|
| A0 | Pretrained gốc, eval-only | Upper bound |
| A1 | Random prune → N′ | Trivial #1 (H4) |
| A2 | Opacity-threshold prune | Trivial #2 |
| A3 | Importance prune, **no** fine-tune | Cô lập; so REFINE zero-shot |
| A4 | Importance prune + FT 5k | H1 |
| A5 | `important_score` (không nhân volume) | Cô lập Eq. 4 |
| A6 | `v_important_score` + clamp `min(·,1)` theo paper | Paper-faithful vs code (D7) |
| A7 | **Rasterizer pin (không `* T`)** | Baseline chính |
| A7′ | **Rasterizer `main` (có `* T`)** | **H7 — chưa ai đo** |
| A8 | SH 3→2 (+FT) | Cô lập SH |
| A9 | Prune + SH2 + FT + VQ | LightGaussian full |
| A10 | PUP Fisher score | Baseline thứ 2 (⚠️ `torch.zeros(N,6,6)` ≈ 470 MB, dùng `--fisher_resolution 4`) |
| A11 | Ảnh GPU vs CPU | H6 / RQ4 |
| **A12** | **Prune TRƯỚC `training_setup`** | **H8 — cứu drjohnson** |
| A13 | REFINE (zero-shot, không rasterizer) | Baseline rẻ, cơ chế khác |

Mọi hàng đo ở **cùng cặp (RAM, VRAM) budget**, ghi cả hai.

---

## 12. Definition of Done

### MVP (dừng ở đây vẫn là project tốt)
- [ ] P0–P2: pipeline chạy trên `train` + `truck` dưới 4 GB/10 GB, **8 bẫy T1–T9 được kiểm và ghi lại**
- [ ] Profiler 10 stage, bảng RAM-vs-VRAM thật (C2)
- [ ] Noise floor 3 seed + A1/A3/A4/A9 trên 2 scene
- [ ] README số thật, không placeholder

MVP này **đã đủ mạnh cho CV** vì C1 (audit) không phụ thuộc kết quả experiment.

### Full
- [ ] MVP + C3 phân rã artifact/cấu trúc + C4 Pareto iso-memory + C5 Deep Blending
- [ ] `drjohnson` chạy được nhờ A12
- [ ] 4 scene × 3 seed headline · quantization sensitivity · H7 (rasterizer pin vs main)
- [ ] Mini-paper + negative results log

---

## 13. Top rủi ro

| # | Rủi ro | Xác suất | Xử lý |
|---|---|---|---|
| 1 | Build 4 CUDA extension thất bại | TB (patch đã biết) | Fallback `gaussian-splatting-lightning` (gsplat, không cần rasterizer riêng) |
| 2 | `drjohnson` OOM cả sau A12 | TB | Hạ `-r 2`, ghi vào bảng OOM — **là kết quả** |
| 3 | **H3 sụp** (trần RAM biến mất sau khi sửa hết artifact) | **TB-Cao** | **Báo cáo negative.** C1 + C2 + C5 vẫn đứng |
| 4 | PSNR không khớp cụm re-run | TB | Kiểm split/res/rasterizer version. **Không chỉnh cho khớp** |
| 5 | Thermal throttle | Cao | Log clock; loại throttled run khỏi timing, giữ quality |
| 6 | Scope creep | Rất cao | Mọi ý mới → Future Work. Không scene thứ 5, không method thứ 3 |
| 7 | Chưa ai chạy bất kỳ method nào trên card 4 GB | — | **Validate trên `train` ngay tuần đầu** trước khi cam kết |

---

## 14. Repo

```
low-3DGS/
├── docs/  PLAN.md(này) · env.md · 02_METHODOLOGY · 03_RISKS · 04_LITERATURE · 05_EXPERIMENTS
│          · 07_RESEARCH_FINDINGS · paper_notes/lightgaussian.md
│          · 00_CRITIQUE, 01_PROJECT_PLAN, 06_WHAT_FITS  (lịch sử, đã superseded)
├── src/memory/     profiler.py · killswitch.py            <- phần thật sự của mình
├── src/{pruning,distillation,quantization,evaluation}/    <- evaluation = COPY của INRIA
├── third_party/    gaussian-splatting · LightGaussian · PUP-3DGS  (submodule)
├── tools/          profile_gpu.py · measure_ply_load.py · ply_to_checkpoint.py · make_plots.py
├── scripts/        setup_wsl_sudo.sh · p1_smoke.sh · run_queue.py
├── configs/ · tests/ · experiments/<run_id>/ · results/ · datasets/
```

---

## 15. Quy tắc

1. Không bịa số. 2. **Không claim novelty** — claim reproduction + measurement + audit.
3. Trivial baseline chạy **trước**. 4. Mọi bảng ghi res · iters · seeds · **cả RAM lẫn VRAM**.
5. Negative result ghi trong 24h. 6. `mean ± std`, không hứa bit-exact.
7. **Đo đúng hàm trong repo, không đo bản mô phỏng đơn giản hoá** (bài học EXP-003).
8. Mỗi lần OOM/kill-switch là data. 9. Đóng Chrome/Zalo; `nvidia-smi` = 0 MiB; `free -g` ≥ 9 GB — ghi vào `env.json`.
10. So với **cụm re-run độc lập**, không với một số đơn lẻ. 11. **Không cite số DB của HAC/HAC++/SymGS** (outlier lan truyền).
12. JSONL ghi ngay. 13. Loại throttled run khỏi timing, giữ quality.
14. Mọi mục "⚠️ chưa verify" trong `07_RESEARCH_FINDINGS.md` **không được cite** cho tới khi tự kiểm.
