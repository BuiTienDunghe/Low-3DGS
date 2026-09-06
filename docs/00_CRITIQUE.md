# Phản biện Plan v1 — 7 vấn đề cần sửa trước khi viết dòng code đầu tiên

Tài liệu này ghi lại các vấn đề phát hiện khi review Plan v1, kèm bằng chứng.
Đọc file này trước `01_PROJECT_PLAN.md`.

---

## V1. [ĐÃ SỬA — xem `06_WHAT_FITS.md`] Circular dependency: post-hoc compression và bài toán 4GB

> ⚠️ **ĐÍNH CHÍNH (2026-09-01):** Kết luận gốc của V1 quá mạnh. Nó bỏ sót việc
> **INRIA phát hành pre-trained models (14 GB)** — bạn không cần tự train baseline,
> tải về là cắt đứt được vòng lặp. Post-hoc compression + fine-tuning **fit thoải mái
> trên 4GB** (chỉ ~1.2-1.9 GB sau khi prune). Đọc `06_WHAT_FITS.md` để có kết luận đúng.
>
> Phần vẫn đúng: peak VRAM ở lúc train; rasterizer workspace bị đánh giá thấp;
> **train from scratch** MipNeRF360 trên 4GB là bất khả thi (README chính chủ ghi 24 GB).
> Phần sai: kết luận rằng contribution *bắt buộc* phải là train-time.

Plan v1 giả định pipeline:

```
Train baseline 3DGS  ->  Prune / Quantize  ->  Compressed model
```

Vấn đề: **LightGaussian, Compact-3DGS, HAC, VQ — tất cả đều yêu cầu một model 3DGS
đã train xong làm input.** Chúng nén *artifact*, không giảm *peak VRAM khi train*.

Nhưng peak VRAM của 3DGS xảy ra **trong lúc train**, tại đỉnh densification —
không phải lúc inference. Nên plan v1 tự mâu thuẫn:

- Để chạy được compression thì cần baseline đã train xong
- Để train baseline thì cần vượt qua đúng cái peak VRAM mà project đang muốn giải

Bằng chứng định lượng (gsplat paper, MipNeRF360):

| Impl        | 7k iters | 30k iters |
|-------------|----------|-----------|
| 3DGS gốc    | 7.7 GB   | 9.0 GB    |
| gsplat      | 4.3 GB   | 5.6 GB    |

**Ngay cả gsplat — implementation tiết kiệm bộ nhớ nhất — vẫn cần 5.6GB cho
MipNeRF360.** Card 4GB (usable ~3.3GB) không **train from scratch** được các scene
này ở setting chuẩn.

Hệ quả **đã đính chính**: contribution *có thể* nằm ở post-hoc (Track A, dùng
checkpoint tải về, chạy được trên scene lớn) **hoặc** train-time (Track B, scene nhỏ).
Làm cả hai. Xem `06_WHAT_FITS.md` §3.

---

## V2. [NGHIÊM TRỌNG] Contribution đề xuất đã bị scoop, nhiều lần

Plan v1 đề xuất: "memory-aware adaptive pruning theo target budget
(`--target-vram 3.0GB` / `--target-gaussians 500000`)".

Đã tồn tại:

| Paper | Venue | Trùng ở đâu |
|---|---|---|
| **Taming 3DGS: High-Quality Radiance Fields with Limited Resources** | SIGGRAPH Asia 2024 | Score-based densification tiến tới **exact user-defined budget**. Đúng ý tưởng "target-gaussians". |
| **Gaussians on a Diet: High-Quality Memory-Bounded 3DGS Training** | arXiv | Kiểm soát việc sinh primitive để giữ **peak training memory gần như hằng số** trên edge device. Đúng ý tưởng "target-vram". |
| **Reducing the Memory Footprint of 3DGS** (`reduced-3dgs`) | I3D 2024, INRIA | Chính nhóm tác giả 3DGS gốc làm memory reduction. Có code. |
| **MEGS²: Memory-Efficient GS via Spherical Gaussians and Unified Pruning** | arXiv 2509.07021 | Unified pruning cho memory. |
| **Mini-Splatting** | ECCV 2024 | Constrained number of Gaussians. |
| **3DGS.zip survey** | CGF 2025 | Khảo sát toàn bộ mảng compression, hàng chục method. |
| **Splatwizard** | arXiv | Đã có sẵn benchmark toolkit cho 3DGS compression. |

Nếu vẫn viết CV là "developed a memory-aware pruning strategy" mà không cite
những paper này, người phỏng vấn có domain knowledge sẽ phát hiện ngay. Đó là
rủi ro lớn hơn nhiều so với việc không có novelty.

**Reposition bắt buộc.** Xem `01_PROJECT_PLAN.md` §3.

---

## V3. Memory model trong plan v1 sai/thiếu

Plan v1 viết "Estimate Gaussian Memory" như một bước, nhưng không định nghĩa.
Sai lầm phổ biến là tính `N × bytes_per_gaussian`. Thực tế khi TRAIN:

Với SH degree 3, fp32, mỗi Gaussian:

| Thành phần | floats | bytes |
|---|---|---|
| means (xyz) | 3 | 12 |
| scales | 3 | 12 |
| quats | 4 | 16 |
| opacity | 1 | 4 |
| SH (deg 3: 16 coeff × 3 ch) | 48 | 192 |
| **params** | **59** | **236** |
| gradients | 59 | 236 |
| Adam `exp_avg` | 59 | 236 |
| Adam `exp_avg_sq` | 59 | 236 |
| densify stats (grad_accum, denom, max_radii2D) | ~3 | ~12 |
| **TỔNG / Gaussian** | | **~956 B, tức ~1 KB** |

Cộng thêm, KHÔNG scale theo N:

- **Rasterizer workspace** (geomBuffer / binningBuffer / imgBuffer) — scale theo
  số cặp *(tile, gaussian)* đã sort, không theo N. Splat to phủ nhiều tile thì
  tốn nhiều. **Đây thường mới là thủ phạm OOM thật sự**, và nó bùng nổ đột ngột.
- Autograd activations của backward pass
- CUDA context + workspace: ~300-500 MB, mất trắng
- Fragmentation của PyTorch caching allocator

**Ngân sách thực tế trên 4GB (usable ~3.3GB sau CUDA context):**

```
3.3 GB - 0.6 GB (rasterizer + activations, ước lượng dè dặt) = 2.7 GB
2.7 GB / 1 KB per Gaussian  =  ~2.7M Gaussians   (giới hạn lý thuyết)
Thực tế an toàn:               ~1.0 - 1.5M Gaussians
```

MipNeRF360 outdoor scene hội tụ ở **2-6M Gaussians**. Xác nhận V1.

**Đòn bẩy lớn nhất bị plan v1 bỏ sót:** giảm SH degree 3 -> 0 làm 59 -> 14 floats,
tức **giảm 4.2x bộ nhớ train mỗi Gaussian**. Trên 4GB, "SH distillation" không
phải trick nén — nó là **điều kiện để train được**. Phải đưa lên sớm, không để ở Phase 5.

---

## V4. Ablation thiếu trivial baselines, nguy cơ straw man

Plan v1 so sánh "adaptive pruning" với "fixed-ratio pruning". Đó là straw man.
Interviewer sẽ hỏi ngay 4 câu mà plan v1 không trả lời được:

| Baseline bắt buộc bổ sung | Tại sao |
|---|---|
| **Hard cap N** — chỉ đơn giản dừng densify khi đạt N Gaussian | Trivial nhất. Không thắng được cái này thì method vô nghĩa. |
| **Random pruning** cùng tỉ lệ | Mạnh một cách đáng ngạc nhiên. Thước đo xem importance score có thật sự làm gì không. |
| **Opacity-threshold pruning** | Đã có sẵn trong vanilla 3DGS. Là "free baseline". |
| **Tăng `densify_grad_threshold`** | Cách chỉnh tay mà cộng đồng vẫn dùng để fit VRAM. Zero-code baseline. |

Một method chỉ có giá trị khi thắng được cả 4 cái trên. Nếu không thắng, đó là
**negative result và vẫn phải báo cáo** (xem V7 và `03_RISKS.md`).

---

## V5. Không có seed/variance protocol, mọi con số đều vô nghĩa

Plan v1 không nhắc tới seed, số lần chạy lặp, hay variance. Lỗi chết người vì hai lý do:

1. **CUDA rasterizer backward dùng `atomicAdd`, nên non-deterministic.**
   Cùng seed, cùng config, chạy 2 lần vẫn ra kết quả khác nhau. Không thể có
   bit-exact reproducibility. Phải chấp nhận và report `mean ± std`.
2. Với budget nhỏ, noise giữa các run thường **~0.1-0.3 dB PSNR** — cùng bậc độ lớn
   với improvement mà project này có thể đạt được.

Nếu report "ours 27.4 dB vs baseline 27.2 dB" từ 1 run mỗi bên, con số đó
**không chứng minh được gì**. Đây chính xác là kiểu lỗi mà interviewer kỹ tính sẽ đâm thủng.

Protocol bắt buộc: xem `02_METHODOLOGY.md`.

---

## V6. Không có compute budget, plan 13 phase không khả thi về thời gian

Plan v1 có 13 phase, 7 experiment, "3-5 scenes", nhưng **không có một ước lượng
giờ GPU nào**. Tính thử:

```
7 methods x 4 scenes x 3 seeds x ~2.0 h/run  =  168 GPU-hours
```

GTX 1650 Ti chậm hơn RTX 3090 khoảng 4-6x cho workload này, và là **card laptop**,
nên thermal throttling khi chạy liên tục nhiều giờ. 168h tương đương 3-4 tuần chạy
gần như 24/7 trên máy chính (máy sẽ không dùng được cho việc khác).

Bắt buộc phải có compute budget table và cắt giảm có chủ đích. Xem `01_PROJECT_PLAN.md` §7.

---

## V7. Thiếu decision gates, risk register, và kill criteria

Plan v1 tuyến tính: Phase 0 -> 13, không có nhánh "nếu thất bại thì sao".
Với hardware này, xác suất một phase thất bại là **cao**:

- Build `diff-gaussian-rasterization` trên **Windows** cần MSVC + CUDA Toolkit
  đúng version. Đây là failure point số 1 của người mới làm 3DGS trên Windows.
  Plan v1 hoàn toàn không nhắc tới, dù môi trường là Windows 11.
- COLMAP trên scene tự chụp có thể fail hoặc ra pose sai, tốn nhiều ngày.
- Method đề xuất có thể **không thắng** trivial baseline.

Cần risk register + kill criteria per phase. Xem `03_RISKS.md`.

---

## Tổng kết phản biện

| Khía cạnh | Đánh giá |
|---|---|
| Research methodology (baseline -> ablation -> benchmark -> report) | **Tốt.** Giữ nguyên. Điểm mạnh nhất của plan v1. |
| Kỷ luật khoa học (§25 "không được làm") | **Rất tốt.** Giữ nguyên, đã bổ sung thêm. |
| Cấu trúc repo | Tốt. Chỉ cần thêm `src/memory/`, `tests/`, `.gitignore` |
| Evaluation matrix | Tốt về metric, thiếu protocol (seed / split / resolution) |
| **Technical framing** | **Sai. Xem V1, V3** |
| **Novelty claim** | **Không đứng vững. Xem V2** |
| **Tính khả thi** | **Chưa được kiểm chứng. Xem V6, V7** |

---

## Nguồn

- Taming 3DGS — https://arxiv.org/abs/2406.15643 · https://humansensinglab.github.io/taming-3dgs/
- gsplat — https://arxiv.org/pdf/2409.06765
- Reducing the Memory Footprint of 3DGS — https://repo-sam.inria.fr/fungraph/reduced_3dgs/ · https://github.com/graphdeco-inria/reduced-3dgs
- 3DGS.zip survey (CGF 2025) — https://onlinelibrary.wiley.com/doi/10.1111/cgf.70078
- Compression in 3DGS: A Survey — https://arxiv.org/abs/2502.19457
- MEGS² — https://arxiv.org/pdf/2509.07021
- Splatwizard benchmark toolkit — https://arxiv.org/html/2512.24742v1
