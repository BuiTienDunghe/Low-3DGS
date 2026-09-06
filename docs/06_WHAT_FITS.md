# Cái gì chạy được trên GTX 1650 Ti 4GB

> ⛔ **SUPERSEDED bởi [`PLAN.md`](PLAN.md) §2 (2026-09-01).** Bảng VRAM ở đây vẫn đúng về mặt
> số học, nhưng **thiếu hoàn toàn phân tích RAM** — thứ hoá ra bind trước VRAM khi fine-tune.
> Track B (train from scratch) đã bị bỏ khỏi phạm vi.

> Tài liệu này trả lời trực tiếp câu hỏi: **"project này có thật sự chạy được trên máy tôi không?"**
> Câu trả lời: **có, phần lớn.** Nhưng phải chia thành 2 track, vì không phải mọi thứ đều fit.

---

## 1. Sửa lại V1 trong `00_CRITIQUE.md`

V1 kết luận: *"post-hoc compression không giải bài toán 4GB vì bạn cần baseline đã train xong."*

**Kết luận đó quá mạnh.** Nó bỏ sót một lối thoát quan trọng:

> **INRIA phát hành pre-trained models (14 GB) cho chính các scene trong paper.**
> Bạn KHÔNG cần tự train baseline. Tải về là xong.

Và LightGaussian được thiết kế đúng cho workflow này — `prune_finetune.py` nhận input là
một thư mục chứa `point_cloud/iteration_30000/point_cloud.ply` của một model **đã train sẵn**.

Vòng lặp mà V1 chỉ ra bị **cắt đứt** bằng cách tải checkpoint thay vì tự train.

**Cái gì trong V1 vẫn đúng:**
- Peak VRAM nằm ở lúc train, không phải lúc render — vẫn đúng
- Rasterizer workspace là thủ phạm OOM bị đánh giá thấp — vẫn đúng
- **Train from scratch** MipNeRF360 trên 4GB là không khả thi — vẫn đúng
  (README chính chủ ghi **24 GB VRAM** để train đạt chất lượng paper)

**Cái gì thay đổi:** contribution KHÔNG bắt buộc phải là train-time. Post-hoc track
hoàn toàn fit trên 4GB, và đó là track an toàn nhất để bắt đầu.

---

## 2. Toán bộ nhớ: cái gì fit, cái gì không

**Ngân sách khả dụng:**
```
4.00 GB VRAM
-0.35 GB CUDA context + driver   (đo lại ở P0)
-0.50 GB rasterizer workspace + activations (ước lượng dè dặt, phụ thuộc resolution)
──────
≈ 3.15 GB cho Gaussian state    -> làm tròn an toàn: 2.6 - 3.1 GB
```

**Bytes per Gaussian:**

| SH degree | floats | params | **train** (params+grads+2×Adam) |
|---|---|---|---|
| 3 | 59 | 236 B | **956 B** |
| 2 | 38 | 152 B | **620 B** |
| 1 | 23 | 92 B | **380 B** |
| 0 | 14 | 56 B | **236 B** |

> LightGaussian giảm SH deg 3 → 2. Riêng bước đó đã cắt **35%** bộ nhớ train mỗi Gaussian.

---

### Bảng khả thi

| # | Việc | N | SH | VRAM (state) | Kết luận |
|---|---|---|---|---|---|
| 1 | **Load** pretrained `bicycle`, tính importance score | 6.0 M | 3 | 1.42 GB | ✅ Thoải mái |
| 2 | **Fine-tune** sau prune 66% | 2.0 M | 3 | 1.91 GB | ✅ Fit |
| 3 | **Fine-tune** sau prune + SH distill | 2.0 M | 2 | 1.24 GB | ✅ Thoải mái |
| 4 | **Fine-tune** sau prune mạnh | 1.0 M | 3 | 0.96 GB | ✅ Rất thoải mái |
| 5 | **Fine-tune** prune mạnh + SH distill | 1.0 M | 2 | 0.62 GB | ✅ Rất thoải mái |
| 6 | **Train from scratch** `lego` 800×800 | ~0.4 M | 3 | 0.38 GB | ✅ Thoải mái |
| 7 | **Train from scratch** `truck` @1/2 | ~2.5 M | 3 | 2.39 GB | ⚠️ Sát ngưỡng |
| 8 | **Train from scratch** `bicycle` full res | 6 M+ | 3 | 5.7 GB+ | ❌ OOM |

**Điểm mấu chốt: fine-tuning rẻ hơn training 5-10 lần**, vì sau khi prune thì `N` giảm mạnh,
và toàn bộ bộ nhớ scale theo `N`. Đây chính xác là lý do plan v1 của bạn **chạy được**.

---

## 3. Hai track — cả hai đều chạy trên máy này

### Track A — Compress + Fine-tune từ checkpoint `[AN TOÀN, làm trước]`

```
Tải pretrained 3DGS (INRIA, 14GB)
      ↓
Load + tính importance score        [#1: 1.4 GB]
      ↓
Prune to budget                     [rẻ]
      ↓
SH distillation deg 3 -> 2          [giảm 35% memory]
      ↓
Fine-tune / recovery                [#3: 1.2 GB]
      ↓
Quantization + VQ + entropy coding  [chạy trên CPU]
      ↓
Evaluate
```

**Chạy được trên scene LỚN** (`bicycle`, `garden`, `truck`, `playroom`) vì bỏ qua bước train.
Đây là toàn bộ pipeline của plan v1 gốc — nó hoạt động, chỉ cần đổi
"tự train baseline" thành "tải baseline".

**Ưu điểm không ngờ:** baseline của bạn là **model reference chính chủ**, không phải bản
tự reproduce có thể bị under-train. Về mặt reproducibility, điều này **tốt hơn**.

### Track B — Train from scratch dưới VRAM budget `[nơi contribution memory-aware sống]`

```
Scene nhỏ (lego, chair, truck@1/2)
      ↓
Train from scratch với VRAM controller   [#6, #7]
      ↓
So với trivial baselines ở cùng budget
```

**Chỉ chạy được trên scene nhỏ/vừa.** Và điều đó **hoàn toàn ổn** — vì đây là nơi
bạn so sánh method, không phải nơi bạn cần điểm số đẹp.

### Vì sao cần cả hai

| | Track A | Track B |
|---|---|---|
| Scene lớn | ✅ | ❌ |
| Reproduce SOTA (C5) | ✅ | — |
| Post-hoc compression (prune/quant/VQ) | ✅ | — |
| Train-time memory control (C4) | ❌ | ✅ |
| Rủi ro | Thấp | Cao |
| Bắt đầu ở đâu | **Đây** | Sau khi A xong |

---

## 4. Flags tiết kiệm VRAM — dùng ngay từ đầu

Từ README chính chủ, và một số cái không hiển nhiên:

| Flag | Tác dụng | Lưu ý |
|---|---|---|
| `--data_device cpu` | Giữ ảnh training trên CPU RAM thay vì VRAM | Chậm hơn chút. Bạn có 16GB RAM, dư sức. **Bật mặc định.** |
| `--test_iterations -1` | Tắt evaluation giữa chừng | **Gotcha lớn:** eval giữa training gây memory spike và giết run ở giờ thứ 2. Eval riêng sau khi xong. |
| `-r 2` / `-r 4` / `-r 8` | Downscale resolution | Ghi vào MỌI bảng kết quả |
| `--optimizer_type sparse_adam` | Nhanh hơn ~2.7×, hiệu quả hơn | Có trong bản INRIA mới |
| `--densify_grad_threshold` ↑ | Sinh ít Gaussian hơn | Đây là **trivial baseline A4** |
| `--densify_until_iter` ↓ | Dừng densify sớm | Cũng là trivial baseline |
| `--sh_degree 2` (hoặc 1) | Cắt 35% / 60% memory | **Knob mạnh nhất.** Xem bảng §2 |
| `gsplat: packed=True` | Intermediates dạng sparse | Chỉ gsplat |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | Giảm fragmentation | Env var, không phải flag |

Ảnh input tự động rescale nếu width > 1.6K — **nên độ phân giải thực tế có thể khác
cái bạn nghĩ**. Log resolution thật, không log resolution của file gốc.

---

## 5. Thứ tự làm lại (thay cho §6 của plan)

| | Phase | Track | Fit? |
|---|---|---|---|
| P0 | Env + profiling | — | ✅ |
| P1 | Train `lego` from scratch | B | ✅ #6 |
| P1b | **Tải pretrained models, verify render + eval** | A | ✅ #1 |
| P2 | Memory cost model (C1) | cả hai | ✅ |
| P3 | Literature review | — | — |
| **P4** | **Prune + fine-tune trên pretrained (C5)** | **A** | ✅ #2-5 |
| P5 | SH distillation study (C3) | A | ✅ |
| P6 | Quantization sensitivity | A | ✅ (CPU) |
| P7 | Trivial baselines iso-memory (C2) | B | ✅ #6,#7 |
| P8 | VRAM controller (C4) | B | ⚠️ #7 |
| P9-13 | Ablation → benchmark → report | cả hai | ✅ |

**Khác biệt lớn nhất so với plan v2 trước:** P4 (Track A) được kéo lên sớm.
Nó ít rủi ro nhất, chạy được trên scene lớn, và cho kết quả nhìn thấy được sớm nhất.

---

## 6. Cái gì KHÔNG chạy được — và phải báo cáo như một kết quả

| Không làm được | Xử lý |
|---|---|
| Train `bicycle`/`garden` full-res from scratch | **Document nó.** README chính chủ ghi 24GB. Đo chính xác N và iteration lúc OOM → data cho C1. |
| Reproduce đúng số PSNR của paper ở full resolution | Chạy ở res thấp hơn, ghi rõ. **Không so trực tiếp với số trong paper.** |
| So FPS với paper (A6000/3090) | Chỉ so tương đối giữa các method của mình |

Đây không phải thất bại — đây là **kết quả định lượng về giới hạn phần cứng**,
và nó chính là chủ đề của project.

---

## Nguồn

- 3DGS README (pre-trained models 14GB, "24 GB VRAM", low-memory flags) — https://github.com/graphdeco-inria/gaussian-splatting/blob/main/README.md
- LightGaussian `prune_finetune.py` — https://github.com/VITA-Group/LightGaussian/blob/main/prune_finetune.py
- LightGaussian issue #18 (prune pre-trained checkpoint) — https://github.com/VITA-Group/LightGaussian/issues/18
- LightGaussian paper — https://arxiv.org/pdf/2311.17245
