# Low-3DGS

**Nén 3D Gaussian Splatting dưới ràng buộc RAM và VRAM đồng thời — trên GTX 1650 Ti 4 GB.**

> *Compressing 3D Gaussian Splatting under joint RAM/VRAM constraints on a 4 GB laptop GPU.
> We show that the standard way of reporting compression results credits extra fine-tuning
> to compression, and give a rule that predicts exactly when this illusion appears.*

📊 **[Xem hình kết quả](https://claude.ai/code/artifact/c3f8262d-fc69-45e1-89bf-8fe70732eff0)** ·
📋 [Nhật ký thí nghiệm](docs/05_EXPERIMENTS.md) · 🔖 [Quy ước phiên bản](docs/VERSIONING.md)

Phiên bản hiện tại: **v0.5.0** — 22 thí nghiệm, 2 scene, 27 lần chạy huấn luyện.

---

## Phát hiện chính

### 1. Cách báo cáo phổ biến tính công của "luyện thêm" thành công của "nén"

Gần như mọi bài nén 3DGS đều: cắt bớt Gaussian → tinh chỉnh lại → so với checkpoint gốc.
Nhưng model đã nén được **tinh chỉnh thêm** mà checkpoint gốc thì không.

Chúng tôi chạy nhánh đối chứng — **không nén gì, tinh chỉnh đúng bằng số bước đó**:

| scene `train`, cắt 50% | |
|---|---|
| Cách so ngây thơ (với checkpoint gốc) | **+0,3667 dB** → "nén còn làm model tốt hơn!" |
| Riêng "luyện thêm" đã cho | **+0,3587 ± 0,0551 dB** |
| **Chi phí nén thật** | **+0,0080 dB** |

Gần như toàn bộ "cải thiện" là của việc luyện thêm.

### 2. Ảo giác này **dự đoán được**

Chạy lại trên scene thứ hai — model đã hội tụ sẵn — thì ảo giác **biến mất**:

```
ảo giác xuất hiện  ⟺  lợi ích luyện thêm  >  chi phí nén

train       : +0,3587  >  0,0080   → CÓ ảo giác
truck-864k  : +0,0475  <  0,2946   → KHÔNG có
```

Hiệu ứng "luyện thêm" **teo 7,6 lần** trên model đã hội tụ, nhất quán ở cả ba thước đo.
⇒ **Confound không phải hằng số — nó là hàm của khoảng cách tới hội tụ.**
Nó sống đúng ở vùng vận hành mà phần lớn bài báo nén 3DGS đang báo cáo.

### 3. PSNR không đủ nhạy để trả lời câu hỏi trung tâm

Tỉ lệ tín hiệu/nhiễu của chi phí nén, đo qua 3 seed:

| | `train` | `truck-864k` |
|---|---|---|
| PSNR | **0,2×** | 58,0× |
| SSIM | 7,9× | 153,0× |
| LPIPS | **64,9×** | **378,5×** |

Ở `train`, PSNR **hoàn toàn không phân biệt được** chi phí nén với nhiễu, còn LPIPS thì thấy rõ
ở mức 64,9 lần nhiễu. Thứ tự LPIPS > SSIM > PSNR tái lập ở **cả hai** scene.
Mà PSNR mới là con số hầu hết bài báo đặt lên tiêu đề.

---

## Vì sao lại làm trên máy yếu

Không phải vì thiếu máy mạnh, mà vì **ràng buộc bộ nhớ làm lộ ra những thứ máy mạnh che mất**:

| Phát hiện | Chỉ thấy được khi thiếu bộ nhớ |
|---|---|
| `save_ply` phình **13,35×** kích thước file trong RAM (`gaussian_model.py:206`) | bị hệ điều hành giết ở 3/4 scene |
| WSL2 **tràn VRAM im lặng** sang host RAM thay vì báo lỗi | mọi số đo thành vô nghĩa mà không ai biết |
| Lấy mẫu bộ nhớ **bỏ sót 99,9%** một transient 400 MB | đỉnh thật cao hơn số báo cáo 36% |
| `dL_dsh` tốn 465 MiB và **độc lập độ phân giải** | khép lại hướng "giảm `-r` cho vừa" |
| Model `truck` chưa nén **không fine-tune nổi** trên card 4 GB | nén là *vé vào cửa*, không phải tối ưu |

Ranh giới đo được trên máy này: **949 MiB VRAM cho mỗi triệu Gaussian**, trần thực tế
**3440 MiB** trên card 4096 MiB ⇒ tối đa ~3,07 triệu Gaussian.

---

## Cấu trúc

```
docs/       05_EXPERIMENTS.md   nhật ký 22 thí nghiệm — nguồn sự thật
            PLAN.md             kế hoạch dự án
            REPO_PATCHES.md     mọi sửa đổi lên mã bên thứ ba
            VERSIONING.md       quy ước phiên bản
            NEXT.md             tiếp tục từ đâu
scripts/    các loạt chạy thí nghiệm, đều có cổng tiền kiểm
src/memory/ bộ đếm bộ nhớ chính xác, profiler, killswitch
tools/      công cụ đo và phân tích
experiments/ số đo (*.csv, *.log, metric.csv) — model .ply KHÔNG lên git
```

Dữ liệu thô (19 GB) và model (861 MB) không nằm trong repo; toàn bộ **số đo** thì có, ~600 KB.

## Tái lập

Cần: WSL2 + CUDA 12.6, [PUP 3D-GS](https://github.com/j-alex-hanson/gaussian-splatting-pup)
@ `e971ea4` (kèm 4 dòng vá ở [`docs/REPO_PATCHES.md`](docs/REPO_PATCHES.md) — **mã công bố
không biên dịch nổi trên toolchain 2026 nếu thiếu chúng**), và checkpoint pretrained của INRIA.

```bash
bash scripts/run_train_pair.sh control   # nhánh đối chứng
bash scripts/run_train_pair.sh prune     # nhánh nén
python tools/cross_scene.py              # so sánh chéo hai scene
```

Mỗi lần chạy nên gắn phiên bản mã: `bash tools/stamp_run.sh experiments/<tên_run>`

---

## Giới hạn — đọc trước khi trích số

- **Hai scene**, và scene thứ hai là model *đã nén* (864.017 Gaussian), không phải checkpoint
  gốc — đó là model hợp lệ duy nhất chạy nổi cặp đối chứng trên card 4 GB.
- **Quy tắc dự đoán mới có hai điểm dữ liệu.** Đúng ở cả hai, nhưng hai điểm chưa phải quy luật;
  cần một điểm ở vùng ranh giới để thử bác bỏ.
- Đường cong đánh đổi 9 điểm chỉ có **seed 0** (trừ ba mức 50/60/66% đã có 3 seed).
- **"Có ý nghĩa thống kê" ≠ "mắt người thấy được".** Chưa có khảo sát người xem.
- Thí nghiệm EXP-001..022 chạy **trước khi bật git** ⇒ được đánh dấu `pre_versioning: true`
  thay vì gán cho chúng một commit không có thật.

## Dự án này đã tự rút lại hai phát biểu

Ghi lại vì đó là một phần của kết quả, không phải điều cần giấu:

1. *"Nén còn làm model tốt hơn +0,112 dB"* → thật ra nén **tốn** 0,088 dB (v0.3.0)
2. *"Nén miễn phí tới 2,00×"* → chỉ đúng theo PSNR; SSIM và LPIPS cho thấy đã tốn từ 50% (v0.4.0)

Xem [CHANGELOG.md](CHANGELOG.md).
