# Low-3DGS

**Nén 3D Gaussian Splatting dưới ràng buộc RAM và VRAM đồng thời — trên GTX 1650 Ti 4 GB.**

> *Compressing 3D Gaussian Splatting under joint RAM/VRAM constraints on a 4 GB laptop GPU.
> We show that the standard way of reporting compression results credits extra fine-tuning
> to compression, and give a rule that predicts exactly when this illusion appears.*

📊 **[Xem hình kết quả](https://claude.ai/code/artifact/c3f8262d-fc69-45e1-89bf-8fe70732eff0)** ·
📋 [Nhật ký thí nghiệm](docs/05_EXPERIMENTS.md) · 🔖 [Quy ước phiên bản](docs/VERSIONING.md)

Phiên bản hiện tại: **v0.8.0** — 23 mục thí nghiệm đã ghi (ID cấp tới EXP-026), 2 scene, 36 lần chạy huấn luyện.

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

### 2. Vì sao cách so ngây thơ lại đánh lừa — và khi nào nó đánh lừa mạnh nhất

Con số ngây thơ **phân tách chính xác** thành hai phần. Đây là phép cộng trừ, luôn đúng:

```
so-với-checkpoint-gốc  =  chi phí nén  +  lợi ích luyện thêm
```

Phần **thực nghiệm** — thứ có thể khác đi và đã được đo — là **độ lớn** của số hạng thứ hai:

| scene `train`, cắt 50% | |
|---|---|
| chi phí nén | +0,0080 dB |
| lợi ích luyện thêm | **+0,3587 ± 0,0551 dB** ← lớn gấp ~45 lần |
| ⇒ tổng (số ngây thơ báo cáo) | **+0,3667 dB** — đổi dấu, thành "nén làm tốt hơn" |

Số hạng "luyện thêm" **có thể đã gần 0** — khi đó cách so ngây thơ vô hại. Nó không gần 0.
Đó mới là phát hiện.

Và nó **co lại 7,6 lần** trên model đã hội tụ sẵn (+0,3587 → +0,0475), nhất quán ở cả ba
thước đo. Nghĩa là méo mó nặng nhất đúng ở vùng vận hành mà phần lớn bài báo báo cáo:
**checkpoint chưa hội tụ, tỉ lệ nén thấp.**

> ⚠️ **Đính chính (v0.5.1).** Bản trước trình bày quan hệ này như một "quy tắc dự đoán
> bác bỏ được". **Sai** — nó là đồng nhất thức, đúng theo đại số ở mọi bộ dữ liệu, nên
> không thí nghiệm nào bác bỏ được nó. Nội dung thực nghiệm nằm ở **độ lớn** của các số
> hạng, không ở quan hệ giữa chúng. Xem [CHANGELOG.md](CHANGELOG.md).

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

### 4. Điểm dừng là một attractor — và nó là thật, không phải artifact

Gộp mọi lần chạy trên `train`: điểm xuất phát trải **1,56 dB** (cắt 20–70%) nhưng điểm dừng
chỉ trải **0,17 dB** — co lại **9,3 lần**, tụ về **22,1762 ± 0,0305**.

Điều đó đọc lại cả hai con số ở trên: C1 chính là *khoảng cách từ checkpoint tới điểm dừng*
(22,1762 − 21,8157 = 0,3605 ≈ 0,3587), và "nén 50% miễn phí" là **hai nhánh cùng rơi về một
điểm dừng** — chênh +0,0080 nằm trong nhiễu 0,0305 của chính attractor.
Đầu gối ở 70–80% không phải ngưỡng mất thông tin mà là **biên của miền hút**.

**Nghi vấn đã được kiểm và bác bỏ.** Chúng tôi nghi attractor do `ExponentialLR` mà PUP thêm
vào tạo ra — nếu đúng thì "vùng miễn phí" là tính chất của công thức chứ không phải của nén.
Tắt anneal và chạy lại ba mức cắt: attractor **không những còn, mà còn chặt hơn** (co 16,1×
thay vì 9,3×), và trải điểm dừng nhỏ hơn cả nhiễu giữa các seed.

Anneal chỉ **dịch chỗ** attractor (+0,153 dB) và làm nó ổn định hơn (nhiễu giảm 1,93×).
Cấu trúc ảo giác giữ nguyên khi bỏ anneal — ở mức cắt 50%, C1 vẫn chiếm **96%** của con số
ngây thơ. **Hiện tượng này không phải artifact của lịch learning rate.**

**Và attractor tái lập trên scene thứ hai** (co 15,7× so với 9,3–16,1× ở `train`) ⇒ nó là tính
chất của **công thức tinh chỉnh**, không phải của một scene. Nhưng **bề rộng miền hút thì phụ
thuộc model**: `train` còn trong miền tới ~70% cắt, còn `truck-864k` (vốn đã bị nén 66%) đã ra
khỏi miền ở ~20–30%. ⇒ **Không có con số "nén tới X% là miễn phí" dùng chung được** — nó là
miền hút của công thức trên model đang xét, rộng hẹp theo lượng dư thừa còn lại.

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
docs/       05_EXPERIMENTS.md   nhật ký thí nghiệm — nguồn sự thật
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
- **Chênh lệch 7,6× của C1 lẫn ba biến** (scene, số Gaussian, trạng thái hội tụ) — chưa tách được.
  Phép tách từng đề xuất (chạy lại trên `train` đã tinh chỉnh) **đã bị bác bỏ trước khi chạy**:
  model đó nằm sẵn trên attractor nên kết quả biết trước, và n=3 không đủ lực. Xem EXP-023.
- Attractor mới kiểm trên **một scene** (`train`), và EXP-025 là **n=1 mỗi mức cắt** — hướng chắc
  (trải điểm dừng nhỏ hơn cả nhiễu seed) nhưng con số 16,1× không tin được về độ lớn.
- **Biên miền hút** chưa xác định chính xác: nằm đâu đó giữa cắt 70% và 80%.
- Đường cong đánh đổi 9 điểm chỉ có **seed 0** (trừ ba mức 50/60/66% đã có 3 seed).
- **"Có ý nghĩa thống kê" ≠ "mắt người thấy được".** Chưa có khảo sát người xem.
- Thí nghiệm EXP-001..022 chạy **trước khi bật git** ⇒ được đánh dấu `pre_versioning: true`
  thay vì gán cho chúng một commit không có thật.

## Dự án này đã tự rút lại hai phát biểu

Ghi lại vì đó là một phần của kết quả, không phải điều cần giấu:

1. *"Nén còn làm model tốt hơn +0,112 dB"* → thật ra nén **tốn** 0,088 dB (v0.3.0)
2. *"Nén miễn phí tới 2,00×"* → chỉ đúng theo PSNR; SSIM và LPIPS cho thấy đã tốn từ 50% (v0.4.0)

Xem [CHANGELOG.md](CHANGELOG.md).
