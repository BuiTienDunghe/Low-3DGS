# Nhật ký thay đổi

Theo [Keep a Changelog](https://keepachangelog.com/vi/1.1.0/).
Quy ước phiên bản: [`docs/VERSIONING.md`](docs/VERSIONING.md) — đánh dấu **trạng thái phát biểu
khoa học**, không phải API.

## [Chưa phát hành]

### Dự định
- Điểm dữ liệu thứ ba để **thử bác bỏ** quy tắc dự đoán ảo giác (cần cấu hình ở vùng ranh giới)
- Đường cong đánh đổi cho scene thứ hai (mới có một tỉ lệ nén)
- Quyết dứt hướng giảm bậc màu: cắt xuống bậc 2 + phục hồi
- Sửa `save_ply` phình 13.35× (`gaussian_model.py:206`)

---

## [0.5.0] — 2026-09-07

Phiên bản **đầu tiên có git**. Toàn bộ trạng thái dự án tại thời điểm này.

### Thêm
- **Scene thứ hai** (EXP-022): `truck` đã prune còn 864.017 hạt, 6 lần chạy.
  Cả ba phát biểu C1/C2/C3 đều tái lập.
- **Quy tắc dự đoán ảo giác** — kết quả sắc nhất của dự án tới nay:
  `ảo giác xuất hiện ⟺ lợi ích luyện thêm > chi phí nén`.
  Đúng ở cả hai scene. Chuyển từ *quan sát* sang *cơ chế bác bỏ được*.
- Đo VRAM **từ trong tiến trình** (`max_memory_allocated`), cài qua `sitecustomize.py`
  ngoài repo ⇒ 0 dòng sửa mã bên thứ ba. NVML báo cao hơn ~220 MiB vì đo *reserved*.
- `tools/cross_scene.py`, `tools/knee_errorbars.py`, `tools/stamp_run.sh`, `tools/bump_version.sh`
- Hệ thống phiên bản: `VERSION`, `CHANGELOG.md`, `docs/VERSIONING.md`

### Phát hiện
- Hiệu ứng "luyện thêm" **teo 7,6 lần** trên model đã hội tụ (+0,3587 → +0,0475 dB),
  nhất quán ở cả ba thước đo ⇒ **confound không phải hằng số**, mà là hàm của khoảng cách tới hội tụ.
- `dL_dsh` = `torch::zeros({P,16,3})` tốn 465 MiB ở 2,54 triệu hạt và **độc lập độ phân giải**
  — khoản mà mô hình bộ nhớ cũ thiếu hẳn; nó khép lại hướng "giảm `-r` cho vừa bộ nhớ".

### Sửa
- Cổng kiểm `B4` loại oan cả 3 nhánh nén (nhánh nén eval **3 lần**, không phải 2).
  Dữ liệu gom lại được từ `metric.csv`, không phải chạy lại.

---

## [0.4.0] — 2026-09-06 *(hồi cố, không có commit)*

### Phát hiện
- **Đường cong đánh đổi** 9 điểm (EXP-019) và thanh sai số ở vùng đầu gối (EXP-021).
- **Ba thước đo không đồng ý đầu gối ở đâu**: PSNR nói miễn phí tới 60%;
  SSIM và LPIPS nói đã tốn từ 50%, và nói chắc chắn (LPIPS = 38σ).

### Rút lại
- **"Nén miễn phí tới 2,00×"** — chỉ dựa vào PSNR, và nhầm *"chưa chứng minh được khác 0"*
  thành *"bằng 0"*. Phát biểu đúng: **không mức nén nào thực sự miễn phí; PSNR chỉ không
  nhìn thấy hoá đơn.**

### Sửa
- Trung bình đối chứng `22.2277` → `22.1744` (con số cũ là giá trị của riêng seed 2).
- Ảnh 16 B/px → **12 B/px** (PUP không giữ `alpha_mask` thường trú; 16 B/px là của nhánh INRIA mới).

---

## [0.3.0] — 2026-09-05 *(hồi cố, không có commit)*

### Phát hiện
- **Thí nghiệm đối chứng** (EXP-016/017): riêng "luyện thêm" cho **+0,3587 ± 0,0551 dB**.
- **PSNR là thước đo ồn nhất**: tỉ lệ tín hiệu/nhiễu kém SSIM/LPIPS 30–40 lần (EXP-018).
- **Không chạy được đối chứng trên `truck`** (EXP-015): model chưa nén cần ~4,7 GB trên card 4 GB.

### Rút lại
- **"+0,1123 dB — nén còn làm model tốt hơn"**: cách so cũ tính công của "luyện thêm"
  thành công của "nén". Trừ đúng phần đó ra thì nén **tốn** −0,0881 dB.

---

## [0.2.0] — 2026-09-04 *(hồi cố, không có commit)*

### Thêm
- **Kết quả nén đầu tiên** (EXP-014): `truck` cắt 66% + phục hồi 5000 bước, 2,94× nhỏ hơn.

---

## [0.1.0] — 2026-09-02 *(hồi cố, không có commit)*

### Thêm
- Môi trường WSL2 + 4 phần mở rộng CUDA biên dịch được trên sm_75
  (cần vá 4 dòng `#include` — xem `docs/REPO_PATCHES.md`).
- Hạ tầng đo: `src/memory/{counters,profiler,killswitch}.py`.

### Phát hiện
- **Trần VRAM thật 3440 MiB** trên card 4096 MiB.
- **WSL2 tràn VRAM im lặng sang host RAM** thay vì báo lỗi; `expandable_segments:True` dập được.
- **Lấy mẫu bỏ sót 99,9%** một transient 400 MB ⇒ phải dùng bộ đếm chính xác của nhân.
- `save_ply` phình **13,35×** và bị hệ điều hành giết ở 3/4 scene.
