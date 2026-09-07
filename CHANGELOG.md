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

## [0.9.0] — 2026-09-08

### Thêm — biên miền hút nay là số đo được
Định nghĩa vận hành: mức cắt làm điểm dừng lệch **2 lần nhiễu giữa các seed**.
`train` = **67,19%** · `truck-864k` = **20,81%**. Thay cho hai khoảng mơ hồ trước đó.

### Bác bỏ — EXP-027
Giả thuyết "biên miền hút nằm ở tỉ lệ **khối lượng quan trọng** bị cắt cố định" — nếu đúng thì
dự đoán được mức nén miễn phí **không cần chạy fine-tune**. **Sai:** trên trục đó hai model lệch
**3,76×**, còn xa hơn cả trục số hạt (3,23×).
⇒ Mức nén miễn phí không suy ra được từ model tĩnh.
Chi phí bác bỏ: **~2 phút GPU, không lần huấn luyện nào.**

### Phép đo phụ vẫn dùng được
Khối lượng quan trọng ở nhóm hạt yếu nhất 50%: `train` 2,13% vs `truck-864k` 8,84% (gấp 4,2×).
Phân bố ít lệch hơn ⇒ còn ít dư thừa. Đúng chiều với miền hút hẹp hơn, nhưng chỉ là **chỉ báo**,
không phải công cụ dự đoán.

### Thêm công cụ
`tools/importance_profile.py` · `tools/basin_vs_importance.py` · `scripts/run_importance_profile.sh`

---

## [0.8.0] — 2026-09-08

### Phát hiện
- **EXP-026 — attractor tái lập trên scene thứ hai.** Co **15,7×** (`truck-864k`, cắt 10–30%),
  so với 9,3× (train, anneal bật) và 16,1× (train, anneal tắt). Phán quyết ghi trước (≥5×) thoả.
  ⇒ Attractor là tính chất của **công thức tinh chỉnh**, không phải của một scene.
- **Miền hút HẸP LẠI khi model đã bị nén.** `train` còn trong miền tới ~70% cắt;
  `truck-864k` (đã nén 66%) đã ra khỏi miền ở ~30% (lệch 5,7 lần nhiễu).
  ⇒ **Không có con số "nén tới X% là miễn phí" dùng chung được.** Bề rộng miền hút phụ thuộc
  lượng dư thừa còn lại của model. Định lượng được trực giác "nén cái đã nén thì đắt hơn".

### Sửa — cờ `dirty` giờ mới dùng được
`tools/stamp_run.sh` báo `dirty: true` cho **mọi** lần chạy vì hai lỗi:
(1) tính cả `experiments/` vào phép kiểm, mà chính lần chạy ghi kết quả vào đó;
(2) tin stat cache của git — trên `/mnt/d` (9p) git báo SẠCH ngay sau khi file vừa bị sửa
(quan sát trực tiếp: WSL git 0 mục / Windows git 1 mục). False negative đúng ở chỗ nguy hiểm nhất.
Nay `update-index --really-refresh` trước, và loại trừ `experiments/`.
**EXP-026 là ba lần chạy đầu tiên có dấu xuất xứ sạch.**

---

## [0.7.0] — 2026-09-08

### Phát hiện
- **EXP-023 — điểm dừng là một attractor.** Phân tích lại dữ liệu đã có, 0 phút GPU:
  điểm xuất phát trải 1.5588 dB → điểm dừng trải 0.1668 dB, **co 9.3 lần**,
  tụ về **22.1762 ± 0.0305** (n=9). Đọc lại ba con số:
  C1 = khoảng cách từ checkpoint tới attractor; "nén 50% miễn phí" = hai nhánh cùng rơi
  về một điểm dừng (+0.0080 nằm trong nhiễu 0.0305); **đầu gối = biên miền hút**.
- **EXP-025 — attractor KHÔNG do anneal tạo ra.** Tắt `ExponentialLR` mà PUP thêm vào:
  co lại **16.1×** (chặt hơn cả khi bật), trải điểm dừng 0.0965 < nhiễu seed 0.1065.
  Anneal chỉ **dịch chỗ** attractor (+0.1532 dB) và giảm nhiễu 1.93×.
  ⇒ **Cấu trúc ảo giác không phải artifact của lịch learning rate**: tắt anneal, ở mức cắt
  50% C1 vẫn chiếm **96%** con số ngây thơ. Phát hiện chính mạnh lên.

### Rút lại
- **Diễn giải của EXP-024** ("anneal tạo ra attractor"). Sai vì **nhầm hai loại phân tán**:
  EXP-024 đo phân tán giữa các *seed* ở một điểm xuất phát; attractor nói về phân tán giữa các
  *điểm xuất phát*. Hai thứ đi ngược chiều nhau — bỏ anneal làm tăng cái thứ nhất nhưng **giảm**
  cái thứ hai. EXP-025 (phán quyết ghi trước) bác bỏ.
- **Thí nghiệm exp017** bị bác bỏ **trước khi chạy**: model đó nằm sẵn trên attractor nên kết quả
  biết trước (~0.003), và n=3 cho nửa KTC ±0.137 dB — rộng hơn cả dải dự đoán.

### Thêm
- Patch `L3DGS_NO_ANNEAL` (`docs/REPO_PATCHES.md` mục 2b), mặc định không đổi hành vi gốc.
- `scripts/run_noanneal.sh`, `scripts/run_noanneal_sweep.sh`.
- EXP-024/025 là những lần chạy đầu tiên gắn được **commit thật**.

### Đã biết còn khiếm khuyết
- `tools/stamp_run.sh` báo `dirty: true` vì **chính lần chạy ghi file vào `experiments/`**.
  Phép kiểm dirty phải loại trừ `experiments/`, nếu không không lần chạy nào sạch được.

---

## [0.5.1] — 2026-09-07

### Rút lại
- **"Quy tắc dự đoán ảo giác"** — phát biểu `ảo giác ⟺ lợi ích luyện thêm > chi phí nén`
  được trình bày ở v0.5.0 như "cơ chế bác bỏ được". **Sai.** Ba đại lượng định nghĩa trên
  cùng một baseline nên `ngây thơ ≡ trung thực + C1` **theo đại số**. Kiểm số: sai khác
  5.55e-17 (train) và 0 (truck-864k). Không thí nghiệm nào bác bỏ được một đồng nhất thức.
- Kèm theo: mô tả "chuyển từ quan sát sang cơ chế bác bỏ được" — ngược với sự thật.

### Phần KHÔNG bị ảnh hưởng
- Độ lớn C1 = **+0.3587 ± 0.0551 dB** trên `train` — đo được, có thể đã gần 0, và không.
  Đây mới là nội dung thực nghiệm: nó đủ lớn để **đổi dấu** con số ngây thơ.
- C1 co **7.6×** trên model đã hội tụ — đo được (nhưng xem giới hạn mới bên dưới).
- Toàn bộ C3 (thứ tự độ nhạy thước đo) và mọi kết quả về bộ nhớ.

### Giới hạn mới được nêu
- Chênh lệch 7.6× của C1 lẫn **ba biến**: scene, số Gaussian, trạng thái hội tụ — cả ba
  đổi cùng lúc giữa `train` và `truck-864k`. Phép tách rẻ: chạy lại cặp đối chứng trên
  `exp017_train_noprune@35000` (đã có sẵn trên đĩa, cùng scene, cùng N=1,026,508).

### Sửa
- `docs/NEXT.md` ghi scene 2 nhánh nén `n=2` trong khi dữ liệu có đủ 3 seed.
- `docs/REPO_PATCHES.md` còn ghi "chưa dùng git".
- `README.md` đếm 22 thí nghiệm, tài liệu thực có 19 mục đã viết.
- `scripts/run_scene2.sh:117` có `
` viết thành ký tự thường (không ảnh hưởng logic).
- 5 thư mục run thiếu `run_meta.json`.

### Thêm
- Skill `bao-cao` (`.claude/skills/bao-cao/`) — báo cáo trạng thái bằng cách **tính lại từ
  dữ liệu thô rồi đối chiếu với tài liệu**. Chính skill này phát hiện toàn bộ các mục trên,
  kể cả lỗi đồng nhất thức, ngay trong lần chạy thử đầu tiên.

---

## [0.5.0] — 2026-09-07

Phiên bản **đầu tiên có git**. Toàn bộ trạng thái dự án tại thời điểm này.

### Thêm
- **Scene thứ hai** (EXP-022): `truck` đã prune còn 864.017 hạt, 6 lần chạy.
  Cả ba phát biểu C1/C2/C3 đều tái lập.
- ~~"Quy tắc dự đoán ảo giác"~~ — **RÚT LẠI ở v0.5.1**, xem mục đó. Quan hệ này là
  đồng nhất thức đại số, không phải quy luật thực nghiệm.
  Phần đứng vững: **độ lớn** của số hạng "luyện thêm" (+0.3587 dB) và mức co lại của nó.
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
