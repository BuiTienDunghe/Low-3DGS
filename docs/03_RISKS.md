# Risk Register & Kill Criteria

Mỗi rủi ro có: xác suất, tác động, dấu hiệu sớm, fallback, và **deadline để quyết định bỏ**.

Nguyên tắc: **thà cắt scope sớm còn hơn chết ở tuần 12.**

---

## R1 — Build CUDA extension trên Windows thất bại

> ✅ **ĐÃ XẢY RA + ĐÃ CÓ LỐI THOÁT (audit 2026-09-01, xem `env.md`).**
> Máy **không có MSVC** (chỉ có VS Installer stub, không có `cl.exe`) → `nvcc` trên Windows
> native không dùng được. **Nhưng WSL2 + Ubuntu đã cài sẵn và GPU passthrough đã verify
> hoạt động.** Fallback #3 được chọn làm đường chính. Rủi ro giảm từ Cao xuống Thấp.

| | |
|---|---|
| Xác suất | ~~**Cao** (~50%)~~ → **Thấp** (đã có WSL2 hoạt động) |
| Tác động | Blocking toàn bộ project |
| Dấu hiệu sớm | `nvcc fatal`, `cl.exe not found`, C++ std version mismatch, `error C2039` |
| Deadline | **5 ngày.** Không quá. |

**Fallback theo thứ tự:**
1. Native Windows + MSVC Build Tools 2019 + CUDA 11.8 (tổ hợp ổn định nhất)
2. Thử `gsplat` prebuilt wheel (tránh compile hoàn toàn)
3. WSL2 + Ubuntu 22.04 — build dễ hơn nhiều. **Đo và ghi VRAM overhead của WSL2.**
4. Nếu cả 3 fail sau 5 ngày: dùng backend không cần compile, và ghi rõ trong report

**Ghi lại toàn bộ quá trình vào `docs/env.md`.** Nghe có vẻ vặt, nhưng "tôi debug được
CUDA toolchain trên Windows" là một tín hiệu engineering thật, và người phỏng vấn hiểu điều đó.

---

## R2 — Không train nổi scene nào ở resolution hợp lý

| | |
|---|---|
| Xác suất | Trung bình |
| Tác động | Cao |
| Dấu hiệu sớm | OOM ngay ở `lego` 800x800 |
| Deadline | 1 tuần sau khi env chạy |

**Fallback bậc thang** (áp dụng lần lượt, ghi lại cái nào cần thiết):
1. `gsplat` với `packed=True` + `sparse_grad` + SparseAdam
2. Giảm SH degree tối đa xuống 2, rồi 1 (**giảm 4.2x memory/Gaussian ở deg 0**)
3. Giảm resolution: 1/2 -> 1/4 -> 1/8
4. Hard cap `N` + tăng `densify_grad_threshold`
5. Giảm `densify_until_iter`
6. Gradient checkpointing nếu backend hỗ trợ

**Quan trọng:** mỗi bậc phải xuống là **một data point cho C1 và C2**, không phải thất bại.
Ghi vào `05_EXPERIMENTS.md`.

---

## R3 — Method đề xuất (C4) không thắng trivial baseline

| | |
|---|---|
| Xác suất | **Cao** (~40%) — Taming 3DGS đã tối ưu vấn đề này kỹ |
| Tác động | Trung bình (không blocking nếu đã theo plan v2) |
| Deadline | 3 tuần sau khi controller chạy được |

**Đây chính là lý do plan v2 đặt C1/C2/C3 làm contribution chính, không phải C4.**
Nếu C4 thua:

1. **Báo cáo nó như negative result** — có phân tích tại sao. Negative result được
   trình bày tốt là tín hiệu maturity mạnh hơn positive result được trình bày ẩu.
2. Reframe C4: không phải "chất lượng cao hơn" mà là **"robustness cao hơn"** —
   nếu controller cho **0 OOM** trong khi count-cap thỉnh thoảng OOM, đó vẫn là
   kết quả thật và hữu ích (H4).
3. Project vẫn hoàn chỉnh nhờ C1 + C2 + C3 + C5.

**Câu trả lời phỏng vấn nếu bị hỏi:**
> "Method của tôi không thắng hard-cap baseline về PSNR ở cùng budget. Nhưng khi
> phân tích, tôi thấy nguyên nhân là [X], và điều đó khớp với việc [Y]. Điều tôi
> thực sự thu được là cost model và bằng chứng rằng rasterizer workspace mới là
> ràng buộc thật, chứ không phải parameter memory."

Đây là câu trả lời tốt. Đừng sợ nó.

---

## R4 — COLMAP thất bại trên scene tự chụp

| | |
|---|---|
| Xác suất | Cao nếu tự chụp |
| Tác động | Thấp (đã đánh dấu optional) |
| Deadline | 3 ngày |

**Fallback:** bỏ hẳn self-captured scene. 4 scene từ dataset chuẩn đã đủ, và **tốt hơn**
vì so sánh được với literature. Self-capture chỉ làm nếu mọi thứ khác đã xong.

---

## R5 — Compute budget vượt quá thời gian thực tế

| | |
|---|---|
| Xác suất | **Cao** |
| Tác động | Cao |
| Dấu hiệu sớm | Sau P4, ngoại suy thời gian còn lại > thời gian có |
| Deadline | Đánh giá lại sau mỗi phase |

**Fallback theo thứ tự cắt (cắt từ trên xuống):**
1. Bỏ scene thứ 4 (`bicycle`) khỏi headline, chỉ dùng cho stress test
2. Giảm ablation từ 30k xuống 15k iters (ghi rõ trong caption)
3. Giảm budget sweep từ 5 mức xuống 3 mức (3.0 / 2.5 / 2.0)
4. Bỏ C5 (SOTA reproduction) — **cắt cuối cùng**, vì nó có giá trị CV cao

**Không bao giờ cắt:** số seed cho headline comparison. Cắt seed = mọi kết quả vô nghĩa (V5).

---

## R6 — Laptop thermal throttling làm nhiễu timing

| | |
|---|---|
| Xác suất | Cao khi chạy nhiều giờ |
| Tác động | Thấp-Trung bình (chỉ ảnh hưởng timing, không ảnh hưởng quality) |

**Xử lý:**
- Log clock + temp ở 2 Hz, đánh dấu run bị throttle
- Loại throttled run khỏi bảng FPS / training-time, **giữ** trong bảng quality
- Ghi rõ trong report: "timing measured on a thermally-constrained laptop GPU;
  relative comparisons only"
- Đo FPS trong burst ngắn sau khi GPU đã nguội, không đo giữa lúc train

---

## R7 — Ổ đĩa đầy

> ✅ **ĐÃ XẢY RA VÀ ĐÃ XỬ LÝ (2026-09-01, xem `env.md` §3).**
> C: từ 9.54 GB (5%) → **35.23 GB (18.5%)**, giải phóng **25.69 GB**:
> xoá cache trùng lặp của VS Code (14.04 GB) + chuyển Ollama models sang D: (11.36 GB).
> Cả hai đều không mất dữ liệu, đã verify byte-exact trước khi xoá nguồn.
>
> **Còn lại:** `wsl --manage Ubuntu --move D:\wsl\Ubuntu` (6.92 GB, vhdx vẫn trên C:).
> Dư địa nếu cần thêm: ~40 GB (hibernation 6.15 — đã quyết định giữ; Docker prune,
> DISM, ZaloData, `.gemini`, MySQL, VMware).

| | |
|---|---|
| Xác suất | ~~Trung bình~~ → **Đã xảy ra** |
| Tác động | ~~**Cao — chặn toàn bộ P0**~~ → **Đã gỡ chặn** |

90GB nghe nhiều nhưng: MipNeRF360 full ~30GB + COLMAP intermediates + checkpoints
(mỗi `.ply` có thể 300MB-1.5GB) x hàng trăm run.

**Xử lý:**
- Chỉ giữ checkpoint cuối + 1 checkpoint giữa mỗi run
- `.gitignore`: `datasets/`, `results/checkpoints/`, `*.ply`
- Script dọn dẹp tự động: xoá checkpoint của exploratory run sau khi đã trích metric
- **Không bao giờ xoá `metrics.jsonl` / `final.json`** — chúng nhỏ và là toàn bộ giá trị của project

---

## R8 — Scope creep

| | |
|---|---|
| Xác suất | **Rất cao** |
| Tác động | Cao |

Cám dỗ: thêm dynamic scenes, thêm SLAM, thêm mesh extraction, thử 5 SOTA method thay vì 1.

**Quy tắc cứng:**
- Mọi ý tưởng mới ghi vào `docs/05_EXPERIMENTS.md` mục "Future work", **không implement**
- Không thêm scene thứ 5
- Không reproduce method thứ 2 trước khi method thứ 1 xong hoàn toàn
- MVP (§9 của plan) là đích. Mọi thứ ngoài MVP là bonus.

---

## Bảng Decision Gates

| Sau phase | Câu hỏi | Nếu KHÔNG |
|---|---|---|
| P0 | Env chạy được, đo được VRAM? | R1 fallback. Cứng deadline 5 ngày. |
| P1 | `lego` train xong, PSNR hợp lý? | R2 fallback bậc thang |
| P2 | Cost model MAPE < 15%? | Đơn giản hoá model, giảm còn 2 số hạng. Vẫn ship. |
| P3 | Chọn được 1 method để reproduce? | Chọn `reduced-3dgs` (khả năng reproduce cao nhất) |
| P4 | Có Pareto front của trivial baselines? | **Blocking** — không được sang P8 |
| P6 | Reproduce khớp paper trong ~1dB? | Ghi lại chênh lệch + giải thích. Không được bịa số. |
| P8 | C4 thắng trivial baseline? | R3 — reframe thành robustness, hoặc report negative |
| P10 | Đủ 4 scene? | R5 — cắt còn 3, ghi rõ |

---

## Nhật ký kích hoạt rủi ro

Khi một rủi ro xảy ra, ghi vào đây:

| Ngày | Risk | Cái gì xảy ra | Fallback đã dùng | Kết quả |
|---|---|---|---|---|
| | | | | |
