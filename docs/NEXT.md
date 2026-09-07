# Tiếp tục từ đâu — chốt 2026-09-08, v0.7.0

Máy sạch, cây làm việc sạch, đã đồng bộ remote. Chạy tiếp được ngay.

---

## Bức tranh hiện tại — đã đổi bản chất hai lần trong hai ngày

**Phát hiện chính, sau khi qua ba lần tự bác bỏ:**

> Cách so ngây thơ (model-đã-nén-đã-tinh-chỉnh vs checkpoint-gốc) **tính công của việc tinh
> chỉnh thành công của việc nén**. Ở `train` cắt 50%: ngây thơ **+0,3667 dB**, trong đó
> **+0,3587** là của tinh chỉnh và chỉ **+0,0080** là của nén.

**Cơ chế, và đây mới là phần đáng giá:** công thức tinh chỉnh có một **attractor**.
Điểm xuất phát trải 1,56 dB → điểm dừng trải 0,17 dB (co 9,3 lần), tụ về **22,1762 ± 0,0305**.
Suy ra ba điều:
- C1 = **khoảng cách từ checkpoint tới attractor**, không phải "giá trị của luyện thêm"
- "nén 50% miễn phí" = **hai nhánh cùng rơi về một điểm dừng** (+0,0080 nằm trong nhiễu 0,0305)
- **đầu gối = biên miền hút**, không phải ngưỡng mất thông tin

**Và attractor là thật, không phải artifact** (EXP-025, phán quyết ghi trước): tắt anneal thì
attractor còn **chặt hơn** (16,1× thay vì 9,3×). Anneal chỉ dịch chỗ nó (+0,153 dB) và giảm
nhiễu seed 1,93×. Cấu trúc ảo giác giữ nguyên khi bỏ anneal — C1 vẫn chiếm **96%** con số ngây thơ.

---

## Việc tiếp theo — chọn một

| # | Việc | Nó trả lời gì | Chi phí |
|---|---|---|---|
| **1** ⟵ đề xuất | **Kiểm attractor trên scene 2** (`truck-864k`, 3 mức cắt, seed 0) | Attractor có phải tính chất của một scene, hay của công thức nói chung. **Đây là thứ quyết định phát hiện có suy rộng được không** | 3 lần chạy, ~45 phút |
| 2 | Xác định **biên miền hút** (chạy 72%, 75%, 78%) | Đầu gối hiện chỉ biết "giữa 70% và 80%" | 3 lần chạy, ~45 phút |
| 3 | Thêm seed cho EXP-025 (đang n=1 mỗi mức) | Con số 16,1× hiện không tin được về độ lớn | 6 lần chạy, ~1,5 giờ |
| 4 | Thêm 2 seed mỗi nhánh cho EXP-024 → n=5 | t=2,889 còn thiếu 0,29 so với ngưỡng 3,182 | 4 lần chạy, ~70 phút |
| 5 | Sửa `save_ply` phình 13,35× | Nợ lâu nhất còn lại; mở đường cho scene lớn | code |

**Vì sao việc 1 trước:** attractor giờ là *cơ chế* chống đỡ toàn bộ cách diễn giải. Nếu nó chỉ
tồn tại trên `train` thì mọi phát biểu phải hạ xuống thành "quan sát trên một scene". Nếu nó
xuất hiện cả trên `truck-864k` thì đó là tính chất của **công thức tinh chỉnh nói chung** — và
đó mới là thứ áp được cho bài của người khác.

Dự đoán ghi trước cho việc 1: baseline `truck-864k` = 25,113122522830963; nếu có attractor thì
điểm dừng ở các mức cắt khác nhau phải tụ trong khoảng ±0,05 dB. **Lưu ý:** ở scene 2, cắt 50%
đã cho chi phí −0,2946 dB — tức đã **ngoài** miền hút. Nên phải chọn mức cắt NHẸ hơn (10%, 20%, 30%).

---

## 🔧 Khiếm khuyết phải sửa trước loạt sau

`tools/stamp_run.sh` báo `dirty: true` cho mọi lần chạy — vì **chính lần chạy ghi file vào
`experiments/`** làm bẩn cây. Theo thiết kế hiện tại **không lần chạy nào có thể sạch**.
Sửa: phép kiểm dirty phải loại trừ `experiments/`.

---

## Số dùng lại được, không phải đo lại

```
--- scene train (N=1,026,508) ---
baseline @30001            21.815698046433297   (trùng 15 chữ số qua mọi lần chạy)
attractor CÓ anneal        22.1762 ± 0.0305     (n=9, N ∈ [513k, 1.03M])
attractor KHÔNG anneal     21.9688              (n=1 × 3 mức cắt)
C1 có anneal               +0.3587 ± 0.0551     (n=3)
C1 không anneal            +0.1587 ± 0.1065     (n=3)
nhiễu ghép cặp             PSNR 0.0181 · SSIM 0.00006 · LPIPS 0.00015

--- scene truck-864k (N=864,017) ---
baseline @30001            25.113122522830963
model nền sha256           5c837cbf1f3f0479c76d3576aa48973a2dacebd5f0ecb122dc9fa4c5fbf5c56c
C1                         +0.0475 ± 0.0110     (n=3)
chi phí nén 50%            −0.2946 ± 0.0051     (n=3)  <- đã NGOÀI miền hút
```

**Hình:** https://claude.ai/code/artifact/c3f8262d-fc69-45e1-89bf-8fe70732eff0
**Báo cáo trạng thái:** `/bao-cao` — nó tính lại từ CSV thô, đừng tin số trong tài liệu này.

---

## Chưa kiểm chứng, đừng phát biểu như thật

- Attractor mới có ở **một scene**. Việc 1 là để trị đúng điều này.
- EXP-025 **n=1 mỗi mức cắt** — hướng chắc (trải điểm dừng < nhiễu seed) nhưng độ lớn thì không.
- EXP-024 **chưa đủ ý nghĩa thống kê** (t=2,889 < 3,182). Ước lượng điểm "anneal chiếm 55,8%"
  không được phát biểu như kết luận.
- Chênh 7,6× của C1 giữa hai scene vẫn **lẫn ba biến** (scene, N, trạng thái hội tụ).
- Đường cong đánh đổi 9 điểm chỉ có seed 0 (trừ 50/60/66%).
- **"Có ý nghĩa thống kê" ≠ "mắt người thấy được"** — chưa có khảo sát người xem.

## Dự án đã tự rút lại ba lần

1. "Nén còn làm model tốt hơn +0,112 dB" → nén **tốn** 0,088 dB (v0.3.0)
2. "Nén miễn phí tới 2,00×" → chỉ đúng theo PSNR (v0.4.0)
3. "Quy tắc dự đoán ảo giác" → là **đồng nhất thức đại số**, không bác bỏ được (v0.5.1)

Cộng thêm một lần rút lại diễn giải: "anneal tạo ra attractor" (v0.7.0, EXP-025).
