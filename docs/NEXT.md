# Tiếp tục từ đâu — chốt phiên 2026-09-06 (23:05)

Máy sạch: GPU 12 MiB, không tiến trình sót, đĩa D còn 101 G.

---

## Scene 2 — ✅ XONG (EXP-022). Ba phát biểu đều tái lập

6/6 lần chạy. Baseline `25.113122522830963` trùng qua cả 6 ⇒ đúng model nền.

| | `train` (chưa hội tụ) | `truck-864k` (đã hội tụ) |
|---|---|---|
| riêng "luyện thêm" | +0.3587 ± 0.0551 | **+0.0475 ± 0.0110** (teo 7.6×) |
| chi phí nén 50% | +0.0080 | **−0.2946** |
| cách so ngây thơ | **+0.3667** "tốt hơn!" | **−0.2471** đã âm sẵn |
| **có ảo giác?** | **CÓ** | **KHÔNG** |

**~~Kết quả sắc nhất — quy tắc dự đoán~~ — ĐÃ RÚT LẠI v0.5.1: đây là đồng nhất thức
đại số, không bác bỏ được. Phần đứng vững là ĐỘ LỚN của C1. Quan hệ dưới đây vẫn đúng
nhưng là phép cộng trừ, không phải phát hiện:**
```
ảo giác xuất hiện  <=>  lợi ích luyện thêm > chi phí nén
train      : +0.3587 > 0.0080  -> CÓ
truck-864k : +0.0475 < 0.2946  -> KHÔNG
```
Ảo giác sống trên **checkpoint chưa hội tụ ở tỉ lệ nén thấp**. Đó đúng là vùng vận hành
mà phần lớn bài nén 3DGS báo cáo.

C3 tái lập ở cả hai scene: LPIPS > SSIM > PSNR (64.9× / 7.9× / 0.2× ở train;
378.5× / 153.0× / 58.0× ở truck-864k).

---

## VIỆC TIẾP THEO — chọn một

| | Việc | Vì sao | Chi phí |
|---|---|---|---|
| **1** | **git init** | 22 thí nghiệm chưa gắn được với phiên bản mã. Món nợ lâu nhất | ~10 phút |
| 2 | Điểm thứ ba cho **quy tắc dự đoán** — chọn cấu hình nằm ở vùng ranh giới (lợi ích ≈ chi phí) để thử **bác bỏ** | quy tắc mới có 2 điểm; 2 điểm chưa phải quy luật | ~1 giờ |
| 3 | Đường cong đánh đổi cho scene 2 (thêm vài tỉ lệ) | scene 2 mới có 1 điểm | ~1.5 giờ |
| 4 | Quyết dứt hướng B: cắt bậc màu 2 + phục hồi | cần patch chặn `oneupSHdegree` | code + 15 phút |

Nghiêng về **1 rồi 2**: quy tắc dự đoán là kết quả mạnh nhất của dự án, và một điểm ở vùng
ranh giới có thể **bác bỏ** nó — đó mới là phép thử thật.

---

## Lỗi đã mắc và đã sửa trong phiên này

**Cổng B4 tự loại oan cả 3 nhánh nén.** Viết `[ "$NEVAL" -eq 2 ]`, nhưng nhánh nén eval
**3 lần** (trước prune @30001, sau prune @30001, và @35000) — đúng cái đã thấy rõ trong
`metric.csv` của EXP-014 từ lâu. Các lần chạy **hoàn toàn tốt**, chỉ là không được ghi vào CSV.
Đã gom lại từ `metric.csv`, **không phải chạy lại**. Cổng đã sửa: đối chứng=2, nén=3.

---

## Đã trả nợ

- **Đo VRAM từ trong tiến trình** (`max_memory_allocated`) thay cho NVML lấy mẫu —
  qua `~/l3dgs/tools_memprobe/sitecustomize.py` + `PYTHONPATH`, **0 dòng sửa repo**.
  Phải dùng `atexit` vì `prune_finetune.py` có `return` ngay trong vòng lặp.
  Chênh lệch thật đo được: NVML 1923 MiB vs trong tiến trình **1691.7 MiB**.

---

## Số dùng lại được, không phải đo lại

```
--- scene train ---
baseline @30001          21.815698046433297   (trùng 15 chữ số qua 14 lần chạy)
đối chứng @35000         22.174395            (trung bình 3 seed)
nhiễu ghép cặp           PSNR 0.0181 · SSIM 0.00006 · LPIPS 0.00015

--- scene truck-864k ---
baseline @30001          25.113122522830963   (trùng qua 5 lần chạy)
model nền sha256         5c837cbf1f3f0479c76d3576aa48973a2dacebd5f0ecb122dc9fa4c5fbf5c56c
N: đối chứng 864017 · nén 50% 432008
```

**Hình:** https://claude.ai/code/artifact/c3f8262d-fc69-45e1-89bf-8fe70732eff0

---

## Vì sao scene 2 phải là model đã prune (đừng thử lại truck nguyên bản)

`truck` 2.541.226 hạt: ba khoản lớn nhất **độc lập độ phân giải**, nên giảm `-r` vô ích:

| khoản | MiB |
|---|---|
| tham số + grad + 2 moment Adam (944 B/hạt) | 2288 |
| `dL_dsh` = `torch::zeros({P,16,3})` — `rasterize_points.cu:157` | 465 |
| CUDA context | 62 |
| **sàn cứng** | **2815** / trần 3440 → chỉ còn 625 |

`playroom` (2.546.116) và `drjohnson` (3.405.153) còn tệ hơn: ảnh **1264×832** và **1332×876**,
gấp ~2 lần số pixel của train/truck, nên công thức workspace khớp ở cỡ 980×546 **đánh giá thấp** chúng.

---

## Nợ kỹ thuật còn lại

1. **Chưa git.** 22 thí nghiệm không gắn được với phiên bản mã. Sửa đổi repo ghi tay ở
   `docs/REPO_PATCHES.md` (sinh lại: `bash tools/collect_patches.sh`).
   Loại trừ khi commit: `datasets/` (19 G) và `experiments/` (874 M) — nhưng **giữ**
   `experiments/*.csv` và `*.json`, đó là toàn bộ số liệu, chỉ vài chục KB.
2. `save_ply` phình 13.35× (`gaussian_model.py:206`). Giờ **sửa được** (mọi loạt đã xong).
3. `tools/knee_errorbars.py` mới đọc `train`; cần mở rộng cho `scene2_truck864.csv`.

---

## Chưa kiểm chứng, đừng phát biểu như thật

- ~~Scene 2 thiếu seed 2 của nhánh nén~~ — **đã xong 2026-09-07, nay n=3 cả hai nhánh.**
  (Dòng này là tàn dư từ lúc loạt chạy còn dở; skill `bao-cao` bắt được vì nó đối chiếu
  tài liệu với `experiments/scene2_truck864.csv` thay vì tin tài liệu.)
- Đường cong đánh đổi của `train` vẫn chỉ **seed 0** (trừ 50/60/66% đã có 3 seed).
- Cắt bậc màu: phần tiết kiệm bộ nhớ là **tính toán**, chưa đo. Chưa thử phục hồi sau khi cắt màu.
- Ghép hai trục nén (cắt hạt × cắt màu) **chưa đo**.
