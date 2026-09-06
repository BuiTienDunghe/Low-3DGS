---
name: bao-cao
description: Báo cáo trạng thái dự án nghiên cứu Low-3DGS — đang có gì, vấn đề gì, trạng thái hệ thống, kế hoạch tiếp theo, và điều hướng phát triển. Số liệu được TÍNH LẠI từ dữ liệu thô rồi đối chiếu với tài liệu, không chép lại kết luận cũ. Dùng skill này bất cứ khi nào người dùng hỏi về tình hình dự án, tiến độ, "đang ở đâu rồi", "còn gì phải làm", "tóm tắt lại", "báo cáo", "status", hoặc cần quyết định làm gì tiếp — kể cả khi họ không gọi tên skill. Cũng dùng khi mở lại phiên làm việc sau một thời gian nghỉ.
---

# Báo cáo trạng thái dự án

## Vì sao skill này tồn tại

Chế độ hỏng đặc trưng của dự án này không phải là code chạy sai — mà là **số liệu cũ và
số liệu chép lại**. Lịch sử có thật:

- hai phát biểu đã phải **rút lại** sau khi có đối chứng và thanh sai số
- một giá trị trung bình bị chép nhầm từ **một seed** thành "trung bình 3 seed"
- một hệ số đo ở code A bị áp cho code B, sai 2,5 lần
- một cổng kiểm viết sai đã **âm thầm loại 3 lần chạy tốt** khỏi file tổng hợp

Một báo cáo chỉ đọc `docs/NEXT.md` rồi nhắc lại sẽ **thừa hưởng và khuếch đại** đúng những
lỗi đó — và nghe rất thuyết phục trong khi đang sai.

Nên nguyên tắc của skill này là: **tính lại từ dữ liệu thô, rồi đối chiếu với tài liệu, và
báo khi hai bên lệch nhau.** Tài liệu là thứ *được kiểm tra*, không phải nguồn để trích.

---

## Bước 1 — Thu thập sự thật sống

```bash
bash .claude/skills/bao-cao/scripts/gather.sh
```

Trả về: phiên bản, git (kể cả cây làm việc có bẩn không), tình trạng máy, kiểm kê dữ liệu,
và hai phép đối chiếu quan trọng — tài liệu có cũ hơn dữ liệu không, và có run nào chạy rồi
mà chưa được gom vào file tổng hợp không.

Nếu script báo cây làm việc **bẩn** hoặc **chưa có git**, hãy nêu ngay trong báo cáo: khi đó
kết quả đang chạy không truy ngược được về phiên bản mã nào.

## Bước 2 — Tính lại các con số

Đừng trích số từ `docs/`. Chạy công cụ phân tích trên dữ liệu thật:

```bash
python tools/cross_scene.py       # so sánh chéo các scene: ba phát biểu chính C1/C2/C3
python tools/knee_errorbars.py    # chi phí nén có thanh sai số ở vùng đầu gối
python tools/tradeoff_curve.py    # đường cong đánh đổi
```

Nếu công cụ báo thiếu dữ liệu hoặc `n=1`, **nói ra**, đừng lấp bằng số trong tài liệu.

Sau đó đọc `docs/05_EXPERIMENTS.md` và `docs/NEXT.md` — nhưng để **đối chiếu**, không phải để
trích. Nếu một con số trong tài liệu khác với con số vừa tính, đó là **phát hiện của báo cáo**,
phải nêu rõ và nói bên nào đúng.

## Bước 3 — Phân tầng bằng chứng

Mọi phát biểu trong báo cáo phải mang đúng một nhãn. Đây là kỷ luật đã cứu dự án hai lần:

| Nhãn | Điều kiện | Được nói kiểu gì |
|---|---|---|
| ✅ **Chắc** | ≥3 seed, khoảng tin cậy 95% không chứa 0 | phát biểu thẳng |
| 🟡 **Tạm** | đo rồi nhưng n=1 hoặc n=2, chưa có thanh sai số | "sơ bộ", "chưa chốt" |
| 🔵 **Ước lượng** | tính ra từ công thức, **chưa đo** | "dự đoán", kèm cách tính |
| ⛔ **Đã rút lại** | từng phát biểu, nay biết là sai | nêu cả lý do sai |

Hai cái bẫy hay mắc, cần tránh chủ động:

- **"Chưa chứng minh được khác 0" không phải "bằng 0".** Khoảng tin cậy rộng do ít mẫu là
  *thiếu lực thống kê*, không phải bằng chứng không có hiệu ứng.
- **"Có ý nghĩa thống kê" không phải "mắt người thấy được".** Chưa có khảo sát người xem.

## Bước 4 — Viết báo cáo

Dùng đúng năm mục dưới đây, theo thứ tự này. Người đọc muốn biết *đang ở đâu* trước, rồi mới
tới *làm gì tiếp*.

```markdown
## Tóm tắt
Một đoạn ngắn: dự án đang ở đâu, phiên bản nào, điều quan trọng nhất lúc này.

## Đang có gì
Tài sản đã chắc — kết quả có thanh sai số, hạ tầng đo, công cụ.
Mỗi kết quả kèm nhãn bằng chứng và con số thật.

## Vấn đề đang gặp
Thứ đang chặn đường, mâu thuẫn giữa tài liệu và dữ liệu, nợ kỹ thuật.
Xếp theo mức nghiêm trọng, không theo thứ tự phát hiện.

## Trạng thái hệ thống
Máy, git, độ tươi của dữ liệu. Có gì đang chạy không. Chạy tiếp được ngay không.

## Kế hoạch tiếp theo
Bảng: việc · vì sao · chi phí (thời gian máy / công viết code).
Xếp theo "cái nào mở khoá cái tiếp theo", không phải theo cái nào dễ.
Nêu rõ một đề xuất, đừng liệt kê rồi để đó.

## Điều hướng phát triển
Nhìn xa hơn việc trước mắt: điều gì sẽ **làm mạnh** luận điểm, điều gì có thể **bác bỏ** nó,
và đâu là ranh giới mà phần cứng này không vượt qua được.
```

Về mục cuối — nó là mục dễ viết thành sáo rỗng nhất. Cái làm nó có giá trị là chỉ ra
**phép thử có thể làm luận điểm sụp đổ**. Một phát biểu chưa từng có cơ hội sai thì chưa
đáng tin. Nếu bạn không nghĩ ra được phép thử nào như vậy, hãy nói thẳng là chưa nghĩ ra.

---

## Ngôn ngữ

Viết **tiếng Việt**. Giữ nguyên tiếng Anh cho thuật ngữ chuyên ngành đã quen dùng trong ngành —
`PSNR`, `SSIM`, `LPIPS`, `VRAM`, `checkpoint`, `seed`, `commit`, `baseline`, `fine-tune`,
`Gaussian`, `prune` — vì dịch ra sẽ khó tra cứu ngược và khó đối chiếu với tài liệu tiếng Anh.

Diễn đạt theo lối **suy luận**, không phải liệt kê: nêu con số, rồi nêu nó *kéo theo* điều gì.
"LPIPS 64,9× nhiễu còn PSNR 0,2×" tự nó không nói gì; "PSNR không đủ nhạy để trả lời câu hỏi
trung tâm, còn LPIPS thì có" mới là điều người đọc cần.

Dùng bảng cho số, văn xuôi cho lập luận. Số phải kèm đơn vị và cỡ mẫu.

## Độ dài

Mặc định vừa phải — người đọc đang muốn quyết định làm gì tiếp, không đọc để giải trí.
Nếu người dùng hỏi ngắn ("đang tới đâu rồi?") thì trả lời gọn: tóm tắt + trạng thái +
một việc nên làm tiếp. Đủ năm mục chỉ khi họ muốn báo cáo đầy đủ.
