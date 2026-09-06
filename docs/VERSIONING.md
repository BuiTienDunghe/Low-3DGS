# Quy ước phiên bản

Dự án này là **nghiên cứu**, không phải thư viện. Nên "phiên bản" ở đây không đánh dấu
API thay đổi, mà đánh dấu **trạng thái của các phát biểu khoa học**.

## Ý nghĩa từng số — `MAJOR.MINOR.PATCH`

| | Khi nào tăng | Ví dụ có thật trong dự án |
|---|---|---|
| **MAJOR** | Một phát biểu đã công bố **bị rút lại hoặc đảo chiều** | sẽ tăng lên `1.0.0` khi nộp bài |
| **MINOR** | Có **loạt thí nghiệm mới** hoặc **phát hiện mới** | `0.5.0`: scene thứ hai + quy tắc dự đoán ảo giác |
| **PATCH** | Sửa tài liệu, công cụ, đính chính **không đổi kết luận** | sửa trung bình đối chứng 22.2277 → 22.1744 |

> **Rút lại một phát biểu KHÔNG phải PATCH.** Dự án này đã rút lại hai lần
> ("nén miễn phí +0.112 dB", "nén miễn phí tới 2×"). Từ `1.0.0` trở đi, những lần như thế
> **tăng MAJOR** — người đọc phải thấy ngay rằng kết luận đã đổi.

## Quy tắc bắt buộc: mỗi kết quả phải gắn với một commit

`docs/PLAN.md` yêu cầu mọi kết quả gắn được với phiên bản mã đã sinh ra nó.
Từ `v0.5.0`, mỗi lần chạy ghi kèm `run_meta.json`:

```bash
bash tools/stamp_run.sh experiments/<tên_run>
```

File đó ghi lại: phiên bản, commit, **cây làm việc có bẩn không**, ngày, GPU,
phiên bản torch/CUDA, và commit của repo bên thứ ba.

**Cây làm việc bẩn (`dirty: true`) làm kết quả KHÔNG tái lập được.** Hãy commit trước khi chạy.
Các thí nghiệm EXP-001..022 chạy **trước khi có git**, nên được đánh dấu
`"pre_versioning": true` — trung thực hơn là gán cho chúng một commit không có thật.

## Nâng phiên bản

```bash
bash tools/bump_version.sh minor "Scene thứ hai và quy tắc dự đoán ảo giác"
```

Script sẽ: sửa `VERSION` → thêm mục vào `CHANGELOG.md` → commit → tạo tag `vX.Y.Z`.
Đẩy tag lên bằng `git push --follow-tags`.

## Các phiên bản trước khi có git

`v0.1.0`–`v0.4.0` là **hồi cố**: chúng mô tả trạng thái thật của dự án tại từng thời điểm,
nhưng **không có commit tương ứng** vì git chỉ được bật ở `v0.5.0`.
Chúng không được tạo thành tag — tag trỏ vào commit sai còn tệ hơn không có tag.
