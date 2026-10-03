# low3dgs-guard

Mod cho Claude Code, chạy sống trong phiên làm việc của dự án này.

## Nó giải quyết gì

Một loạt thí nghiệm ở đây tốn **45–100 phút GPU**. Nếu nó chạy khi cây làm việc đang **bẩn**,
`run_meta.json` ghi `dirty: true`, và theo đúng quy ước của chính dự án
([`docs/VERSIONING.md`](../../../docs/VERSIONING.md)) **kết quả đó không tái lập được** —
phát hiện ra sau một tiếng rưỡi thì đã muộn. Trong một phiên làm việc, chuyện này suýt xảy ra ba lần.

## Nó làm gì

**1. Chặn trước khi chạy.** Mọi lệnh khớp `scripts/run_*.sh` đều bị kiểm:

| Điều kiện | Hành động |
|---|---|
| Cây bẩn (ngoài `experiments/`) | chặn, kèm số mục bẩn và lý do |
| GPU đang có job | chặn — hai job trên card 4 GB tranh bộ nhớ, làm hỏng mọi số đo VRAM |
| Có `ALLOW_DIRTY=1` ở đầu lệnh | cho qua |

Chặn xảy ra **trước** khi script nạp ảnh, tức mất 0 phút chứ không phải 45.

Phép kiểm bẩn **loại trừ `experiments/`**: chính lần chạy ghi kết quả vào đó, nên nếu tính cả thì
không lần chạy nào sạch được. Thứ cần biết là **mã nguồn** lúc chạy có khớp commit không.

**2. Một dòng trạng thái trên ô nhập:** tag hiện tại, cây sạch/bẩn, VRAM, số job đang chạy.
Làm tươi mỗi 20 giây nên vẫn sống trong lúc loạt dài chạy nền. Dim khi ổn, lên màu khi không.

## Bật

```bash
claude --plugin-dir .claude/mods
```

Hoặc đặt `CLAUDE_CODE_PLUGIN_DIRS` trỏ vào thư mục đó.

Đang phát triển thì chép sang `~/.claude/dev-mods/<session-id>/` để nó **tự nạp lại mỗi khi sửa file**;
nhớ chép ngược lại đây khi xong, vì hai bản không tự đồng bộ.

## Type cho editor

`.claude-plugin/types/` là file sinh theo build, **không commit** (engine tự đặt `.gitignore` ở đó).
Tạo lại bằng lệnh `/plugin-types` trong Claude Code.

## Đã kiểm

Cả ba đường đã thử thật trong phiên: cây sạch → cho qua · cây bẩn 1 mục → chặn đúng thông điệp ·
cây bẩn + `ALLOW_DIRTY=1` → cho qua.
