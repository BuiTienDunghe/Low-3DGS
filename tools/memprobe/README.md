# memprobe — đo đỉnh VRAM từ trong tiến trình

File này **vốn sống ngoài repo**, ở `~/l3dgs/tools_memprobe/sitecustomize.py`, và được
nạp qua `PYTHONPATH` để không phải sửa một dòng nào trong mã bên thứ ba
(xem `docs/REPO_PATCHES.md` mục 3).

Nhưng vì nó ở ngoài repo, nó **không được đẩy lên GitHub** cùng dự án — trong khi
5 script trong `scripts/` lại trỏ tới nó. Nếu máy bị xóa thì các script đó tham chiếu
tới một file không còn tồn tại ở đâu nữa. Bản sao này đóng lại lỗ hổng đó.

## Dựng lại khi cài máy mới

```bash
mkdir -p ~/l3dgs/tools_memprobe
cp tools/memprobe/sitecustomize.py ~/l3dgs/tools_memprobe/
```

Rồi chạy như các script vẫn làm:

```bash
PYTHONPATH=~/l3dgs/tools_memprobe python prune_finetune.py ...
```

## Vì sao phải dùng `atexit`

`prune_finetune.py` có `return` **ngay trong vòng lặp huấn luyện**:

```python
if iteration < opt.iterations:
    gaussians.optimizer.step()
    ...
else:
    return          # <- thoát từ trong vòng lặp
```

Nên **mọi dòng đặt sau vòng lặp đều không bao giờ chạy**. Chỉ `atexit` mới chắc chắn
được gọi. Đây là lý do duy nhất file này không phải là vài dòng `print` ở cuối hàm.

## Nó trả nợ gì

EXP-015 kết luận 3: mọi số VRAM trước đó đều lấy từ NVML lấy mẫu 1 Hz, tức là
**chặn dưới** — bỏ sót đỉnh ngắn hơn một giây. EXP-014 báo 2621 MiB trong khi thao tác
thật cần ~3567 MiB. `torch.cuda.max_memory_allocated()` đọc từ trong tiến trình nên
không bỏ sót đỉnh.
