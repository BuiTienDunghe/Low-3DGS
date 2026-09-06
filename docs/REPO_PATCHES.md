# Sửa đổi lên mã nguồn bên thứ ba

Mọi thay đổi dự án này áp lên mã của người khác, để người sau tái lập được **chính xác**.
Từ v0.5.0 dự án đã có git, nhưng mã bên thứ ba nằm **ngoài** repo này nên vẫn phải ghi tay —
cập nhật **mỗi khi** chạm vào nó. Sinh lại bảng: `bash tools/collect_patches.sh`.

**Repo gốc:** `j-alex-hanson/gaussian-splatting-pup` @ `e971ea4802908c69eab7e8601bb1eddd2e443754`
**Sinh lại bảng này:** `bash tools/collect_patches.sh`

---

## 1. Bắt buộc để BIÊN DỊCH ĐƯỢC trên CUDA 12 / GCC 13

Bốn phần mở rộng CUDA không biên dịch được với toolchain hiện đại vì thiếu `#include`.
Trước CUDA 12 các header này được kéo vào gián tiếp; nay thì không.
**Không đổi hành vi — chỉ thêm khai báo.**

| # | file | thay đổi |
|---|---|---|
| 1 | `submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h` | `+#include <cstdint>` |
| 2 | `submodules/compress-diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h` | `+#include <cstdint>` |
| 3 | `submodules/rasterization_and_pup_fisher/cuda_rasterizer/rasterizer_impl.h` | `+#include <cstdint>` |
| 4 | `submodules/simple-knn/simple_knn.cu` | `+#include <cfloat>` |

Submodule ở các commit: `59f5f77` · `79b24db` · `eaef2fa` · `44f7642`.

> **Đây tự nó là một phát hiện về khả năng tái lập:** mã công bố kèm bài báo
> **không biên dịch nổi** trên toolchain mặc định năm 2026 nếu không sửa 4 dòng này.

---

## 2. Thêm khả năng ĐỔI SEED (2026-09-04)

**File:** `utils/general_utils.py`, hàm `safe_state`

`safe_state` cố định cả ba nguồn ngẫu nhiên về 0 và **không có cờ dòng lệnh nào** để đổi.
Không đổi được seed thì không chạy lặp được, mà không chạy lặp thì không có thanh sai số.

```diff
+import os
 import torch

-    random.seed(0)
-    np.random.seed(0)
-    torch.manual_seed(0)
+    _seed = int(os.environ.get("L3DGS_SEED", "0"))
+    random.seed(_seed)
+    np.random.seed(_seed)
+    torch.manual_seed(_seed)
```

**Tương thích ngược tuyệt đối:** không đặt `L3DGS_SEED` → seed = 0 → hành vi y hệt bản gốc.
Đã kiểm chứng: không đặt biến thì `random.randint(0,10**6)` = `885440`, đúng giá trị chuẩn
của Mersenne Twister với `seed(0)`. Nhờ vậy **EXP-014/016/017 vẫn hợp lệ và vẫn là "seed 0"**.

---

## 3. CHƯA sửa, nhưng đã biết là có vấn đề

Ghi lại để không quên, và vì bản thân chúng là kết quả nghiên cứu.

| chỗ | vấn đề | vì sao chưa sửa |
|---|---|---|
| `scene/gaussian_model.py:206` | `elements[:] = list(map(tuple, attributes))` dựng N tuple Python 62 float ⇒ **~13.35× kích thước file trong RAM**. Ở N=2.54M là ~8.3 GB, vượt trần 9.7 GiB. Đây là lý do `truck` không lưu nổi model chưa nén. | Sửa được bằng `.view(dtype_full)` (không sao chép). Nhưng sửa lúc này sẽ làm các lần chạy **khác mã** với EXP-014/016/017. Để sau khi xong loạt đo hiện tại. |
| `utils/logger_utils.py:31-33` | cắt cụt `cfg_args` | không ảnh hưởng số đo |
| `prune_finetune.py` (cuối `training`) | **không in đỉnh VRAM từ trong tiến trình.** Mọi số VRAM hiện có đều lấy từ NVML lấy mẫu 1 Hz ⇒ **chặn dưới**, bỏ sót đỉnh dưới 1 giây (EXP-015 kết luận 3: EXP-014 báo 2621 MiB trong khi thao tác thật cần ~3567 MiB). **Việc phải làm:** in `torch.cuda.max_memory_allocated()` / `max_memory_reserved()`. | Sửa lúc này sẽ làm 6 lần chạy của loạt EXP-018 **khác mã nhau**. Làm ngay sau khi loạt xong. |
| `prune_finetune.py:272` | `viewpoint_stack = None` sau prune ⇒ hai nhánh đối chứng **không** thấy cùng thứ tự ảnh | sửa thì lệch khỏi hành vi gốc của LightGaussian/PUP; đã ghi là nguồn nhiễu thay vì sửa |

---

## 4. Không phải sửa mã, nhưng bắt buộc khi chạy

| biến | giá trị | vì sao |
|---|---|---|
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | Không có nó, WSL2 **tràn VRAM im lặng sang host RAM** thay vì báo lỗi — mọi số đo thành vô nghĩa (EXP-006/008). Tốn 368 MiB trần, đổi lại được lỗi OOM sạch. |
| `--data_device` | `cpu` | ảnh trên GPU thì playroom/drjohnson OOM chắc chắn |
