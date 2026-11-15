# Dựng lại dự án từ con số không

Viết ngày 2026-11-15, lúc **bản làm việc trên máy bị xóa có chủ ý** và repo này
trở thành bản duy nhất. Mọi số trong tài liệu này đo trực tiếp trên bản làm việc
trước khi xóa, không phải nhớ lại.

---

## Cái gì còn, cái gì không

| | |
|---|---|
| **Còn trên GitHub** | 189 file trong `experiments/` — **818 KB** gồm `*.csv`, `metric.csv`, `run_meta.json`, `cfg_args`, log. Cộng toàn bộ `docs/`, `scripts/`, `tools/`, `src/`. |
| **Đã xóa, tải lại được** | `datasets/` — **19 GB**. Model dựng sẵn của INRIA (18 GB) + tandt (398 MB) + db (352 MB). |
| **Đã xóa, phải chạy lại GPU** | `experiments/**/*.ply` — **42 file, 899,6 MB**. Mỗi file 45–100 phút GPU. |
| **Đã xóa, dựng lại được** | `~/l3dgs/venv` (7,0 GB) + `~/l3dgs/third_party` (650 MB). |

**Quan trọng:** không kết luận nào phụ thuộc vào phần đã xóa. Mọi PSNR/SSIM/LPIPS,
mọi `N`, mọi `run_meta.json` gắn commit đều ở trên GitHub. File `.ply` chỉ cần nếu
muốn **mở lại model để xem**, hoặc **đo thêm một chỉ số mới** mà lúc chạy chưa đo.

---

## Bố cục hai nửa — và vì sao phải tách

```
D:\low-3DGS\          <- repo này (git) + datasets/ 19 GB (không vào git)
~/l3dgs/              <- trong WSL: venv + third_party + tools_memprobe
```

Repo nằm trên `/mnt/d` (ổ Windows, qua 9p) vì dataset 19 GB không nên nằm trong
ổ ảo của WSL. Nhưng `venv` và `third_party` **phải** nằm trong hệ tập tin Linux
thật: biên dịch extension CUDA trên 9p vừa chậm vừa hay lỗi quyền.

---

## Các bước

### 1. Clone

```bash
git clone https://github.com/BuiTienDunghe/Low-3DGS.git /mnt/d/low-3DGS
```

`.gitattributes` buộc LF. Nếu clone bằng Git for Windows với `core.autocrlf=true`
thì mọi `.sh` sẽ hỏng trong WSL với `bad interpreter: No such file or directory`.

### 2. Toolchain trong WSL

```bash
wsl -d Ubuntu -u root bash /mnt/d/low-3DGS/scripts/setup_wsl_sudo.sh
```

Script này cài `cuda-toolkit-12-6` (**không** phải gói `cuda` hay `cuda-drivers` —
chúng sẽ cài driver Linux và xung đột với driver Windows mà WSL2 đang dùng), cùng
hai header mà Ubuntu 24.04 thiếu. Tải ~3 GB, mất 5–15 phút.

### 3. Mã bên thứ ba — đúng commit, rồi vá

```bash
mkdir -p ~/l3dgs/third_party && cd ~/l3dgs/third_party
git clone --recursive https://github.com/j-alex-hanson/gaussian-splatting-pup.git
git -C gaussian-splatting-pup checkout e971ea4
git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting.git
git -C gaussian-splatting checkout 54c035f
git clone --recursive https://github.com/VITA-Group/LightGaussian.git
git -C LightGaussian checkout 6676b98
```

Rồi áp **toàn bộ** `docs/REPO_PATCHES.md`. Mục 1 (bốn dòng `#include`) là **bắt buộc**:
mã công bố không biên dịch nổi trên toolchain 2026 nếu thiếu. Mục 2 và 2b thêm
`L3DGS_SEED` và `L3DGS_NO_ANNEAL` — thiếu chúng thì không chạy lại được loạt nào có
nhiều seed, cũng không chạy lại EXP-024/025.

Trạng thái lúc xóa, để đối chiếu: `gaussian-splatting-pup` có 6 mục khác upstream,
`LightGaussian` 2, `gaussian-splatting` 1 — phần lớn là con trỏ submodule.

### 4. venv

`venv` phải ở `~/l3dgs/venv` (hệ tập tin Linux), PyTorch bản `cu126` — `nvcc` phải
khớp major.minor với torch, nếu không rasterizer không build được.

### 5. memprobe

```bash
mkdir -p ~/l3dgs/tools_memprobe
cp /mnt/d/low-3DGS/tools/memprobe/sitecustomize.py ~/l3dgs/tools_memprobe/
```

**Không bỏ bước này.** Năm script trong `scripts/` nạp nó qua `PYTHONPATH`; thiếu
nó thì chúng chạy nhưng **im lặng không đo đỉnh VRAM**, và mọi số VRAM lại rơi về
chặn dưới lấy mẫu 1 Hz — đúng cái nợ mà EXP-015 đã trả.

### 6. Dataset

```
datasets/pretrained/models/{drjohnson, playroom, train, truck}/
datasets/tandt/{train, truck}/
datasets/db/{drjohnson, playroom}/
```

Nguồn: README của [3DGS gốc](https://github.com/graphdeco-inria/gaussian-splatting)
có link cho cả model dựng sẵn và hai dataset Tanks&Temples / Deep Blending.
Model dựng sẵn ~14 GB tải về, giải nén thành 18 GB.

---

## Kiểm đã dựng đúng chưa

Chạy lại baseline và so với số đã ghi. Hai số này **trùng tới 15 chữ số** qua mọi
lần chạy, nên lệch là dấu hiệu môi trường khác, không phải nhiễu:

```
scene train       @30001   21.815698046433297
scene truck-864k  @30001   25.113122522830963
```

Model nền `truck-864k`, sha256:

```
5c837cbf1f3f0479c76d3576aa48973a2dacebd5f0ecb122dc9fa4c5fbf5c56c
```

Nếu baseline khớp mà kết quả tinh chỉnh lệch, nghi `L3DGS_SEED` chưa có hiệu lực —
`safe_state` mặc định `0` nghĩa là **giữ hành vi gốc**, không phải seed 0.

---

## Cạm bẫy đã mất thời gian một lần rồi

- **`first_iter`**: `prune_finetune.py:59` đọc số vòng lặp từ **tên thư mục** —
  `int(path.split('/')[-2].split('_')[-1])` — và chạy **trước** `if args.first_iter`.
  File `.ply` phải nằm trong thư mục tên `iteration_NNNNN`, nếu không là crash.
- **`save_ply` phình 13,35×** (`gaussian_model.py:206`). Ở N=2,54M cần ~8,3 GB RAM,
  vượt trần 9,7 GiB. Đây là lý do `truck` chưa nén không lưu nổi. Chưa sửa — xem
  `docs/REPO_PATCHES.md` mục 3 để biết vì sao chưa.
- **`dL_dsh`** (`rasterize_points.cu:157`) cấp `{P,16,3}` float — **theo số Gaussian,
  không theo độ phân giải**. 465 MiB ở N=2,54M. Nên hạ `-r` **không** giúp vừa VRAM.
- **Mod `low3dgs-guard`**: JSX biên dịch thành `h(...)` trần. Đừng đặt tên biến nào
  là `h` hay `Fragment` trong file có JSX — hook sẽ bị bỏ qua và UI im lặng không vẽ.

---

## Đọc theo thứ tự nào

1. `README.md` — bốn phát hiện chính
2. `docs/05_EXPERIMENTS.md` — EXP-001…EXP-027, nhật ký đầy đủ
3. `docs/REPO_PATCHES.md` — mọi sửa đổi lên mã bên thứ ba
4. `CHANGELOG.md` — gồm cả **bốn lần tự rút lại**; đọc chúng trước khi tin con số nào
5. `/bao-cao` — skill báo cáo; nó **tính lại từ CSV thô**, không chép kết luận cũ
