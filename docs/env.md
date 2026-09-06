# Environment Audit — máy thật

**Ngày audit:** 2026-09-01 · **Hostname model:** 82B5 (Lenovo)
**Kết luận: đây ĐÚNG là máy trong plan.** Specs khớp hoàn toàn.

---

## 1. Phần cứng (đã đo, không phải giả định)

| Thành phần | Giá trị đo được | Khớp plan? |
|---|---|---|
| GPU | **NVIDIA GeForce GTX 1650 Ti**, 4096 MiB | ✅ |
| Compute capability | **7.5** (Turing) | ✅ |
| Driver | 560.70 (hỗ trợ CUDA **12.6**) | — |
| Driver model | **WDDM** (không có TCC trên GeForce) | ⚠️ xem §5 |
| iGPU | AMD Radeon Graphics (Renoir) | ✅ quan trọng, xem §4 |
| CPU | AMD Ryzen 5 4600H, 6C/12T, 3.0 GHz | ✅ |
| RAM | **15.37 GB** khả dụng | ✅ |
| OS | Windows 11 Home Single Language | ✅ |
| Disk C: | 190 GB tổng, **9.54 GB trống (5%)** | ❌ **BLOCKER** |
| Disk D: | 285.6 GB tổng, **80.32 GB trống (28%)** | ✅ |

---

## 2. Toolchain hiện có

### Windows (native)

| Tool | Trạng thái | Ghi chú |
|---|---|---|
| Python | ✅ 3.11.9 (`C:\Users\dungbui\AppData\Local\Programs\Python\Python311`) | Trên C: |
| PyTorch | ⚠️ **2.12.0+cpu — CUDA build = None, `is_available() = False`** | **Phải cài lại bản CUDA** |
| conda | ❌ không có | Dùng `venv` là đủ |
| CUDA toolkit (`nvcc`) | ✅ **12.1** (V12.1.66) | Nhưng vô dụng nếu thiếu MSVC — xem dưới |
| **MSVC / `cl.exe`** | ❌ **KHÔNG CÓ** | Chỉ có VS *Installer* stub, không có toolset nào |
| git | ✅ 2.45.2 | |
| cmake | ✅ 3.29.2 | |
| COLMAP | ❌ không có trên PATH | Chưa cần cho Track A |

> **`nvcc` trên Windows BẮT BUỘC cần MSVC làm host compiler.**
> Không có `cl.exe` → **không build được CUDA extension trên Windows native ngay lúc này.**
> Đây chính là R1 trong `03_RISKS.md` đã hiện thực hoá.

### WSL2

| Tool | Trạng thái |
|---|---|
| WSL2 | ✅ Đã cài, Ubuntu là default, version 2 (`docker-desktop` cũng có) |
| **GPU passthrough** | ✅ **HOẠT ĐỘNG** — `nvidia-smi` trong WSL thấy GTX 1650 Ti, 4096 MiB |
| Python | ✅ 3.12.3 |
| gcc | ❌ chưa cài (`build-essential`) |
| nvcc | ❌ chưa cài (CUDA toolkit for WSL) |
| Dung lượng `/` | 1007 GB, 954 GB "trống" — **CON SỐ NÀY LÀ ẢO**, xem §3 |

---

## 3. ✅ BLOCKER #1 — Ổ C: — ĐÃ XỬ LÝ (2026-09-01)

> **Đã giải phóng 25.69 GB. C: từ 9.54 GB (5%) → 35.23 GB (18.5%).**
>
> | Việc đã làm | GB |
> |---|---|
> | Xoá `Code\WebStorage\{11,10}\CacheStorage` — cache trùng lặp, VS Code đã đóng | **14.04** |
> | Chuyển `.ollama\models` → `D:\ollama\models`, đặt `OLLAMA_MODELS` (User env) | **11.36** |
> | Chuyển WSL Ubuntu `ext4.vhdx` → `D:\wsl\Ubuntu` | **6.92** |
> | **Tổng giải phóng trên C:** | **~32.6** |
>
> **C: 9.54 GB (5%) → 42.13 GB (22.2%)** · D: 80.32 → 62.04 GB (21.7%)
>
> Kiểm chứng: robocopy 16 files / 11.356 GB, 0 failed; so khớp **chính xác 12,194,120,000 bytes**
> cả nguồn lẫn đích trước khi xoá nguồn. `ollama list` sau đó vẫn thấy đủ 4 model.
> `C:\Users\dungbui\.ollama` giờ chỉ còn `id_ed25519` (SSH keys) — giữ nguyên, đúng.
>
> **Chưa làm** (còn dư địa ~40 GB nếu cần): tắt hibernation (6.15 GB — *đã quyết định giữ lại*,
> xem §5b), Docker prune (~4.4), DISM WinSxS (~2-5), npm cache (2.11), ZaloData (6.75),
> `.gemini` (3.4), MySQL (2.16), VMware (1.03).
>
> Trạng thái hiện tại: **C: 42.13 GB / D: 66.83 GB**.

### Dọn ổ D: (2026-09-01)

D: là nơi chứa dataset + checkpoint, nên đã rà luôn. Đang dùng 223.6 / 285.65 GB.

**Đã làm:** xoá 3 installer đã dùng xong ở gốc D: (`cuda_12.1.0_531.14_windows.exe` 3.12 GB,
`OllamaSetup.exe` 1.46 GB, `Antigravity IDE.exe` 0.21 GB) → **+4.79 GB**. D: 62.04 → 66.83 GB.

**Phát hiện quan trọng — Docker vhdx phình rỗng:**

```
docker system df:  Images 1.718 GB · Containers 426 kB
                   Volumes 173.7 MB · Build cache 977.7 MB
                   -> noi dung THAT ~ 2.87 GB
docker_data.vhdx:  65.35 GB
                   -> ~62.5 GB la SLACK
```

vhdx không bao giờ tự co lại — nó giữ mức cao nhất từng đạt. Prune vô nghĩa ở đây
(chỉ 11.36 MB build cache thu hồi được, đã prune). **Toàn bộ phần thắng nằm ở nén vhdx.**

**Chưa làm — cần Admin:** `diskpart compact vdisk`. Các đường không-elevation đều tắc:
- `Optimize-VHD`: không có (Win 11 Home thiếu module Hyper-V)
- `wsl --manage --set-sparse`: Microsoft đã vô hiệu hoá vì **nguy cơ hỏng dữ liệu**
  (cần `--allow-unsafe` — không dùng). Và sai đích: distro `docker-desktop` đăng ký
  `ext4.vhdx` 0.1 GB trên C:, không phải `docker_data.vhdx`.

Script sẵn sàng ở scratchpad: `compact_docker_vhdx.ps1` (chạy bằng Terminal Admin).
**Kỳ vọng thu hồi ~57-61 GB → D: ≈ 125-128 GB.**

**Đã làm — xoá VMware VMs (2026-09-01):** xoá cả 4 VM trong `D:\VM`
(`Ubuntu Blue Hat`, `Windows 10 and later x64`, `Ubuntu 64-bit (4)`, `vCenter`).

> Kích thước logic 86.82 GB nhưng **thực thu 73.08 GB** — chênh lệch vì vmdk của VMware
> là thin-provisioned/sparse: tổng độ dài file lớn hơn dung lượng thật chiếm trên đĩa.
>
> Lý do bỏ: VMware Workstation **không cho VM truy cập CUDA** (3D acceleration của nó là
> SVGA ảo hoá, không phải passthrough; PCI passthrough là tính năng ESXi, Workstation không có).
> Project cần CUDA → WSL2 là đường duy nhất, và đã verify chạy. VMs bất động 17 tháng.

**D: 66.83 → 139.91 GB (49%)**

**Còn tồn — VMware app chưa gỡ (registry gỡ cài đặt bị hỏng):**

- Files: `C:\Program Files (x86)\VMware` = **1.1 GB**
- Services đang chạy: `VMAuthdService`, `VMnetDHCP`, `VMware NAT Service`, `VMUSBArbService`
- Adapter: `VMnet1`, `VMnet8` (Status Up)
- ProductCode: `{F3CC3EA6-E281-4DE0-833A-AC84D226C2C6}` — có trong `Installer\Products`
  nhưng **thiếu `InstallProperties` và thiếu key `Uninstall`** → không hiện trong Apps & Features,
  `msiexec /x` nhiều khả năng thất bại. Xem README/chat để biết 3 lựa chọn xử lý.

**Chưa làm — bạn tự quyết:** `D:\0. Downloads` 28.07 GB (file cá nhân) ·
`D:\$RECYCLE.BIN` 1.28 GB (839 items).

---

### Kết luận dung lượng

| | Trước | Sau |
|---|---|---|
| C: | 9.54 GB (5%) | **40.91 GB (21.5%)** |
| D: | 80.32 GB (28%) | **139.91 GB (49%)** |

**Nhu cầu project: ~95-105 GB** (pretrained 14 + MipNeRF360 30 + dataset nhỏ 1.7
+ WSL grow 20 + checkpoints 30-40).

→ **D: đã dư (~35-45 GB headroom). Nén Docker vhdx giờ là tuỳ chọn, không còn bắt buộc.**
Nếu làm thì được thêm ~57-61 GB → D: ≈ 197-201 GB.

### Tình trạng gốc (giữ lại để tham chiếu)

```
C:  190 GB tổng, chỉ còn 9.54 GB trống (5%)
```

Windows trở nên không ổn định khi dưới ~10% trống. Và mọi thứ project cần đều đang nhắm vào C::

| Thứ cần cài/tải | Dung lượng | Mặc định vào đâu |
|---|---|---|
| PyTorch + CUDA runtime libs | ~3-5 GB | C: (Python đang ở C:) |
| VS Build Tools (nếu chọn native) | ~3-7 GB | C: |
| CUDA toolkit trong WSL | ~3-4 GB | C: (qua vhdx) |
| Pre-trained models (INRIA) | **14 GB** | tuỳ chọn |
| MipNeRF360 dataset | **~30 GB** | tuỳ chọn |
| Checkpoints các run | hàng trăm MB × N run | tuỳ chọn |

**Và điểm chí mạng:**

```
Ubuntu WSL ext4.vhdx
  -> C:\Users\dungbui\AppData\Local\wsl\{fa6ceac7-...}\ext4.vhdx   (6.92 GB)
```

WSL báo "954 GB trống" nhưng nó được backing bởi một vhdx nằm **trên ổ C: chỉ còn 9.54 GB**.
Con số 954 GB là **ảo hoàn toàn**. WSL chỉ phình được thêm ~9.5 GB nữa là C: đầy và Windows hỏng.

### Phải làm trước mọi thứ khác

1. **Di chuyển WSL Ubuntu sang D::**
   ```
   wsl --manage Ubuntu --move D:\wsl\Ubuntu
   ```
   (`wsl --shutdown` trước. Cách cũ: `wsl --export` rồi `wsl --import` vào D:.)
2. **Dọn C:** — `Disk Cleanup`, xoá Windows.old, pip cache, npm cache, Docker images không dùng
3. **Mọi thứ của project vào D::** venv, datasets, checkpoints, pip cache (`PIP_CACHE_DIR`)
4. Mục tiêu: **C: >= 30 GB trống** trước khi bắt đầu P0

---

## 4. ✅ Tin tốt — dGPU đang rảnh hoàn toàn

```
nvidia-smi: 0 MiB / 4096 MiB used, "No running processes found"
```

Máy có iGPU AMD Radeon, và **màn hình đang chạy trên iGPU**, không phải trên 1650 Ti.
Nghĩa là gần như **toàn bộ 4096 MiB dành cho compute**.

Ước lượng ngân sách cần **sửa lại tốt hơn** so với `06_WHAT_FITS.md`:

| | Ước lượng cũ | Sửa lại |
|---|---|---|
| Trừ display + CUDA context | −0.7 GB | **−0.35 GB** (chỉ CUDA context) |
| Usable | ~3.3 GB | **~3.7 GB** |

**Cần verify khi làm P0:** mở trình duyệt / VS Code có đẩy tiến trình sang dGPU không.
Nếu có, đặt Windows Graphics Settings → các app đó dùng "Power saving" (iGPU).
**Đo `nvidia-smi` ngay trước mỗi run và ghi lại.**

---

## 5. Quyết định: WSL2, không phải Windows native

| | Windows native | **WSL2 (khuyến nghị)** |
|---|---|---|
| Host compiler | ❌ Phải cài VS Build Tools ~3-7 GB **lên C: chỉ còn 9.5 GB** | ✅ `apt install build-essential`, nhẹ hơn nhiều |
| GPU | ✅ Trực tiếp | ✅ **Đã verify hoạt động** |
| Build CUDA ext (3DGS/gsplat) | ❌ Nổi tiếng khó trên Windows | ✅ Repos đều Linux-first, mọi hướng dẫn giả định Linux |
| VRAM overhead | Không (nhưng có WDDM overhead) | ⚠️ **Chưa đo** — paravirtualization có overhead |
| Trạng thái hiện tại | Thiếu MSVC | **Đã cài sẵn, GPU đã thông** |

**Điều PHẢI đo ở P0 (không được đoán):**
> CUDA context overhead trong WSL2 so với Windows native.
> Nếu WSL2 tốn thêm > 300 MB, đó là ~8% của 4GB — đáng kể, và có thể phải xét lại.
> **Đây là data point đầu tiên của cost model C1.**

### WDDM
GeForce không hỗ trợ TCC mode, nên luôn chạy WDDM: Windows quản lý VRAM và có thể
preempt. Điều này thêm overhead và làm `nvidia-smi` reporting kém chính xác hơn Linux bare-metal.
Ghi vào limitations của report.

---

## 6. Sửa lại kế hoạch dataset vì ràng buộc dung lượng

`06_WHAT_FITS.md` giả định tải được cả pretrained models (14 GB) và MipNeRF360 (~30 GB).
Với **80 GB trên D:**, làm được nhưng phải có thứ tự. Ưu tiên:

| Ưu tiên | Dataset | Size | Vì sao |
|---|---|---|---|
| 1 | **NeRF Synthetic** (`lego`, `chair`) | ~1 GB | Train from scratch được ngay (Track B, #6) |
| 2 | **Tanks&Temples + Deep Blending** (bản 3DGS phát hành) | **~650 MB** | Scene thật, **nhỏ**, `truck`/`train`/`playroom`/`drjohnson` |
| 3 | **Pre-trained models** | 14 GB | Mở khoá Track A |
| 4 | MipNeRF360 | **~30 GB** | Chỉ tải khi 1-3 đã xong và còn chỗ |

> **Điều chỉnh quan trọng:** T&T + DB chỉ ~650 MB nhưng vẫn là scene thật, được dùng rộng rãi
> trong 3DGS papers. **Bắt đầu ở đó, không phải MipNeRF360.** Rẻ hơn 45× về dung lượng
> và vẫn so sánh được với literature.

---

## 5b. Hibernation — quyết định GIỮ LẠI

Đã cân nhắc tắt (`powercfg /h off`, tiết kiệm 6.15 GB) nhưng **quyết định không làm**, vì:

| Cấu hình đo được | Hệ quả nếu tắt |
|---|---|
| `HiberbootEnabled = 1` (Fast Startup ON) | Mất Fast Startup, cold boot chậm thêm ~10-25s |
| `Hibernate after`: AC 120 phút / pin 120 phút | Máy nằm mãi ở S3, ăn pin chậm khi để lâu |
| **`Critical battery action = 0x02 (Hibernate)` ở 5%** | **Fallback sang Shutdown → mất việc chưa lưu.** Rủi ro thật cho run training dài |
| `Hybrid Sleep`: không khả dụng (hypervisor/WSL2 chặn) | Không mất thêm gì |

Không cần thiết: hai việc không-mặt-trái đã giải phóng 25.69 GB, vượt mục tiêu 30 GB.
Nếu sau này cần thêm: `powercfg /h /type reduced` giữ Fast Startup, bỏ full hibernate (~3 GB).
Lưu ý `hiberfil.sys` đang ở 6.15/15.37 GB = **đúng 40%**, mức tối thiểu Windows cho phép
với kiểu `full` — `powercfg /h /size` không giảm thêm được.

---

## 6b. Tiến độ P0 (2026-09-02)

| Việc | Trạng thái | Ghi chú |
|---|---|---|
| `.wslconfig` (10 GB RAM, 4 GB swap trên D:) | ✅ | WSL thấy **9.7 GiB / 9.2 avail**, swap `/dev/sdc` 4G, `D:\wsl\swap.vhdx` |
| Clone `gaussian-splatting` (54c035f, 2024-10-30) | ✅ | `~/l3dgs/third_party/`, submodules: diff-gaussian-rasterization, fused-ssim, simple-knn |
| Clone `LightGaussian` (6676b98, 2024-12-29) | ✅ | submodules: **compress-diff-gaussian-rasterization** (fork riêng, có `count_render`), simple-knn |
| Dataset tier 1 `tandt_db` | ✅ | `datasets/{tandt,db}/` = 739 MB. truck 251 · train 301 · playroom 225 · drjohnson 263 ảnh. COLMAP sparse có sẵn |
| **Pretrained `models.zip` (INRIA)** | ✅ | 14,660,630,999 B, HTTP 200, 40 phút @ ~6 MB/s. 117 file / 16.76 GB nén. Giữ zip (còn MipNeRF360 cho P7) |
| Giải nén 4 scene tier 1 | ✅ | `datasets/pretrained/models/{truck,train,playroom,drjohnson}/` = 3.58 GB. ⚠️ `unzip` Git Bash không cho `*` khớp `/` → phải dùng Python `zipfile` |
| N Gaussian (header PLY) | ✅ | truck 2.54M · train 1.03M · playroom 2.55M · drjohnson 3.41M (@30k); @7k ≈ 55–68% → 8 điểm N cho cost model |
| Kích thước ảnh thật (đọc SOF JPEG — `file` báo sai 72×72 vì đọc thumbnail EXIF) | ✅ | Xem bảng dưới |

**Ảnh training theo scene** (mọi ảnh trong scene cùng kích thước; ≤ 1.6K nên `-r 1` không bị auto-rescale):

| scene | n ảnh | dims | MB/ảnh f32 | **tổng ảnh f32** | params @30k | ảnh + params |
|---|---|---|---|---|---|---|
| train | 301 | 980×545 | 6.1 | **1.80 GB** | 0.24 GB | 2.04 GB |
| truck | 251 | 979×546 | 6.1 | **1.50 GB** | 0.60 GB | 2.10 GB |
| playroom | 225 | 1264×832 | 12.0 | **2.64 GB** | 0.60 GB | 3.24 GB |
| drjohnson | 263 | 1332×876 | 13.4 | **3.43 GB** | 0.80 GB | **4.23 GB** |

→ Với `data_device=cuda` (mặc định INRIA): drjohnson **4.23 GB > 4 GB VRAM → chắc chắn OOM**, playroom sát ngưỡng.
→ Với `data_device=cpu`: drjohnson cần 3.43 GB ảnh + ~2.4 GB đỉnh load + runtime ≈ **6–7 GB RSS** trong 9.7 GB WSL — đây là
scene stress đầu tiên **trước cả bicycle**, và có thể là nơi streaming-load / ảnh uint8 trở nên bắt buộc.
(Ảnh lưu float32 là cách INRIA làm; uint8 + convert on-the-fly sẽ giảm 4× — ứng viên cho C5.)
| `tools/profile_gpu.py` (NVML+psutil sampler) | ✅ viết, chưa test | Cần venv |
| `src/memory/profiler.py` (StageProfiler) | ✅ viết, chưa test | Cần venv |
| `src/memory/killswitch.py` | ✅ viết, chưa test | Cần venv |
| **venv + PyTorch cu126** | ❌ **BLOCKED** | `python3-venv` chưa có → cần `sudo` → **bạn chạy `scripts/setup_wsl_sudo.sh`** |
| build-essential + CUDA 12.6 toolkit | ❌ BLOCKED | cùng script trên |
| Build rasterizers | ⏳ | sau sudo |
| Đo CUDA context + WSL overhead | ⏳ | sau torch |

**Rủi ro mới phát hiện:** LightGaussian pin **PyTorch 1.12.1 / CUDA 11.6 / Python 3.9** (environment.yml).
Ta dùng torch 2.x / cu126 / py3.12. `compress-diff-gaussian-rasterization` có thể không build.
Fallback đã định: port `count_render` (kernel đếm hit/contribution) sang `diff-gaussian-rasterization`
hiện đại của INRIA — nhỏ, và là một implementation difference phải document trong paper note.

**Deviation #1 (đã biết trước khi chạy):** script chính thức nạp `chkpnt30000.pth` (có Adam state);
bundle INRIA nhiều khả năng chỉ có `.ply` → dùng `--start_pointcloud`, optimizer khởi tạo lại.

---

## 7. Checklist P0 (thứ tự bắt buộc)

- [x] ~~**Dọn C: lên >= 30 GB trống**~~ ✅ **XONG 2026-09-01 — C: = 42.13 GB (22.2%)**
- [x] ~~`wsl --manage Ubuntu --move D:\wsl\Ubuntu`~~ ✅ **XONG** — WSL 2.6.1.0, 12 giây,
      GPU passthrough verify lại vẫn OK (GTX 1650 Ti, 4096 MiB thấy trong WSL sau move)
- [ ] Trong WSL: `apt install build-essential`, cài CUDA toolkit for WSL
- [ ] Cài PyTorch **bản CUDA** (hiện đang là `+cpu`), verify `torch.cuda.is_available() == True`
- [ ] **Đo CUDA context overhead** trong WSL2 vs Windows native ← data point đầu tiên của C1
- [ ] Verify màn hình vẫn ở iGPU, `nvidia-smi` = 0 MiB khi idle
- [ ] Viết `tools/profile_gpu.py` (NVML sampling 2 Hz: used / clock / temp / power)
- [ ] Đo nhiệt độ + clock khi tải nặng để định lượng thermal throttling
- [ ] Tải dataset theo thứ tự ưu tiên §6

---

## 8. Tóm tắt rủi ro đã thay đổi

| Risk | Trạng thái trước audit | Sau audit |
|---|---|---|
| R1 (build fail trên Windows) | "Xác suất cao ~50%" | ⚠️ **Đã xảy ra** — không có MSVC. Nhưng WSL2 fallback **đã verify hoạt động** → rủi ro giảm mạnh |
| R7 (ổ đĩa đầy) | "Xác suất trung bình" | 🚨 **Đã xảy ra, là blocker #1.** C: còn 5% |
| Ngân sách VRAM | ~3.3 GB usable | ✅ **~3.7 GB** — tốt hơn dự kiến (display ở iGPU) |
| WSL2 VRAM overhead | Không nhắc tới | ⚠️ **Rủi ro mới, chưa đo.** Phải đo ở P0 |
