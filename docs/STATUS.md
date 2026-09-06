# Trạng thái dự án — ảnh chụp 2026-09-03

Bản tổng hợp một trang: **điều kiện đang có · ngân sách tính toán · cấu trúc · kế hoạch**.
Số trong file này là **đo được**, trừ chỗ ghi rõ "tính toán" hoặc "chưa verify".
Chi tiết đầy đủ: [`PLAN.md`](PLAN.md) · [`05_EXPERIMENTS.md`](05_EXPERIMENTS.md) · [`07_RESEARCH_FINDINGS.md`](07_RESEARCH_FINDINGS.md)

---

## 1. ĐIỀU KIỆN ĐANG CÓ

### 1.1 Phần cứng

| | Giá trị | Ghi chú |
|---|---|---|
| GPU | GTX 1650 Ti, **4096 MiB**, sm_75 (Turing), driver 560.70 | CUDA ≤ 12.6 |
| Màn hình | Chạy trên **iGPU AMD** → dGPU idle **0 MiB** | Trọn 4 GB cho compute |
| Băng thông VRAM | **53–57 GB/s** ổn định (37–38 khi clock chưa boost) | **Phải warm-up trước mọi phép đo tốc độ** |
| CPU | Ryzen 5 4600H, 6C/12T, 3.0 GHz | 12 luồng trong WSL |
| RAM vật lý | **15.37 GB** | |
| Đĩa | C: 41 GB · **D: 122 GB** | Project + dataset ở D: |
| OS | Win 11 Home + WSL2 Ubuntu 24.04.3, kernel 6.6.87, systemd | GPU passthrough đã verify |

### 1.2 Ngân sách VRAM — đã đo (EXP-007, EXP-008)

```
4096.0 MiB  tổng
-  34.5 MiB  driver giữ (NVML đỉnh chỉ tới 4061.5)
- 161.0 MiB  đường nền WSL2 GPU-PV
-  62.5 MiB  CUDA context          (ước tính ban đầu 350 MiB — sai 5.6 lần)
-  32.0 MiB  cuBLAS workspace (nạp lazy ở matmul đầu tiên)
- 368.0 MiB  chi phí expandable_segments (đổi lấy OOM sạch)
─────────────
= 3440.0 MiB  = 3.36 GiB  DÙNG ĐƯỢC CHO TENSOR      <- SỐ CHÍNH THỨC
```

### 1.3 Ngân sách RAM — phụ thuộc trạng thái máy

| Trạng thái | Windows khả dụng | WSL đạt được cap 10 GB? |
|---|---|---|
| Chrome mở (26 proc, 4.24 GB) | 5.30 GB | ❌ **thiếu 3.95 GB → page** |
| Chrome đóng | ~10.2 GB | ✅ dư ~0.9 GB |

`free -h` trong WSL báo 9.2 GB available, nhưng đó là **góc nhìn của máy ảo**, không phải cam kết
Windows cấp được. WSL2 lấy RAM theo nhu cầu; vấn đề chỉ lộ ra giữa run dài.

**Kill-switch đặt ở 9.5 GB RSS.** Quy tắc vận hành: đóng Chrome/Zalo trước run dài,
`nvidia-smi` = 0 MiB, ghi trạng thái vào `env.json`.

### 1.4 🔴 Hai đường chết im lặng

| | Cơ chế | Triệu chứng | Trạng thái |
|---|---|---|---|
| **RAM** | pagefile 13.5 GB (Win) + swap 4 GB (WSL) | Run không crash, chậm 10–50× | `killswitch` theo RSS ✅ |
| **VRAM** | Driver WSL2/WDDM tràn sang host qua PCIe | `OutOfMemoryError` **không** kích hoạt, chậm 3.6× | ✅ **ĐÃ DẬP** bằng `expandable_segments:True` |

Bằng chứng gốc (EXP-006): cấp phát **10816 MiB thành công trên card 4096 MiB**, NVML đứng ở 4061.5,
băng thông tụt 54 → 15 GB/s. Sau khi bật `expandable_segments`, GPU OOM sạch trở lại.

### 1.5 Phần mềm — đã cài và smoke-test

```
gcc 13.3.0 · nvcc 12.6.85 · cmake 3.28.3 · ninja 1.11.1 · zip
torch 2.14.0+cu126 · torchvision 0.29.0+cu126 · cuda.is_available() = True
gsply · lpips · pynvml · psutil · numpy 2.5.2 · plyfile

/etc/profile.d/cuda126.sh:
  CUDA_HOME=/usr/local/cuda-12.6 · TORCH_CUDA_ARCH_LIST=7.5 · MAX_JOBS=2
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Đặt trong `/etc/profile.d/` chứ không phải `~/.bashrc` vì `.bashrc` của Ubuntu có guard
thoát sớm với shell non-interactive — mọi lệnh `bash -lc` sẽ không đọc tới cuối file.

**Công cụ tự viết, đã test:** `src/memory/profiler.py` (RAM+VRAM theo stage) ·
`src/memory/killswitch.py` (RSS + VRAM + phát hiện tràn) · `tools/profile_gpu.py` (NVML 2 Hz) ·
`tools/measure_ply_load.py` · `tools/measure_cuda_overhead.py` · `tools/ply_to_checkpoint.py`

### 1.6 Dữ liệu — đã tải và giải nén

| Scene | Bộ | N @30k | N @7k | `.ply` @30k | Ảnh | Kích thước ảnh |
|---|---|---|---|---|---|---|
| `train` | T&T | **1,026,508** | 559,263 | 243 MB | 301 | 980×545 |
| `truck` | T&T | **2,541,226** | 1,732,378 | 601 MB | 251 | 979×546 |
| `playroom` | DB | **2,546,116** | 1,734,607 | 602 MB | 225 | 1264×832 |
| `drjohnson` | DB | **3,405,153** | 1,902,253 | 805 MB | 263 | 1332×876 |

Đủ COLMAP `sparse/` + `images/` + checkpoint pretrained INRIA. Có cả @7k → **8 điểm N** cho cost model.
`models.zip` 13.65 GB giữ lại (còn MipNeRF360 nếu cần sau).

---

## 2. ĐIỀU KIỆN MỤC TIÊU — theo tính toán

### 2.1 Chi phí đơn vị (đã verify trong code)

| Hạng mục | Chi phí | Nguồn |
|---|---|---|
| Params SH3 (3+3+4+1+48 = 59 float) | **236 B/Gaussian** | tính |
| Fine-tune SH3 (params + grad + 2×Adam + stats) | **956 B/Gaussian** | tính |
| Fine-tune SH2 (38 float) | **620 B/Gaussian** | tính |
| Fine-tune SH1 (23 float) | **380 B/Gaussian** | tính |
| **Ảnh training** | **16 B/pixel** (RGB + alpha_mask) | verify `cameras.py:48` |
| Nạp `.ply` — INRIA | **4.58 × file** (có float64 lãng phí 1.2 GB) | **đo** |
| Nạp `.ply` — **gsply** | **1.95 × file**, nhanh hơn 6× | **đo** |
| Ghi `.ply` — INRIA | **12.08 × file** | đo (ngoại suy) |
| Ghi `.ply` — **gsply** | **1.30 × file** | **đo** |

### 2.2 VRAM: fine-tune có chạy được không?

Ngưỡng **3440 MiB = 3.36 GiB**. Pipeline LightGaussian prune 66% → N′ = 0.34 N.

| Scene | Score (N×236B) | FT **thứ tự gốc**¹ | Còn cho rasterizer | FT **A12**² | Còn lại |
|---|---|---|---|---|---|
| `train` 1.03M | 0.24 GB | **0.98 GB** | 2.38 GB ✅ | 0.33 GB | 3.03 GB ✅ |
| `truck` 2.54M | 0.60 GB | **2.43 GB** | 0.93 GB ✅ | 0.82 GB | 2.54 GB ✅ |
| `playroom` 2.55M | 0.60 GB | **2.44 GB** | 0.92 GB ✅ | 0.83 GB | 2.53 GB ✅ |
| `drjohnson` 3.41M | 0.80 GB | **3.26 GB** | **0.18 GB** ❌ | **1.11 GB** | 2.25 GB ✅ |

¹ Thứ tự gốc: `load_ply` → `training_setup` (Adam cấp phát cho **toàn bộ** N) → prune.
² **A12** = chuyển prune lên **trước** `training_setup`. ~5 dòng code. Adam chỉ cấp cho N′.

→ **A12 không còn là tối ưu tuỳ chọn — nó bắt buộc cho `drjohnson`.**
Đây cũng là một đóng góp kỹ thuật báo cáo được (H8 trong plan).

### 2.3 RAM: fine-tune có chạy được không?

Ngưỡng **9.5 GB** (kill-switch), `--data_device cpu`.

| Scene | Ảnh (16 B/px) | Nạp gsply | Runtime | **Đỉnh** | vs INRIA loader |
|---|---|---|---|---|---|
| `train` | 2.40 GB | 0.47 | 0.5 | **3.37 GB** ✅ | 4.02 GB ✅ |
| `truck` | 2.00 GB | 1.17 | 0.5 | **3.67 GB** ✅ | 5.25 GB ✅ |
| `playroom` | 3.53 GB | 1.17 | 0.5 | **5.20 GB** ✅ | 6.79 GB ⚠️ |
| `drjohnson` | 4.57 GB | 1.57 | 0.5 | **6.64 GB** ✅ | **8.76 GB** ⚠️ sát 9.5 |

**Ghi `.ply` là chỗ nguy hiểm nhất:** INRIA `save_ply` cho model đầy đủ drjohnson ≈ **9.7 GB** —
vượt kill-switch. Với gsply chỉ 1.05 GB. → **Bắt buộc dùng gsply cho I/O.**

### 2.4 Train from scratch — vì sao ngoài phạm vi

Cần 24 GB VRAM theo README chính chủ. Ngay cả `gsplat` (tiết kiệm nhất) cũng cần 5.6 GB cho MipNeRF360.
Với 3.36 GiB → **không khả thi**. Đây là lý do project chuyển hẳn sang **fine-tune từ checkpoint có sẵn**.

MipNeRF360 `bicycle` (~5.7M Gaussian): FT thứ tự gốc = 5.45 GB → **ngoài phạm vi**, kể cả với A12
thì ảnh + RAM vẫn quá lớn. Đã bỏ khỏi kế hoạch.

### 2.5 Điều gì CHƯA được kiểm chứng

- **Chưa chạy một dòng 3DGS nào.** Mọi số ở §2.2/§2.3 là **số học**, chưa phải run thật.
- Rasterizer workspace **chưa đo** — nó là ẩn số lớn nhất còn lại, và là thứ hay gây OOM bất ngờ.
- PUP 3D-GS **chưa clone, chưa build**.
- Khảo sát cộng đồng: **không tìm thấy ai từng chạy bất kỳ method nén 3DGS nào trên card 4 GB.**
  Bằng chứng gần nhất là Mini-Splatting trên GTX 1060 **6 GB**.

---

## 3. CẤU TRÚC

```
D:\low-3DGS\
├── docs/
│   ├── PLAN.md                  ⭐ nguồn sự thật (v4)
│   ├── STATUS.md                (file này)
│   ├── 05_EXPERIMENTS.md        EXP-003…008, nhật ký OOM, đính chính
│   ├── 07_RESEARCH_FINDINGS.md  khảo sát 6 agent, có URL từng claim
│   ├── env.md                   audit phần cứng/toolchain
│   ├── 02_METHODOLOGY.md        protocol đo
│   ├── 03_RISKS.md · 04_LITERATURE.md
│   ├── paper_notes/lightgaussian.md   8 bẫy đã verify + mốc re-run độc lập
│   └── 00_CRITIQUE · 01_PROJECT_PLAN · 06_WHAT_FITS   (lịch sử, superseded)
├── src/memory/     profiler.py · killswitch.py        ← phần thật sự của mình
├── tools/          profile_gpu · measure_ply_load · measure_cuda_overhead · ply_to_checkpoint
├── scripts/        setup_wsl_sudo.sh · p1_smoke.sh
├── datasets/       tandt/ · db/ · pretrained/models/  (4 scene)
├── experiments/    p0_setup/ (log + JSON đo được)
└── configs/ · tests/ · results/ · third_party/

~/l3dgs/  (trong WSL)
├── venv/                        torch cu126
└── third_party/                 gaussian-splatting · LightGaussian  (đã vá header)
```

**Chưa có:** `git init` · PUP 3D-GS · `src/{pruning,distillation,quantization,evaluation}/` · configs

---

## 4. KẾ HOẠCH

### 4.1 Câu hỏi nghiên cứu

| | |
|---|---|
| **RQ1** | Ở từng stage, ràng buộc nào bind trước — RAM hay VRAM — và bao nhiêu phần là **artifact sửa được** vs **cấu trúc**? |
| **RQ2** | Code LightGaussian đã phát hành có tái tạo được kết quả paper không, sai lệch từ đâu? |
| **RQ3** | Cùng ngân sách: nhiều Gaussian + SH thấp, hay ít Gaussian + SH cao? |
| **RQ4** | Ảnh training nên ở CPU hay GPU khi cả hai đều thiếu? |

### 4.2 Đóng góp, xếp theo độ chắc chắn

| | Nội dung | Trạng thái |
|---|---|---|
| **C1** | **Reproducibility audit LightGaussian** — 8 bẫy im lặng đã verify + chuỗi trích dẫn sai (HAC → HAC++ → SymGS) | ✅ **đã có, chưa cần chạy gì** |
| **C2** | Per-stage RAM+VRAM profiler | ✅ tool xong, chờ dữ liệu |
| **C3** | Phân rã trần bộ nhớ: artifact vs cấu trúc | 🔄 3/5 thành phần đã đo |
| **C4** | Benchmark cùng ngân sách bộ nhớ (không claim đặt tên) | ⏳ |
| **C5** | Deep Blending gap (LightGaussian chưa từng đánh giá) | ⏳ |

### 4.3 Tiến độ

| Phase | Nội dung | Trạng thái |
|---|---|---|
| **P0** | Env + toolchain + đo VRAM/RAM cơ sở | ✅ **XONG** |
| **P1** | Clone/build PUP · gate: render `train` rồi `truck` | 🔄 **đang ở đây** |
| P2 | Reproduce trên 4 scene, kiểm 8 bẫy | ⏳ |
| P3 | Literature review | ⏳ |
| P4 | Profiler + phân rã C3 | ⏳ |
| P5 | Trivial baselines + noise floor 3 seed | ⏳ |
| P6 | Allocation study + A12 | ⏳ |
| P7–P11 | Stress · quantization · benchmark · report · đóng gói | ⏳ |

**Tổng: 24 tuần part-time.** Compute ước tính ~100 GPU-giờ.

### 4.4 Gate kế tiếp (P1) — câu hỏi sống-chết

> Render test set của `truck` pretrained (2.54M Gaussian) trên máy này, `--data_device cpu`.
> **PASS khi:** không OOM · PSNR = **25.187 ± 0.3 dB** (3DGS paper Table 8) · RSS < 9 GB · không chạm swap.

Không đạt → hạ `-r 2`, ghi vào bảng OOM, và **chính điều đó là kết quả C3**.

### 4.5 Mốc so sánh (dùng cụm re-run độc lập, KHÔNG dùng số paper)

| Dataset | 3DGS baseline | LightGaussian — cụm độc lập |
|---|---|---|
| T&T | 23.14 / 411.0 MB | **22.83 – 23.62** |
| Deep Blending | 29.41 / 676.0 MB | **28.8 – 29.6** (bỏ outlier HAC 27.01) |
| `truck` riêng | **25.187** PSNR / 0.148 LPIPS | — |

⚠️ Không cite số Deep Blending của HAC/HAC++/SymGS — 27.01 là outlier lan truyền qua 3 paper.

### 4.6 Bốn quy tắc đo (sai là hỏng toàn bộ số liệu)

1. **LPIPS** = bản vendored `lpipsPyTorch/` của INRIA, backbone **VGG**, ảnh giữ `[0,1]`.
   Bản này có lỗi chuẩn hoá đã biết (issue #1239, còn mở) — **ngành cố ý giữ** để so sánh được.
2. **SSIM** = `utils/loss_utils.py` của INRIA (Gauss 11×11, σ=1.5). **Không** dùng `skimage`.
3. **MB = 10⁶ byte**, không phải 2²⁰.
4. **Size** = tổng byte trên đĩa sau bước coding cuối. `npz_fp16` chỉ là chẩn đoán, không phải số headline.
