# Methodology & Measurement Protocol

Cố định protocol **trước** khi chạy experiment. Mọi thay đổi protocol sau đó phải
ghi ngày + lý do, và **mọi kết quả cũ phải chạy lại** hoặc được đánh dấu là không so sánh được.

---

## 1. Seeds & Variance

### Vì sao bắt buộc

CUDA rasterizer backward dùng `atomicAdd` với thứ tự float không xác định.
**Cùng seed, cùng config, chạy 2 lần vẫn ra kết quả khác nhau.** Bit-exact
reproducibility là bất khả thi với 3DGS — đừng hứa nó trong README.

### Protocol

| Loại run | Số seed | Report |
|---|---|---|
| Exploratory / debug | 1 | Chỉ dùng nội bộ, **không đưa vào bảng nào** |
| Ablation | 1 | Ghi rõ "single seed" trong caption |
| **Headline comparison** | **3** (seed = 0, 1, 2) | **`mean ± std`**, luôn luôn |

### Quy tắc claim improvement

> Một improvement chỉ được gọi là improvement khi
> `mean_ours - mean_baseline > 2 x max(std_ours, std_baseline)`.

Nếu không đạt, viết: **"within noise"**. Không viết "slightly better".

Trước tiên phải đo noise floor: chạy **baseline y hệt 3 lần** trên `lego`, ghi std.
Đây là con số quan trọng nhất của cả project — nó quyết định improvement nào là thật.

### Seed phải set cho

`random`, `numpy.random`, `torch.manual_seed`, `torch.cuda.manual_seed_all`,
`PYTHONHASHSEED`, và thứ tự shuffle của camera trong training loop.

---

## 2. Đo VRAM — định nghĩa chính xác

Ba con số khác nhau, **không được lẫn lộn**:

| Metric | API | Ý nghĩa |
|---|---|---|
| `peak_allocated` | `torch.cuda.max_memory_allocated()` | Tensor thật sự đang dùng. Dùng cho **cost model (C1)**. |
| `peak_reserved` | `torch.cuda.max_memory_reserved()` | PyTorch caching allocator đã giữ. Luôn >= allocated. |
| `peak_device` | `nvidia-smi` / NVML `used` | Bao gồm CUDA context + driver + process khác. **Đây mới là cái quyết định OOM.** |

**Số báo cáo trong mọi bảng là `peak_device`**, vì đó là thứ so được với "4GB".
`peak_allocated` chỉ dùng nội bộ để fit cost model.

### Quy trình đo

1. Đóng mọi ứng dụng dùng GPU (trình duyệt cũng chiếm VRAM). Ghi lại baseline `nvidia-smi` trước khi chạy.
2. `torch.cuda.reset_peak_memory_stats()` ở đầu run.
3. Sample NVML ở tần số 2 Hz trong suốt run, ghi vào JSONL.
4. Ghi cả `peak`, và **iteration nào đạt peak** (thường là ngay sau densification step).
5. Đặt `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` và ghi lại có/không dùng — nó ảnh hưởng fragmentation.

### CUDA context overhead

Đo một lần ở P0: chạy `torch.zeros(1).cuda()` rồi đọc NVML. Con số đó (~250-400MB)
bị mất trắng và phải trừ khỏi 4GB trong mọi tính toán budget.

---

## 3. Đo Model Size — chỗ dễ gian lận nhất

Bắt buộc báo cáo **cả 4 con số**, không được chỉ chọn con số đẹp nhất:

| Định dạng | Cách tính | Ghi chú |
|---|---|---|
| `size_ply_fp32` | File `.ply` chuẩn 3DGS | Baseline, cách mọi người hay report |
| `size_raw` | `N x bytes_per_gaussian` sau quantization, chưa nén | "Kích thước lý thuyết" |
| `size_npz_fp16` | `np.savez_compressed` với fp16 | Cách lưu thực dụng |
| **`size_entropy`** | Sau codebook VQ + entropy coding (zstd/Huffman) | **Đây là con số các compression paper báo cáo** |

**Cạm bẫy:** quantize xuống INT8 rồi lưu vào mảng `int8` là 4x. Nhưng LightGaussian
báo cáo ~15x nhờ VQ + entropy coding. So 4x của mình với 15x của họ là **so sai**.

Compression ratio luôn tính so với `size_ply_fp32` của **chính baseline của mình**,
không phải của paper.

---

## 4. Quality Metrics

| Metric | Implementation | Lưu ý |
|---|---|---|
| PSNR | Trên RGB, `[0,1]`, không gamma correction | Ghi rõ có tính trên toàn ảnh hay crop |
| SSIM | `skimage` hoặc impl trong 3DGS repo | 2 impl này **ra số khác nhau** — chọn 1 và giữ nguyên |
| LPIPS | `lpips` package, **backbone VGG** (không phải AlexNet) | VGG là chuẩn trong 3DGS papers |

**Trên 4GB:** LPIPS-VGG chiếm VRAM đáng kể. Chạy evaluation trong **process riêng**
sau khi train xong, hoặc trên CPU. Không bao giờ để eval làm nhiễu `peak_device` của training.

### Split
- Ghi **hash của danh sách test image** vào mỗi run's metadata
- Không đổi split giữa các experiment

---

## 5. Đo Speed / FPS

- **Warm-up 50 frame** trước khi đo, bỏ qua kết quả warm-up
- `torch.cuda.synchronize()` trước và sau mỗi lần đo — thiếu cái này thì số FPS là vô nghĩa
- Đo **median** của 200 frame, không phải mean (tránh outlier do OS scheduling)
- Ghi **GPU clock + temperature** cùng lúc. Run bị throttle (clock < 90% base) thì
  **loại khỏi timing benchmark**, nhưng quality metrics vẫn giữ
- FPS trên 1650 Ti **không so được** với FPS trong paper (A6000/3090). Chỉ so **tương đối
  giữa các method của mình**. Ghi câu này vào caption mọi bảng có FPS.

---

## 6. Cost Model (C1) — thiết kế

### Dạng model đề xuất

```
peak_allocated ≈ a * N * bytes_per_gaussian(sh_degree, dtype)
               + b * E[tile_gaussian_pairs]
               + c * (H * W)
               + d
```

Trong đó:
- Số hạng 1: params + grads + Adam states, đã biết analytically
- Số hạng 2: rasterizer workspace — **cần đo, không suy ra được từ N**
- Số hạng 3: image buffers + activations
- Số hạng 4: overhead cố định

### Validation
- **Fit** trên 2 scene (`lego`, `truck`)
- **Test** trên 3 scene chưa thấy (`playroom`, `bicycle`, + 1)
- Report **MAPE** (mean absolute percentage error). Mục tiêu < 15%.
- Report cả **failure case** — nếu model dự đoán sai ở scene nào, phân tích tại sao.
  Đây là phần thú vị nhất, không phải phần cần giấu.

---

## 7. Ghi log mỗi run

Mỗi run tạo `experiments/<run_id>/`:

```
config.yaml         # config đầy đủ, đã resolve, không phải template
env.json            # python/torch/cuda/gpu driver version, git commit hash
metrics.jsonl       # 1 dòng/iteration: iter, loss, N, peak_alloc, peak_device, lr
memory.jsonl        # NVML sampling 2Hz: t, used, clock, temp, power
final.json          # PSNR/SSIM/LPIPS, sizes (4 con số), FPS, wall time, throttled?
stdout.log
```

`run_id` = `{method}_{scene}_{budget}_{seed}_{YYYYMMDD-HHMMSS}`

**Ghi ngay lập tức, không buffer.** Một crash ở giờ thứ 40 không được làm mất 39 giờ dữ liệu.

---

## 8. Bảng kết quả — template bắt buộc

Mọi bảng phải có header ghi rõ:

> Scene: `truck` · Resolution: 979x546 (1/2) · Iterations: 30k · Seeds: 3 (mean ± std)
> · GPU: GTX 1650 Ti 4GB · VRAM = `peak_device` (NVML) · Size = `size_entropy`

Thiếu bất kỳ dòng nào trong header thì bảng **không được đưa vào report**.
