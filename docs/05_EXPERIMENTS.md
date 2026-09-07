# Experiment Log

Ghi **mọi** experiment, kể cả thất bại. Quy tắc: viết trong vòng 24h, khi còn nhớ context.

Nguyên tắc: **hypothesis trước, kết quả sau.** Không được viết hypothesis sau khi đã thấy kết quả
(HARKing — Hypothesizing After Results are Known). Đây là cách tự lừa mình phổ biến nhất.

---

## Noise Floor `[PHẢI LÀM ĐẦU TIÊN, trước mọi so sánh]`

Chạy baseline **y hệt nhau 3 lần** trên `lego`. Đây là con số quan trọng nhất project.

| Seed | PSNR | SSIM | LPIPS | N | peak_device |
|---|---|---|---|---|---|
| 0 | | | | | |
| 1 | | | | | |
| 2 | | | | | |
| **std** | | | | | |

> **Ngưỡng claim improvement: `> 2 x std` PSNR = ____ dB.**
> Mọi cải thiện nhỏ hơn con số này là **within noise**, không được gọi là improvement.

---

## Template

```markdown
### EXP-XXX — <tiêu đề ngắn>
**Ngày:** YYYY-MM-DD · **Phase:** PX · **Run IDs:** ...

**Hypothesis**
H: ... (viết TRƯỚC khi chạy)
Điều kiện bác bỏ: nếu ... thì H sai.

**Config**
scene / resolution / iterations / seeds / method / budget / commit hash

**Hardware**
GPU / driver / CUDA / torch / OS(native hay WSL2) / throttled?

**Kết quả**
| metric | value (mean ± std) |

**Quan sát**
Cái gì bất ngờ? Cái gì khớp/không khớp dự đoán?

**Kết luận**
H được ủng hộ / bị bác bỏ / không kết luận được (thiếu power).

**Việc tiếp theo**
```

---

## Nhật ký thất bại

Bảng này **có giá trị CV cao hơn** bảng kết quả tốt. Nó chứng minh bạn thật sự chạy thí nghiệm.

| Ngày | Đã thử gì | Kỳ vọng | Thực tế xảy ra | Nguyên nhân gốc | Rút ra |
|---|---|---|---|---|---|
| | | | | | |

---

## Nhật ký OOM `[data cho cost model C1]`

Mỗi lần OOM là một điểm dữ liệu, không phải một lỗi. Ghi hết.

| Ngày | Scene | Res | N tại lúc OOM | Iteration | SH deg | Cấu hình | Ghi chú |
|---|---|---|---|---|---|---|---|
| | | | | | | | |

---

## Future work — KHÔNG implement trong scope này

Chỗ để "chôn" ý tưởng mới mà không làm hỏng scope (R8 trong `03_RISKS.md`).

- [ ] ...

---

## Experiments

### EXP-001 — P1 gate: render pretrained `truck` trên máy này
**Ngày đăng ký:** 2026-09-02 (viết TRƯỚC khi chạy) · **Phase:** P1 · **Script:** `scripts/p1_smoke.sh truck 1`
**Trạng thái:** ⏳ chờ toolchain (sudo) + `models.zip`

**Số đo thật của checkpoint (2026-09-02, đọc header PLY):**

| scene | N @7k | N @30k | .ply @30k | props |
|---|---|---|---|---|
| truck | 1,732,378 | **2,541,226** | 601 MB | 62 (SH deg 3) |
| train | 559,263 | 1,026,508 | 243 MB | 62 |
| playroom | 1,734,607 | 2,546,116 | 602 MB | 62 |
| drjohnson | 1,902,253 | **3,405,153** | 805 MB | 62 |

→ 8 điểm N miễn phí cho cost model. `cfg_args`: `resolution=1, eval=True, sh_degree=3`.
In-memory 59 float/Gaussian (bỏ 3 normal) → truck 30k = **600 MB params**.

**Phát hiện trước khi chạy (thay đổi thiết kế gate):** `render.py` INRIA mặc định `data_device=cuda`, và
`Scene.__init__` nạp **tất cả** 251 ảnh train dù `--skip_train` → ~1.6 GB ảnh lên **VRAM**. Cộng 0.6 GB
params + context + rasterizer ≈ 2.7–3.2 GB — sát 3.7 GB usable. Với drjohnson (3.4M, 263 ảnh) gần chắc OOM.
→ Gate chạy `--data_device cpu`. Biến thể `cuda` tách thành **EXP-002** = điểm dữ liệu đầu tiên của RQ4/H6.

**Hypothesis (đăng ký trước, đã tinh chỉnh theo N thật)**
- H-a: `render.py` truck @30k (2.54M, SH3), `-r 1`, `data_device=cpu` → render 31 ảnh test **không OOM**.
- H-b: Đỉnh **RSS 3–4 GB** (ply 601 MB × ~3 bản sao khi load ≈ 1.8 GB + 251 ảnh float32 trên CPU ≈ 1.6 GB + runtime ~0.5 GB).
  Bác bỏ nếu **> 6 GB** (nghĩa là load tốn > 3× file, phải streaming-load ngay từ P2).
- H-c: Đỉnh **VRAM device 1.2–1.6 GB** (params 0.6 + context ~0.35 + rasterizer 0.2–0.5). Bác bỏ nếu > 2.2 GB.
- H-d: PSNR trong **±0.3 dB** so với **25.187 dB** (3DGS paper, Table 8, truck @30k, A6000, native res, llffhold=8; LPIPS 0.148 Table 9).
  ⚠️ KHÔNG so với 25.37 của LightGaussian — đó là "3D-GS*" **họ tự train lại**, khác checkpoint INRIA +0.18 dB.
  Tham chiếu INRIA cho 4 scene: truck 25.187 · train 21.097 · playroom 30.044 · drjohnson 28.766 (PSNR); LPIPS 0.148 / 0.218 / 0.241 / 0.244.

**Điều kiện bác bỏ chung:** OOM → H-a sai → thử `-r 2`, ghi bảng OOM. RSS > 9 GB → RAM bind ngay ở scene vừa → kết quả lớn.
PSNR lệch > 0.3 dB → nghi split/res; **không** chỉnh cho khớp, ghi chênh lệch và nguyên nhân nghi ngờ.

**Config:** scene=truck · `-r 1` · `data_device=cpu` · llffhold=8 (cfg `eval=True`) · iteration_30000 · GTX 1650 Ti · WSL2 10 GB
**Sẽ đo:** `summary_render.json` (rss_peak_gb, vram_peak_mb, throttled_frac) · `summary_metrics.json` · PSNR/SSIM/LPIPS

**KẾT QUẢ: ✅ GATE PASS** (2026-09-02, `experiments/p1_smoke_train_r1_cpu_20260902-190359`)

Chạy trên `train` (scene nhỏ nhất) trước `truck`, repo **PUP 3D-GS** `e971ea4`, torch 2.14.0+cu126.

| Tiêu chí | Ngưỡng | Đo được | |
|---|---|---|---|
| Không OOM | — | render + metrics đều exit 0 | ✅ |
| RSS đỉnh | < 9 GB | **4.48 GB** (render) · 2.10 GB (metrics) | ✅ |
| VRAM đỉnh | < 3440 MiB | **1069.5 MiB** (render) · 1547.5 MiB (metrics) | ✅ |
| Swap | không chạm | 0 | ✅ |
| Resolution | native | **980×545 = ảnh gốc**, không rescale | ✅ |
| Split | llffhold=8 | **38 / 301 = 1/7.9** | ✅ |

**Chất lượng vs paper (3DGS Table 7/8/9, Ours-30K, `train`):**

| | Paper | Đo được | Δ |
|---|---|---|---|
| PSNR | 21.097 | **21.7869816** | **+0.690** |
| SSIM | 0.802 | 0.8076466 | +0.006 |
| LPIPS | 0.218 | 0.2143095 | −0.004 |

> 🟢 **ĐÃ GIẢI DỨT ĐIỂM (2026-09-03) — bằng lời tác giả, không phải suy đoán của tôi.**
>
> Giải thích tôi đưa ra bên dưới (*"mọi nhóm độc lập đều đo cao hơn paper"*) **SAI như một quy luật** —
> `truck` sau đó ra **thấp hơn** paper. Câu trả lời thật nằm ở chính README của INRIA:
>
> > *"The pre-trained models were created with the release codebase. This code base has been cleaned up
> > and includes bugfixes, hence the metrics you get from evaluating them **will differ from those in the paper**."*
>
> Và Bernhard Kerbl (tác giả thứ nhất) trong [issue #82](https://github.com/graphdeco-inria/gaussian-splatting/issues/82):
>
> > *"We did a major overhaul on the code for the release. **Some scenes can have slightly lower quality**
> > because of this, but **on average the scores should now be higher** than what we report in the paper."*
> > […] *"If you are doing a scientific evaluation, feel free to compare against the numbers you prefer."*
>
> **Điều này dự đoán đúng số đo của ta:** `truck` −0.194 (scene "slightly lower"), `train` +0.690,
> trung bình hai scene 23.390 vs paper 23.142 = **+0.248 ("on average higher")**.
>
> **Kiểm chứng độc lập — FlexGaussian đo cùng checkpoint:** `train` **21.78** · `truck` **24.93**
> so với của ta **21.787** · **24.993**. Lệch **0.007 dB** và **0.06 dB**. Số của ta đúng và tái lập được.
>
> → **Chênh lệch là hành vi được ghi rõ, không phải lỗi setup. GATE PASS.**
> → Baseline chính thức của project = **số đo của chính ta trên checkpoint**, và mọi Δ nén tính trên đó.

**Giải thích ban đầu của tôi (SAI như quy luật chung, giữ để đối chiếu):**

SSIM và LPIPS khớp trong sai số, chỉ PSNR lệch +0.69 dB. Không phải lỗi protocol (res/split/checkpoint
đều đã verify đúng). Nguyên nhân: **PSNR nhạy hơn SSIM rất nhiều với thay đổi MSE nhỏ** — +0.69 dB
chỉ tương đương MSE giảm 15%, trong khi SSIM chỉ nhích +0.7%.

Và mức lệch này **khớp đúng mẫu đã biết**: mọi nhóm đánh giá độc lập 3DGS đều ra **cao hơn** bảng của paper.
Baseline 3DGS trên T&T: paper **23.14**, còn cụm độc lập là PUP 23.77 · NeuralGS 23.75 · SPARE-GS 23.74 ·
ACE-GS 23.74 · FCGS 23.71 · VEDAL 23.68 · GS² 23.63 · POTR 23.36 — tức **+0.2 đến +0.63 dB**.
Của ta +0.69 nằm đúng trong dải đó.

→ **Không "chỉnh cho khớp" paper.** Mốc so sánh chính thức của project là **cụm re-run độc lập**,
đúng như quy tắc #10 đã đặt ra *trước khi* có dữ liệu này. Đây là lần đầu quy tắc đó được kiểm chứng bằng số đo.

**Tái lập:** chạy lại lần hai ra **bit-identical** (21.7869816 / 0.8076466 / 0.2143095). Race `atomicAdd`
nằm ở đường `count_render`, không ở đường render thường → render là tất định. Ghi nhận cho protocol seed.

**Phụ:** FPS 114 frame ở ~32–35 it/s (chưa warm-up đúng chuẩn, chỉ để tham khảo).

---

### EXP-009 — 🔴 Repo INRIA hiện tại KHÔNG render nổi checkpoint chính chủ của họ
**Ngày:** 2026-09-02 · Phát hiện khi làm thí nghiệm đối chứng cho EXP-001

Định render lại `train` bằng repo INRIA gốc (`54c035f`, 30/10/2024) để tách "PUP render khác" khỏi
"checkpoint khác paper". Không chạy được:

```
AttributeError: 'GroupParams' object has no attribute 'depths'
```

Nguyên nhân đã verify:

| | |
|---|---|
| `cfg_args` của bundle pretrained (2023) | `Namespace(eval, images, model_path, resolution, sh_degree, source_path, white_background)` |
| `arguments/__init__.py` của INRIA hiện tại | thêm `self._depths = ""` (dòng 53), `self.train_test_exp = False` (dòng 56) |
| PUP (fork từ bản cũ hơn) | **0 tham chiếu** tới `depths`/`train_test_exp` → chạy được |

→ Xác nhận [INRIA issue #198](https://github.com/graphdeco-inria/gaussian-splatting/issues/198) bằng thực nghiệm.
**Repo chính chủ mới nhất không đánh giá được model pretrained chính chủ nếu không vá.**
Đây là một mục cho C1 (reproducibility audit), và là một lý do nữa để dùng PUP.

---

### EXP-010 — Metric throttle bị sai: NVML tính GPU idle là throttle
**Ngày:** 2026-09-02

Gate EXP-001 báo `throttled_frac = 0.868` (render) và `0.927` (metrics) — nghe như GPU bị bóp 87–93% thời gian.
Nhưng nhiệt chỉ **63–67°C** và clock max chạm đủ 2100 MHz. Mâu thuẫn.

Nguyên nhân: NVML liệt `GpuIdle` và `ApplicationsClocksSetting` là "clocks event reason". Giữa các frame
GPU idle → bị đếm là throttle. Nếu không sửa, **quy tắc "loại throttled run khỏi bảng timing" sẽ loại nhầm gần hết run.**

Đã sửa `tools/profile_gpu.py`: lọc `{GpuIdle, ApplicationsClocksSetting, DisplayClockSetting}`.
Chỉ giữ throttle thật (nhiệt / nguồn / HW slowdown).

---

### EXP-026 — ✅ **Attractor TÁI LẬP trên scene thứ hai. Và miền hút hẹp lại khi model đã bị nén**
**Ngày:** 2026-09-08 · **Script:** `scripts/run_scene2_attractor.sh` · **Dữ liệu:** `experiments/exp026_scene2_attractor.csv`
**Commit:** `0225da02` — **ba lần chạy đầu tiên có dấu xuất xứ SẠCH** (không còn `dirty: true`)

**Câu hỏi:** attractor là tính chất của scene `train`, hay của **công thức tinh chỉnh** nói chung?
Nếu chỉ của một scene thì mọi phát biểu phải hạ xuống "quan sát trên một scene".

**Chọn mức cắt nhẹ (10/20/30%)** vì ở scene 2 mức 50% đã cho chi phí −0.2946 dB — **đã ngoài
miền hút**. Đo attractor thì phải đo trong miền.

**Phán quyết ghi trước:** co ≥5 lần → có attractor · ≤2 lần → không.

#### Kết quả

| cắt | xuất phát | điểm dừng | lệch đối chứng | = mấy lần nhiễu |
|---|---|---|---|---|
| 0% (mốc, n=3) | 25.1131 | **25.1606** | — | SD 0.0110 |
| 10% | 24.9608 | **25.1613** | +0.0007 | **0.1×** |
| 20% | 24.5753 | **25.1421** | −0.0185 | 1.7× |
| 30% | 23.9629 | **25.0977** | −0.0629 | 5.7× |
| **trải** | **0.9979** | **0.0636** | | |
| **co lại** | | **15.7×** | | |

**Đo được 15.7× ⇒ CÓ ATTRACTOR trên scene 2.** Phán quyết ghi trước được thoả.

#### Attractor có cùng độ mạnh ở cả hai scene

| | hệ số co | dải mức cắt |
|---|---|---|
| `train`, anneal bật | 9.3× | 20–70% |
| `train`, anneal tắt | 16.1× | 20–70% |
| **`truck-864k`** | **15.7×** | **10–30%** |

⇒ **Attractor là tính chất của CÔNG THỨC TINH CHỈNH, không phải của một scene cụ thể.**
Đây là điều kiện cần để cách diễn giải của dự án suy rộng được sang model của người khác.

#### 🔑 Phát hiện phụ: miền hút HẸP LẠI khi model đã bị nén

| | còn trong miền hút tới | |
|---|---|---|
| `train` (checkpoint gốc, 1.03M) | **~70%** | ở 50% chi phí chỉ +0.0080, nằm trong nhiễu |
| `truck-864k` (đã nén 66%, 864k) | **~20%** | ở 30% đã lệch **5.7 lần nhiễu** |

Model đã bị cắt 66% một lần thì **còn ít dư thừa để cắt tiếp**, và miền hút thu hẹp tương ứng.
Điều này định lượng được trực giác "nén cái đã nén thì đắt hơn" — và nó khớp với EXP-022
(cắt 50% trên `truck-864k` tốn −0.2946 dB, trong khi cùng mức trên `train` gần như miễn phí).

⇒ Phát biểu về "vùng miễn phí" phải kèm điều kiện: **nó là miền hút của công thức tinh chỉnh
trên model ĐANG XÉT, và bề rộng của nó phụ thuộc lượng dư thừa còn lại.** Không có một con số
"nén tới X% là miễn phí" dùng chung được.

#### Xuất xứ — lần đầu hoạt động đúng

`run_meta.json` của cả ba lần chạy ghi `commit 0225da02` với `dirty: false`.
Trước đó mọi lần chạy đều `dirty: true` vì hai lỗi trong `stamp_run.sh`:
loại kiểm tính cả `experiments/` (mà chính lần chạy ghi vào đó), và tin stat cache của git
(trên `/mnt/d` git báo SẠCH ngay sau khi file vừa bị sửa — đã quan sát trực tiếp).

#### Còn thiếu

- **n=1 mỗi mức cắt.** Hệ số 15.7× không tin được về độ lớn; nhưng hướng chắc vì trải điểm dừng
  (0.0636) chỉ bằng 5.8 lần nhiễu seed trong khi trải xuất phát bằng 91 lần.
- Biên miền hút của cả hai scene mới biết trong khoảng, chưa xác định điểm.
- Vẫn chưa kiểm attractor ở scene thứ ba — nhưng `playroom`/`drjohnson` không chạy nổi cặp
  đối chứng trên card 4 GB (EXP-015, EXP-022).

---

### EXP-025 — ✅ **Attractor KHÔNG do anneal tạo ra. Diễn giải EXP-024 của tôi SAI — và phát hiện chính mạnh lên**
**Ngày:** 2026-09-08 · **Script:** `scripts/run_noanneal_sweep.sh` · **Dữ liệu:** `experiments/exp025_noanneal_sweep.csv`
**Commit:** `b7750be7` · 3 lần chạy, seed 0, `L3DGS_NO_ANNEAL=1`

**Giả thuyết đang kiểm (của EXP-024):** attractor là do `ExponentialLR` mà PUP thêm vào tạo ra.
Nếu đúng thì *"vùng nén miễn phí"* là tính chất của **công thức tinh chỉnh**, không phải của **phép nén**.

**Phán quyết ghi trước khi chạy:** hệ số co ≤2 → do anneal · ≥5 → độc lập với anneal.

#### Kết quả

| cắt | \_\_\_\_ANNEAL BẬT\_\_\_\_ | | \_\_\_\_ANNEAL TẮT\_\_\_\_ | |
|---|---|---|---|---|
| | xuất phát | điểm dừng | xuất phát | điểm dừng |
| 20% | 21.8144 | 22.1757 | 21.8144 | 22.0034 |
| 50% | 21.5694 | 22.1787 | 21.5697 | 21.9963 |
| 70% | 20.2556 | 22.0118 | 20.2586 | 21.9068 |
| **trải** | **1.5588** | **0.1668** | **1.5557** | **0.0965** |
| **co lại** | | **9.3×** | | **16.1×** |

**Đo được 16.1× — vượt ngưỡng ≥5 đã ghi trước. GIẢ THUYẾT BỊ BÁC BỎ.**

Không những attractor tồn tại khi tắt anneal, nó còn **chặt hơn**. Đối chiếu với nhiễu seed:

| | trải điểm dừng | nhiễu seed (SD) | tỉ lệ |
|---|---|---|---|
| có anneal | 0.1668 | 0.0551 | 3.0× nhiễu — **có** phụ thuộc điểm xuất phát |
| không anneal | 0.0965 | 0.1065 | **0.9× nhiễu — KHÔNG phát hiện được phụ thuộc** |

Tắt anneal thì điểm xuất phát **hoàn toàn không còn ảnh hưởng** tới điểm dừng, ở mức nhiễu hiện có.

#### Anneal làm gì, nếu không tạo ra attractor

| | mức attractor | nhiễu seed (SD) |
|---|---|---|
| có anneal | 22.1221 | 0.0551 |
| không anneal | 21.9688 | 0.1065 |
| **anneal** | **nâng +0.1532 dB** | **giảm nhiễu 1.93×** |

Anneal **dịch chỗ** attractor và **làm nó ổn định hơn giữa các lần chạy**, chứ không **tạo ra** nó.
Hai phép đo độc lập khớp nhau: EXP-024 cho C1 không-anneal = +0.1587 ± 0.1065 (n=3, 0% cắt);
EXP-025 cho 21.9688 − 21.8157 = **+0.1531** (n=1 × 3 mức cắt). Sai khác 0.006.

#### 🔑 Phát hiện chính SỐNG SÓT, và mạnh hơn trước

Tại mức cắt 50%, seed 0, **tắt anneal**:

```
ngây thơ   = 21.9963 − 21.8157 = +0.1806
C1         = 21.9890 − 21.8157 = +0.1733     <- vẫn chiếm 96% của số ngây thơ
chi phí nén= 21.9963 − 21.9890 = +0.0073
```

Cấu trúc ảo giác **giữ nguyên khi bỏ anneal**, chỉ nhỏ đi về độ lớn. Nghĩa là:

> **Hiện tượng "cách so ngây thơ tính công của luyện thêm thành công của nén" KHÔNG phải
> artifact của lịch learning rate mà PUP thêm vào.** Nó tồn tại cả khi dùng lịch LR của 3DGS gốc.

Chuỗi phụ thuộc mà EXP-024 lo ngại — *"vùng miễn phí ← attractor ← anneal ← thứ PUP thêm vào"* —
**đứt ở mắt xích cuối**. Điều còn lại phụ thuộc công thức chỉ là **mức** của attractor (±0.15 dB),
không phải **sự tồn tại** của nó.

#### Tự phê bình

EXP-024 quan sát "SD tăng 1.93× khi tắt anneal" rồi tôi suy ra "anneal tạo ra attractor".
**Sai lầm: nhầm hai loại phân tán.** EXP-024 đo phân tán **giữa các seed** ở *một* điểm xuất phát;
attractor thì nói về phân tán **giữa các điểm xuất phát**. Hai đại lượng khác nhau, và chúng
đi **ngược chiều nhau**: bỏ anneal làm tăng nhiễu seed nhưng lại **giảm** phụ thuộc điểm xuất phát.
Suy luận từ cái này sang cái kia là không có cơ sở, và chỉ một thí nghiệm 50 phút mới lộ ra.

#### Còn thiếu

- EXP-025 là **n=1 mỗi mức cắt**. Kết luận "16.1×" dựa trên ba lần chạy đơn; con số chính xác
  không tin được, nhưng **hướng** thì chắc vì nó vượt ngưỡng rất xa và trải điểm dừng còn nhỏ hơn
  cả nhiễu seed.
- Vẫn **một scene**. Attractor chưa được kiểm trên `truck`.
- Vẫn chưa biết **biên miền hút** chính xác (nằm giữa 70% và 80% khi anneal bật).

---

### EXP-024 — ⚖️ **Anneal learning rate chiếm ~56% của C1 — nhưng n=3 chưa đủ chứng minh.** Điều chắc chắn: anneal là thứ TẠO RA attractor
**Ngày:** 2026-09-07 · **Script:** `scripts/run_noanneal.sh` · **Dữ liệu:** `experiments/exp024_noanneal.csv`
**Patch:** `L3DGS_NO_ANNEAL` — xem `docs/REPO_PATCHES.md` mục 2b
**Commit:** `1ee1c052` — **ba lần chạy đầu tiên của dự án gắn được với commit thật**

**Câu hỏi:** C1 = +0.3587 dB là "checkpoint chưa hội tụ", hay chỉ là phần thưởng một lần của
`ExponentialLR(gamma=0.95)` mà PUP thêm vào còn 3DGS gốc không có trong 30k bước đầu?

**Thiết kế:** nhánh đối chứng × 3 seed, khác EXP-018 **đúng một biến**: `L3DGS_NO_ANNEAL=1`.
Cổng chặn: nếu điểm dừng seed 0 trùng khít giá trị có-anneal thì cờ không ăn ⇒ loại.
Đã qua cổng (21.9890 ≠ 22.1799).

#### Kết quả

| seed | điểm dừng CÓ anneal | KHÔNG anneal | C1 có | C1 không |
|---|---|---|---|---|
| 0 | 22.179857 | 21.988998 | +0.3642 | +0.1733 |
| 1 | 22.116719 | 21.861285 | +0.3010 | +0.0456 |
| 2 | 22.226609 | 22.072802 | +0.4109 | +0.2571 |
| **TB** | | | **+0.3587 ± 0.0551** | **+0.1587 ± 0.1065** |

**Ước lượng điểm: anneal chiếm 55.8% của C1** (chênh +0.2000 dB).

**Nhưng chưa đủ ý nghĩa thống kê.** Welch: t = 2.889, df ≈ 3.0, t tới hạn 95% = 3.182.
Rơi vào dải "0.15–0.30 → đóng góp một phần, cần thêm seed" của dự đoán ghi trước. Sát ngưỡng,
không vượt. **Không được phát biểu "anneal chiếm hơn một nửa" như một kết luận** — mới là ước lượng điểm.

#### 🔑 Điều CHẮC CHẮN, và quan trọng hơn con số trên: anneal TẠO RA attractor

| | biên độ điểm dừng qua 3 seed |
|---|---|
| CÓ anneal | **0.1099 dB** |
| KHÔNG anneal | **0.2115 dB** |
| | **SD tăng 1.93 lần** |

Bỏ anneal thì điểm dừng **tán ra**, không còn tụ về một chỗ. Nghĩa là:

> **Attractor 22.1762 ± 0.0305 của EXP-023 không phải tính chất của scene hay của model —
> nó là tính chất của LỊCH LEARNING RATE mà công thức fine-tune này áp vào.**

Và vì attractor là thứ giải thích "nén 50% miễn phí" (EXP-023, hai nhánh bằng nhau vì cùng
rơi về một điểm dừng), nên chuỗi phụ thuộc là:

```
"nén tới 50% gần như miễn phí"
   <- vì cả hai nhánh rơi về cùng attractor
      <- vì anneal LR kéo chúng về đó
         <- mà anneal là thứ PUP THÊM VÀO, 3DGS gốc không có trong 30k bước đầu
```

⇒ **"Vùng miễn phí" có thể là tính chất của công thức tinh chỉnh, không phải của phép nén.**
Đây là phát biểu mạnh hơn và khó chịu hơn phát biểu cũ, và nó kiểm được.

#### Việc phải làm tiếp

1. **Thêm seed.** n=3 mỗi nhánh cho t=2.889, thiếu 0.29 so với ngưỡng. Thêm 2 seed mỗi nhánh
   (n=5, df≈8, t tới hạn ≈2.31) gần như chắc chắn kết luận được — **4 lần chạy, ~70 phút**.
2. **Kiểm attractor không-anneal.** Nếu attractor thật sự do anneal tạo ra thì chạy các mức cắt
   khác nhau với `L3DGS_NO_ANNEAL=1` phải cho điểm dừng **tán ra theo điểm xuất phát**,
   thay vì tụ lại. Đó là phép thử trực tiếp, và nó **có thể thất bại**.

#### Ghi chú kỹ thuật

- VRAM 1847.6 MiB ở cả 3 lần chạy (đo trong tiến trình, tất định). Cao hơn nhánh có anneal
  (1691.7) — chưa rõ vì sao, đáng xem lại; có thể do LR cao hơn làm Gaussian bung rộng hơn.
- ⚠️ `run_meta.json` ghi `commit 1ee1c052` nhưng `dirty: true` — vì **chính lần chạy ghi file vào
  `experiments/`** làm bẩn cây. Đây là khiếm khuyết của `stamp_run.sh`: phép kiểm dirty nên
  **loại trừ `experiments/`**, nếu không thì không lần chạy nào có thể sạch.

---

### EXP-023 — 🔑 **ĐIỂM DỪNG LÀ MỘT ATTRACTOR.** Phải đọc lại C1 và C2 theo cách khác
**Ngày:** 2026-09-07 · **Nguồn:** phân tích lại dữ liệu đã có, **không chạy thêm lần nào**
**Phát hiện bởi:** workflow phản biện thiết kế thí nghiệm exp017 — nó bác luôn thí nghiệm đó

**Quan sát.** Gộp mọi lần chạy trên `train`, ghép cặp "điểm xuất phát → điểm dừng":

| cắt | xuất phát | điểm dừng | phục hồi |
|---|---|---|---|
| 0% (checkpoint gốc) | 21.8157 | 22.1799 / 22.1167 / 22.2266 | +0.36 |
| 20% | 21.8144 | **22.1757** | +0.3613 |
| 40% | 21.7463 | **22.1663** | +0.4200 |
| 50% | 21.5694 | **22.1787** | +0.6093 |
| 60% | 21.1379 | **22.1308** | +0.9928 |
| 66% | 20.6676 | **22.0764** | +1.4089 |
| 70% | 20.2556 | **22.0118** | +1.7563 |
| 80% | 18.8438 | 21.6612 | +2.8174 |
| 90% | 16.7321 | 20.8177 | +4.0856 |
| 95% | 15.1066 | 19.7303 | +4.6236 |

**Nhóm 20–70%: điểm xuất phát trải 1.5588 dB, điểm dừng chỉ trải 0.1668 dB — co lại 9.3 lần.**
Trung bình điểm dừng của 9 lần chạy ở N ≥ 513k: **22.1762 ± 0.0305 dB**.
Từ 80% trở đi attractor gãy (21.66 / 20.82 / 19.73).

→ **Công thức tinh chỉnh này kéo mọi thứ về cùng một điểm dừng, gần như bất kể xuất phát từ đâu,
miễn là còn trong miền hút.**

#### Ba con số phải đọc lại

**1. C1 không phải "giá trị của việc luyện thêm".**
```
attractor 22.1762 − checkpoint gốc 21.8157 = 0.3605
C1 đo được                                 = 0.3587   (khớp trong SD 0.0305)
```
C1 chỉ là **khoảng cách từ checkpoint công bố tới điểm dừng của công thức này**.
Không phải "train thêm thì tốt lên" — mà "checkpoint gốc chưa nằm ở chỗ công thức này đưa nó tới".

**2. "Nén 50% miễn phí" có lời giải thích đơn giản hơn.**
Chi phí đo được +0.0080; SD của chính attractor là 0.0305. Con số đó **nằm trong nhiễu của attractor**.
Hai nhánh bằng nhau không phải vì nén vô hại, mà vì **cả hai rơi về cùng một điểm dừng**.

**3. Điều này giải thích luôn ĐẦU GỐI.** Vùng "gần như miễn phí" chính là **miền hút của attractor**;
cắt sâu quá thì văng ra khỏi miền đó. Đầu gối không phải ngưỡng của "thông tin bị mất" mà là
**biên của miền hút** — một phát biểu cơ chế, kiểm được, thay cho một quan sát.

**4. Chênh 7.6× ở `truck-864k` có lời giải tầm thường hơn.** Model đó **chính là đầu ra** của cùng
công thức nên đã nằm sẵn trên attractor của nó; liều thứ hai mua gần như không gì.
Đó không phải "đã hội tụ" theo nghĩa tổng quát mà là **công thức này gần như luỹ đẳng**.

#### 🔴 Hệ quả: thí nghiệm exp017 bị BÁC BỎ TRƯỚC KHI CHẠY

Kế hoạch trước đó (chạy lại nhánh đối chứng trên `exp017_train_noprune@35000`) hỏng vì hai lẽ:
- Model đó **đã nằm trên attractor** (22.1736, lệch attractor 0.0026) ⇒ C1′ ≈ **0.003**, biết trước
  từ dữ liệu đã có. 45 phút GPU mua ~0 thông tin.
- Với n=3, nửa KTC 95% = 4.303 × 0.0551/√3 = **±0.137 dB**, **rộng hơn cả dải dự đoán 0.05–0.15**.
  Thí nghiệm không đủ lực để phân biệt hai giả thuyết mà nó định phân biệt.

#### 🔴 Biến chống đỡ luận điểm chính mà TỪ ĐẦU CHƯA AI KIỂM: anneal learning rate

`prune_finetune.py:81,124-125` tạo `ExponentialLR(gamma=0.95)` và gọi `scheduler.step()` mỗi 400
bước — 12 lần trong 5000 bước, đưa LR cuối về **0.54×**. **3DGS gốc KHÔNG anneal**
`feature_lr / opacity_lr / scaling_lr / rotation_lr` trong 30k bước đầu (chỉ `xyz` có lịch riêng).

Nên +0.3587 có thể **không phải** "checkpoint chưa luyện đủ" mà là **phần thưởng một lần của anneal**
mà giai đoạn fine-tune tự mang theo. Hai cách hiểu suy rộng **ngược chiều nhau**:

| cách hiểu | suy rộng thành |
|---|---|
| "checkpoint của các bài báo chưa hội tụ" | phê phán checkpoint của người khác |
| "bản thân giai đoạn fine-tune tặng +0.36 dB cho bất kỳ ai thêm nó" | confound nằm trong **phương pháp**, không phải checkpoint |

→ Đây là thứ EXP-024 phải kiểm.

#### Giới hạn của chính EXP-023

- Attractor mới quan sát trên **một scene** (`train`), một công thức, một khoảng N.
- "Miền hút" chưa được xác định biên chính xác — mới biết nó nằm giữa 70% và 80%.
- Toàn bộ suy ra từ dữ liệu có sẵn; chưa có lần chạy nào **thiết kế riêng** để kiểm attractor.
  Phép thử rẻ đã xác định: chạy từ `datasets/pretrained/models/train/point_cloud/iteration_7000`
  (N=559,263, có sẵn) — xa hội tụ hơn hẳn, cùng scene, N trong dải đã kiểm.

---

### EXP-022 — ✅ **SCENE THỨ HAI: cả ba phát biểu tái lập. Và ảo giác trở thành thứ DỰ ĐOÁN ĐƯỢC**

> ### ⛔ ĐÍNH CHÍNH 2026-09-07 — phần "DỰ ĐOÁN ĐƯỢC" trong tiêu đề là SAI
> Quan hệ `ảo giác ⟺ lợi ích luyện thêm > chi phí nén` là **đồng nhất thức đại số**,
> không phải quy luật thực nghiệm. Ba đại lượng cùng định nghĩa trên một baseline nên
> `ngây thơ ≡ trung thực + C1` luôn đúng. Kiểm số: sai khác **5.55e-17** (train) và **0**
> (truck-864k) — tức là đúng tới độ chính xác máy, ở mọi bộ dữ liệu.
> **Không thí nghiệm nào bác bỏ được nó**, nên nói nó là "cơ chế bác bỏ được" là ngược.
>
> **Phần vẫn đứng vững, và mới là nội dung thực nghiệm:** *độ lớn* của C1
> (+0.3587 ± 0.0551 dB trên `train`) — nó **có thể đã gần 0** và không hề gần 0, đủ lớn
> để đổi dấu con số ngây thơ. Cùng với mức co lại 7.6× trên model đã hội tụ.
>
> **Giới hạn mới phải nêu:** chênh lệch 7.6× lẫn **ba biến** (scene, số Gaussian, trạng thái
> hội tụ) đổi cùng lúc. Phép tách rẻ, checkpoint đã có sẵn: chạy lại cặp đối chứng trên
> `experiments/exp017_train_noprune/point_cloud/iteration_35000/point_cloud.ply`
> — cùng scene, cùng N=1,026,508, chỉ khác trạng thái hội tụ. Dự đoán ghi trước:
> nếu C1' vẫn ~0.36 dB thì phát biểu "confound là hàm của khoảng cách tới hội tụ" **SAI**.
**Ngày:** 2026-09-07 · **Script:** `scripts/run_scene2.sh` · **Phân tích:** `tools/cross_scene.py`
**Dữ liệu:** `experiments/scene2_truck864.csv` · 6 lần chạy (đối chứng×3, cắt 50%×3)

**Thiết lập:** model nền = `truck` đã prune còn **864.017 hạt** (đầu ra EXP-014 @35000,
sha256 `5c837cbf…`). Baseline tái lập **25.113122522830963** qua cả 6 lần chạy — trùng khít
giá trị EXP-014 ghi ở iteration 35000 ⇒ cổng xuất xứ đạt, đúng model.

**Vì sao không dùng `truck` nguyên bản:** ba khoản bộ nhớ lớn nhất **độc lập độ phân giải**
— tham số+grad+Adam 2288 MiB, `dL_dsh` = `torch::zeros({P,16,3})` 465 MiB (`rasterize_points.cu:157`),
context 62 MiB ⇒ **sàn cứng 2815 MiB** trên trần 3440, chỉ còn 625 MiB. Không `-r` nào cứu được.
Khoản `dL_dsh` là thứ mô hình bộ nhớ cũ của dự án **thiếu hẳn**.

#### C1 — hiệu ứng "luyện thêm" TEO 7.6 LẦN trên model đã hội tụ

| | `train` (checkpoint gốc, chưa tinh chỉnh) | `truck-864k` (đã tinh chỉnh 5000 bước) |
|---|---|---|
| **PSNR** | **+0.3587 ± 0.0551** | **+0.0475 ± 0.0110** |
| SSIM | +0.00506 ± 0.00011 | +0.00065 ± 0.00025 |
| LPIPS | −0.00504 ± 0.00014 | −0.00137 ± 0.00015 |

Nhất quán ở **cả ba** thước đo (7.6× / 7.8× / 3.7×) ⇒ không phải hiện tượng riêng của PSNR.

→ **Confound KHÔNG phải hằng số. Nó là hàm của việc checkpoint còn cách hội tụ bao xa.**
Đây là phát biểu **cơ chế**, mạnh hơn con số "+0.3587 dB" của EXP-016/017.

#### 🔑 C2 — và đây là kết quả sắc nhất của cả dự án

| | ngây thơ (so checkpoint gốc) | trung thực (so đối chứng) | có ảo giác? |
|---|---|---|---|
| `train` | **+0.3667** ← "nén còn làm tốt hơn!" | +0.0080 | **CÓ** |
| `truck-864k` | **−0.2471** ← đã âm sẵn | −0.2946 | **KHÔNG** |

**Trên scene 2, ảo giác không xuất hiện.** Cách so ngây thơ ở đó đã cho số âm rồi.
Ghép với C1 thì ra một **quy tắc dự đoán**:

> **Ảo giác xuất hiện khi và chỉ khi: lợi ích của luyện thêm > chi phí của nén.**
> `train`      : +0.3587 > 0.0080  → có ảo giác
> `truck-864k` : +0.0475 < 0.2946  → không có

Ta không chỉ *đo* được ảo giác nữa, mà **nói trước được khi nào nó xảy ra**: nó sống trên
checkpoint chưa hội tụ ở tỉ lệ nén thấp, và biến mất khi model đã hội tụ hoặc nén đủ sâu.
Đây chính là vùng vận hành mà phần lớn bài nén 3DGS báo cáo.

#### C3 — thứ tự thước đo tái lập trên CẢ HAI scene

Tỉ lệ tín hiệu/nhiễu của chi phí nén (|hiệu| / độ lệch chuẩn):

| | `train` | `truck-864k` |
|---|---|---|
| PSNR | **0.2×** | 58.0× |
| SSIM | 7.9× | 153.0× |
| LPIPS | **64.9×** | **378.5×** |

**LPIPS > SSIM > PSNR ở cả hai scene.** LPIPS nhạy hơn PSNR 325× (scene 1) và 6.5× (scene 2).
Khoảng cách hẹp lại ở scene 2 chỉ vì hiệu ứng ở đó quá lớn (−0.295 dB) nên PSNR cũng thấy được.

#### Prune trên model đã prune thì đắt hơn nhiều

| | thiệt hại ngay sau cắt 50% | phục hồi lấy lại | ròng |
|---|---|---|---|
| `train` (từ 1.03M) | −0.2463 | +0.6093 | +0.3630 |
| `truck-864k` (từ 864k, đã cắt 66%) | **−3.0951** | **+2.8479** | −0.2471 |

Cắt cùng 50% nhưng thiệt hại **gấp 12.6 lần**, và phục hồi tuy gắng gấp 4.7 lần vẫn không bù nổi.
Model đã sát giới hạn dung lượng của nó.

#### Tài nguyên — lần đầu có số ĐO TỪ TRONG TIẾN TRÌNH

| nhánh | NVML (lấy mẫu) | `max_memory_allocated` | chênh | thời gian |
|---|---|---|---|---|
| đối chứng (864k) | 1903–1923 MiB | **1691.7 MiB** | ~+220 | 937–998 s |
| nén (432k) | 1583 MiB | **1404.9 MiB** | ~+178 | 802–848 s |

Con số trong tiến trình **giống hệt nhau qua mọi seed** (1691.7 / 1404.9) ⇒ cấp phát tất định.
NVML báo cao hơn vì nó đo *reserved* (pool của allocator), không phải *allocated*.
Đo qua `sitecustomize.py` + `atexit` đặt **ngoài repo** (`PYTHONPATH`) ⇒ **0 dòng sửa repo**;
phải dùng `atexit` vì `prune_finetune.py` có `return` ngay trong vòng lặp.

#### Lỗi đã mắc trong loạt này

**Cổng B4 loại oan cả 3 nhánh nén.** Viết `[ "$NEVAL" -eq 2 ]`, nhưng nhánh nén eval **3 lần**
(trước prune @30001, sau prune @30001, và @35000) — điều đã nằm rõ trong `metric.csv` của
EXP-014 từ lâu. Các lần chạy hoàn toàn tốt, chỉ là không được ghi vào CSV; đã gom lại từ
`metric.csv`, **không phải chạy lại**. Cổng đã sửa: đối chứng=2, nén=3.

#### Còn thiếu

- **Hai scene vẫn là ít.** Và scene 2 là model *đã nén*, không phải checkpoint gốc — chọn thế
  vì đó là model hợp lệ duy nhất chạy nổi cặp đối chứng trên card 4 GB. Phải nói rõ điều này,
  đừng trình bày như hai checkpoint gốc độc lập.
- Chỉ **một tỉ lệ nén (50%)** trên scene 2. Chưa có đường cong.
- Quy tắc dự đoán ảo giác mới có **hai điểm dữ liệu**. Nó đúng ở cả hai, nhưng hai điểm thì
  chưa đủ để gọi là quy luật — cần một điểm nữa nằm ở vùng ranh giới để thử bác bỏ.

---

### EXP-021 — 🔴 **Ba thước đo KHÔNG đồng ý đầu gối ở đâu. "Nén miễn phí tới 2×" phải rút lại**
**Ngày:** 2026-09-06 · **Script:** `scripts/run_knee_seeds.sh` · **Gộp:** `tools/knee_errorbars.py`
**Chạy:** 4 lần (50% và 60%, seed 1 và 2) · ~52 phút · N khớp dự đoán 4/4

**Mục đích:** đường cong EXP-019 chỉ có seed 0. Ranh giới "miễn phí ↔ tốn" nằm giữa 50% và 60%,
mà điểm 60% mới ở mức 2.4σ. Thêm seed để chốt. 66% đã có sẵn 3 seed từ EXP-018.

**Đổi cách tính cho đúng:** trước đây so với **trung bình** đối chứng 3 seed.
Nay **ghép cặp theo seed** (đối chứng seed k trừ nén seed k) — đúng theo phát hiện EXP-018
rằng cùng seed tác động lên cả hai nhánh như nhau. Cùng một điểm ra số hơi khác:
50% từ +0.0043 (so trung bình) thành **−0.0012** (ghép cặp). Ghép cặp là ước lượng đúng hơn.

#### Chi phí nén, ghép cặp theo seed, n=3, KTC 95% (t Student, 2 bậc tự do)

| cắt | **PSNR** | KTC | **SSIM** | KTC | **LPIPS** | KTC |
|---|---|---|---|---|---|---|
| 50% | +0.0080 ± 0.0324 | [−0.073, +0.088] ⟵ **chứa 0** | −0.00158 ± 0.00020 | [−0.00208, −0.00108] **TỐN** | +0.00573 ± 0.00009 | [+0.00551, +0.00595] **TỐN** |
| 60% | −0.0343 ± 0.0267 | [−0.101, +0.032] ⟵ **chứa 0** | −0.00608 ± 0.00012 | [−0.00636, −0.00579] **TỐN** | +0.01478 ± 0.00007 | [+0.01460, +0.01495] **TỐN** |
| 66% | −0.0881 ± 0.0181 | [−0.133, −0.043] **TỐN** | −0.01186 ± 0.00006 | **TỐN** | +0.02447 ± 0.00015 | **TỐN** |

#### 🔴 Kết luận 1 — ba thước đo cho ba câu trả lời khác nhau

| thước đo | đầu gối nằm ở đâu |
|---|---|
| PSNR | giữa **60% và 66%** |
| SSIM | **trước 50%** — đã tốn ở mọi mức đo |
| LPIPS | **trước 50%** — đã tốn ở mọi mức đo |

**PSNR nói nén miễn phí tới 60%. SSIM và LPIPS nói đã phải trả giá từ 50%, và nói CHẮC CHẮN**
(KTC cách xa 0; LPIPS ở 50% là +0.00573 với nhiễu 0.00015 — **38σ**).

#### 🔴 Kết luận 2 — RÚT LẠI phát biểu "nén miễn phí tới 2.00×"

Phát biểu đó (EXP-019, ghi trong `NEXT.md` và trong hình đã dựng) dựa **chỉ vào PSNR**.
Sai theo hai cách:
1. **Nhầm "chưa chứng minh được khác 0" thành "bằng 0".** Ước lượng điểm ở 60% là −0.0343,
   không phải 0. Với n=3, t=4.303 nên KTC rất rộng — đây là **thiếu lực thống kê**, không phải
   bằng chứng của "không có hiệu ứng".
2. **Dùng đúng cái thước tệ nhất.** Chính EXP-018/019 đã chứng minh PSNR là thước ồn nhất;
   vậy mà kết luận "miễn phí" lại rút ra từ nó.

→ **Phát biểu đúng:** *không có mức nén nào thực sự miễn phí; PSNR chỉ là không nhìn thấy hoá đơn.*
Theo LPIPS, vùng gần như không mất gì chỉ kéo tới khoảng **20–30% (~1.3×)**
(seed 0: 20% → −0.00009 ≈ 0; 40% → +0.00156 = **10σ**).

#### Kết luận 3 — bằng chứng mạnh thêm cho "PSNR mù"

Đây là lần thứ ba cùng một hiện tượng, nay có thanh sai số ở cả ba mức:
tỉ lệ tín hiệu/nhiễu của LPIPS lớn hơn PSNR khoảng **30–60 lần**. PSNR **không đủ lực**
để trả lời câu hỏi trung tâm của bài (đầu gối ở đâu) ngay cả với 3 seed —
trong khi LPIPS trả lời dứt khoát chỉ với 3 seed.

#### Phải nói rõ, đừng thổi phồng

- **"Có ý nghĩa thống kê" ≠ "mắt người thấy được".** LPIPS ở 50% tăng 0.00573 trên nền 0.21521,
  tức **+2.7% tương đối**. Là thật, nhưng nhỏ. Chưa làm khảo sát người xem nên **không được
  phát biểu là "nhìn thấy khác biệt"**.
- **n=3 vẫn ít.** Độ lệch chuẩn PSNR ở 50% (0.0324) và 60% (0.0267) còn *lớn hơn* ở 66% (0.0181)
  — với 2 bậc tự do thì bản thân ước lượng độ lệch chuẩn cũng rất nhiễu. Muốn chốt đầu gối
  theo PSNR cần n≥8. Nhưng **không cần**: SSIM/LPIPS đã chốt xong với n=3.
- Vẫn **một scene** (`train`).

---

### EXP-020 — 🔻 **Giảm bậc màu THUA cắt hạt ở mọi tỉ lệ nén.** Bác bỏ kỳ vọng ban đầu về hướng B
**Ngày:** 2026-09-05 · **Script:** `scripts/run_sh_probe.sh` · `tools/measure_sh_truncation.py`
**Dữ liệu:** `experiments/sh_truncation.csv` · scene `train`, 38 view test, không huấn luyện

**Giả thuyết đang kiểm:** 45/59 số mỗi hạt (76%) là hệ số màu phụ thuộc góc nhìn.
Bỏ chúng cho tỉ lệ nén tới 4.2× trên tham số — **lớn hơn cả cắt hạt 66%**.
Tôi đã xếp đây là "đòn mạnh nhất chưa dùng". **Số liệu bác bỏ điều đó.**

**Cách đo:** `render()` truyền `sh_degree=pc.active_sh_degree` xuống rasterizer, nên đặt
`active_sh_degree = d` là mô phỏng **chính xác** việc chỉ lưu tới bậc d. Dùng đúng bộ hàm đo
của repo. **Cổng tự kiểm:** bậc 3 cho PSNR `21.8157` — trùng baseline đã biết ⇒ đường đo đúng.

#### Thiệt hại thuần của việc cắt bậc màu (chưa phục hồi)

| bậc | float/hạt | MB | nhỏ hơn | ΔPSNR | ΔSSIM | ΔLPIPS |
|---|---|---|---|---|---|---|
| 3 | 59 | 254.57 | 1.00× | — | — | — |
| 2 | 38 | 168.35 | 1.51× | **−0.795** | −0.01397 | +0.01126 |
| 1 | 23 | 106.76 | 2.38× | **−1.681** | −0.03254 | +0.02632 |
| 0 | 14 | 69.80 | 3.65× | **−2.308** | −0.04805 | +0.03854 |

#### 🔴 So ở CÙNG tỉ lệ nén, cắt hạt thắng đậm

Thiệt hại của cắt hạt nội suy từ EXP-019 tại đúng các tỉ lệ trên:

| tỉ lệ nén | cắt bậc màu | cắt hạt | chênh |
|---|---|---|---|
| 1.51× | −0.795 | **−0.043** | cắt hạt ít hại hơn **18×** |
| 2.38× | −1.681 | **−0.574** | ít hại hơn **2.9×** |
| 3.65× | −2.308 | **−1.831** | ít hại hơn **1.26×** |

**Kết luận 1: trên trục kích-thước-đổi-chất-lượng, cắt hạt trội hơn cắt bậc màu ở mọi mức đo được.**
Xếp hạng ban đầu của tôi cho hướng B là **sai**, và sai vì suy từ *tỉ lệ nén danh nghĩa*
(76% dữ liệu!) chứ không từ *chất lượng mất đi*. Đúng loại lỗi đã ghi ở EXP-018:
lấy con số to nhất làm bằng chứng thay vì đo.

**Kết luận 2 (đáng chú ý): khoảng cách THU HẸP rất nhanh khi tỉ lệ tăng** — 18× → 2.9× → 1.26×.
Thiệt hại của cắt hạt **tăng tốc** (tăng dần: 0.18 → 0.43 → 0.47 → 1.41 dB mỗi nấc),
còn của cắt bậc màu **giảm tốc** (0.795 → 0.886 → 0.627). Rất có thể có điểm giao **quá 3.65×**.
Nhưng bậc 0 là **hết đường** của trục màu — không đi xa hơn được.
⇒ Chiến lược hợp lý: **dùng cắt hạt cho phần rẻ (miễn phí tới 2.00×), rồi mới cân nhắc trục màu.**

#### Điều EXP-020 KHÔNG bác bỏ

- **Chưa thử phục hồi.** Cắt hạt 66% trước phục hồi cũng xấu (−1.148) rồi về −0.098.
  Cắt bậc màu có thể cũng hồi được. Nhưng có **bất đối xứng thật**: cắt hạt bỏ phần *thừa*,
  các hạt còn lại học bù được; cắt bậc màu bỏ mất một *khả năng* — ở bậc 0 thì phản quang
  không còn biểu diễn được nữa, tinh chỉnh bao nhiêu cũng không tạo lại. Dự đoán: hồi kém hơn.
- **Chưa thử ghép hai trục.** Chúng **nhân** với nhau: cắt hạt 50% (miễn phí, 2.00×) + bậc 2 (1.51×)
  = 3.02×. Chưa đo.
- **Trục màu vẫn giữ một lợi thế RIÊNG mà cắt hạt không có: bộ nhớ lúc nạp.**
  Nó giảm chi phí *mỗi hạt* mà không đổi N, nên **bỏ được ngay lúc đọc file, trên CPU,
  trước khi đưa lên card** — 956 → 236 B/hạt khi fine-tune. Đây vẫn là con đường khả dĩ nhất
  cho `drjohnson` (3.41M hạt, hiện không nạp nổi — EXP-015 hệ). Muốn cắt hạt thì phải nạp được đã;
  còn cắt bậc màu thì không cần.

#### Cảnh báo về số liệu

Đỉnh VRAM đo được **1079 MiB ở cả 4 bậc** — vì phép đo chỉ đổi `active_sh_degree`,
không thực sự bỏ tensor. **Mọi con số MB/tỉ lệ ở đây là TÍNH TOÁN, không phải đo.**
Chúng đúng theo định nghĩa (đếm float), nhưng phần tiết kiệm bộ nhớ thật chưa được kiểm chứng.
(Đỉnh VRAM lấy bằng `torch.cuda.max_memory_allocated()` từ trong tiến trình — đúng theo
việc-phải-làm ở EXP-015 kết luận 3, lần đầu áp dụng.)

---

### EXP-019 — 📈 **ĐƯỜNG CONG ĐÁNH ĐỔI.** Đầu gối ở 50–60%; vùng ảo giác 40–70%
**Ngày:** 2026-09-05 · **Script:** `scripts/run_sweep.sh` · **Phân tích:** `tools/tradeoff_curve.py`
**Dữ liệu:** `experiments/sweep_train.csv` · `experiments/tradeoff.json`
**Hình:** `docs/tradeoff.html` → https://claude.ai/code/artifact/c3f8262d-fc69-45e1-89bf-8fe70732eff0

**Thiết kế tiết kiệm một nửa số lần chạy:** nhánh đối chứng **không phụ thuộc tỉ lệ cắt**
(không cắt gì thì bàn % là vô nghĩa), nên chỉ chạy các nhánh prune và dùng lại đối chứng
của EXP-018. 8 mức mới + điểm 0.66 sẵn có = **9 điểm**, seed 0, ~96 phút.

**9/9 dự đoán N khớp CHÍNH XÁC** (công thức `N - (int(p*(N-1))+1)` từ `gaussian_model.py:405`).

#### ⚠️ Điều chỉnh quan trọng về cách vẽ

Ban đầu định vẽ **hai đường** "ngây thơ" và "trung thực". **Sai về mặt thông tin:**
```
ngây thơ − trung thực = (đối chứng − gốc) = +0.3587 dB, HẰNG SỐ với mọi tỉ lệ
```
Hai "đường" đó là **cùng một đường dịch đi**, không phải hai phép đo độc lập.
Vẽ chúng cạnh nhau sẽ gợi ý sai rằng có hai xu hướng.
→ Cách vẽ đúng: **một đường chất lượng, hai vạch mốc.** Thông tin nằm ở chỗ *vạch số 0 đặt ở đâu*.

#### Kết quả

| cắt | hạt còn lại | MB | nén | sau cắt | phục hồi | so gốc | **so đối chứng** |
|---|---|---|---|---|---|---|---|
| 20% | 821,206 | 203.66 | 1.25× | 21.814 | 22.176 | +0.360 | **+0.001** ✅ |
| 40% | 615,905 | 152.75 | 1.67× | 21.746 | 22.166 | +0.351 | **−0.008** ✅ |
| 50% | 513,254 | 127.29 | 2.00× | 21.569 | 22.179 | +0.363 | **+0.004** ✅ |
| **60%** | 410,603 | 101.83 | 2.50× | 21.138 | 22.131 | +0.315 | **−0.044** ⟵ đầu gối |
| 66% | 349,013 | 86.56 | 2.94× | 20.668 | 22.076 | +0.261 | **−0.098** |
| 70% | 307,953 | 76.37 | 3.33× | 20.256 | 22.012 | +0.196 | **−0.163** |
| 80% | 205,302 | 50.92 | 5.00× | 18.844 | 21.661 | −0.155 | **−0.513** |
| 90% | 102,651 | 25.46 | 10.0× | 16.732 | 20.818 | −0.998 | **−1.357** |
| 95% | 51,326 | 12.73 | 20.0× | 15.107 | 19.730 | −2.085 | **−2.444** |

**Đầu gối ở giữa 50% và 60%.** Ba mức 20/40/50% nằm **trong** dải nhiễu ±0.018 dB (EXP-018)
→ nén tới 2× thực sự **không mất gì**. Từ 60% trở đi ra khỏi dải nhiễu.

**Vùng ảo giác 40–70%:** cột "so gốc" **dương** (báo cáo kiểu thường sẽ nói "nén còn tốt hơn")
trong khi "so đối chứng" đã **âm**. Cách so ngây thơ nói "miễn phí" đến tận **70%**,
sự thật chỉ miễn phí đến **50%**.

#### 🔎 PSNR mù ở chỗ LPIPS thấy rõ

Số lần độ lệch chuẩn (nhiễu ghép cặp từ EXP-018: PSNR 0.0181 · SSIM 0.00006 · LPIPS 0.00015):

| cắt | PSNR | SSIM | LPIPS |
|---|---|---|---|
| 40% | 0.4σ — không thấy | 1.7σ — không thấy | **10.4σ — thấy rõ** |
| 50% | 0.2σ — không thấy | **22.5σ** | **37.3σ** |
| 60% | 2.4σ — ranh giới | **99σ** | **98σ** |
| 66% | **5.4σ** | **196σ** | **162σ** |

**LPIPS phát hiện suy giảm từ 40%, PSNR phải tới 66% — chậm hơn 26 điểm phần trăm.**
Mà PSNR mới là con số hầu hết bài nén 3DGS đặt lên tiêu đề.
→ Củng cố kết luận phụ của EXP-018, nay có 9 điểm dữ liệu thay vì 1.

#### Phục hồi: càng phá nhiều càng gắng, nhưng không bao giờ về lại

| cắt | prune lấy đi | phục hồi trả lại | ròng |
|---|---|---|---|
| 20% | −0.001 | +0.361 | +0.360 |
| 50% | −0.246 | +0.609 | +0.363 |
| 66% | −1.148 | +1.409 | +0.261 |
| 80% | −2.972 | +2.817 | −0.155 |
| 95% | −6.709 | +4.624 | −2.085 |

#### Còn thiếu

- **Đường cong chỉ có seed 0.** Thanh sai số lấy từ EXP-018 tại 0.66 và giả định áp cho mọi điểm.
  Điểm 60% (−0.044, tức 2.4σ) là **ranh giới** — cần thêm seed để chốt.
- Một scene. Chưa biết đầu gối dịch đi thế nào ở scene khác.
- Chưa ghép với các trục nén khác (giảm bậc SH, lượng tử hoá) — chúng nhân với nhau chứ không cộng.

---

### EXP-018 — ✅ **THANH SAI SỐ: kết luận của EXP-016/017 đứng vững.** Và PSNR là chỉ số ồn nhất
**Ngày:** 2026-09-04 · **Script:** `scripts/run_repeats.sh` · **Gộp:** `tools/aggregate_repeats.py`
**Dữ liệu:** `experiments/repeats_train.csv` · 6 lần chạy (3 seed × 2 nhánh) + cặp gốc EXP-016/017
**Patch cần có:** `L3DGS_SEED` trong `utils/general_utils.py` — xem `docs/REPO_PATCHES.md` mục 2

**Vì sao cần:** EXP-016/017 kết luận "nén tốn −0.1059 dB" từ **đúng một** lần chạy mỗi nhánh.
Con số đó nhỏ; chưa có gì bảo đảm nó không phải nhiễu.

#### Tách được hai nguồn biến thiên

| nguồn | cách đo | độ lớn (PSNR) |
|---|---|---|
| **Nhân CUDA không tất định** | seed 0 chạy hai lần | **~0.006–0.009 dB** |
| Seed (thứ tự ảnh) — giá trị **tuyệt đối** | độ lệch chuẩn qua seed 0/1/2 | **~0.055 dB** |
| Seed — **hiệu ghép cặp** (prune − control) | như trên | **~0.018 dB** |

→ **Thiết kế ghép cặp có tác dụng thật:** cùng một seed ảnh hưởng lên cả hai nhánh như nhau,
nên lấy hiệu trong cùng seed **triệt tiêu 2/3 nhiễu** (0.055 → 0.018).

#### Kết quả có thanh sai số (n=3 seed, KTC 95% dùng t Student)

| | do **luyện thêm** | **CHI PHÍ NÉN** | KTC 95% chi phí | chứa 0? |
|---|---|---|---|---|
| **PSNR** | **+0.3587 ± 0.0551** | **−0.0881 ± 0.0181** | [−0.1330, −0.0431] | **KHÔNG** |
| **SSIM** | +0.00506 ± 0.00011 | **−0.01186 ± 0.00006** | [−0.01200, −0.01172] | **KHÔNG** |
| **LPIPS** | −0.00504 ± 0.00014 | **+0.02447 ± 0.00015** | [+0.02410, +0.02484] | **KHÔNG** |

**Cả ba chỉ số: khoảng tin cậy không chứa 0. Nén thực sự làm chất lượng giảm.**
Và **+0.3587 dB do luyện thêm vẫn lớn gấp ~4 lần chi phí nén** — kết luận cốt lõi của
EXP-016/017 không những đứng vững mà còn được củng cố.

#### 🔎 Phát hiện phụ (đáng giá độc lập): **PSNR là chỉ số ồn nhất**

Tỉ lệ tín hiệu/nhiễu của chi phí nén, tính bằng |hiệu| / độ lệch chuẩn:

| chỉ số | hiệu | độ lệch chuẩn | **tín hiệu/nhiễu** |
|---|---|---|---|
| PSNR | −0.0881 | 0.0181 | **4.9** |
| SSIM | −0.01186 | 0.00006 | **198** |
| LPIPS | +0.02447 | 0.00015 | **163** |

**SSIM và LPIPS phân biệt tốt hơn PSNR khoảng 30–40 lần** trong thí nghiệm này.
Nghịch lý: PSNR là con số mà gần như mọi bài nén 3DGS đặt lên tiêu đề, nhưng ở đây
nó **ồn nhất và dễ ru ngủ nhất** — chính PSNR là chỗ hiệu ứng "nén miễn phí" giả tạo
xuất hiện, trong khi SSIM/LPIPS lộ ngay chi phí thật với thanh sai số cực hẹp.
→ Đề xuất cho báo cáo cuối: **luôn kèm SSIM và LPIPS có thanh sai số**, không báo PSNR trần.

#### Quỹ đạo đầy đủ (trung bình 3 seed)

```
21.8157  baseline
20.6679  ngay sau prune 66%        (-1.1478 ± 0.0028)   <- rất chặt: prune gần như tất định
22.0862  sau phục hồi 5000 bước    (+1.4184)
22.1744  ĐỐI CHỨNG: không nén, cũng 5000 bước   <- CAO HƠN nhánh nén
```

> **Đính chính (cùng ngày):** dòng đối chứng ban đầu ghi `22.2277` — đó là giá trị của
> **riêng seed 2**, không phải trung bình 3 seed. Đúng là **22.1744**. Chi phí nén −0.0881
> vốn đã tính từ trung bình đúng nên **không kết luận nào thay đổi**; chỉ con số hiển thị sai.

#### Kiểm tra vệ sinh

`baseline@30001` qua **cả 6 lần chạy**: biên độ **0.00e+00** — giống hệt tới từng bit.
Đúng như phải thế (chưa qua bước tối ưu nào), và xác nhận eval là tất định.
`N` ổn định tuyệt đối: control 1,026,508 · prune 349,013 ở mọi seed.

#### Tài nguyên (ổn định qua các lần chạy)

| nhánh | VRAM đỉnh (NVML 1 Hz, **chặn dưới**) | thời gian |
|---|---|---|
| control (1.03M) | 2023 / 2043 / 2023 MiB | 884 / 881 / 880 s |
| prune (349k) | 1663 / 1643 / 1643 MiB | 723 / 728 / 724 s |

#### Còn thiếu

- **n=3 vẫn ít.** KTC dùng t với 2 bậc tự do nên rất rộng (t=4.303). PSNR đủ để bác bỏ 0,
  nhưng để nói **độ lớn** chính xác thì cần n≥5.
- **Một scene, một tỉ lệ nén.** Chưa biết chi phí nén thay đổi thế nào theo tỉ lệ cắt.
- Vẫn **chưa kiểm chứng được trên `truck`** (EXP-015: model chưa nén tràn VRAM).
- Số VRAM vẫn từ NVML lấy mẫu → chặn dưới. Xem việc phải làm trong `docs/REPO_PATCHES.md` mục 3.

---

### EXP-016 / EXP-017 — 🎯 **ĐỐI CHỨNG: nén KHÔNG miễn phí. Phần lớn "cải thiện" của EXP-014 là do luyện thêm**
**Ngày:** 2026-09-04 · **Script:** `scripts/run_train_pair.sh {prune,control}` · **So sánh:** `tools/compare_arms.py`
**Kết quả:** `experiments/exp016_train_v66_ft5k` · `experiments/exp017_train_noprune`

**Câu hỏi:** EXP-014 báo `truck` +0.1123 dB **sau khi nén**. Nhưng model đã nén được luyện thêm
5000 bước mà baseline không có. Bao nhiêu là "nén không mất gì", bao nhiêu chỉ là "luyện thêm"?

**Thiết kế:** hai nhánh khác nhau **đúng một thứ** — prune có chạy hay không.
Scene `train` (N=1,026,508) vì EXP-015 đã chứng minh `truck` chưa nén **không fine-tune nổi** trên card này.

| | EXP-016 (prune) | EXP-017 (control) |
|---|---|---|
| `--prune_iterations` | 30001 | 99999 (không bao giờ chạy) |
| mọi cờ khác | giống hệt | giống hệt |

**Cổng nhất quán — hai nhánh phải xuất phát từ đúng một điểm.** Đạt tuyệt đối:
`PSNR 21.815698046433297` **trùng tới 15 chữ số thập phân** ở cả hai nhánh (eval@30001).
→ eval tất định, và so sánh là hợp lệ.

**Dự đoán ghi trước khi chạy đều đúng:** N sau prune = **349,013** ✓;
kích thước file = 86,556,75**6** B dự đoán vs **86,556,755** B thật — lệch **đúng 1 byte**, giải thích trọn vẹn:
header PLY chứa chuỗi `element vertex <N>`, mà `349013` có **6 chữ số** còn `1026508` có **7**.
Mô hình định dạng chính xác tới từng byte.

#### Kết quả

Model nhỏ đi **2.94×** (1,026,508 → 349,013 Gaussian; 254.58 → 86.56 MB) ở cả hai cách đọc.

| | baseline | control<br>(không nén, +5k bước) | prune<br>(nén, +5k bước) | do **luyện thêm** | do **nén**<br>(prune − control) |
|---|---|---|---|---|---|
| **PSNR** | 21.8157 | **22.1736** | 22.0677 | **+0.3579** | **−0.1059** |
| **SSIM** | 0.80813 | **0.81337** | 0.80130 | +0.00524 | **−0.01207** |
| **LPIPS** | 0.21521 | **0.20992** | 0.23441 | −0.00529 | **+0.02450** |

*(ngay sau prune, chưa phục hồi: PSNR 20.6715 = **−1.1442 dB**)*

#### 🔴 Kết luận

**1. "Luyện thêm" một mình cho +0.3579 dB — lớn hơn TOÀN BỘ mức tăng +0.1123 dB mà EXP-014 báo.**

**2. Nén thực sự TỐN, không miễn phí:** −0.1059 dB PSNR, −0.0121 SSIM, **+0.0245 LPIPS**.
LPIPS là chỗ lộ rõ nhất: so với baseline model nén còn **xấu đi** (+0.0192), trong khi
control **tốt lên** (−0.0053). Nghĩa là mắt người sẽ thấy khác biệt mà PSNR gần như che mất.

**3. Cách so sánh của EXP-014 (model-nén-đã-fine-tune vs checkpoint-gốc) TÍNH NHẦM
công của "luyện thêm" thành công của "nén".** Ở `train`, cách so đó cho "+0.2520 dB, nén làm model
tốt hơn!" — trong khi sự thật là nén **lấy đi** 0.1059 dB.

**4. Đây là một cổng phương pháp luận, không phải chi tiết vụn.** Bất kỳ so sánh nào giữa
model đã fine-tune sau nén với checkpoint gốc, mà không có nhánh đối chứng "luyện thêm cùng số bước",
đều đang **cộng công của tối ưu thêm vào cho nén**. Dự án này từ nay bắt buộc chạy cặp đối chứng.

#### Giới hạn (phải nói rõ)

- **Một scene, một lần chạy mỗi nhánh, không có thanh sai số.** Con số −0.1059 dB nhỏ; chưa loại trừ
  được biến thiên giữa các lần chạy. Ngược lại **+0.3579 dB thì lớn và rõ ràng** — đó mới là phát hiện chắc.
- **Hai nhánh KHÔNG thấy cùng thứ tự ảnh** sau iter 30001: `prune_finetune.py:272` đặt
  `viewpoint_stack = None`. Là nhiễu chứ không thiên lệch hệ thống, nhưng có góp vào sai số.
- **Chưa kiểm chứng lại trên `truck`** vì EXP-015: model chưa nén không chạy nổi. Hướng của kết luận
  đã rõ, nhưng **độ lớn trên `truck` vẫn chưa đo được.**
- Số dùng ở đây là eval **trong vòng lặp** (cùng mã, cùng 38 view test cho cả hai nhánh).
  Chưa chạy `render.py` + `metrics.py` ngoại tuyến; cả hai nhánh đều đã lưu `.ply` nên làm được.

#### Tài nguyên

| | VRAM đỉnh (NVML 1 Hz) | RSS đỉnh | thời gian 5000 bước |
|---|---|---|---|
| control (1.03M) | 2023 MiB | 6.52 GiB | 13:12 (6.31 it/s) |
| prune (349k) | 1663 MiB | 5.04 GiB | 10:39 (7.82 it/s) |

⚠️ Cả hai vẫn là **chặn dưới** — xem EXP-015, kết luận 3: NVML 1 Hz bỏ sót đỉnh dưới 1 giây.

---

### EXP-015 — 🔴 **Không thể chạy đối chứng trên `truck`.** Fine-tune model CHƯA nén tràn VRAM
**Ngày:** 2026-09-04 · **Script:** `scripts/run_control_noprune.sh --probe` · **Log:** `experiments/ctrl_truck_noprune_PROBE.{log,mon}`

**Mục đích:** EXP-014 báo +0.1123 dB sau khi nén, nhưng model đã nén được luyện thêm 5000 bước
mà baseline không có. Cần đối chứng: fine-tune model **chưa prune** cùng cấu hình, rồi so.

**Cấu hình:** giống hệt EXP-014, khác đúng một thứ — `--prune_iterations 99999` (prune không bao giờ chạy).
Đã đọc mã để chắc chắn thay đổi này không kéo theo gì khác:
`init_report` (`prune_finetune.py:176`) chỉ đổi **nhãn in ra**; `prune_idx` cũng chỉ dùng cho nhãn;
**ĐÍNH CHÍNH (2026-09-04, cùng ngày):** bản đầu của mục này ghi *“prune không tiêu thụ RNG
nên thứ tự camera hai lần chạy là như nhau”* — **SAI**. `prune_finetune.py:272` đặt
`viewpoint_stack = None`, tức **reset ngăn xếp camera**, nên từ iter 30002 trở đi hai nhánh
**không** thấy cùng thứ tự ảnh. Đây là *nhiễu* (cả hai vẫn lấy mẫu không thiên lệch từ cùng
tập train), không phải thiên lệch hệ thống — nhưng không được phát biểu là “giống hệt”.
Thứ **đúng là** giống nhau: lịch learning rate (cả hai đều tạo optimizer mới ở 30001 với cùng
LR ban đầu và đi đúng 12 bước `ExponentialLR(gamma=0.95)`), và Adam chưa có state khi prune chạy. Chạy thăm dò 60 iteration.

**DỰ ĐOÁN GHI TRƯỚC KHI CHẠY** (in ra trong log, không sửa về sau):
cố định 2350 MiB (tham số 572 + grad 572 + 2 moment Adam 1144 + context 62)
+ workspace ~1415 MiB → **~3765 MiB** so với trần **3440 MiB** → **dự đoán tràn**.

**KẾT QUẢ: tràn, đúng như dự đoán — nhưng sớm hơn và nặng hơn.**

```
Initial Model: Number of Gaussians is 2541226
Training progress: 0%|  | 0/60
  File ".../diff_gaussian_rasterization/__init__.py", line 141, in backward
RuntimeError: CUDA driver error: out of memory
NVML đỉnh 3567 MiB (trần 3440) · RSS đỉnh 5.11 GiB
```

Vết VRAM (1 Hz): `683 → 3143 → 3567 → chết`, trong 2 giây.

**Chết ở `loss.backward()` đầu tiên — TRƯỚC khi `optimizer.step()` chạy lần nào.**
Nghĩa là 3567 MiB này **chưa hề có 2 moment Adam**. Cộng thêm 1144 MiB nữa thì nhu cầu thật là
**~4.7 GB trên card 4 GB**. Không phải sát nút — thiếu hẳn một khoảng lớn.

→ **Kết luận 1: trên máy này, model `truck` chưa nén KHÔNG fine-tune được, ở bất kỳ cấu hình nào.**
Nén ở đây không phải là "tối ưu thêm", mà là **điều kiện cần để chạm được vào model.**

---

#### ⚠️ EXP-015 buộc phải SỬA một số của EXP-014

EXP-014 ghi "VRAM đỉnh **2621 MiB**". Con số đó **là chặn dưới, không phải đỉnh thật.**

EXP-014 cũng chạy đúng một lần `loss.backward()` ở N=2.54M (tại iter 30001, trước khi prune).
Đối chiếu mốc thời gian trong `lg_truck_v66_ft5k.mon`:

| thời điểm | NVML | việc đang chạy |
|---|---|---|
| 19:29:21 | 1603 | model vừa lên GPU |
| *(giữa 2 mẫu)* | **không đo được** | **backward ở N đầy đủ** |
| 19:29:23 | 2421 | đã sang phần eval |

Backward đó kéo dài **dưới 1 giây**, còn NVML chỉ lấy mẫu **1 Hz** → **đỉnh của nó chưa bao giờ được đo.**
EXP-015 cho thấy chính thao tác đó cần ~3567 MiB. Vậy EXP-014 gần như chắc chắn cũng chạm ~3.5 GB
và **sống sót trong gang tấc**, chỉ khác EXP-015 ở camera ngẫu nhiên / trạng thái phân mảnh bộ nhớ.

→ **Kết luận 2: EXP-014 chạy được là may, không phải do có biên an toàn.** Lặp lại có thể tràn.

→ **Kết luận 3 (tự phê bình): đây đúng là lỗi mình đã ghi thành quy tắc rồi vẫn vi phạm.**
`src/memory/counters.py` tồn tại chính vì đã đo được rằng **lấy mẫu bỏ sót 99.9% một transient 400 MB**.
Vậy mà `run_prune_finetune.sh` vẫn dùng NVML 1 Hz làm nguồn số VRAM chính thức.
**Việc phải làm:** mọi script chạy thật phải lấy đỉnh VRAM từ `torch.cuda.max_memory_allocated()`
/ `max_memory_reserved()` in ra từ trong tiến trình, chứ không phải từ NVML lấy mẫu ngoài.
NVML chỉ dùng để phát hiện tiến trình lạ.

---

**Hệ quả với kế hoạch:** cặp đối chứng chuyển sang scene `train` (N=1,026,508 — vừa bộ nhớ):
EXP-016 (prune 66% + ft5k) và EXP-017 (không prune + ft5k). Xem `scripts/run_train_pair.sh`.

**Chi phí bên lề đã tránh được:** nhánh control ở `truck` còn không lưu nổi model.
`save_ply` (`gaussian_model.py:206`) dùng `list(map(tuple, attributes))` → dựng N tuple Python
62 float ⇒ ~13.35× kích thước file trong RAM. Ở N=2.54M là **~8.3 GB** so với trần **9.7 GiB**.
EXP-014 sống sót chỉ vì nó lưu **sau** khi prune (N=864k → 2.86 GB, khớp đỉnh RSS 5.67 GiB đã đo).

---

### EXP-014 — 🎯 **KẾT QUẢ NÉN ĐẦU TIÊN.** LightGaussian prune 66% + recovery 5k trên `truck`

> ### ⛔ ĐỌC EXP-016/017 TRƯỚC KHI TRÍCH BẤT KỲ SỐ NÀO Ở ĐÂY (sửa 2026-09-04)
> **Phát biểu "+0.1123 dB — nén còn làm model tốt hơn" KHÔNG đứng vững.**
> Đối chứng (EXP-017) cho thấy trên `train`, **luyện thêm 5000 bước mà không nén gì**
> đã tự cho **+0.3579 dB**. Cách so của mục này — model-nén-đã-fine-tune vs checkpoint-gốc —
> **tính công của "luyện thêm" thành công của "nén"**. Khi trừ đúng phần đó ra,
> nén **tốn** −0.1059 dB PSNR / −0.0121 SSIM / **+0.0245 LPIPS**.
> Không đo lại được trực tiếp trên `truck` vì EXP-015: model `truck` chưa nén **tràn VRAM**,
> nên độ lớn trên `truck` vẫn chưa biết — nhưng **hướng thì đã rõ**.
>
> **Còn đúng và vẫn dùng được:** N=864,017 và 630.23 → 214.28 MB (2.94×) đều khớp dự đoán từng byte;
> quỹ đạo trong vòng lặp 25.0008 → 24.2253 → 25.1131; toàn bộ phần đo tài nguyên.
> **Sai thêm:** "VRAM đỉnh 2621 MiB" là chặn dưới do NVML lấy mẫu 1 Hz — xem EXP-015.
**Ngày:** 2026-09-04 · **Script:** `scripts/run_prune_finetune.sh` · **Kết quả:** `experiments/lg_truck_v66_ft5k`
**Chuẩn bị:** workflow pre-flight 78 agent (126 phát hiện · 71 rủi ro · 69 xác nhận / 2 bác bỏ)

**Cấu hình:** PUP 3D-GS `e971ea4` · `--prune_type v_important_score --prune_percent 0.66 --v_pow 0.1`
· prune tại iter 30001 · fine-tune tới 35000 · `-r 1` · `--data_device cpu` · `--eval` (llffhold=8, 32 view)
· `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` · 11/11 cổng tiền kiểm đạt.

#### Kết quả — đo bằng `render.py` + `metrics.py` (cùng đường với baseline, quy tắc L1/L2)

| | PSNR | SSIM | LPIPS | N | Kích thước |
|---|---|---|---|---|---|
| Baseline (checkpoint INRIA) | 24.9927 | 0.87562 | 0.15130 | 2,541,226 | 630.23 MB |
| **Sau nén + fine-tune** | **25.1050** | **0.87671** | **0.15078** | **864,017** | **214.28 MB** |
| **Δ** | **+0.1123** | **+0.0011** | **−0.0005** | **−66.0%** | **2.94×** |

**Cả ba chỉ số đều TỐT LÊN, với 2.94× ít Gaussian hơn.**

#### Diễn biến trong vòng lặp (cùng 32 view)

| Mốc | PSNR | SSIM | LPIPS |
|---|---|---|---|
| Trước prune (@30001) | 25.0008 | 0.87614 | 0.15215 |
| Ngay sau prune, chưa fine-tune | **24.2253** | 0.86157 | 0.16604 |
| Sau 5k fine-tune (@35000) | 25.1131 | 0.87724 | 0.15167 |

→ Prune mất **−0.776 dB**, recovery lấy lại **+0.888 dB**.
Lệch giữa in-loop (25.1131) và metrics.py (25.1050) là **0.008 dB** — khớp sai lệch ước tính ~0.006 dB của pre-flight.

#### Tài nguyên

| | Đo được | Dự đoán pre-flight |
|---|---|---|
| VRAM đỉnh (NVML) | **2621 MiB** / trần 3440 | 2829–2841 |
| RSS đỉnh | **5.67 GiB** | 5.3–6.0 |
| Thời gian | **22 phút** | 20–45 |
| N sau prune | **864,017** | 864,017 ✓ chính xác |
| Kích thước | **214,277,747 B** | 214,277,747 ✓ chính xác |

#### So với dự đoán và với literature

Pre-flight dự đoán PSNR cuối **24.4–24.9** (dựa trên literature: LightGaussian re-run báo −0.2 đến −0.7 dB
trên T&T). Thực tế **25.105 — vượt khoảng dự đoán và ngược dấu**. PSNR sau-prune-chưa-fine-tune (24.2253)
thì khớp gần như hoàn hảo với dự đoán 24.20–24.26.

Tức là: **tiêu chí prune hành xử đúng như literature; phần recovery mới là chỗ vượt kỳ vọng.**

#### ⚠️ Giới hạn — phải nêu trong mọi báo cáo

1. **Confound lớn nhất: model đã nén được thêm 5000 bước tối ưu mà baseline không có.**
   Một phần của +0.112 dB có thể chỉ là "train thêm", không phải "nén miễn phí". **Đối chứng bắt buộc:**
   fine-tune model **chưa prune** thêm 5000 iteration với cùng cấu hình rồi so. Chưa làm.
2. **Một scene, một tỉ lệ, một lần chạy, không có thanh sai số.** Kernel importance không dùng `atomicAdd`
   nên score không tất định — chạy lại sẽ đổi tập sống sót ~1.1%.
3. **Tiêu chí prune và lịch recovery không tách được** trong một run. Δ 30002→35000 trộn cả
   `ExponentialLR(gamma=0.95)` (12 bước, LR cuối còn 0.54×, **không có flag tắt**) lẫn xyz-LR phẳng.
4. **Không phải endpoint của LightGaussian.** 214.28 MB là `.ply` thô. Paper báo 22.43 MB sau khi thêm
   SH distillation + vector quantization — cả hai đều chưa làm (T9: `distill_train.py` cần `.pth`).
5. **Rasterizer thiếu nhân transmittance** (`forward.cu:474`). Nhưng bản pin của PUP **byte-identical**
   với bản LightGaussian pin → ta so được với mọi re-run đã công bố, chỉ **không** được nói "tái tạo đúng
   phương trình trong paper".
6. **FPS không tính là kết quả** — baseline đo nguội, model nén đo nóng ngay sau fine-tune.

#### Đính chính T6 (nửa đúng)
PUP dùng **`torch.optim.Adam`**, **không phải AdamW** → không có weight decay trên tham số Gaussian.
Nhưng `ExponentialLR(gamma=0.95)` mỗi 400 iteration thì **có thật** (`prune_finetune.py:81,124-125`).
Cập nhật `paper_notes/lightgaussian.md` T6.

#### Điều pre-flight cứu được
Script nháp của tôi có 3 lỗi chí mạng, tất cả đều do agent phát hiện khi đọc source:
`--prune_decay 1` (không tồn tại trong PUP → thoát mã 2) · `--quiet` (giết mọi chuỗi cổng kiểm tra) ·
thiếu `--checkpoint_iterations 0` (lưu thêm 622 MB optimizer state, hỏng số kích thước).

**Ghi chú A12:** `--prune_iterations 30001` khiến prune chạy **trước** `optimizer.step()` đầu tiên, nên
`_prune_optimizer` đi nhánh không-có-state và Adam **không bao giờ** cấp moment ở N đầy đủ.
→ **H8 (A12) đạt được miễn phí bằng cách chọn đúng iteration, không cần sửa code.** Hạ H8 khỏi danh sách contribution.

---

### EXP-012 — ✅ PLY I/O đo trên **đúng code sẽ chạy**. `save_ply` là blocker, đã xác nhận bằng OOM thật
**Ngày:** 2026-09-04 · **Tool:** `tools/bench_plyio.py` · **Dữ liệu:** `experiments/plyio/results.jsonl`

**Phương pháp — sửa cả 3 lỗi của phép đo cũ:** gọi thẳng `GaussianModel.load_ply/.save_ply` của repo PUP
(không phải bản mô phỏng) · **có torch đã import + CUDA đã init** · đo bằng **VmHWM** (bộ đếm kernel,
không lấy mẫu) · mỗi phép đo một tiến trình riêng dưới `systemd-run --scope MemoryMax=8G MemorySwapMax=0`
(biến swap âm thầm thành OOM rõ ràng) · tách `RssAnon` khỏi `RssFile`.

| op | ×file | train | truck | playroom | drjohnson |
|---|---|---|---|---|---|
| `load_inria` | **3.56–3.57×** | 0.85 | 2.09 | 2.09 | 2.80 GiB |
| `load_gsply` | **2.03–2.21×** | 0.52 | 1.21 | 1.21 | 1.60 GiB |
| **`save_inria`** | **13.35×** | **3.17** | ❌ **OOM-KILL** | ❌ **OOM-KILL** | ❌ **OOM-KILL** |
| `save_gsply` | **2.98–3.88×** | 0.92 | 1.82 | 1.82 | 2.35 GiB |

**1. Hai hệ số cũ ĐỀU SAI — và đều là lỗi "đo code A, áp cho code B":**

| | Đã dùng | Nguồn của số sai | **Đo thật** |
|---|---|---|---|
| `load_inria` | 4.58× | agent đo trên **INRIA main**, Windows Python, **không có torch** | **3.56×** |
| `load_gsply` | 1.95× | agent | **2.03–2.21×** |
| `save_inria` | 12.08× (**ngoại suy** từ N=400k/900k) | agent | **13.35×** (đo tại N đầy đủ) |

Con số 13.35× cũng cho thấy README của dự án đang ghi *"ví dụ **đã đo**: `save_ply` 12× (10.2 GB)"* —
lúc đó nó là **ngoại suy**, không phải đo. Giờ mới thật sự đo.

**2. `save_ply` của INRIA là blocker CỨNG — xác nhận bằng OOM thật, không phải ngoại suy.**
Chỉ `train` (254.6 MB) sống sót ở 3.17 GiB. Ba scene còn lại **bị kernel giết** ở cap 8G.
Mọi vòng prune/nén đều ghi `.ply` → **không dùng `save_ply` của INRIA được cho scene ≥ truck.**

**3. `gsply` — claim "bitwise khớp" ĐÚNG, và giờ có bằng chứng.**
Tài liệu trước ghi *"verify `torch.equal` bitwise trên cả 6 tensor"* nhưng **không script nào làm việc đó**
(audit defect #15a). Đã làm thật: **4 scene × 6 tensor, `np.array_equal` = True toàn bộ.**
Lưu ý shape khác nhau nhưng nội dung khớp sau khi flatten:
`_features_dc` INRIA `(N,1,3)` ↔ gsply `sh0 (N,3)` · `_features_rest (N,15,3)` ↔ `shN (N,15,3)`.

**4. `load_gsply` dùng 0 MiB VRAM** (trả numpy trên CPU) trong khi `load_inria` đỉnh **1.63× params**
(train 375.9 · truck 930.6 · drjohnson 1247.0 MiB) — đúng bằng transient của `.transpose(1,2).contiguous()`
giữ cả bản trước và sau. Hệ số 1.63× ổn định qua cả 4 scene.

**5. `gsply.plywrite` bỏ `nx,ny,nz`** → file ra = **95%** file vào. `load_ply` của INRIA không đọc normal
nên round-trip vẫn đúng, nhưng phải ghi rõ khi so sánh kích thước model.

**Ảnh hưởng:** `gsply` chuyển từ "tối ưu nên làm" thành **bắt buộc cho cả đọc lẫn ghi** ở mọi scene ≥ truck.

---

### EXP-006 — 🔴 VRAM trên WSL2 **KHÔNG OOM sạch — nó tràn im lặng sang host RAM**
**Ngày:** 2026-09-02 · **Phase:** P0 · **Tool:** `tools/measure_cuda_overhead.py`
**Dữ liệu:** `experiments/p0_setup/cuda_overhead.json` · GPU idle, torch 2.14.0+cu126, driver 560.70

**Phát hiện tình cờ:** lần dò trần đầu tiên báo *"OOM sau khi giữ **10816 MiB**"* trên card **4096 MiB**.
Không phải lỗi đo. Driver NVIDIA trên WSL2/WDDM cho phép cấp phát **tràn sang host RAM qua PCIe**
thay vì báo lỗi.

**Xác nhận bằng hai dấu hiệu độc lập:**

| Đã giữ (MiB) | NVML (MiB) | Băng thông | Ghi chú |
|---|---|---|---|
| 512 → 3584 | 669 → 3805 | 53–57 GB/s | bình thường, NVML tăng theo |
| **4096** | 4061.5 | **23.1 GB/s** | bắt đầu chậm |
| **4224** | **4061.5** | — | ⚠️ **NVML PLATEAU** — cấp phát vẫn "thành công" |
| **5120** | 4061.5 | **15.0 GB/s (41%)** | ⚠️ **BĂNG THÔNG SỤP** — đang chạy trên PCIe |
| 5632 | 4061.5 | 15.3 GB/s | |

→ Vượt VRAM **không crash**. Nó chỉ **chậm đi 3.6×** và chạy tiếp.

**Hệ quả nghiêm trọng cho toàn bộ plan:**
1. **`torch.cuda.OutOfMemoryError` sẽ KHÔNG kích hoạt** khi vượt VRAM trên máy này.
2. Mọi ngân sách VRAM phải được **cưỡng chế tường minh**, không được "phát hiện bằng OOM".
3. `killswitch.py` cần một nhánh VRAM dựa trên **plateau NVML**, không phải bắt exception.
4. Đây đúng là bản sao của vấn đề swap ở RAM — nhưng cho VRAM, và tôi **đã không lường trước**.
   Giờ máy này có **hai** đường chết im lặng: pagefile cho RAM, PCIe fallback cho VRAM.
5. Mọi phép đo thời gian đều vô nghĩa nếu không kiểm tra plateau — một run "chạy được" có thể
   đang chạy 3.6× chậm trên host RAM mà không dấu hiệu gì.

---

### EXP-007 — Ngân sách VRAM thật: **3808 MiB**, tốt hơn ước tính 25%
**Cùng run với EXP-006.**

```
4096.0 MiB  tổng
-  34.5 MiB  driver giữ (NVML đỉnh chỉ đạt 4061.5)
- 161.0 MiB  đường nền WSL2 GPU-PV   <- ĐÂY LÀ "WSL2 overhead" mà plan hỏi từ đầu
-  62.5 MiB  CUDA context            <- ước tính cũ 350 MiB, SAI 5.6×
-  32.0 MiB  cuBLAS workspace (nạp lazy ở matmul đầu tiên)
─────────────
= 3808.0 MiB dùng được cho tensor
```

Băng thông VRAM ổn định **53–57 GB/s** (lần đo đầu 37 GB/s vì clock chưa boost — **phải warm-up**).

**Ngân sách Gaussian (chưa trừ rasterizer workspace và ảnh):**

| Kịch bản | B/Gaussian | N tối đa | truck 2.54M | playroom 2.55M | drjohnson 3.41M |
|---|---|---|---|---|---|
| Inference (params SH3) | 236 | **16.92 M** | ✅ | ✅ | ✅ |
| Fine-tune SH3 | 956 | **4.18 M** | ✅ 2.42 GB | ✅ 2.43 GB | ⚠️ 3.26 GB (còn 0.55 GB) |
| Fine-tune SH2 | 620 | 6.44 M | ✅ | ✅ | ✅ 2.11 GB |
| **A12: prune trước `training_setup`** | 240 | **16.64 M** | ✅ | ✅ | ✅ **0.82 GB** |

**So với PLAN v4 §3 (ước tính 3.0–3.2 GB):** thực tế **3.81 GB**, dư thêm ~600–800 MB.
→ `truck` và `playroom` fine-tune **thoải mái**, không còn "chật" như dự đoán.
→ `drjohnson` fine-tune SH3 vừa đủ nhưng chỉ còn 0.55 GB cho rasterizer — **A12 hoặc SH2 vẫn cần**.

**Cảnh báo diễn giải:** đường nền 161 MiB đo được lúc NVML khởi tạo *từ trong WSL*, trong khi
`nvidia-smi` gọi riêng lại báo 0 MiB. Nhiều khả năng đó là phần GPU-PV giữ khi có client CUDA.
Cần đo lại khi đã đóng Chrome/Zalo để loại nhiễu — hiện có 29 tiến trình Chrome đang chạy.

---

### EXP-008 — ✅ `expandable_segments:True` DẬP được đường chết im lặng ở VRAM
**Ngày:** 2026-09-02 · **Phase:** P0 · Dữ liệu: `cuda_overhead.json` vs `cuda_overhead_expandable.json`

Cùng máy, cùng lúc, chỉ khác một biến môi trường:

| Metric | `unset` | `expandable_segments:True` | Δ |
|---|---|---|---|
| CUDA context | 62.5 MiB | 62.5 MiB | 0 |
| cuBLAS | 32.0 MiB | 32.0 MiB | 0 |
| NVML đỉnh | 4061.5 | 3693.5 | **−368** |
| **Trần thật** | **3808.0 MiB** | **3440.0 MiB** | **−368** |
| NVML plateau tại | 4224 MiB | **không xảy ra** | — |
| Băng thông sụp tại | 5120 MiB | **không xảy ra** | — |
| **`host_spill_observed`** | **True** ⚠️ | **False** ✅ | — |
| Ngân sách fine-tune SH3 | 4.18 M | 3.77 M | −0.41 M |

**Kết luận:** biến này **loại bỏ hoàn toàn** hành vi tràn sang host RAM của EXP-006. GPU **OOM sạch**
thay vì chạy tiếp chậm 3.6× trên PCIe. Giá: mất **368 MiB (9.7%)** trần.

**Quyết định: BẬT, đặt trong `/etc/profile.d/cuda126.sh`.** Lý do: một con số sai mà nhanh nguy hiểm hơn
một lỗi rõ ràng. Với A12 (prune trước `training_setup`), drjohnson chỉ cần 0.82 GB nên 368 MiB không còn quyết định.

**Hệ quả cho các experiment:**
- EXP-006 và EXP-007 đo khi **CHƯA** đặt biến này → trần 3808 MiB chỉ đúng cho cấu hình đó.
  **Số dùng cho mọi thứ về sau là 3440 MiB.**
- Cơ chế phát hiện tràn trong `killswitch.py` chuyển từ **phòng tuyến chính** sang **lưới an toàn**
  (đề phòng biến bị unset ở đâu đó). Giữ nguyên.
- `PYTORCH_CUDA_ALLOC_CONF` phải được ghi vào `env.json` mỗi run — `profiler.py` đã làm.
- ⚠️ Nếu về sau có run nào cho trần ≈ 3808 thay vì 3440, tức biến đã **không được nạp** → kiểm tra
  shell có phải login shell không (bài học `.bashrc` non-interactive guard).

---

### EXP-003 — Chi phí RAM khi nạp `.ply` ⚠️ **ĐÃ SỬA — phép đo gốc THẤP 2.5×**

> 🔴 **ĐÍNH CHÍNH (2026-09-02, cùng ngày).** Phép đo bên dưới **không đo code INRIA thật**.
> Biến thể A của tôi ép `dtype=np.float32`. INRIA **không** truyền dtype:
>
> ```python
> features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))   # -> float64!
> ```
> (`scene/gaussian_model.py`, đã verify trong repo đã clone: dòng `np.zeros` cho
> `features_dc`, `features_extra`, `scales`, `rots` đều **thiếu dtype → float64**.)
> Cộng thêm `torch.tensor(...).transpose(1,2).contiguous()` giữ bản trước và sau transpose cùng lúc.
>
> **Số đúng cho drjohnson (đo độc lập, cùng file):**
>
> | Đường nạp | Đỉnh | ×file | Thời gian |
> |---|---|---|---|
> | `PlyData.read()` đơn thuần (mmap mặc định) | 0.3 MB | 0.00× | 0.01 s |
> | **INRIA `load_ply()` nguyên bản** | **3864 MB** | **4.58×** | 9.0 s |
> | + sửa một dòng `dtype=np.float32` | 3114 MB | 3.69× | 7.3 s |
> | `np.memmap` tự viết | 2434 MB | 2.88× | 7.4 s |
> | chunked `readinto` tự viết | 1655 MB | 1.96× | 7.5 s |
> | **`gsply.plyread()`** | **1648 MB** | **1.95×** | **1.5 s** |
> | *(đo của tôi — float32, chưa có torch)* | *1560 MB* | *1.96×* | *4.5 s* |
>
> Model giải tích dự đoán 3841 MB vs đo 3864 MB (**sai 0.6%**) → đáng tin.
> Trong đó **1.2 GB là lãng phí float64 thuần tuý**.
>
> **Bài học phương pháp (ghi vào 02_METHODOLOGY):** đo một *phiên bản đơn giản hoá* của code
> rồi báo cáo như thể đó là code thật là lỗi nghiêm trọng. Từ nay mọi phép đo bộ nhớ phải gọi
> **đúng hàm trong repo**, không phải bản mô phỏng.
>
> **Cái vẫn đúng:** plyfile **memory-map** (0.3 MB) → chi phí nằm ở bước rút thuộc tính, không ở đọc.
> `np.memmap` không giúp nhiều (2.88×). Streaming/chunked là hướng đúng.
> **Cái phải sửa:** không tự viết loader nữa — [**gsply**](https://github.com/OpsiClear/gsply)
> (MIT, wheel cp312 manylinux, **không cần build CUDA**) đạt 1.95× và **nhanh hơn 6×**, đã được
> verify `torch.equal` bitwise trên cả 6 tensor so với loader INRIA. Trả về giá trị **thô trước activation**
> (scale dạng log, opacity dạng logit) — khớp đúng cái 3DGS cần.

---

#### Nội dung gốc (giữ lại để đối chiếu — số float32, KHÔNG phải code INRIA)
**Ngày:** 2026-09-02 · **Phase:** P1/P4 (C2) · **Tool:** `tools/measure_ply_load.py`
**Dữ liệu:** `experiments/p0_setup/ply_load_all.json` · 8 checkpoint × 4 phương pháp, mỗi lần một tiến trình con sạch

**Hypothesis (từ PLAN.md §2, đăng ký trước):** `plyfile` + `load_ply` tạo ~3 bản sao → đỉnh RAM ≈ **3× kích thước file**.

**Kết quả: BỊ BÁC BỎ. Đỉnh thật = 1.96–2.08× file, cực kỳ ổn định.**

| scene | N | file MB | plyfile_full | numpy_direct | numpy_mmap | **chunked_500k** |
|---|---|---|---|---|---|---|
| drjohnson@30k | 3,405,153 | 805 | 1.56 GB (1.96×) | 1.57 | 1.57 | **0.24 GB (0.27×)** |
| playroom@30k | 2,546,116 | 602 | 1.18 (1.96×) | 1.18 | 1.18 | — |
| truck@30k | 2,541,226 | 601 | 1.17 (1.97×) | 1.18 | 1.18 | **0.24 (0.37×)** |
| drjohnson@7k | 1,902,253 | 450 | 0.89 (1.97×) | 0.89 | 0.89 | — |
| playroom@7k | 1,734,607 | 410 | 0.81 (1.98×) | 0.82 | 0.82 | — |
| truck@7k | 1,732,378 | 410 | 0.81 (1.98×) | 0.82 | 0.82 | — |
| train@30k | 1,026,508 | 243 | 0.50 (2.01×) | 0.50 | 0.50 | **0.24 (0.91×)** |
| train@7k | 559,263 | 132 | 0.29 (2.07×) | 0.29 | 0.29 | — |

**Fit tuyến tính (8 điểm, sai số ≤ 0.01 GB ở MỌI điểm — thành phần đầu tiên của cost model C2):**

```
peak_RSS_load(GB)  =  0.446 × N_triệu  +  0.04
```

**Bốn phát hiện:**

1. **`PlyData.read()` gần như miễn phí (0.04 GB, ~0.03× file)** — plyfile **memory-map** binary PLY.
   Toàn bộ chi phí nằm ở bước **rút từng thuộc tính** (`np.stack` per attribute), không phải ở việc đọc.
2. **`np.memmap` KHÔNG giúp gì** (1.97–2.08×, y hệt đọc thẳng). Vì vẫn phải copy mọi cột cần ra mảng
   riêng, và page đã chạm của memmap vẫn tính vào RSS. → Gạch bỏ "dùng mmap" khỏi danh sách giải pháp.
3. **2× = 1× output + 1× bản sao tạm.** Useful data ≈ 0.93× file (giữ 59/62 property, bỏ 3 normal).
4. **Streaming theo khối hàng thắng tuyệt đối:** đỉnh **0.24 GB CỐ ĐỊNH** không phụ thuộc N
   (O(chunk) chứ không O(N)), và **không chậm hơn** (drjohnson 4.1s vs 4.6s). Giảm 6.5× ở drjohnson.

**Ảnh hưởng tới plan:** ước tính "4–6 GB để nạp bicycle 6M" trong PLAN.md §2 là **quá bi quan ~2×**.
Số đúng: 6M → `0.446×6+0.04` = **2.72 GB**, và với streaming loader chỉ **0.24 GB**.
→ RAM vẫn là ràng buộc đáng theo dõi, nhưng **ảnh training (1.5–3.4 GB) mới là khoản chi lớn hơn model**, không phải chiều ngược lại.

**Cảnh báo — đây là CẬN DƯỚI, chưa phải số cuối:**
- Đo bằng Windows Python 3.11.9 + numpy + plyfile, **chưa có torch**. `GaussianModel.load_ply` thật còn
  `torch.tensor(...).requires_grad_(True)` → thêm một bản sao CPU nữa trước `.cuda()`. **Phải đo lại sau khi có torch.**
- Chưa đo trong WSL (allocator khác). Lặp lại ở P1.

**Việc tiếp:** ~~`src/loading/ply_stream.py`~~ → **dùng `gsply` thay vì tự viết** (xem đính chính đầu mục).

---

### EXP-004 — 🔴 `save_ply` là nút thắt THẬT, không phải `load_ply`
**Ngày:** 2026-09-02 · **Nguồn:** khảo sát cộng đồng + verify source · **Trạng thái:** cần đo lại tại chỗ ở P1

Tôi tập trung sai chỗ. Ghi `.ply` **đắt gấp 2.6× lần đọc**, và nó **vượt toàn bộ ngân sách RAM**.

Nguyên nhân, đã verify trong `scene/gaussian_model.py:save_ply`:

```python
elements[:] = list(map(tuple, attributes))    # tạo N tuple, mỗi tuple 62 numpy scalar OBJECT
```

| Đường ghi | Byte/Gaussian | ×file ra | Ngoại suy cho drjohnson 3.4M |
|---|---|---|---|
| **INRIA `save_ply()` nguyên bản** | **2997** | **12.08×** | **≈ 10.2 GB** ❌ |
| `gsply.plywrite()` | — | 1.30× | 1042 MB (đo thật, full file) |
| chunked writer tự viết | 267 | 1.08× | ≈ 910 MB |

**10.2 GB > 9.7 GB RAM của WSL.** Và **mỗi vòng prune/compress đều ghi một `.ply`**.
→ Đây là blocker cứng, không phải `load_ply`. Nếu không sửa, pipeline chết ở lần lưu đầu tiên trên drjohnson.

*(Con số 12.08× là ngoại suy từ N=400k và N=900k, tuyến tính tới 0.25%. Chưa đo tại N đầy đủ — cố ý,
vì 10.2 GB sẽ làm swap máy. **Phải đo lại tại chỗ ở P1 với kill-switch bật.**)*

⚠️ `gsply.plywrite()` **bỏ `nx,ny,nz`** (804 vs 844 MB). `load_ply` của INRIA không đọc normal nên
round-trip vẫn đúng, nhưng vài viewer cần. `gsplat.utils.save_ply` ghi normal bằng 0 nếu cần byte-identity.

---

### EXP-005 — Ảnh training tốn 16 byte/pixel, không phải 12

> ### ⛔ MỤC NÀY SAI — ĐÃ SỬA 2026-09-04 (phát hiện lại khi lập kế hoạch EXP-015)
> Kết luận 16 B/px **không áp dụng cho code chúng ta chạy**. Đã kiểm chứng trực tiếp
> trên `gaussian-splatting-pup/scene/cameras.py:39-46`:
> ```python
> self.original_image = image.clamp(0.0, 1.0).to(self.data_device)   # 3 kênh
> if gt_alpha_mask is not None:  self.original_image *= gt_alpha_mask.to(...)
> else:                          self.original_image *= torch.ones((1,H,W), ...)  # TẠM, huỷ ngay
> ```
> PUP **không** giữ `self.alpha_mask`; tensor ones chỉ là trung gian của phép nhân rồi được giải phóng.
> Dòng `self.alpha_mask = ...` ở `cameras.py:48` là của **nhánh main mới của INRIA**, không phải PUP.
> → **12 B/px là đúng** cho toàn bộ dự án này. Bảng bên dưới (cột "đúng 16 B/px") **không dùng**.
> Đo lại 2026-09-04: train 301 ảnh 980×545 = **1.80 GiB**; truck 251 ảnh 979×546 = **1.50 GiB**.
> Thuộc nhóm lỗi "sửa tại chỗ phát hiện, không lan sang nơi khác".
**Ngày:** 2026-09-02 · **Verify:** `scene/cameras.py:48`

```python
self.alpha_mask = torch.ones_like(resized_image_rgb[0:1, ...].to(self.data_device))
```

`alpha_mask` **luôn** được materialize kể cả khi ảnh không có kênh alpha → thêm 1 kênh float32.
Tức **16 byte/pixel (RGB + alpha), không phải 12**.

→ Mọi số ảnh trong `env.md` và `PLAN.md` **thấp 33%**. Sửa:

| scene | cũ (12 B/px) | **đúng (16 B/px)** |
|---|---|---|
| train | 1.80 GB | **2.40 GB** |
| truck | 1.50 GB | **2.00 GB** |
| playroom | 2.64 GB | **3.52 GB** |
| drjohnson | 3.43 GB | **4.57 GB** |

**Hệ quả nghiêm trọng:** với `data_device=cuda` (mặc định INRIA), drjohnson cần
**4.57 GB ảnh + 0.80 GB params = 5.37 GB > 4 GB VRAM** — OOM chắc chắn, còn tệ hơn ước tính cũ.
Và playroom 3.52 + 0.60 = 4.12 GB cũng **vượt 4 GB**. → `--data_device cpu` là **bắt buộc**, không phải tuỳ chọn.

---

### EXP-002 — Image placement: `data_device=cuda` vs `cpu` (RQ4 / H6, điểm dữ liệu đầu)
**Ngày đăng ký:** 2026-09-02 · **Phase:** P1 · **Script:** `scripts/p1_smoke.sh truck 1 cuda`
**Hypothesis:** với `cuda`, VRAM device đỉnh tăng **+1.4–1.7 GB** so với EXP-001 (= 251 ảnh × 979×546×3×4 B ≈ 1.6 GB),
RSS giảm tương ứng, wall-time render giảm < 20%. Bác bỏ H6 nếu wall-time giảm > 40% (khi đó CPU placement đắt hơn tưởng).
**Rủi ro:** có thể OOM → chính nó là kết quả: *"scene 2.5M Gaussian không render được với ảnh trên GPU ở 4 GB"*.
**Kết quả:** _(chưa chạy)_

---

<!-- EXP-002 trở đi ghi từ đây xuống -->
