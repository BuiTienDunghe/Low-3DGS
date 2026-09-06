# Literature Review

> **Trạng thái: SEED — chưa verify.** Bảng dưới được khởi tạo từ web search ngày 2026-09-01
> để bạn có điểm bắt đầu. Cột `Verified` = ☐ nghĩa là **chưa tự đọc paper**.
> Nhiệm vụ đầu tiên của P3: đọc và tự điền. **Không cite gì có ☐ trong report.**

---

## 0. Bắt đầu từ đâu

Đọc theo thứ tự này, không đọc lung tung:

1. **3DGS gốc** (Kerbl et al., SIGGRAPH 2023) — nền tảng bắt buộc
2. **3DGS.zip survey** (CGF 2025) — bản đồ toàn bộ mảng compression, tiết kiệm hàng tuần
3. **Compression in 3DGS: A Survey** (arXiv 2502.19457) — survey thứ hai, đối chiếu
4. **Taming 3DGS** — vì nó trùng ý tưởng gốc của bạn, phải đọc kỹ nhất
5. **gsplat** — vì đây là codebase sẽ dùng
6. Chọn 1 method để reproduce từ danh sách shortlist

---

## 1. Bảng chính

| Paper | Năm/Venue | Ý tưởng chính | Loại | Ảnh hưởng train-time VRAM? | Reproducibility | Verified |
|---|---|---|---|---|---|---|
| 3D Gaussian Splatting for Real-Time Radiance Field Rendering | SIGGRAPH 2023 | Explicit Gaussian primitives + differentiable tile rasterizer | Baseline | — | Cao (code chính chủ) | ☑ Table 1/8/9 đã trích (A6000, native res, llffhold=8) |
| **Taming 3DGS: High-Quality Radiance Fields with Limited Resources** | SIGGRAPH Asia 2024 | Score-based **constructive** densification tiến tới **exact budget**; backward nhanh hơn | Train-time | **CÓ** | Cao (code public) | ☐ |
| **Reducing the Memory Footprint of 3DGS** | I3D 2024 (INRIA) | Giảm memory footprint từ chính nhóm tác giả gốc | Cả hai | Có | Cao (`graphdeco-inria/reduced-3dgs`) | ☐ |
| **Gaussians on a Diet: Memory-Bounded 3DGS Training** | arXiv | Giữ **peak training memory gần hằng số** cho edge device | Train-time | **CÓ** | ? | ☐ |
| LightGaussian | NeurIPS 2024 | Prune theo importance + SH distillation + VectorQuant | Post-hoc | Không | Trung bình | ☑ Eq.3/4, Table 1/2/8 + code đã map → `paper_notes/lightgaussian.md` |
| Compact 3D Gaussian Representation | CVPR 2024 | Learnable masking + residual VQ cho SH | Cả hai | Một phần | Trung bình | ☐ |
| Mini-Splatting | ECCV 2024 | Densification + simplification, số Gaussian bị ràng buộc | Cả hai | Có | Trung bình | ☐ |
| Scaffold-GS | CVPR 2024 | Anchor-based, Gaussian sinh ra từ anchor feature | Structured | Có | Trung bình | ☐ |
| HAC | ECCV 2024 | Hash-grid assisted context cho entropy coding | Post-hoc | Không | Thấp-TB | ☐ |
| EAGLES | ECCV 2024 | Quantized latent encoding | Cả hai | Một phần | Trung bình | ☐ |
| MEGS² | arXiv 2509.07021 | Spherical Gaussians + unified pruning cho memory | Cả hai | Có | ? | ☐ |
| Revising Densification in GS | 2024 | Phân tích lại cơ chế densification | Train-time | Có | ? | ☐ |
| Self-Organizing Gaussians | 2024 | Sort Gaussian vào lưới 2D, nén bằng image codec | Post-hoc | Không | ? | ☐ |
| **gsplat** | arXiv 2409.06765 | Library: packed mode, sparse grad, fused Adam | Infra | **CÓ** | Rất cao | ☐ |
| Splatwizard | arXiv | Benchmark toolkit cho 3DGS compression | Infra | — | ? | ☐ |
| 3DGS.zip survey | CGF 2025 | Survey compression methods + leaderboard | Survey | — | — | ☐ |
| Compression in 3DGS: A Survey | arXiv 2502.19457 | Survey thứ hai | Survey | — | — | ☐ |

**Cột quan trọng nhất là "Ảnh hưởng train-time VRAM?"** — vì V1 trong `00_CRITIQUE.md`.
Chỉ những paper có "CÓ" mới giải được bài toán train trên 4GB.

---

## 2. Template cho mỗi paper note

Lưu vào `docs/paper_notes/<slug>.md`:

```markdown
# <Paper Title>
Venue / Year / Link / Code

## 1. Problem
Vấn đề cụ thể nào? Ai bị ảnh hưởng?

## 2. Prior work & gap
Trước đó làm thế nào? Thiếu gì?

## 3. Key idea (1 câu)
Nếu không tóm được trong 1 câu, chưa hiểu.

## 4. Method chi tiết
- Kiến trúc / thuật toán
- Loss function (viết ra công thức)
- Training procedure

## 5. Equation -> Code mapping   <-- BẮT BUỘC, quan trọng nhất
| Eq trong paper | File:line trong code | Ghi chú / khác biệt |
|---|---|---|

## 6. Experiments trong paper
Dataset / resolution / iterations / hardware / metrics

## 7. Results đã báo cáo
(chép nguyên số, ghi rõ điều kiện đo)

## 8. Limitations
Cả cái tác giả thừa nhận lẫn cái mình tự thấy.

## 9. Liên quan tới project này
- Có giảm train-time VRAM không? Tại sao?
- Có reproduce được trên 4GB không? Cần sửa gì?
- Nó là baseline, là method để reproduce, hay là related work?

## 10. Reproduction của mình
| Metric | Paper | Của mình | Điều kiện khác nhau | Giải thích chênh lệch |
|---|---|---|---|---|
```

---

## 3. Câu hỏi phải trả lời được sau P3

Nếu chưa trả lời được, literature review chưa xong:

1. Trong tất cả method đã đọc, cái nào thực sự giảm **peak training VRAM** (không phải model size)?
2. Taming 3DGS target primitive count. Nó có **guarantee** peak VRAM không? Tại sao có/không?
3. Rasterizer workspace scale theo cái gì? Có paper nào model hoá nó chưa?
4. Ở cùng bit-budget, giảm số Gaussian hay giảm SH degree tốt hơn? Có ai trả lời rõ chưa?
5. Method nào **reproduce được** trên 4GB với công sức hợp lý? (đây là tiêu chí chọn C5)
6. Các paper báo cáo model size theo định dạng nào? Có entropy coding không? (xem `02_METHODOLOGY.md` §3)

---

## 4. Nguồn đã tra (2026-09-01)

- Taming 3DGS — https://arxiv.org/abs/2406.15643 · https://humansensinglab.github.io/taming-3dgs/ · https://dl.acm.org/doi/10.1145/3680528.3687694
- gsplat — https://arxiv.org/pdf/2409.06765 · https://docs.gsplat.studio/
- Reduced-3DGS (INRIA) — https://repo-sam.inria.fr/fungraph/reduced_3dgs/ · https://github.com/graphdeco-inria/reduced-3dgs
- 3DGS gốc — https://github.com/graphdeco-inria/gaussian-splatting
- 3DGS.zip survey (CGF 2025) — https://onlinelibrary.wiley.com/doi/10.1111/cgf.70078
- Compression in 3DGS: A Survey — https://arxiv.org/abs/2502.19457
- MEGS² — https://arxiv.org/pdf/2509.07021
- Splatwizard — https://arxiv.org/html/2512.24742v1
