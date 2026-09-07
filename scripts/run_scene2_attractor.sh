#!/usr/bin/env bash
# =============================================================================
#  EXP-026 — ATTRACTOR CÓ TỒN TẠI TRÊN SCENE THỨ HAI KHÔNG?
#
#  EXP-023/025 tìm ra attractor trên `train`: điểm xuất phát trải 1.56 dB nhưng
#  điểm dừng chỉ trải 0.17 dB (co 9.3x, tụ về 22.1762). EXP-025 chứng minh nó
#  không phải artifact của lịch learning rate.
#
#  Nhưng nó mới có ở MỘT scene. Attractor giờ là CƠ CHẾ chống đỡ toàn bộ cách
#  diễn giải của dự án:
#     C1 = khoảng cách từ checkpoint tới attractor
#     "nén 50% miễn phí" = hai nhánh cùng rơi về một điểm dừng
#     đầu gối = biên miền hút
#  Nếu attractor chỉ có trên `train` thì mọi phát biểu phải hạ xuống
#  "quan sát trên một scene". Nếu nó có cả ở scene 2 thì đó là tính chất của
#  CÔNG THỨC TINH CHỈNH nói chung — thứ áp được cho bài của người khác.
#
#  CHỌN MỨC CẮT NHẸ: ở scene 2, cắt 50% đã cho chi phí -0.2946 dB, tức ĐÃ NGOÀI
#  miền hút. Nên dùng 10/20/30% để ở trong miền.
# =============================================================================
set -uo pipefail
PROJ=/mnt/d/low-3DGS
REPO=~/l3dgs/third_party/gaussian-splatting-pup
PY=~/l3dgs/venv/bin/python
SRC=$PROJ/datasets/tandt/truck
PLY=$PROJ/experiments/lg_truck_v66_ft5k/point_cloud/iteration_35000/point_cloud.ply
SHA_OK=5c837cbf1f3f0479c76d3576aa48973a2dacebd5f0ecb122dc9fa4c5fbf5c56c
RESULTS=$PROJ/experiments/exp026_scene2_attractor.csv
export PYTHONPATH=$HOME/l3dgs/tools_memprobe

# N sau prune: N - (int(p*(N-1)) + 1),  N = 864017
PCTS=(0.10 0.20 0.30)
EXPN=(777615 691213 604812)

[ -f "$RESULTS" ] || echo "pct,seed,iteration,occurrence,psnr,ssim,lpips,l1,n_gauss,vram_alloc_mib,wall_s" > "$RESULTS"

echo "===== CỔNG MỘT LẦN ====="
SHA=$(sha256sum "$PLY" | cut -d' ' -f1)
[ "$SHA" = "$SHA_OK" ] && echo "  ✓ model nền đúng (truck-864k @35000)" || { echo "  ✗ sha256 sai: $SHA"; exit 1; }
[ "$(ls "$SRC/images" | wc -l)" -eq 251 ] && echo "  ✓ 251 ảnh truck" || { echo "  ✗ sai số ảnh"; exit 1; }
cd "$PROJ"; git update-index -q --really-refresh 2>/dev/null || true
[ -z "$(git status --porcelain -- ':!experiments')" ] && echo "  ✓ mã nguồn SẠCH — commit $(git rev-parse --short HEAD)" || echo "  ⚠ mã nguồn bẩn"
echo
cat <<'PRED'
DỰ ĐOÁN GHI TRƯỚC KHI CHẠY
  baseline truck-864k              = 25.113122522830963
  điểm dừng nhánh đối chứng (n=3)  = 25.1606  (SD giữa seed chỉ 0.0110 — chặt hơn train nhiều)
  chi phí nén ở 50%                = -0.2946  -> ĐÃ NGOÀI miền hút, nên không dùng mức đó

  PHÁN QUYẾT theo hệ số co lại (điểm xuất phát trải / điểm dừng trải):
     co ≥ 5 lần  -> CÓ attractor ở scene 2. Cơ chế suy rộng được, không phải
                    tính chất riêng của scene train.
     co ≤ 2 lần  -> KHÔNG có. Mọi phát biểu về attractor phải hạ xuống
                    "quan sát trên một scene".
  Phụ: nếu có attractor, điểm dừng ba mức phải tụ quanh ~25.16 trong vài lần nhiễu seed (0.0110).
PRED
echo

for i in "${!PCTS[@]}"; do
  P=${PCTS[$i]}; EN=${EXPN[$i]}
  TAG=exp026_s2attr_p$(echo "$P" | tr -d '.')
  OUT=$PROJ/experiments/$TAG; LOG=$OUT.log
  echo "############################################################"
  echo "# [$((i+1))/3] cắt=$P (N kỳ vọng $EN)   ($(date +%H:%M:%S))"
  echo "############################################################"
  [ -e "$OUT" ] && { echo "  ! đã có, bỏ qua"; continue; }
  k=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)
  [ "${k:-0}" -eq 0 ] || { echo "  ✗ có tiến trình CUDA lạ"; exit 1; }

  T0=$(date +%s); cd "$REPO"
  L3DGS_SEED=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  $PY -u prune_finetune.py \
    -s "$SRC" -m "$OUT" -r 1 --eval --data_device cpu --port 6471 \
    --start_pointcloud "$PLY" --first_iter 30000 \
    --iterations 35000 --prune_iterations 30001 \
    --prune_percent "$P" --prune_type v_important_score --v_pow 0.1 \
    --test_iterations 30001 35000 \
    --save_iterations 0 --checkpoint_iterations 0 \
    > "$LOG" 2>&1
  RC=$?; T1=$(date +%s); WALL=$((T1-T0)); cd "$PROJ"

  [ "$RC" -ne 0 ] && { echo "  ✗ THẤT BẠI (exit $RC)"; grep -iE "error|Traceback|out of memory" "$LOG" | head -3 | sed 's/^/    /'; continue; }
  NEVAL=$(grep -c 'Evaluating test' "$LOG")
  [ "$NEVAL" -eq 3 ] || { echo "  ✗ thấy $NEVAL lần eval, nhánh nén cần 3 — vòng lặp rỗng?"; continue; }
  B=$(grep -oP 'PSNR \K[0-9.]+' "$LOG" | head -1)
  [ "${B:0:9}" = "25.113122" ] || { echo "  ✗ baseline $B ≠ 25.113122... -> nạp nhầm model"; continue; }
  NG=$(grep -oP 'Number of Gaussians is \K[0-9]+' "$LOG" | tail -1)
  [ "$NG" = "$EN" ] && echo "  ✓ N=$NG khớp" || echo "  ⚠ N=$NG khác dự đoán $EN"
  VA=$(grep -oP 'max_allocated_MiB=\K[0-9.]+' "$LOG" | tail -1)

  awk -F, -v p="$P" -v ng="$NG" -v va="$VA" -v w="$WALL" '
    NR>1 { c[$1]++; printf "%s,0,%s,%d,%s,%s,%s,%s,%s,%s,%s\n", p,$1,c[$1]-1,$4,$5,$6,$3,ng,va,w }
  ' "$OUT/metric.csv" >> "$RESULTS"

  bash tools/stamp_run.sh "experiments/$TAG"
  S=$(grep -oP 'PSNR \K[0-9.]+' "$LOG" | sed -n 2p); E=$(grep -oP 'PSNR \K[0-9.]+' "$LOG" | tail -1)
  echo "  ✓ xong: xuất phát $S -> điểm dừng $E · VRAM $VA MiB · ${WALL}s"
  echo
done
echo "############################################################"
echo "# XONG ($(date +%H:%M:%S))"
echo "############################################################"
column -s, -t "$RESULTS"
