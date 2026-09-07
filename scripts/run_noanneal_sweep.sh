#!/usr/bin/env bash
# =============================================================================
#  EXP-025 — ATTRACTOR CÓ PHẢI DO ANNEAL TẠO RA KHÔNG?
#
#  EXP-023: với anneal BẬT, điểm xuất phát trải 1.5588 dB (cắt 20-70%) nhưng
#  điểm dừng chỉ trải 0.1668 dB — co lại 9.3 lần. Gọi đó là attractor.
#  EXP-024: tắt anneal ở nhánh ĐỐI CHỨNG thì độ phân tán điểm dừng tăng 1.93 lần.
#
#  Nhưng EXP-024 chỉ đo ở MỘT điểm xuất phát (0% cắt). Nó không cho biết attractor
#  có còn kéo các điểm xuất phát KHÁC NHAU về cùng chỗ hay không. Đây là phép đo đó.
#
#  Vì sao quan trọng: nếu attractor do anneal tạo ra thì chuỗi phụ thuộc là
#     "nén tới 50% gần như miễn phí" <- cùng attractor <- anneal <- thứ PUP THÊM VÀO
#  tức phát hiện chính của dự án nói về CÔNG THỨC TINH CHỈNH, không phải về NÉN.
#
#  Chạy 3 mức cắt với L3DGS_NO_ANNEAL=1, seed 0 — đúng các mức đã có số với anneal bật.
# =============================================================================
set -uo pipefail
PROJ=/mnt/d/low-3DGS
REPO=~/l3dgs/third_party/gaussian-splatting-pup
PY=~/l3dgs/venv/bin/python
SRC=$PROJ/datasets/tandt/train
PLY=$PROJ/datasets/pretrained/models/train/point_cloud/iteration_30000/point_cloud.ply
RESULTS=$PROJ/experiments/exp025_noanneal_sweep.csv
export PYTHONPATH=$HOME/l3dgs/tools_memprobe

PCTS=(0.20 0.50 0.70)
EXPN=(821206 513254 307953)

[ -f "$RESULTS" ] || echo "pct,anneal,seed,iteration,occurrence,psnr,ssim,lpips,l1,n_gauss,vram_alloc_mib,wall_s" > "$RESULTS"

echo "===== CỔNG MỘT LẦN ====="
[ "$(stat -c %s "$PLY")" -eq 254575516 ] && echo "  ✓ checkpoint đúng" || { echo "  ✗ sai"; exit 1; }
grep -q "L3DGS_NO_ANNEAL" $REPO/prune_finetune.py && echo "  ✓ đã patch tắt anneal" || { echo "  ✗ chưa patch"; exit 1; }
cd "$PROJ"; [ -z "$(git status --porcelain)" ] && echo "  ✓ cây SẠCH — commit $(git rev-parse --short HEAD)" || echo "  ⚠ cây bẩn"
echo
cat <<'PRED'
DỰ ĐOÁN GHI TRƯỚC KHI CHẠY
  Với anneal BẬT (seed 0):
     cắt 20%  xuất phát 21.8144  ->  điểm dừng 22.1757
     cắt 50%  xuất phát 21.5694  ->  điểm dừng 22.1787
     cắt 70%  xuất phát 20.2556  ->  điểm dừng 22.0118
     => xuất phát trải 1.5588 dB, điểm dừng trải 0.1669 dB, CO LẠI 9.3 lần

  PHÁN QUYẾT khi tắt anneal, dựa trên hệ số co lại:
     co ≤ 2 lần   -> attractor DO ANNEAL TẠO RA. "Vùng miễn phí" là tính chất
                     của công thức tinh chỉnh, không phải của phép nén.
     co ≥ 5 lần   -> attractor TỒN TẠI ĐỘC LẬP với anneal. Diễn giải EXP-024 SAI,
                     "vùng miễn phí" vẫn là chuyện của nén.
     2 - 5 lần    -> anneal góp phần nhưng không phải nguyên nhân duy nhất.
PRED
echo

for i in "${!PCTS[@]}"; do
  P=${PCTS[$i]}; EN=${EXPN[$i]}
  TAG=exp025_noanneal_p$(echo "$P" | tr -d '.')
  OUT=$PROJ/experiments/$TAG; LOG=$OUT.log
  echo "############################################################"
  echo "# [$((i+1))/3] cắt=$P (N kỳ vọng $EN), anneal TẮT   ($(date +%H:%M:%S))"
  echo "############################################################"
  [ -e "$OUT" ] && { echo "  ! đã có, bỏ qua"; continue; }
  k=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)
  [ "${k:-0}" -eq 0 ] || { echo "  ✗ có tiến trình CUDA lạ"; exit 1; }

  T0=$(date +%s); cd "$REPO"
  L3DGS_NO_ANNEAL=1 L3DGS_SEED=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  $PY -u prune_finetune.py \
    -s "$SRC" -m "$OUT" -r 1 --eval --data_device cpu --port 6461 \
    --start_pointcloud "$PLY" \
    --iterations 35000 --prune_iterations 30001 \
    --prune_percent "$P" --prune_type v_important_score --v_pow 0.1 \
    --test_iterations 30001 35000 \
    --save_iterations 0 --checkpoint_iterations 0 \
    > "$LOG" 2>&1
  RC=$?; T1=$(date +%s); WALL=$((T1-T0)); cd "$PROJ"

  [ "$RC" -ne 0 ] && { echo "  ✗ THẤT BẠI (exit $RC)"; grep -iE "error|Traceback|out of memory" "$LOG" | head -3 | sed 's/^/    /'; continue; }
  NEVAL=$(grep -c 'Evaluating test' "$LOG")
  [ "$NEVAL" -eq 3 ] || { echo "  ✗ thấy $NEVAL lần eval, nhánh nén cần 3"; continue; }
  NG=$(grep -oP 'Number of Gaussians is \K[0-9]+' "$LOG" | tail -1)
  [ "$NG" = "$EN" ] && echo "  ✓ N=$NG khớp" || echo "  ⚠ N=$NG khác dự đoán $EN"
  VA=$(grep -oP 'max_allocated_MiB=\K[0-9.]+' "$LOG" | tail -1)

  awk -F, -v p="$P" -v ng="$NG" -v va="$VA" -v w="$WALL" '
    NR>1 { c[$1]++; printf "%s,off,0,%s,%d,%s,%s,%s,%s,%s,%s,%s\n", p,$1,c[$1]-1,$4,$5,$6,$3,ng,va,w }
  ' "$OUT/metric.csv" >> "$RESULTS"

  bash tools/stamp_run.sh "experiments/$TAG"
  S=$(grep -oP 'PSNR \K[0-9.]+' "$LOG" | sed -n 2p)
  E=$(grep -oP 'PSNR \K[0-9.]+' "$LOG" | tail -1)
  echo "  ✓ xong: xuất phát $S -> điểm dừng $E · VRAM $VA MiB · ${WALL}s"
  echo
done

echo "############################################################"
echo "# XONG ($(date +%H:%M:%S))"
echo "############################################################"
column -s, -t "$RESULTS"
