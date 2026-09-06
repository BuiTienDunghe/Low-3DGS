#!/usr/bin/env bash
# =============================================================================
#  EXP-021 — CHỐT ĐẦU GỐI: thêm seed cho mức cắt 50% và 60%.
#
#  Vì sao chỉ hai mức này: đường cong EXP-019 chỉ có seed 0. Ranh giới giữa
#  "nén miễn phí" và "nén tốn" nằm giữa 50% (−0.004, trong nhiễu) và
#  60% (−0.044, mới 2.4σ — CHƯA CHẮC). Khuyến nghị chính của dự án
#  ("cắt 50%, đừng cắt 66%") dựa đúng vào điểm đó.
#
#  66% KHÔNG cần chạy: đã có seed 0/1/2 từ EXP-018
#  (rep_train_prune_s{0,1,2} = 22.076434 / 22.048626 / 22.133967).
#  Đối chứng cũng đã có 3 seed. => chỉ thiếu 4 lần chạy.
#
#  Ghi thêm vào sweep_train.csv (đã có cột seed). tradeoff_curve.py đã được
#  sửa để LỌC seed 0 khi vẽ đường cong, nên thêm seed không làm hỏng hình cũ.
# =============================================================================
set -uo pipefail

PROJ=/mnt/d/low-3DGS
REPO=~/l3dgs/third_party/gaussian-splatting-pup
PY=~/l3dgs/venv/bin/python
SRC=$PROJ/datasets/tandt/train
PLY=$PROJ/datasets/pretrained/models/train/point_cloud/iteration_30000/point_cloud.ply
RESULTS=$PROJ/experiments/sweep_train.csv

PCTS=(0.50 0.50 0.60 0.60)
SEEDS=(1    2    1    2)
EXPN=(513254 513254 410603 410603)   # N không phụ thuộc seed: prune chạy trước bước tối ưu đầu tiên

echo "===== KIỂM TRA MỘT LẦN ====="
[ -f "$RESULTS" ] || { echo "✗ thiếu $RESULTS"; exit 1; }
[ "$(stat -c %s "$PLY")" -eq 254575516 ] || { echo "✗ checkpoint sai"; exit 1; }
grep -q "L3DGS_SEED" $REPO/utils/general_utils.py || { echo "✗ chưa patch seed"; exit 1; }
echo "  ✓ checkpoint · ✓ patch seed · ✓ đĩa $(df -BG /mnt/d | tail -1 | awk '{print $4}')"
echo "  4 lần chạy, ước lượng ~52 phút."
echo

for i in "${!PCTS[@]}"; do
  P=${PCTS[$i]}; S=${SEEDS[$i]}; EN=${EXPN[$i]}
  TAG=knee_train_p$(echo "$P" | tr -d '.')_s$S
  OUT=$PROJ/experiments/$TAG
  LOG=$OUT.log; MON=$OUT.mon

  echo "############################################################"
  echo "# [$((i+1))/4] cắt=$P seed=$S  ->  $TAG   ($(date +%H:%M:%S))"
  echo "############################################################"
  if [ -e "$OUT" ]; then echo "  ! đã có, bỏ qua"; continue; fi

  n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)
  if [ "$n" -ne 0 ]; then echo "  ✗ có $n tiến trình CUDA lạ — dừng loạt"; exit 1; fi

  ( while true; do printf '%s\n' \
      "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)"; sleep 1; done ) > "$MON" 2>/dev/null &
  MONPID=$!

  T0=$(date +%s)
  cd "$REPO"
  L3DGS_SEED=$S PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  $PY -u prune_finetune.py \
    -s "$SRC" -m "$OUT" -r 1 --eval --data_device cpu --port 6431 \
    --start_pointcloud "$PLY" \
    --iterations 35000 \
    --prune_iterations 30001 \
    --prune_percent "$P" --prune_type v_important_score --v_pow 0.1 \
    --test_iterations 30001 35000 \
    --save_iterations 0 --checkpoint_iterations 0 \
    > "$LOG" 2>&1
  RC=$?; T1=$(date +%s)
  kill $MONPID 2>/dev/null || true

  WALL=$((T1-T0))
  VRAM=$(awk 'NF>=1{if($1>m)m=$1} END{print m+0}' "$MON" 2>/dev/null)
  NG=$(grep -oP 'Number of Gaussians is \K[0-9]+' "$LOG" | tail -1)

  if [ "$RC" -ne 0 ]; then
    echo "  ✗ THẤT BẠI (exit $RC)"; grep -iE "out of memory|Error|Killed" "$LOG" | head -3 | sed 's/^/    /'; continue
  fi
  grep -qi "prune round" "$LOG" || { echo "  ✗ prune KHÔNG chạy — bỏ điểm này"; continue; }
  [ "$NG" = "$EN" ] && echo "  ✓ N=$NG khớp" || echo "  ⚠ N=$NG khác dự đoán $EN"

  awk -F, -v p="$P" -v s="$S" -v ng="$NG" -v v="$VRAM" -v w="$WALL" '
    NR>1 { c[$1]++; printf "%s,%s,%s,%d,%s,%s,%s,%s,%s,%s,%s\n", p,s,$1,c[$1]-1,$4,$5,$6,$3,ng,v,w }
  ' "$OUT/metric.csv" >> "$RESULTS"

  echo "  ✓ xong: VRAM đỉnh ${VRAM} MiB · ${WALL}s"
  grep -oP 'PSNR \K[0-9.]+' "$LOG" | tail -1 | sed 's/^/    PSNR sau phục hồi: /'
  echo
done

echo "############################################################"
echo "# XONG ($(date +%H:%M:%S))"
echo "############################################################"
