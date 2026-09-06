#!/usr/bin/env bash
# =============================================================================
#  EXP-019 — ĐƯỜNG CONG ĐÁNH ĐỔI: chất lượng theo tỉ lệ cắt hạt.
#
#  ĐIỂM MẤU CHỐT của thiết kế: nhánh đối chứng KHÔNG phụ thuộc tỉ lệ cắt
#  (không cắt gì thì cắt bao nhiêu % cũng vô nghĩa). Nên chỉ cần các nhánh prune;
#  đối chứng dùng lại 3 lần chạy của EXP-018 (trung bình 22.1744 dB).
#  => tiết kiệm một nửa số lần chạy.
#
#  Mỗi điểm cho ra HAI đường:
#    - đường "ngây thơ"  : so với checkpoint gốc 21.8157   <- các bài báo hay vẽ
#    - đường "trung thực": so với đối chứng      22.1744   <- đã trừ phần luyện thêm
#  Khoảng cách giữa hai đường chính là +0.3587 dB "luyện thêm" (EXP-018).
#
#  seed 0 cho mọi điểm. Căn cứ EXP-018: nhiễu ghép cặp của SSIM/LPIPS ~0.0001
#  (coi như bằng 0), của PSNR ~0.018. n=1 đủ để dựng HÌNH DẠNG đường cong;
#  thêm seed sau ở những điểm đáng quan tâm.
#
#  0.66 KHÔNG chạy lại — đã có sẵn ở experiments/rep_train_prune_s0 (cùng seed,
#  cùng mọi cờ, cùng --save_iterations 0). Gộp lúc vẽ.
#
#  Dùng:  bash scripts/run_sweep.sh
# =============================================================================
set -uo pipefail

PROJ=/mnt/d/low-3DGS
REPO=~/l3dgs/third_party/gaussian-splatting-pup
PY=~/l3dgs/venv/bin/python
SRC=$PROJ/datasets/tandt/train
PLY=$PROJ/datasets/pretrained/models/train/point_cloud/iteration_30000/point_cloud.ply
RESULTS=$PROJ/experiments/sweep_train.csv
SEED=0

# tỉ lệ -> N kỳ vọng.  N_new = N - (int(p*(N-1)) + 1),  N = 1026508
# Công thức lấy từ gaussian_model.py:405-410, đã khớp từng đơn vị ở 0.66 -> 349013.
PCTS=(0.20 0.40 0.50 0.60 0.70 0.80 0.90 0.95)
EXPN=(821206 615905 513254 410603 307953 205302 102651 51326)

[ -f "$RESULTS" ] || echo "pct,seed,iteration,occurrence,psnr,ssim,lpips,l1,n_gauss,vram_mib,wall_s" > "$RESULTS"

echo "===== KIỂM TRA MỘT LẦN ====="
sz=$(stat -c %s "$PLY" 2>/dev/null || echo 0)
[ "$sz" -eq 254575516 ] || { echo "✗ checkpoint sai: $sz"; exit 1; }
grep -q "L3DGS_SEED" $REPO/utils/general_utils.py || { echo "✗ chưa patch seed"; exit 1; }
DF=$(df -BG /mnt/d | tail -1 | awk '{gsub("G","",$4); print $4}')
echo "  ✓ checkpoint · ✓ patch seed · ✓ đĩa ${DF}G"
echo "  Sẽ chạy ${#PCTS[@]} điểm, ước lượng ~$(( ${#PCTS[@]} * 12 )) phút."
echo

for i in "${!PCTS[@]}"; do
  P=${PCTS[$i]}
  EN=${EXPN[$i]}
  TAG=sweep_train_p$(echo "$P" | tr -d '.')
  OUT=$PROJ/experiments/$TAG
  LOG=$OUT.log
  MON=$OUT.mon

  echo "############################################################"
  echo "# [$((i+1))/${#PCTS[@]}] prune_percent=$P  N kỳ vọng=$EN   ($(date +%H:%M:%S))"
  echo "############################################################"

  if [ -e "$OUT" ]; then echo "  ! đã có, bỏ qua"; continue; fi

  n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)
  if [ "$n" -ne 0 ]; then echo "  ✗ có $n tiến trình CUDA lạ — dừng loạt"; exit 1; fi

  ( while true; do printf '%s\n' \
      "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)"; sleep 1; done ) > "$MON" 2>/dev/null &
  MONPID=$!

  T0=$(date +%s)
  cd "$REPO"
  L3DGS_SEED=$SEED PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  $PY -u prune_finetune.py \
    -s "$SRC" -m "$OUT" -r 1 --eval --data_device cpu --port 6421 \
    --start_pointcloud "$PLY" \
    --iterations 35000 \
    --prune_iterations 30001 \
    --prune_percent "$P" --prune_type v_important_score --v_pow 0.1 \
    --test_iterations 30001 35000 \
    --save_iterations 0 --checkpoint_iterations 0 \
    > "$LOG" 2>&1
  RC=$?
  T1=$(date +%s)
  kill $MONPID 2>/dev/null || true

  WALL=$((T1-T0))
  VRAM=$(awk 'NF>=1{if($1>m)m=$1} END{print m+0}' "$MON" 2>/dev/null)
  NG=$(grep -oP 'Number of Gaussians is \K[0-9]+' "$LOG" | tail -1)

  if [ "$RC" -ne 0 ]; then
    echo "  ✗ THẤT BẠI (exit $RC)"; grep -iE "out of memory|Error|Killed" "$LOG" | head -3 | sed 's/^/    /'; continue
  fi
  grep -qi "prune round" "$LOG" || { echo "  ✗ prune KHÔNG chạy — bỏ điểm này"; continue; }
  if [ "$NG" != "$EN" ]; then echo "  ⚠ N=$NG khác dự đoán $EN (vẫn ghi lại)"; else echo "  ✓ N=$NG khớp dự đoán"; fi

  awk -F, -v p="$P" -v s="$SEED" -v ng="$NG" -v v="$VRAM" -v w="$WALL" '
    NR>1 { c[$1]++; printf "%s,%s,%s,%d,%s,%s,%s,%s,%s,%s,%s\n", p,s,$1,c[$1]-1,$4,$5,$6,$3,ng,v,w }
  ' "$OUT/metric.csv" >> "$RESULTS"

  echo "  ✓ xong: VRAM đỉnh ${VRAM} MiB · ${WALL}s"
  grep -oP 'PSNR \K[0-9.]+' "$LOG" | tail -2 | paste -sd' / ' | sed 's/^/    PSNR (sau prune \/ sau phục hồi): /'
  echo
done

echo "############################################################"
echo "# XONG ($(date +%H:%M:%S))"
echo "############################################################"
column -s, -t "$RESULTS"
