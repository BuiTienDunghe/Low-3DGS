#!/usr/bin/env bash
# =============================================================================
#  EXP-018 — CHẠY LẶP LẤY THANH SAI SỐ cho cặp đối chứng (EXP-016/017).
#
#  Vì sao cần: EXP-016/017 kết luận "nén tốn -0.1059 dB". Con số đó NHỎ.
#  Với đúng một lần chạy mỗi nhánh, không thể phân biệt nó với nhiễu.
#
#  `safe_state` (utils/general_utils.py) cố định random/np/torch seed = 0 và
#  KHÔNG có cờ dòng lệnh. Đã patch cho đọc biến môi trường L3DGS_SEED
#  (không đặt -> 0 = hành vi gốc y hệt). Nhờ vậy tách được HAI loại biến thiên:
#
#    seed 0 chạy lại  -> chỉ khác do NHÂN CUDA KHÔNG TẤT ĐỊNH
#                        (atomicAdd trong backward của rasterizer cộng số thực
#                         theo thứ tự khác nhau mỗi lần). Trả lời:
#                        "chạy lại hai lần có ra cùng số không?"
#    seed 1, 2        -> thêm biến thiên do THỨ TỰ ẢNH khác. Trả lời:
#                        "-0.1059 dB có phân biệt được với 0 không?"
#
#  6 lần chạy, mỗi cặp control+prune ~27 phút => tổng ~80 phút.
#  Chạy theo thứ tự seed để dừng lúc nào cũng có các CẶP hoàn chỉnh.
#
#  Không lưu .ply (--save_iterations 0): số dùng cho thanh sai số lấy từ eval
#  trong vòng lặp, giống hệt EXP-016/017. Lưu file không ảnh hưởng kết quả
#  (chạy dưới no_grad, sau huấn luyện) nhưng tốn ~4 phút và 1 GB.
#
#  Dùng:  bash scripts/run_repeats.sh
# =============================================================================
set -uo pipefail

PROJ=/mnt/d/low-3DGS
REPO=~/l3dgs/third_party/gaussian-splatting-pup
PY=~/l3dgs/venv/bin/python
SRC=$PROJ/datasets/tandt/train
PLY=$PROJ/datasets/pretrained/models/train/point_cloud/iteration_30000/point_cloud.ply
RESULTS=$PROJ/experiments/repeats_train.csv

[ -f "$RESULTS" ] || echo "seed,arm,iteration,occurrence,psnr,ssim,lpips,l1,n_gauss,vram_mib,wall_s" > "$RESULTS"

echo "===== KIỂM TRA MỘT LẦN TRƯỚC TOÀN BỘ LOẠT ====="
sz=$(stat -c %s "$PLY" 2>/dev/null || echo 0)
[ "$sz" -eq 254575516 ] || { echo "✗ checkpoint train sai kích thước: $sz"; exit 1; }
grep -q "L3DGS_SEED" $REPO/utils/general_utils.py || { echo "✗ chưa patch seed"; exit 1; }
echo "  ✓ checkpoint đúng · ✓ đã patch seed"
DF=$(df -BG /mnt/d | tail -1 | awk '{gsub("G","",$4); print $4}')
echo "  ✓ đĩa còn ${DF}G"
echo

RUN=0
for SEED in 0 1 2; do
for ARM in control prune; do
  RUN=$((RUN+1))
  if [ "$ARM" = prune ]; then PRUNE_IT=30001; PORT=6411; else PRUNE_IT=99999; PORT=6412; fi
  TAG=rep_train_${ARM}_s${SEED}
  OUT=$PROJ/experiments/$TAG
  LOG=$OUT.log
  MON=$OUT.mon

  echo "############################################################"
  echo "# [$RUN/6] arm=$ARM seed=$SEED  ->  $TAG   ($(date +%H:%M:%S))"
  echo "############################################################"

  if [ -e "$OUT" ]; then echo "  ! đã tồn tại, bỏ qua"; continue; fi

  n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)
  if [ "$n" -ne 0 ]; then echo "  ✗ có $n tiến trình CUDA lạ — dừng loạt"; exit 1; fi

  ( while true; do printf '%s %s\n' \
      "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)" \
      "$(ps -o rss= -C python 2>/dev/null | sort -n | tail -1)"; sleep 1; done ) > "$MON" 2>/dev/null &
  MONPID=$!

  T0=$(date +%s)
  cd "$REPO"
  L3DGS_SEED=$SEED PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  $PY -u prune_finetune.py \
    -s "$SRC" -m "$OUT" -r 1 --eval --data_device cpu --port $PORT \
    --start_pointcloud "$PLY" \
    --iterations 35000 \
    --prune_iterations $PRUNE_IT \
    --prune_percent 0.66 --prune_type v_important_score --v_pow 0.1 \
    --test_iterations 30001 35000 \
    --save_iterations 0 --checkpoint_iterations 0 \
    > "$LOG" 2>&1
  RC=$?
  T1=$(date +%s)
  kill $MONPID 2>/dev/null || true

  WALL=$((T1-T0))
  VRAM=$(awk 'NF>=1{if($1>m)m=$1} END{print m+0}' "$MON" 2>/dev/null)
  NG=$(grep -oP 'Number of Gaussians is \K[0-9]+' "$LOG" | tail -1)
  [ -z "$NG" ] && NG=?

  if [ "$RC" -ne 0 ]; then
    echo "  ✗ THẤT BẠI (exit $RC) — xem $LOG"
    grep -iE "out of memory|Error|Killed" "$LOG" | head -3 | sed 's/^/    /'
    continue
  fi

  # cổng: prune phải chạy đúng ý đồ của nhánh
  if grep -qi "prune round" "$LOG"; then HASP=1; else HASP=0; fi
  if [ "$ARM" = prune ] && [ "$HASP" -ne 1 ]; then echo "  ✗ nhánh prune mà prune KHÔNG chạy"; continue; fi
  if [ "$ARM" = control ] && [ "$HASP" -ne 0 ]; then echo "  ✗ nhánh control mà prune ĐÃ chạy"; continue; fi

  # gom metric.csv -> repeats_train.csv
  awk -F, -v s="$SEED" -v a="$ARM" -v ng="$NG" -v v="$VRAM" -v w="$WALL" '
    NR>1 { c[$1]++; printf "%s,%s,%s,%d,%s,%s,%s,%s,%s,%s,%s\n", s,a,$1,c[$1]-1,$4,$5,$6,$3,ng,v,w }
  ' "$OUT/metric.csv" >> "$RESULTS"

  echo "  ✓ xong: N=$NG · VRAM đỉnh ${VRAM} MiB · ${WALL}s"
  grep -oP '\[ITER \K[0-9]+\] Evaluating test: L1 [0-9.]+ PSNR [0-9.]+' "$LOG" | sed 's/^/    ITER /'
  echo
done
done

echo "############################################################"
echo "# XONG TOÀN BỘ LOẠT ($(date +%H:%M:%S))"
echo "############################################################"
column -s, -t "$RESULTS"
