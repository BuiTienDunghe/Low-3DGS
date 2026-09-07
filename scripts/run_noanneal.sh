#!/usr/bin/env bash
# =============================================================================
#  EXP-024 — TẮT ANNEAL LEARNING RATE. Kiểm biến chống đỡ luận điểm chính.
#
#  CÂU HỎI: C1 = +0.3587 dB là "checkpoint chưa hội tụ", hay chỉ là phần thưởng
#  MỘT LẦN của việc anneal learning rate mà giai đoạn fine-tune tự mang theo?
#
#  PUP tạo ExponentialLR(gamma=0.95) và step mỗi 400 bước -> 12 lần trong 5000 bước
#  -> LR cuối = 0.54x. 3DGS GỐC KHÔNG anneal feature/opacity/scaling/rotation_lr
#  trong 30k bước đầu. Vậy anneal là thứ giai đoạn fine-tune THÊM VÀO, không phải
#  thứ model "còn thiếu".
#
#  Hai cách hiểu suy rộng NGƯỢC CHIỀU nhau:
#    "checkpoint của các bài báo chưa hội tụ"          -> phê phán checkpoint người khác
#    "fine-tune tặng +0.36 dB cho bất kỳ ai thêm nó"   -> confound nằm trong PHƯƠNG PHÁP
#  Không phép đo nào hiện có phân biệt được chúng. Đây là phép đo đó.
#
#  Nhánh ĐỐI CHỨNG (không prune) x 3 seed, chỉ khác EXP-018 đúng một biến:
#  L3DGS_NO_ANNEAL=1.
# =============================================================================
set -uo pipefail
PROJ=/mnt/d/low-3DGS
REPO=~/l3dgs/third_party/gaussian-splatting-pup
PY=~/l3dgs/venv/bin/python
SRC=$PROJ/datasets/tandt/train
PLY=$PROJ/datasets/pretrained/models/train/point_cloud/iteration_30000/point_cloud.ply
RESULTS=$PROJ/experiments/exp024_noanneal.csv
export PYTHONPATH=$HOME/l3dgs/tools_memprobe

[ -f "$RESULTS" ] || echo "arm,anneal,seed,iteration,occurrence,psnr,ssim,lpips,l1,n_gauss,vram_alloc_mib,wall_s" > "$RESULTS"

echo "===== CỔNG MỘT LẦN ====="
[ "$(stat -c %s "$PLY")" -eq 254575516 ] && echo "  ✓ checkpoint train đúng" || { echo "  ✗ checkpoint sai"; exit 1; }
grep -q "L3DGS_NO_ANNEAL" $REPO/prune_finetune.py && echo "  ✓ đã patch tắt anneal" || { echo "  ✗ CHƯA patch"; exit 1; }
cd "$PROJ" && [ -z "$(git status --porcelain)" ] && echo "  ✓ cây làm việc SẠCH — run sẽ gắn được commit thật" \
  || echo "  ⚠ cây bẩn — run_meta sẽ ghi dirty:true"
echo
echo "DỰ ĐOÁN GHI TRƯỚC KHI CHẠY:"
echo "  C1 CÓ anneal (EXP-018, n=3) = +0.3587 ± 0.0551 dB"
echo "  Kỳ vọng C1 KHÔNG anneal < C1, vì anneal thường giúp ổn định ở cuối."
echo "  Phán quyết:  ≥ 0.30  -> anneal KHÔNG phải nguồn chính của C1"
echo "               ≤ 0.15  -> anneal chiếm ≥ 58% của C1, luận điểm phải viết lại"
echo "               0.15–0.30 -> đóng góp một phần, cần thêm seed"
echo

for S in 0 1 2; do
  TAG=exp024_noanneal_s$S
  OUT=$PROJ/experiments/$TAG; LOG=$OUT.log
  echo "############################################################"
  echo "# [$((S+1))/3] seed=$S  ->  $TAG   ($(date +%H:%M:%S))"
  echo "############################################################"
  [ -e "$OUT" ] && { echo "  ! đã có, bỏ qua"; continue; }
  k=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)
  [ "${k:-0}" -eq 0 ] || { echo "  ✗ có tiến trình CUDA lạ"; exit 1; }

  T0=$(date +%s)
  cd "$REPO"
  L3DGS_NO_ANNEAL=1 L3DGS_SEED=$S PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  $PY -u prune_finetune.py \
    -s "$SRC" -m "$OUT" -r 1 --eval --data_device cpu --port 6451 \
    --start_pointcloud "$PLY" \
    --iterations 35000 \
    --prune_iterations 99999 \
    --prune_percent 0.50 --prune_type v_important_score --v_pow 0.1 \
    --test_iterations 30001 35000 \
    --save_iterations 0 --checkpoint_iterations 0 \
    > "$LOG" 2>&1
  RC=$?; T1=$(date +%s); WALL=$((T1-T0))
  cd "$PROJ"

  [ "$RC" -ne 0 ] && { echo "  ✗ THẤT BẠI (exit $RC)"; grep -iE "error|Traceback|out of memory" "$LOG" | head -3 | sed 's/^/    /'; continue; }
  NEVAL=$(grep -c 'Evaluating test' "$LOG")
  [ "$NEVAL" -eq 2 ] || { echo "  ✗ thấy $NEVAL lần eval, cần 2"; continue; }
  grep -qi "prune round" "$LOG" && { echo "  ✗ prune ĐÃ chạy — sai nhánh"; continue; }

  VA=$(grep -oP 'max_allocated_MiB=\K[0-9.]+' "$LOG" | tail -1)
  NG=$(grep -oP 'Number of Gaussians is \K[0-9]+' "$LOG" | tail -1)
  END=$(grep -oP 'PSNR \K[0-9.]+' "$LOG" | tail -1)

  # CỔNG QUAN TRỌNG: nếu bằng ĐÚNG kết quả có-anneal thì cờ đã không ăn.
  if [ "$S" = 0 ] && [ "$END" = "22.179856877577930" ]; then
    echo "  ✗ điểm dừng TRÙNG KHÍT run có anneal -> L3DGS_NO_ANNEAL KHÔNG ăn"; continue
  fi

  awk -F, -v s="$S" -v ng="$NG" -v va="$VA" -v w="$WALL" '
    NR>1 { c[$1]++; printf "control,off,%s,%s,%d,%s,%s,%s,%s,%s,%s,%s\n", s,$1,c[$1]-1,$4,$5,$6,$3,ng,va,w }
  ' "$OUT/metric.csv" >> "$RESULTS"

  bash tools/stamp_run.sh "experiments/$TAG"
  echo "  ✓ xong: N=$NG · VRAM $VA MiB · ${WALL}s · điểm dừng $END"
  echo
done

echo "############################################################"
echo "# XONG ($(date +%H:%M:%S))"
echo "############################################################"
column -s, -t "$RESULTS"
