#!/usr/bin/env bash
# =============================================================================
#  EXP-016 (prune) / EXP-017 (control) — CẶP ĐỐI CHỨNG trên scene `train`.
#
#  BỐI CẢNH: EXP-015 đã chứng minh không thể chạy đối chứng trên `truck`.
#  Fine-tune model CHƯA nén ở N=2.54M tràn bộ nhớ ngay tại loss.backward()
#  đầu tiên (đỉnh 3567 MiB > trần 3440), và đó là TRƯỚC KHI Adam cấp phát
#  2 moment (thêm 1144 MiB). Tổng nhu cầu thật ~4.7 GB trên card 4 GB.
#  Vì vậy cặp đối chứng chuyển sang `train` (N=1,026,508) — vừa bộ nhớ.
#
#  HAI NHÁNH khác nhau ĐÚNG MỘT THỨ: prune có chạy hay không.
#    prune   : --prune_iterations 30001  (như EXP-014)
#    control : --prune_iterations 99999  (không bao giờ chạy)
#  Mọi thứ khác giống hệt: cùng seed, cùng số bước, cùng lịch LR,
#  cùng thứ tự camera (prune KHÔNG tiêu thụ RNG).
#
#  CÂU HỎI: trong +0.1123 dB của EXP-014, bao nhiêu là "nén không mất gì"
#           và bao nhiêu chỉ là "được luyện thêm 5000 bước"?
#           Trả lời = (prune@35000 - control@35000).
#
#  Dùng:  bash scripts/run_train_pair.sh prune
#         bash scripts/run_train_pair.sh control
# =============================================================================
set -euo pipefail
set -o pipefail

MODE="${1:-}"
case "$MODE" in
  prune)   PRUNE_IT=30001; TAG=exp016_train_v66_ft5k;  PORT=6403; EXPECT_N=349013;  EXPECT_B=86556756  ;;
  control) PRUNE_IT=99999; TAG=exp017_train_noprune;   PORT=6404; EXPECT_N=1026508; EXPECT_B=254575516 ;;
  *) echo "Dùng: $0 prune|control"; exit 2 ;;
esac

PROJ=/mnt/d/low-3DGS
REPO=~/l3dgs/third_party/gaussian-splatting-pup
PY=~/l3dgs/venv/bin/python
SCENE=train
SRC=$PROJ/datasets/tandt/$SCENE
PLY=$PROJ/datasets/pretrained/models/$SCENE/point_cloud/iteration_30000/point_cloud.ply
OUT=$PROJ/experiments/$TAG
LOG=$OUT.log
MON=$OUT.mon

fail() { echo "  ✗ $1"; FAILED=1; }
pass() { echo "  ✓ $1"; }
FAILED=0

echo "===== NHÁNH: $MODE  ($TAG) ====="
echo "===== CỔNG TRƯỚC KHI CHẠY ====="

n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)
[ "$n" -eq 0 ] && pass "A1 GPU không có compute app" || fail "A1 có $n tiến trình CUDA"

n=$(pgrep -af '/l3dgs/venv/bin/python' | grep -vc "$$" || true)
[ "$n" -eq 0 ] && pass "A2 không có python venv đang chạy" || fail "A2 còn $n tiến trình python"

ALLOC=$(PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -c "
import torch; torch.cuda.init(); torch.zeros(1,device='cuda'); b=[]
try:
    while True: b.append(torch.empty(64*1024*1024,dtype=torch.uint8,device='cuda'))
except RuntimeError: pass
print(int(torch.cuda.memory_allocated()/2**20))
" 2>/dev/null || echo 0)
[ "$ALLOC" -ge 3300 ] && pass "A3 cấp phát được $ALLOC MiB (≥3300)" || fail "A3 chỉ được $ALLOC MiB"

MEMAV=$(awk '/MemAvailable/{printf "%d", $2/1024}' /proc/meminfo)
[ "$MEMAV" -ge 7000 ] && pass "A4 RAM khả dụng $MEMAV MiB" || fail "A4 RAM chỉ $MEMAV MiB"

[ ! -e "$OUT" ] && pass "A5 thư mục đích chưa tồn tại" || fail "A5 $OUT đã tồn tại"

n=$(ss -ltn 2>/dev/null | grep -c ":$PORT" || true)
[ "$n" -eq 0 ] && pass "A7 cổng $PORT trống" || fail "A7 cổng $PORT bị chiếm"

sz=$(stat -c %s "$PLY" 2>/dev/null || echo 0)
[ "$sz" -eq 254575516 ] && pass "A8 checkpoint train đúng ($sz B)" || fail "A8 checkpoint sai: $sz"
n=$(ls "$SRC/images" 2>/dev/null | wc -l)
[ "$n" -eq 301 ] && pass "A9 301 ảnh" || fail "A9 có $n ảnh"

ls ~/.cache/torch/hub/checkpoints/vgg16-*.pth >/dev/null 2>&1 \
  && pass "A10 có vgg16 cache" || fail "A10 thiếu vgg16 cache"

DF=$(df -BG /mnt/d | tail -1 | awk '{gsub("G","",$4); print $4}')
[ "$DF" -ge 2 ] && pass "A6 đĩa còn ${DF}G" || fail "A6 đĩa chỉ còn ${DF}G"

echo
[ "$FAILED" -ne 0 ] && { echo "==> CÓ CỔNG KHÔNG ĐẠT. Không chạy."; exit 1; }
echo "==> TẤT CẢ CỔNG ĐẠT."
echo
echo "DỰ ĐOÁN GHI TRƯỚC (nhánh $MODE):"
echo "  N cuối = $EXPECT_N   ·   file = $EXPECT_B B"
if [ "$MODE" = control ]; then
  echo "  VRAM đỉnh ~2374 MiB (tham số 231 + grad 231 + 2 moment Adam 462"
  echo "                       + context 62 + workspace ~1388)"
else
  echo "  VRAM đỉnh ~1912 MiB (đỉnh ở lúc chấm điểm quan trọng, N đầy đủ, chưa có Adam)"
fi
echo

( while true; do printf '%s %s %s\n' "$(date +%T)" \
    "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)" \
    "$(ps -o rss= -C python 2>/dev/null | sort -n | tail -1)"; sleep 1; done ) > "$MON" 2>/dev/null &
MONPID=$!
trap 'kill $MONPID 2>/dev/null || true' EXIT

echo "===== CHẠY ====="
cd "$REPO"
set +e
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
$PY -u prune_finetune.py \
  -s "$SRC" \
  -m "$OUT" \
  -r 1 \
  --eval \
  --data_device cpu \
  --port $PORT \
  --start_pointcloud "$PLY" \
  --iterations 35000 \
  --prune_iterations $PRUNE_IT \
  --prune_percent 0.66 \
  --prune_type v_important_score \
  --v_pow 0.1 \
  --test_iterations 30001 35000 \
  --save_iterations 35000 \
  --checkpoint_iterations 0 \
  2>&1 | tee "$LOG"
RC=${PIPESTATUS[0]}
set -e
kill $MONPID 2>/dev/null || true

echo
echo "===== KIỂM TRA SAU KHI CHẠY ====="
echo "exit=$RC"

# B1 — prune phải chạy đúng như ý đồ của nhánh
if grep -qi "prune round" "$LOG"; then
  [ "$MODE" = prune ] && pass "B1 prune ĐÃ chạy (đúng cho nhánh prune)" \
                      || echo "  ✗ B1 prune chạy trong nhánh CONTROL -> đối chứng vô hiệu"
else
  [ "$MODE" = control ] && pass "B1 prune KHÔNG chạy (đúng cho nhánh control)" \
                        || echo "  ✗ B1 nhánh prune mà không thấy prune -> bẫy T1"
fi

# B2 — N và kích thước so với dự đoán ghi trước
P="$OUT/point_cloud/iteration_35000/point_cloud.ply"
if [ -f "$P" ]; then
  N=$(head -c 4000 "$P" | grep -a -m1 'element vertex' | awk '{print $3}')
  B=$(stat -c %s "$P")
  echo "  B2 N=$N (dự đoán $EXPECT_N) · $B B (dự đoán $EXPECT_B)"
  [ "$N" = "$EXPECT_N" ] && pass "     N khớp" || echo "     ✗ N LỆCH"
  [ "$B" = "$EXPECT_B" ] && pass "     kích thước khớp" || echo "     ✗ kích thước LỆCH"
else
  echo "  ✗ B2 thiếu $P"
fi

# B3 — đỉnh bộ nhớ
[ -s "$MON" ] && awk 'NF>=2{if($2>m)m=$2; if($3>r)r=$3} END{
  printf "  B3 NVML đỉnh %d MiB (trần 3440) · RSS đỉnh %.2f GiB (trần 9.7)\n", m, r/1048576}' "$MON"

# B4 — số đo trong vòng lặp: đây là số dùng để so sánh hai nhánh
echo "  B4 metric.csv:"
[ -f "$OUT/metric.csv" ] && sed 's/^/     /' "$OUT/metric.csv"
