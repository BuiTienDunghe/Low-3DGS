#!/usr/bin/env bash
# =============================================================================
#  EXP-015 — ĐỐI CHỨNG cho EXP-014.
#
#  Câu hỏi: trong +0.1123 dB của EXP-014, bao nhiêu là "nén không mất gì"
#           và bao nhiêu chỉ là "được luyện thêm 5000 bước"?
#
#  Cấu hình GIỐNG HỆT EXP-014, khác đúng MỘT thứ: prune không bao giờ chạy
#  (--prune_iterations 99999). Đã kiểm chứng trong mã nguồn:
#    * init_report chỉ đổi NHÃN in ra, không ảnh hưởng tính toán.
#    * prune_idx chỉ dùng cho nhãn.
#    * ĐÍNH CHÍNH: KHÔNG đúng khi nói "cùng thứ tự ảnh". prune_finetune.py:272 đặt
#      viewpoint_stack = None -> reset ngăn xếp camera, nên từ iter 30002 hai nhánh
#      thấy thứ tự ảnh KHÁC nhau. Đó là nhiễu, không phải thiên lệch hệ thống.
#      Thứ THỰC SỰ giống nhau giữa hai nhánh: lịch learning rate (cùng LR đầu,
#      cùng 12 bước ExponentialLR(0.95)) và điểm xuất phát (eval@30001 trùng khớp
#      tới 15 chữ số thập phân: 21.815698046433297 trên scene train).
#
#  KHÁC BIỆT BẮT BUỘC so với EXP-014: --save_iterations 0 (không lưu).
#    Lý do: save_ply dòng 206 dùng list(map(tuple, attributes)) -> ~13.35x
#    kích thước file trong RAM. Ở N=2.54M là ~8.3 GB, vượt trần 9.7 GiB.
#    EXP-014 sống sót vì nó lưu SAU khi prune (N=864k -> 2.86 GB).
#    Số liệu lấy từ eval trong vòng lặp, cùng mã, cùng 32 view test.
#
#  Dùng:  bash scripts/run_control_noprune.sh --probe   (60 iter, thử bộ nhớ)
#         bash scripts/run_control_noprune.sh           (5000 iter, chạy thật)
# =============================================================================
set -euo pipefail
set -o pipefail

PROJ=/mnt/d/low-3DGS
REPO=~/l3dgs/third_party/gaussian-splatting-pup
PY=~/l3dgs/venv/bin/python
SCENE=truck
SRC=$PROJ/datasets/tandt/$SCENE
BASE=$PROJ/datasets/pretrained/models/$SCENE
PLY=$BASE/point_cloud/iteration_30000/point_cloud.ply
PORT=6402

PROBE=0
[ "${1:-}" = "--probe" ] && PROBE=1

if [ "$PROBE" -eq 1 ]; then
  OUT=$PROJ/experiments/ctrl_truck_noprune_PROBE
  ITERS=30060; TESTIT="30050"
  echo "### CHẾ ĐỘ THĂM DÒ: 60 iteration, chỉ để đo đỉnh VRAM ###"
else
  OUT=$PROJ/experiments/ctrl_truck_noprune_ft5k
  ITERS=35000; TESTIT="30001 35000"
fi
LOG=$OUT.log
MON=$OUT.mon

fail() { echo "  ✗ $1"; FAILED=1; }
pass() { echo "  ✓ $1"; }
FAILED=0

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
[ "$sz" -eq 630225580 ] && pass "A8 checkpoint đúng ($sz B)" || fail "A8 checkpoint sai: $sz"
n=$(ls "$SRC/images" 2>/dev/null | wc -l)
[ "$n" -eq 251 ] && pass "A9 251 ảnh" || fail "A9 có $n ảnh"

ls ~/.cache/torch/hub/checkpoints/vgg16-*.pth >/dev/null 2>&1 \
  && pass "A10 có vgg16 cache" || fail "A10 thiếu vgg16 cache"

echo
[ "$FAILED" -ne 0 ] && { echo "==> CÓ CỔNG KHÔNG ĐẠT. Không chạy."; exit 1; }
echo "==> TẤT CẢ CỔNG ĐẠT."
echo
echo "DỰ ĐOÁN GHI TRƯỚC: cố định 2350 MiB (tham số+grad+2 moment Adam ở N=2.54M"
echo "  + context) cộng workspace ~1415 MiB đo được ở EXP-014 = ~3765 MiB."
echo "  Trần thật 3440 MiB  ->  DỰ ĐOÁN LÀ TRÀN BỘ NHỚ. Chạy để kiểm chứng."
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
  --iterations $ITERS \
  --prune_iterations 99999 \
  --prune_percent 0.66 \
  --prune_type v_important_score \
  --v_pow 0.1 \
  --test_iterations $TESTIT \
  --save_iterations 0 \
  --checkpoint_iterations 0 \
  2>&1 | tee "$LOG"
RC=${PIPESTATUS[0]}
set -e
kill $MONPID 2>/dev/null || true

echo
echo "===== KIỂM TRA SAU KHI CHẠY ====="
echo "exit=$RC"

# B1 — prune TUYỆT ĐỐI không được chạy. Nếu chạy thì đối chứng vô nghĩa.
if grep -qi "prune round" "$LOG"; then
  echo "  ✗ THẤY dòng 'Prune Round' -> prune ĐÃ CHẠY. Đối chứng KHÔNG hợp lệ."
else
  pass "B1 prune không chạy (đúng ý đồ đối chứng)"
fi

# B2 — đỉnh bộ nhớ
if [ -s "$MON" ]; then
  awk 'NF>=2{if($2>m)m=$2; if($3>r)r=$3} END{
    printf "  B2 NVML đỉnh %d MiB (trần tensor 3440) · RSS đỉnh %.2f GiB\n", m, r/1048576}' "$MON"
fi

# B3 — có tràn bộ nhớ không
if grep -qiE "out of memory|CUDA error|Killed" "$LOG"; then
  echo "  → CÓ TRÀN BỘ NHỚ. Đây là KẾT QUẢ, không phải sự cố: đối chứng"
  echo "    không chạy nổi trên máy này ở N=2.54M."
  grep -iE "out of memory|CUDA error|Killed" "$LOG" | head -3 | sed 's/^/    /'
fi

# B4 — số PSNR trong vòng lặp
grep -iE "Evaluating|PSNR|psnr" "$LOG" | tail -12 | sed 's/^/  /'
