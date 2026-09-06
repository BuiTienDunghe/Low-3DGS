#!/usr/bin/env bash
# =============================================================================
#  EXP-014 — LightGaussian prune(0.66) + recovery-finetune 5k trên checkpoint pretrained.
#  THÍ NGHIỆM NÉN THẬT ĐẦU TIÊN của dự án.
#
#  Mọi flag dưới đây do workflow pre-flight chốt: 78 agent, 126 phát hiện, 71 rủi ro,
#  69 xác nhận / 2 bác bỏ, có agent chạy thật pipeline 5 lần để đo. KHÔNG sửa flag mà
#  không đọc docs/05_EXPERIMENTS.md EXP-014 và phần lý do bên dưới.
#
#  BA LỖI CHÍ MẠNG trong bản nháp trước đã được sửa:
#   1. --prune_decay 1     -> KHÔNG TỒN TẠI trong PUP (0 grep hit). argparse thoát mã 2 ngay.
#   2. --quiet             -> safe_state thay stdout bằng sink rỗng, giết mọi chuỗi cổng kiểm tra.
#   3. thiếu --checkpoint_iterations 0 -> mặc định [35000,40000] torch.save 622 MB kèm
#                             optimizer state, làm hỏng số kích thước model (quy tắc L4).
#
#  Dùng:  bash scripts/run_prune_finetune.sh [--check-only]
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
OUT=$PROJ/experiments/lg_truck_v66_ft5k
LOG=$OUT.log
MON=$OUT.mon
PORT=6401

CHECK_ONLY=0
[ "${1:-}" = "--check-only" ] && CHECK_ONLY=1

fail() { echo "  ✗ $1"; FAILED=1; }
pass() { echo "  ✓ $1"; }
FAILED=0

echo "===== CỔNG TRƯỚC KHI CHẠY (A1–A11) ====="

# A1 — không có tiến trình CUDA nào khác. (BỎ QUA cột used_memory: WSL báo [N/A].)
n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)
[ "$n" -eq 0 ] && pass "A1 không có compute app nào trên GPU" || fail "A1 có $n tiến trình CUDA đang chạy"

# A2 — không có run python cũ còn sót
n=$(pgrep -af '/l3dgs/venv/bin/python' | grep -vc "$$" || true)
[ "$n" -eq 0 ] && pass "A2 không có python venv nào đang chạy" || fail "A2 còn $n tiến trình python"

# A3 — CỔNG QUYẾT ĐỊNH. Thăm dò cấp phát thật.
#      KHÔNG dùng nvidia-smi memory.free: đo được 3807 free khi CÓ job khác, và 2007 khi rảnh.
ALLOC=$(PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -c "
import torch; torch.cuda.init(); torch.zeros(1,device='cuda'); b=[]
try:
    while True: b.append(torch.empty(64*1024*1024,dtype=torch.uint8,device='cuda'))
except RuntimeError: pass
print(int(torch.cuda.memory_allocated()/2**20))
" 2>/dev/null || echo 0)
[ "$ALLOC" -ge 3300 ] && pass "A3 cấp phát được $ALLOC MiB (≥3300)" || fail "A3 chỉ cấp phát được $ALLOC MiB — có tiến trình lạ dùng GPU"

# A4 — RAM. (free -g cắt cụt, dùng /proc/meminfo.)
MEMAV=$(awk '/MemAvailable/{printf "%d", $2/1024}' /proc/meminfo)
[ "$MEMAV" -ge 7000 ] && pass "A4 RAM khả dụng $MEMAV MiB (≥7000)" || fail "A4 RAM khả dụng chỉ $MEMAV MiB"

# A5 — thư mục đích phải MỚI. logger_utils.py:31-33 cắt cụt cfg_args;
#      scene/__init__.py:52-63 ghi đè input.ply + cameras.json. Tuyệt đối không trỏ vào $BASE.
[ ! -e "$OUT" ] && pass "A5 thư mục đích chưa tồn tại" || fail "A5 $OUT đã tồn tại — đổi tên hoặc xoá"

# A6 — đĩa
DF=$(df -BG /mnt/d | tail -1 | awk '{gsub("G","",$4); print $4}')
[ "$DF" -ge 2 ] && pass "A6 đĩa còn ${DF}G" || fail "A6 đĩa chỉ còn ${DF}G"

# A7 — cổng network_gui
n=$(ss -ltn 2>/dev/null | grep -c ":$PORT" || true)
[ "$n" -eq 0 ] && pass "A7 cổng $PORT trống" || fail "A7 cổng $PORT đang bị chiếm"

# A8/A9 — dữ liệu đúng
sz=$(stat -c %s "$PLY" 2>/dev/null || echo 0)
[ "$sz" -eq 630225580 ] && pass "A8 checkpoint đúng ($sz B)" || fail "A8 checkpoint sai kích thước: $sz"
n=$(ls "$SRC/images" 2>/dev/null | wc -l)
[ "$n" -eq 251 ] && pass "A9 251 ảnh" || fail "A9 có $n ảnh, cần 251"

# A10 — trọng số VGG cho LPIPS phải có sẵn: lúc chạy KHÔNG có mạng
if ls ~/.cache/torch/hub/checkpoints/vgg16-*.pth >/dev/null 2>&1; then pass "A10 có vgg16 cache"
else fail "A10 thiếu ~/.cache/torch/hub/checkpoints/vgg16-*.pth (LPIPS sẽ cần tải)"; fi

# A11 — sao lưu baseline. Bảo hiểm nếu gõ nhầm -m.
SNAP=$PROJ/experiments/_baseline_truck_snapshot
mkdir -p "$SNAP"
for f in cfg_args cameras.json input.ply; do
  [ -f "$BASE/$f" ] && cp -pn "$BASE/$f" "$SNAP/" 2>/dev/null || true
done
pass "A11 đã sao lưu baseline vào $SNAP"

echo
if [ "$FAILED" -ne 0 ]; then echo "==> CÓ CỔNG KHÔNG ĐẠT. Không chạy."; exit 1; fi
echo "==> TẤT CẢ CỔNG ĐẠT."
[ "$CHECK_ONLY" -eq 1 ] && { echo "(--check-only, dừng ở đây)"; exit 0; }

# ---- giám sát 1 Hz ở nền: NVML used > 3600 MiB kéo dài = có kẻ lạ dùng GPU ----
( while true; do printf '%s %s %s\n' "$(date +%T)" \
    "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)" \
    "$(ps -o rss= -C python 2>/dev/null | sort -n | tail -1)"; sleep 1; done ) > "$MON" 2>/dev/null &
MONPID=$!
trap 'kill $MONPID 2>/dev/null || true' EXIT

echo
echo "===== CHẠY  (kỳ vọng 20–45 phút) ====="
echo "  Mốc kiểm tra:"
echo "   ~phút 9  : 'Prune Round 1: Number of Gaussians is 864017'  <- cổng quyết định"
echo "   ~phút 11 : eval sau prune, PSNR ≈ 24.22"
echo "   ~phút 12 : save 30002 — cùng đỉnh RAM 2.65 GiB mà lần save 35000 sẽ gặp"
echo "  Huỷ nếu: chưa thấy dòng prune sau phút 15, hoặc NVML > 3600 MiB kéo dài"
echo

cd "$REPO"
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
  --prune_iterations 30001 \
  --prune_percent 0.66 \
  --prune_type v_important_score \
  --v_pow 0.1 \
  --test_iterations 30001 35000 \
  --save_iterations 30002 35000 \
  --checkpoint_iterations 0 \
  2>&1 | tee "$LOG"
RC=${PIPESTATUS[0]}
kill $MONPID 2>/dev/null || true

echo
echo "===== KIỂM TRA SAU KHI CHẠY ====="
echo "exit=$RC"

# B1 — bẫy T1: prune có THỰC SỰ chạy không
if grep -qiE "prune round|number of gaussians" "$LOG"; then
  grep -iE "prune round|number of gaussians" "$LOG" | head -4 | sed 's/^/  /'
else
  echo "  ✗ KHÔNG thấy dòng prune trong log -> PRUNE ĐÃ KHÔNG CHẠY (bẫy T1). Kết quả vô giá trị."
fi

# B2 — N và kích thước, so với giá trị đã đo trước
for it in 30002 35000; do
  P="$OUT/point_cloud/iteration_$it/point_cloud.ply"
  if [ -f "$P" ]; then
    N=$(head -c 4000 "$P" | grep -a -m1 'element vertex' | awk '{print $3}')
    B=$(stat -c %s "$P")
    printf "  iter %-6s N=%-10s %s B = %.2f MB" "$it" "$(printf "%'d" "$N")" "$B" "$(echo "$B/1000000"|bc -l)"
    [ "$N" = "864017" ] && echo "   ✓ khớp N kỳ vọng" || echo "   ✗ kỳ vọng 864017"
  else
    echo "  ✗ thiếu $P"
  fi
done

# B3 — đỉnh bộ nhớ từ log giám sát
if [ -s "$MON" ]; then
  awk 'NF>=2{if($2>m)m=$2; if($3>r)r=$3} END{
    printf "  NVML đỉnh %d MiB (trần tensor 3440, card 4096) · RSS đỉnh %.2f GiB\n", m, r/1048576}' "$MON"
fi

echo
echo "Kết quả: $OUT"
echo "Bước tiếp: render + metrics trên $OUT để lấy số SAU, so với baseline 24.9927 / 0.87562 / 0.15130"
