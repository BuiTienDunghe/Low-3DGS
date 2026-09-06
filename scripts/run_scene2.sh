#!/usr/bin/env bash
# =============================================================================
#  EXP-022 — SCENE THỨ HAI: truck (đã prune còn 864.017 hạt).
#
#  Mọi kết quả tới nay nằm trên MỘT scene (train). Đây là phép kiểm chứng chéo.
#
#  VÌ SAO LÀ MODEL ĐÃ PRUNE, KHÔNG PHẢI truck NGUYÊN BẢN:
#  truck nguyên bản (2.541.226 hạt) KHÔNG fine-tune nổi trên card này — EXP-015
#  đã đo OOM ở backward đầu tiên. Và giảm -r KHÔNG cứu được, vì ba khoản lớn nhất
#  đều ĐỘC LẬP ĐỘ PHÂN GIẢI:
#     tham số+grad+Adam  2.54M × 944 B = 2288 MiB
#     dL_dsh             {P,16,3} float = 465 MiB   (rasterize_points.cu:157)
#     context                            62 MiB
#     -------------------------------------------
#     sàn cứng                         2815 MiB / trần 3440  -> chỉ còn 625 MiB
#  => model 864k là model hợp lệ DUY NHẤT của scene này chạy được đủ cặp đối chứng.
#     Nó KHÔNG được dùng để chứng minh "nén miễn phí"; nó chỉ là ĐIỂM XUẤT PHÁT.
#
#  VÀ ĐÂY LÀ CÂU HỎI PHỤ ĐÁNG GIÁ: model nền này ĐÃ được tinh chỉnh 5000 bước rồi
#  (đầu ra EXP-014). Nếu "+0.3587 dB do luyện thêm" vẫn còn ở đây thì đó là hiệu ứng
#  bền; nếu nó teo đi thì ta biết confound chỉ sống trên checkpoint chưa hội tụ.
#
#  BA CẠM BẪY ĐÃ KIỂM CHỨNG TRONG MÃ, ĐỪNG SỬA CẤU HÌNH MÀ KHÔNG ĐỌC:
#   1. prune_finetune.py:59 làm int(path.split('/')[-2].split('_')[-1]) và chạy TRƯỚC
#      `if args.first_iter`. Đường dẫn phải nằm trong thư mục tên iteration_NNNNN.
#      Ta dùng thẳng iteration_35000 của EXP-014 -> parse ra 35000, rồi --first_iter 30000
#      ghi đè, giữ đúng dải 30001..35000 và đúng 12 bước ExponentialLR như loạt train.
#   2. Hai file .ply của EXP-014 (@30002 và @35000) CÙNG 214.277.747 B và CÙNG N=864017,
#      chỉ khác nội dung. Cổng kích thước vô dụng -> phải dùng sha256.
#   3. `else: return` ngay trong vòng lặp => mọi mã sau vòng lặp không bao giờ chạy.
#      Đo VRAM bằng sitecustomize + atexit đặt NGOÀI repo (PYTHONPATH), 0 dòng sửa repo.
# =============================================================================
set -uo pipefail

PROJ=/mnt/d/low-3DGS
REPO=~/l3dgs/third_party/gaussian-splatting-pup
PY=~/l3dgs/venv/bin/python
SRC=$PROJ/datasets/tandt/truck
PLY=$PROJ/experiments/lg_truck_v66_ft5k/point_cloud/iteration_35000/point_cloud.ply
SHA_OK=5c837cbf1f3f0479c76d3576aa48973a2dacebd5f0ecb122dc9fa4c5fbf5c56c
RESULTS=$PROJ/experiments/scene2_truck864.csv
export PYTHONPATH=$HOME/l3dgs/tools_memprobe

# N sau prune: N - (int(p*(N-1)) + 1), N=864017
#   p=0.50 -> 864017 - (432008+1) = 432008
ARMS=(control prune  control prune  control prune)
SEEDS=(0       0      1       1      2       2)
PCT=0.50
EXPN_CTRL=864017
EXPN_PRUNE=432008

[ -f "$RESULTS" ] || echo "arm,pct,seed,iteration,occurrence,psnr,ssim,lpips,l1,n_gauss,vram_nvml_mib,vram_alloc_mib,wall_s" > "$RESULTS"

echo "===== CỔNG MỘT LẦN ====="
SHA=$(sha256sum "$PLY" 2>/dev/null | cut -d' ' -f1)
[ "$SHA" = "$SHA_OK" ] && echo "  ✓ A8 sha256 model nền đúng (@35000, KHÔNG phải @30002)" \
  || { echo "  ✗ A8 sha256 SAI: $SHA"; echo "    (nếu ra ff25a6e0... thì bạn đang trỏ vào @30002 — model chưa phục hồi)"; exit 1; }
n=$(ls "$SRC/images" 2>/dev/null | wc -l)
[ "$n" -eq 251 ] && echo "  ✓ A9 251 ảnh truck" || { echo "  ✗ A9 có $n ảnh"; exit 1; }
ls ~/.cache/torch/hub/checkpoints/vgg16-*.pth >/dev/null 2>&1 && echo "  ✓ A10 vgg16 cache" || { echo "  ✗ A10 thiếu vgg16"; exit 1; }
grep -q L3DGS_SEED $REPO/utils/general_utils.py && echo "  ✓ patch seed" || { echo "  ✗ chưa patch seed"; exit 1; }
[ -f "$HOME/l3dgs/tools_memprobe/sitecustomize.py" ] && echo "  ✓ memprobe sẵn sàng" || { echo "  ✗ thiếu memprobe"; exit 1; }
echo "  ✓ đĩa $(df -BG /mnt/d | tail -1 | awk '{print $4}')"
echo "  6 lần chạy (đối chứng×3, cắt ${PCT}×3), ước lượng ~100 phút."
echo

for i in "${!ARMS[@]}"; do
  ARM=${ARMS[$i]}; S=${SEEDS[$i]}
  if [ "$ARM" = prune ]; then PRUNE_IT=30001; EN=$EXPN_PRUNE; else PRUNE_IT=99999; EN=$EXPN_CTRL; fi
  TAG=s2_truck864_${ARM}_s${S}
  OUT=$PROJ/experiments/$TAG; LOG=$OUT.log; MON=$OUT.mon

  echo "############################################################"
  echo "# [$((i+1))/6] $ARM seed=$S  ->  $TAG   ($(date +%H:%M:%S))"
  echo "############################################################"
  [ -e "$OUT" ] && { echo "  ! đã có, bỏ qua"; continue; }

  k=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)
  [ "$k" -eq 0 ] || { echo "  ✗ có $k tiến trình CUDA lạ — dừng loạt"; exit 1; }
  MEMAV=$(awk '/MemAvailable/{printf "%d", $2/1024}' /proc/meminfo)
  [ "$MEMAV" -ge 7000 ] || { echo "  ✗ RAM chỉ $MEMAV MiB"; exit 1; }

  ( while true; do printf '%s\n' \
      "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)"; sleep 1; done ) > "$MON" 2>/dev/null &
  MONPID=$!

  T0=$(date +%s)
  cd "$REPO"
  L3DGS_SEED=$S PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  $PY -u prune_finetune.py \
    -s "$SRC" -m "$OUT" -r 1 --eval --data_device cpu --port 6441 \
    --start_pointcloud "$PLY" \
    --first_iter 30000 \
    --iterations 35000 \
    --prune_iterations $PRUNE_IT \
    --prune_percent $PCT --prune_type v_important_score --v_pow 0.1 \
    --test_iterations 30001 35000 \
    --save_iterations 0 --checkpoint_iterations 0 \
    > "$LOG" 2>&1
  RC=$?; T1=$(date +%s)
  kill $MONPID 2>/dev/null || true

  WALL=$((T1-T0))
  VNVML=$(awk 'NF>=1{if($1>m)m=$1} END{print m+0}' "$MON" 2>/dev/null)
  VALLOC=$(grep -oP 'max_allocated_MiB=\K[0-9.]+' "$LOG" | tail -1)
  NG=$(grep -oP 'Number of Gaussians is \K[0-9]+' "$LOG" | tail -1)
  NEVAL=$(grep -c 'Evaluating test' "$LOG")

  if [ "$RC" -ne 0 ]; then
    echo "  ✗ THẤT BẠI (exit $RC)"; grep -iE "out of memory|Error|Traceback|Killed" "$LOG" | head -4 | sed 's/^/    /'; continue
  fi
  # B4 — cạm bẫy hỏng ÂM THẦM: nếu first_iter sai thì vòng lặp rỗng, exit 0, in đúng N,
  #      qua sạch mọi cổng cũ mà KHÔNG huấn luyện gì. Phải thấy đúng 2 lần eval.
  # SỬA 2026-09-06: nhánh NÉN eval 3 lần (trước prune @30001, sau prune @30001, và @35000);
  # nhánh ĐỐI CHỨNG eval 2 lần. Bản đầu bắt cứng "= 2" nên loại oan cả 3 nhánh nén.
  if [ "$ARM" = prune ]; then NEVAL_OK=3; else NEVAL_OK=2; fi
  [ "$NEVAL" -eq "$NEVAL_OK" ] && echo "  ✓ B4 có đúng $NEVAL lần eval" \n    || { echo "  ✗ B4 thấy $NEVAL lần eval, cần $NEVAL_OK — VÒNG LẶP RỖNG? bỏ điểm này"; continue; }
  if grep -qi "prune round" "$LOG"; then HASP=1; else HASP=0; fi
  { [ "$ARM" = prune ] && [ "$HASP" -eq 1 ]; } || { [ "$ARM" = control ] && [ "$HASP" -eq 0 ]; } \
    || { echo "  ✗ B1 prune chạy sai ý đồ nhánh"; continue; }
  [ "$NG" = "$EN" ] && echo "  ✓ B2 N=$NG khớp" || echo "  ⚠ B2 N=$NG khác dự đoán $EN"

  awk -F, -v a="$ARM" -v p="$PCT" -v s="$S" -v ng="$NG" -v v="$VNVML" -v va="${VALLOC:-}" -v w="$WALL" '
    NR>1 { c[$1]++; printf "%s,%s,%s,%s,%d,%s,%s,%s,%s,%s,%s,%s,%s\n", a,p,s,$1,c[$1]-1,$4,$5,$6,$3,ng,v,va,w }
  ' "$OUT/metric.csv" >> "$RESULTS"

  echo "  ✓ xong: NVML đỉnh ${VNVML} MiB · trong tiến trình ${VALLOC:-?} MiB (trần 3440) · ${WALL}s"
  grep -oP 'PSNR \K[0-9.]+' "$LOG" | sed 's/^/    PSNR: /'
  echo
done

echo "############################################################"
echo "# XONG ($(date +%H:%M:%S))"
echo "############################################################"
column -s, -t "$RESULTS"
