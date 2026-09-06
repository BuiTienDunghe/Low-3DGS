#!/usr/bin/env bash
# =============================================================================
#  P1 GATE — "câu hỏi sống-chết" (PLAN.md §9):
#    Model pretrained `truck` của INRIA có RENDER được test set trên máy này không,
#    và đỉnh RAM / VRAM là bao nhiêu?
#
#  Không sửa gì trong third_party. Chạy render.py + metrics.py nguyên bản của INRIA,
#  song song với tools/profile_gpu.py bám theo PID để lấy RSS + VRAM device 2 Hz.
#
#  Chạy trong WSL sau khi setup_wsl_sudo.sh + venv xong:
#      bash /mnt/d/low-3DGS/scripts/p1_smoke.sh [scene=truck] [res=1]
#
#  Gate PASS khi: render xong không OOM · PSNR trong ±0.3 dB so với paper cùng res ·
#                 RSS peak < 9 GB · swap không bị chạm.
# =============================================================================
set -euo pipefail

SCENE="${1:-truck}"
RES="${2:-1}"
DATA_DEVICE="${3:-cpu}"   # cpu = ảnh nằm RAM (an toàn VRAM) · cuda = mặc định INRIA (EXP-002, RQ4)
# LƯU Ý 1: Scene nạp TOÀN BỘ ảnh train dù --skip_train.
# LƯU Ý 2: ảnh tốn 16 B/pixel, KHÔNG phải 12 — `cameras.py:48` luôn tạo alpha_mask=ones_like (EXP-005).
#          truck 251×979×546×16 = 2.00 GB · playroom 3.52 GB · drjohnson 4.57 GB.
#          -> data_device=cuda: playroom 4.12 GB và drjohnson 5.37 GB đều VƯỢT 4 GB VRAM. cpu là bắt buộc.
# LƯU Ý 3: `cfg_args` của bundle pretrained có TRƯỚC khi `data_device` tồn tại -> nếu không truyền
#          --data_device tường minh sẽ AttributeError. (INRIA issue #198)

PROJ=/mnt/d/low-3DGS
# Dung repo PUP 3D-GS: torch 2.4.1-compatible, da va loi iteration cua LightGaussian,
# co san ca 3 tieu chi prune, va rasterizer doi ten nen khong dung INRIA.
# metrics.py cua no dung lpipsPyTorch vendored + net_type=vgg (quy tac L1),
# loss_utils.py dung Gauss window sigma=1.5 (quy tac L2) -> khop protocol san.
GS=~/l3dgs/third_party/gaussian-splatting-pup
VENV=~/l3dgs/venv
DATA_ROOT=$PROJ/datasets
PRE_ROOT=$PROJ/datasets/pretrained/models        # sau khi unzip models.zip

case "$SCENE" in
  truck|train)         SRC=$DATA_ROOT/tandt/$SCENE ;;
  playroom|drjohnson)  SRC=$DATA_ROOT/db/$SCENE ;;
  *) echo "scene la gi? truck|train|playroom|drjohnson"; exit 1 ;;
esac
MODEL=$PRE_ROOT/$SCENE

RUN_ID="p1_smoke_${SCENE}_r${RES}_${DATA_DEVICE}_$(date +%Y%m%d-%H%M%S)"
OUT=$PROJ/experiments/$RUN_ID
mkdir -p "$OUT"

# --- điều kiện tiên quyết -----------------------------------------------------
[ -d "$SRC/images" ]  || { echo "Thieu dataset: $SRC"; exit 1; }
[ -f "$MODEL/point_cloud/iteration_30000/point_cloud.ply" ] || {
  echo "Thieu pretrained: $MODEL/point_cloud/iteration_30000/point_cloud.ply"
  echo "  (da unzip models.zip chua? layout co the khac -> ls $PRE_ROOT)"; exit 1; }
[ -x "$VENV/bin/python" ] || { echo "Thieu venv $VENV — chay setup truoc"; exit 1; }

source "$VENV/bin/activate"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- snapshot môi trường TRƯỚC run (quy tắc §16.9) ----------------------------
{
  echo "run_id=$RUN_ID  scene=$SCENE  res=$RES  data_device=$DATA_DEVICE  date=$(date -Is)"
  echo "n_gaussians=$(head -c 4000 "$MODEL/point_cloud/iteration_30000/point_cloud.ply" | grep -a -m1 'element vertex' | awk '{print $3}')"
  echo "n_images=$(ls "$SRC/images" | wc -l)  sample_dims=$(file "$SRC/images/$(ls "$SRC/images" | head -1)" | grep -o '[0-9]\+x[0-9]\+' | head -1)"
  echo "model=$MODEL"; echo "src=$SRC"
  echo "git_gs=$(git -C $GS rev-parse --short HEAD)"
  echo "torch=$(python -c 'import torch;print(torch.__version__, torch.version.cuda)')"
  echo "--- free -g (WSL) ---"; free -g
  echo "--- nvidia-smi (phai ~0 MiB) ---"; nvidia-smi --query-gpu=memory.used,temperature.gpu --format=csv,noheader
  echo "--- ply size ---"; ls -la "$MODEL/point_cloud/iteration_30000/point_cloud.ply"
} | tee "$OUT/env.txt"

# --- STAGE A: render test set (INRIA render.py nguyên bản) --------------------
echo; echo "===== [A] render.py  (skip_train, res=$RES, data_device=$DATA_DEVICE) ====="
cd "$GS"
python render.py -m "$MODEL" -s "$SRC" --skip_train -r "$RES" --data_device "$DATA_DEVICE" --quiet \
    > "$OUT/render.log" 2>&1 &
RPID=$!
python "$PROJ/tools/profile_gpu.py" --out "$OUT/memory_render.jsonl" --pid $RPID --hz 2 --quiet
wait $RPID; RRC=$?
echo "render.py exit=$RRC" | tee -a "$OUT/env.txt"
tail -3 "$OUT/render.log"
[ $RRC -eq 0 ] || { echo "RENDER THAT BAI — xem $OUT/render.log (OOM? -> ghi vao 05_EXPERIMENTS.md bang OOM)"; exit 2; }

# --- STAGE B: metrics (PSNR/SSIM/LPIPS) — process riêng, LPIPS-VGG -------------
echo; echo "===== [B] metrics.py ====="
python metrics.py -m "$MODEL" > "$OUT/metrics.log" 2>&1 &
MPID=$!
python "$PROJ/tools/profile_gpu.py" --out "$OUT/memory_metrics.jsonl" --pid $MPID --hz 2 --quiet
wait $MPID; MRC=$?
echo "metrics.py exit=$MRC" | tee -a "$OUT/env.txt"
grep -E "PSNR|SSIM|LPIPS" "$OUT/metrics.log" | tail -3 | tee -a "$OUT/env.txt"
cp "$MODEL/results.json" "$OUT/results.json" 2>/dev/null || true

# --- tóm tắt ------------------------------------------------------------------
echo; echo "===== TOM TAT ====="
echo "--- render ---";  python "$PROJ/tools/profile_gpu.py" --summary "$OUT/memory_render.jsonl"  | tee "$OUT/summary_render.json"
echo "--- metrics ---"; python "$PROJ/tools/profile_gpu.py" --summary "$OUT/memory_metrics.jsonl" | tee "$OUT/summary_metrics.json"
echo
echo "Ket qua o: $OUT"
echo "GATE: kiem tra  rss_peak_gb < 9  ·  vram_peak_mb  ·  PSNR vs paper  -> ghi vao docs/05_EXPERIMENTS.md (EXP-001)"
