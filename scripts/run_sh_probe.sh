#!/usr/bin/env bash
# EXP-020 bước 1 — thiệt hại của việc cắt bậc màu, chưa phục hồi.
set -uo pipefail
PROJ=/mnt/d/low-3DGS
PY=~/l3dgs/venv/bin/python
SRC=$PROJ/datasets/tandt/train
PLY=$PROJ/datasets/pretrained/models/train/point_cloud/iteration_30000/point_cloud.ply
OUT=$PROJ/experiments/sh_trunc_train
LOG=$OUT.log

n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)
[ "$n" -eq 0 ] || { echo "✗ có $n tiến trình CUDA đang chạy"; exit 1; }
[ -e "$OUT" ] && { echo "✗ $OUT đã tồn tại"; exit 1; }
[ "$(stat -c %s "$PLY")" -eq 254575516 ] || { echo "✗ checkpoint sai"; exit 1; }
echo "✓ cổng đạt — chạy (ước lượng ~7 phút, 4 bậc × 38 view)"
echo

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
$PY -u "$PROJ/tools/measure_sh_truncation.py" \
  -s "$SRC" -m "$OUT" -r 1 --eval --data_device cpu \
  --start_pointcloud "$PLY" \
  --out "$PROJ/experiments/sh_truncation.csv" 2>&1 | tee "$LOG"
