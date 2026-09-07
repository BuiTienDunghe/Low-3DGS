#!/usr/bin/env bash
# EXP-027 — đo phân bố khối lượng quan trọng của hai model. Chỉ forward pass.
set -uo pipefail
PROJ=/mnt/d/low-3DGS
PY=~/l3dgs/venv/bin/python
OUT=$PROJ/experiments/importance_profile.csv
export PYTHONPATH=$HOME/l3dgs/tools_memprobe
rm -f "$OUT"

run() {  # nhãn  scene  ply
  echo "############################################################"
  echo "# $1   ($(date +%H:%M:%S))"
  echo "############################################################"
  local tmp=$PROJ/experiments/_impprof_$1
  rm -rf "$tmp"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  $PY -u "$PROJ/tools/importance_profile.py" \
    -s "$2" -m "$tmp" -r 1 --eval --data_device cpu \
    --start_pointcloud "$3" --label "$1" --out "$OUT" 2>&1 | grep -vE "^Reading camera"
  rm -rf "$tmp"
  echo
}

run train      "$PROJ/datasets/tandt/train" \
               "$PROJ/datasets/pretrained/models/train/point_cloud/iteration_30000/point_cloud.ply"
run truck864   "$PROJ/datasets/tandt/truck" \
               "$PROJ/experiments/lg_truck_v66_ft5k/point_cloud/iteration_35000/point_cloud.ply"
echo "XONG. Kết quả: $OUT"
