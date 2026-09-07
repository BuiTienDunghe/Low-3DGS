#!/usr/bin/env bash
# =============================================================================
#  stamp_run.sh — gắn MỘT lần chạy thí nghiệm với ĐÚNG phiên bản mã đã sinh ra nó.
#
#  docs/PLAN.md yêu cầu mọi kết quả phải truy ngược được về phiên bản mã.
#  Trước v0.5.0 dự án không có git nên không làm được; đây là công cụ đóng nợ đó.
#
#  Dùng:
#    bash tools/stamp_run.sh experiments/<tên_run>          # lần chạy mới
#    bash tools/stamp_run.sh experiments/<tên_run> --retro  # lần chạy TRƯỚC khi có git
#    bash tools/stamp_run.sh --all-retro                    # đánh dấu hồi cố mọi run cũ
#
#  Ghi ra <dir>/run_meta.json. Trường quan trọng nhất là "dirty":
#  cây làm việc bẩn nghĩa là mã lúc chạy KHÔNG khớp commit nào => KHÔNG tái lập được.
#
#  GHI CHÚ KỸ THUẬT: JSON do python sinh (json.dumps), KHÔNG ghép chuỗi trong bash.
#  Thoát ký tự JSON bằng bash là bẫy — đường dẫn có \ hoặc " sẽ tạo ra JSON hỏng.
# =============================================================================
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
THIRD_PARTY=${L3DGS_REPO:-$HOME/l3dgs/third_party/gaussian-splatting-pup}
VENV_PY=${L3DGS_PY:-$HOME/l3dgs/venv/bin/python}
PY=$(command -v python3 || echo "$VENV_PY")

# Phát hiện môi trường MỘT LẦN. Nạp torch mất ~5 s; làm trong vòng lặp thì
# --all-retro với 30 run sẽ mất 7 phút thay vì 10 giây.
if [ -x "$VENV_PY" ]; then
  eval "$("$VENV_PY" - <<'DETECT' 2>/dev/null
import sys
try:
    import torch; tv, cv = torch.__version__, torch.version.cuda
except Exception:
    tv = cv = ""
pv = ".".join(map(str, sys.version_info[:3]))
print(f'export SR_PY={pv!r} SR_TORCH={tv!r} SR_CUDA={cv!r}'.replace("'", '"'))
DETECT
)"
fi
export SR_GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
export SR_DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)

stamp_one() {
  local dir="$1"
  [ -d "$dir" ] || { echo "  ✗ không có thư mục: $dir"; return 1; }

  export SR_RETRO="$2" SR_DIR="$dir"
  export SR_VERSION=$(cat "$ROOT/VERSION" 2>/dev/null || echo "")

  local c
  c=$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo "")
  # repo chưa có commit nào: rev-parse in ra đúng chữ "HEAD" -> coi như rỗng
  case "$c" in *[!0-9a-f]* | "") c="" ;; esac
  export SR_COMMIT="$c"
  local b; b=$(git -C "$ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "")
  [ "$b" = HEAD ] && b=""
  export SR_BRANCH="$b"
  # Buộc git so NỘI DUNG, không tin stat cache. Trên /mnt/d (9p) git có thể báo
  # SẠCH ngay sau khi file vừa bị sửa — false negative đúng ở chỗ nguy hiểm nhất.
  git -C "$ROOT" update-index -q --really-refresh 2>/dev/null || true

  # LOẠI TRỪ experiments/ khỏi phép kiểm bẩn: chính lần chạy ghi kết quả vào đó,
  # nên nếu tính cả thì KHÔNG lần chạy nào có thể sạch — cờ dirty thành vô nghĩa.
  # Cái ta cần biết là: MÃ NGUỒN lúc chạy có khớp commit không.
  if [ -n "$(git -C "$ROOT" status --porcelain -- ':!experiments' 2>/dev/null)" ]; then
    export SR_DIRTY=1
  else
    export SR_DIRTY=0
  fi

  export SR_TP_NAME=$(basename "$THIRD_PARTY")
  export SR_TP_COMMIT=$(git -C "$THIRD_PARTY" rev-parse HEAD 2>/dev/null || echo "")
  if [ -n "$(git -C "$THIRD_PARTY" status --porcelain 2>/dev/null)" ]; then export SR_TP_DIRTY=1; else export SR_TP_DIRTY=0; fi


  "$PY" - <<'PY'
import json, os, subprocess, sys
g = lambda k: os.environ.get(k, "") or None
meta = {
    "pre_versioning": os.environ["SR_RETRO"] == "true",
    "project_version": g("SR_VERSION"),
    "git": {"commit": g("SR_COMMIT"), "branch": g("SR_BRANCH"),
            "dirty": os.environ.get("SR_DIRTY") == "1"},
    "third_party": {"repo": g("SR_TP_NAME"), "commit": g("SR_TP_COMMIT"),
                    "dirty": os.environ.get("SR_TP_DIRTY") == "1",
                    "patches": "xem docs/REPO_PATCHES.md"},
    "env": {"python": g("SR_PY"), "torch": g("SR_TORCH"), "cuda": g("SR_CUDA"),
            "gpu": g("SR_GPU"), "driver": g("SR_DRIVER")},
    "stamped_at": subprocess.run(["date", "-Iseconds"], capture_output=True,
                                 text=True).stdout.strip(),
}
with open(os.path.join(os.environ["SR_DIR"], "run_meta.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)
    f.write("\n")
PY

  if [ "$2" = true ]; then
    echo "  ~ $dir  (hồi cố — chạy trước khi có git)"
  elif [ -z "$SR_COMMIT" ]; then
    echo "  ⚠ $dir  CHƯA CÓ COMMIT nào — không truy ngược được về mã nguồn"
  elif [ "$SR_DIRTY" = 1 ]; then
    echo "  ⚠ $dir  commit ${SR_COMMIT:0:8} nhưng CÂY LÀM VIỆC BẨN => không tái lập được"
  else
    echo "  ✓ $dir  commit ${SR_COMMIT:0:8} sạch"
  fi
}

if [ "${1:-}" = "--all-retro" ]; then
  echo "Đánh dấu hồi cố mọi lần chạy đã có (chúng chạy trước khi bật git):"
  n=0
  for d in "$ROOT"/experiments/*/; do
    [ -f "${d}metric.csv" ] || [ -f "${d}cfg_args" ] || continue
    stamp_one "${d%/}" true && n=$((n+1))
  done
  echo "=> đã đánh dấu $n lần chạy"
  exit 0
fi

[ $# -ge 1 ] || { sed -n '2,18p' "$0" | sed 's/^# \?//'; exit 2; }
RETRO=false; [ "${2:-}" = "--retro" ] && RETRO=true
stamp_one "$1" "$RETRO"
