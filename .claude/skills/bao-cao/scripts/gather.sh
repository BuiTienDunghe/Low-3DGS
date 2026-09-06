#!/usr/bin/env bash
# =============================================================================
#  gather.sh — thu thập SỰ THẬT SỐNG về dự án, không đọc kết luận từ tài liệu.
#
#  Vì sao tồn tại: chế độ hỏng đặc trưng của dự án này là số liệu cũ và số liệu
#  chép lại. Báo cáo mà đọc NEXT.md rồi nhắc lại sẽ thừa hưởng đúng lỗi đó.
#  Script này chỉ lấy thứ đo được ngay lúc chạy: git, máy, file dữ liệu.
#
#  Dùng:  bash .claude/skills/bao-cao/scripts/gather.sh
#  Chạy được cả trong WSL lẫn Git Bash; phần nào không có thì báo "không đo được"
#  chứ không đoán.
# =============================================================================
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." || exit 1

hr() { printf '\n===== %s =====\n' "$1"; }

hr "PHIÊN BẢN & GIT"
echo "  VERSION: $(cat VERSION 2>/dev/null || echo 'chưa có')"
if git rev-parse --git-dir >/dev/null 2>&1; then
  echo "  commit : $(git log --oneline -1 2>/dev/null)"
  echo "  nhánh  : $(git rev-parse --abbrev-ref HEAD 2>/dev/null)"
  echo "  tag    : $(git tag -l | tr '\n' ' ')"
  n=$(git status --porcelain | wc -l)
  if [ "$n" -eq 0 ]; then
    echo "  cây làm việc: SẠCH"
  else
    echo "  cây làm việc: BẨN ($n mục) — kết quả chạy lúc này KHÔNG tái lập được"
    git status --short | head -8 | sed 's/^/      /'
  fi
  if up=$(git rev-list --count '@{u}..HEAD' 2>/dev/null); then
    if [ "$up" -eq 0 ]; then echo "  đã đồng bộ với remote"
    else echo "  commit chưa đẩy: $up"; fi
  else
    echo "  (nhánh chưa gắn với remote)"
  fi
else
  echo "  CHƯA CÓ GIT — không kết quả nào truy ngược được về phiên bản mã"
fi

hr "MÁY"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu \
             --format=csv,noheader 2>/dev/null | sed 's/^/  GPU: /'
  k=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)
  echo "  tiến trình đang dùng GPU: ${k:-0}"
else
  echo "  GPU: (không đo được ở môi trường này)"
fi
[ -r /proc/meminfo ] && awk '/MemAvailable/{printf "  RAM khả dụng: %d MiB\n", $2/1024}' /proc/meminfo
df -BG . 2>/dev/null | tail -1 | awk '{print "  Đĩa còn: "$4}'
p=$(pgrep -c -f 'prune_finetune.py' 2>/dev/null || true)
echo "  tiến trình huấn luyện đang chạy: ${p:-0}"

hr "DỮ LIỆU ĐO ĐƯỢC"
for f in experiments/*.csv; do
  [ -f "$f" ] || continue
  printf '  %-34s %4d dòng\n' "$(basename "$f")" "$(( $(wc -l < "$f") - 1 ))"
done
echo "  thư mục run   : $(find experiments -maxdepth 1 -mindepth 1 -type d | wc -l)"
echo "  có run_meta   : $(ls experiments/*/run_meta.json 2>/dev/null | wc -l)"
echo "  file .ply còn : $(find experiments -name '*.ply' 2>/dev/null | wc -l) (không lên git)"

hr "ĐỐI CHIẾU: TÀI LIỆU CÓ CŨ HƠN DỮ LIỆU KHÔNG"
newest=$(ls -t experiments/*.csv 2>/dev/null | head -1)
if [ -n "$newest" ]; then
  echo "  dữ liệu mới nhất: $(basename "$newest")  ($(date -r "$newest" '+%F %H:%M' 2>/dev/null))"
  for d in docs/NEXT.md docs/05_EXPERIMENTS.md CHANGELOG.md; do
    [ -f "$d" ] || continue
    if [ "$d" -ot "$newest" ]; then
      echo "  ⚠ $d CŨ HƠN dữ liệu — có thể chưa ghi kết quả mới nhất"
    else
      echo "  ✓ $d mới hơn dữ liệu"
    fi
  done
fi

hr "ĐỐI CHIẾU: DỮ LIỆU ĐÃ CHẠY CÓ VÀO HẾT FILE TỔNG HỢP CHƯA"
# Cạm bẫy đã xảy ra thật: một cổng kiểm sai âm thầm loại 3 lần chạy TỐT khỏi file
# tổng hợp, trong khi thư mục run vẫn còn đủ dữ liệu — nhìn vào file tổng hợp thì
# không thấy gì bất thường. Phép so dưới đây bắt đúng loại lỗi đó: đếm dòng trong
# các thư mục run của một họ, so với số dòng trong file tổng hợp của họ.
check_family() {
  local csv="$1"; shift
  [ -f "experiments/$csv" ] || return 0
  local want have=0 dirs=0 d pat
  want=$(( $(wc -l < "experiments/$csv") - 1 ))
  for pat in "$@"; do
    for d in experiments/$pat/; do
      [ -f "${d}metric.csv" ] || continue
      dirs=$((dirs+1))
      have=$(( have + $(wc -l < "${d}metric.csv") - 1 ))
    done
  done
  if [ "$want" -eq "$have" ]; then
    printf '  ✓ %-22s %3d dòng = %2d thư mục run (%d dòng)\n' "$csv" "$want" "$dirs" "$have"
  else
    printf '  ⚠ %-22s %3d dòng ≠ %2d thư mục run (%d dòng) -- LỆCH %d, có run chưa được gom\n' \
           "$csv" "$want" "$dirs" "$have" "$(( have - want ))"
  fi
}
check_family repeats_train.csv   'rep_train_*'
check_family sweep_train.csv     'sweep_train_p*' 'knee_train_p*'
check_family scene2_truck864.csv 's2_truck864_*'
echo "  (exp016/exp017/lg_truck có trước quy ước CSV, không thuộc họ nào)"

hr "MỤC LỤC TÀI LIỆU (đọc trực tiếp, đừng để script tóm tắt hộ)"
# Cố ý KHÔNG grep nội dung: grep theo từ khoá kéo về cả số liệu trong bảng kết quả
# và tạo ấn tượng sai. Chỉ liệt kê tiêu đề để biết chỗ nào cần đọc.
for d in docs/NEXT.md docs/05_EXPERIMENTS.md; do
  [ -f "$d" ] || continue
  echo "  $d:"
  grep -n '^## ' "$d" 2>/dev/null | head -12 | sed 's/^/      /'
done

echo
echo "(Hết phần sự thật sống. Con số KẾT QUẢ phải lấy từ tools/, không phải từ docs/.)"
