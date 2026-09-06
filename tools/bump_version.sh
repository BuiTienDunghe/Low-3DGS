#!/usr/bin/env bash
# =============================================================================
#  bump_version.sh — nâng phiên bản, ghi CHANGELOG, commit, tạo tag.
#
#  Dùng:  bash tools/bump_version.sh <major|minor|patch> "<mô tả một dòng>"
#  Ví dụ: bash tools/bump_version.sh minor "Đường cong đánh đổi cho scene thứ hai"
#
#  Ý nghĩa từng số: xem docs/VERSIONING.md.
#  Nhắc lại phần dễ nhầm nhất: RÚT LẠI một phát biểu là MAJOR, không phải PATCH.
#
#  Ghi chú: Git Bash trên Windows có `python` nhưng KHÔNG có `python3`.
#  Và nếu hỏng giữa chừng thì script tự hoàn tác — không để lại trạng thái nửa vời.
# =============================================================================
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

KIND=${1:-}; DESC=${2:-}
case "$KIND" in
  major|minor|patch) ;;
  *) echo "Dùng: $0 <major|minor|patch> \"<mô tả>\""; exit 2 ;;
esac
[ -n "$DESC" ] || { echo "Thiếu mô tả."; exit 2; }

PY=$(command -v python3 2>/dev/null || command -v python 2>/dev/null || true)
[ -n "$PY" ] || { echo "✗ không tìm thấy python (cần để sửa CHANGELOG)"; exit 1; }

if [ -n "$(git status --porcelain)" ]; then
  echo "✗ Cây làm việc bẩn. Commit hoặc stash trước đã."
  git status --short
  exit 1
fi

CUR=$(cat VERSION)
IFS=. read -r MA MI PA <<< "$CUR"
case "$KIND" in
  major) MA=$((MA+1)); MI=0; PA=0 ;;
  minor) MI=$((MI+1)); PA=0 ;;
  patch) PA=$((PA+1)) ;;
esac
NEW="$MA.$MI.$PA"

echo "$CUR  ->  $NEW   ($KIND)"

# Hoàn tác nguyên tử nếu bất kỳ bước nào hỏng.
rollback() {
  if [ "${DONE:-0}" != 1 ]; then
    git checkout -- VERSION CHANGELOG.md 2>/dev/null || true
    echo "  (có lỗi — đã hoàn tác, không đổi gì)"
  fi
}
trap rollback EXIT

printf '%s\n' "$NEW" > VERSION

"$PY" - "$NEW" "$(date +%F)" "$DESC" <<'PY'
import io, re, sys
new, today, desc = sys.argv[1], sys.argv[2], sys.argv[3]
p = "CHANGELOG.md"
s = io.open(p, encoding="utf-8").read()
m = re.search(r"^## \[\d", s, re.M)
if m is None:
    raise SystemExit("không tìm thấy mục phát hành nào trong CHANGELOG.md")
entry = f"## [{new}] — {today}\n\n{desc}\n\n---\n\n"
io.open(p, "w", encoding="utf-8").write(s[:m.start()] + entry + s[m.start():])
print(f"  đã thêm mục [{new}] vào CHANGELOG.md")
PY

git add VERSION CHANGELOG.md
git commit -q -m "chore(release): v$NEW

$DESC"
git tag -a "v$NEW" -m "v$NEW — $DESC"
DONE=1
echo "  ✓ đã commit và tạo tag v$NEW"
echo
echo "Đẩy lên bằng:  git push --follow-tags"
