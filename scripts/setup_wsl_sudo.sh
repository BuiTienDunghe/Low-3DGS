#!/usr/bin/env bash
# =============================================================================
#  P0 - Cai toolchain trong WSL2 Ubuntu 24.04
#
#  CHAY KIEU NAO CUNG DUOC:
#
#   (A) KHONG CAN MAT KHAU  <-- dung cai nay neu ban quen pass sudo
#       wsl -d Ubuntu -u root bash /mnt/d/low-3DGS/scripts/setup_wsl_sudo.sh
#
#   (B) Kieu thuong (hoi pass sudo mot lan)
#       wsl -d Ubuntu -- bash /mnt/d/low-3DGS/scripts/setup_wsl_sudo.sh
#
#  Script tu nhan biet dang chay bang root hay user thuong, va LUON ghi file
#  cau hinh vao home cua USER THUONG (khong phai /root), dung chu so huu.
#
#  Cai gi:  build-essential (gcc/g++/make) + CUDA toolkit 12.6 + vai lib he thong
#           + va 2 header bat buoc cho Ubuntu 24.04 (issue #923)
#  KHONG cai: goi 'cuda' hay 'cuda-drivers' -> se cai driver Linux, XUNG DOT voi
#             driver Windows ma WSL2 dang dung. Chi cai 'cuda-toolkit-12-6'.
#  Vi sao 12.6: driver Windows 560.70 ho tro toi da CUDA 12.6; PyTorch co wheel cu126.
#               nvcc phai khop major.minor voi torch de build rasterizer.
#  Tai ~3 GB. Mat 5-15 phut tuy mang.
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------- ai dang chay?
if [ "$(id -u)" -eq 0 ]; then
    SUDO=""                       # da la root roi, khong can sudo
    # Tim user thuong de ghi ~/.bashrc dung cho: SUDO_USER -> wsl.conf -> uid 1000/1001
    TARGET_USER="${SUDO_USER:-}"
    [ -z "$TARGET_USER" ] && TARGET_USER="$(awk -F= '/^default=/{gsub(/ /,"");print $2}' /etc/wsl.conf 2>/dev/null || true)"
    [ -z "$TARGET_USER" ] && TARGET_USER="$(getent passwd 1000 | cut -d: -f1 || true)"
    [ -z "$TARGET_USER" ] && TARGET_USER="$(getent passwd 1001 | cut -d: -f1 || true)"
    if [ -z "$TARGET_USER" ]; then echo "LOI: khong xac dinh duoc user thuong"; exit 1; fi
    MODE="root"
else
    SUDO="sudo"
    TARGET_USER="$(id -un)"
    MODE="user (se hoi pass sudo)"
fi
TARGET_HOME="$(getent passwd "$TARGET_USER" | cut -d: -f6)"
run_as_user() { if [ "$(id -u)" -eq 0 ]; then su - "$TARGET_USER" -c "$1"; else bash -c "$1"; fi; }

echo "=== Che do: $MODE | user dich: $TARGET_USER ($TARGET_HOME) ==="
echo

echo "==> [1/6] build-essential + thu vien he thong"
$SUDO apt-get update
$SUDO apt-get install -y --no-install-recommends \
    build-essential ninja-build cmake pkg-config git curl wget ca-certificates \
    python3-venv python3-dev python3-pip \
    libgl1 libglib2.0-0 libjpeg-dev libpng-dev \
    zip unzip htop || {
      echo "  (thu lai khong co goi tuy chon)"
      $SUDO apt-get install -y --no-install-recommends \
        build-essential ninja-build cmake pkg-config git curl wget ca-certificates \
        python3-venv python3-dev python3-pip libgl1 libglib2.0-0 zip unzip; }
# 'zip' can cho LightGaussian vectree.py (no goi os.system("zip -r ...") va KHONG bao loi neu thieu)

echo "==> [2/6] NVIDIA CUDA repo keyring (wsl-ubuntu)"
TMP=$(mktemp -d)
wget -q -O "$TMP/cuda-keyring.deb" \
  https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
$SUDO dpkg -i "$TMP/cuda-keyring.deb"
$SUDO apt-get update
rm -rf "$TMP"

echo "==> [3/6] CUDA toolkit 12.6 (CHI toolkit, KHONG driver) - ~3 GB"
$SUDO apt-get install -y --no-install-recommends cuda-toolkit-12-6

echo "==> [4/6] PATH cho nvcc -> $TARGET_HOME/.bashrc"
BRC="$TARGET_HOME/.bashrc"
if ! grep -q 'cuda-12.6/bin' "$BRC" 2>/dev/null; then
  cat >> "$BRC" <<'EOF'

# CUDA 12.6 (low-3DGS)
export CUDA_HOME=/usr/local/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export TORCH_CUDA_ARCH_LIST="7.5"       # GTX 1650 Ti = Turing sm_75
EOF
  chown "$TARGET_USER":"$TARGET_USER" "$BRC" 2>/dev/null || true
  echo "    da them"
else
  echo "    da co san"
fi
export CUDA_HOME=/usr/local/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH

echo "==> [5/6] Va header CUDA 12 cho rasterizer (BAT BUOC tren Ubuntu 24.04)"
# Nguon: https://github.com/graphdeco-inria/gaussian-splatting/issues/923
#  - gcc 13 bo include bac cau -> can <cstdint> cho uint32_t/std::uintptr_t
#  - CUDA 12 headers bo FLT_MAX/FLT_MIN -> can <cfloat>
# Da verify 2026-09-02: CA HAI deu THIEU trong upstream (grep -c = 0).
patch_include() {
  [ -f "$1" ] || return 0
  if grep -q "$2" "$1"; then echo "    da co <$2> trong $(basename "$1")"; else
    sed -i "0,/#include/s|#include|#include <$2>\n#include|" "$1"
    chown "$TARGET_USER":"$TARGET_USER" "$1" 2>/dev/null || true
    echo "    them <$2> vao $(basename "$1")"
  fi
}
for repo in "$TARGET_HOME"/l3dgs/third_party/*/; do
  [ -d "$repo" ] || continue
  echo "  $(basename "$repo")"
  for f in "$repo"submodules/*rasterization*/cuda_rasterizer/rasterizer_impl.h; do patch_include "$f" cstdint; done
  for f in "$repo"submodules/simple-knn/simple_knn.cu;                          do patch_include "$f" cfloat; done
done

echo "==> [6/6] Kiem tra"
echo "  gcc  : $(gcc --version | head -1)"
echo "  nvcc : $(nvcc --version | grep release || echo 'KHONG THAY - kiem tra PATH')"
echo "  GPU  : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'nvidia-smi loi')"
echo "  venv : $(run_as_user 'python3 -c "import ensurepip; print(\"OK\")"' 2>/dev/null || echo 'THIEU python3-venv')"
echo
echo "XONG."
[ "$(id -u)" -eq 0 ] && echo "LUU Y: da chay bang root. File trong $TARGET_HOME da duoc chown ve $TARGET_USER."
echo "Mo terminal WSL moi (hoac 'source ~/.bashrc') roi bao Claude tiep tuc."
