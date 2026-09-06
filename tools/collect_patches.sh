#!/usr/bin/env bash
# Liệt kê mọi sửa đổi của dự án lên mã nguồn bên thứ ba, để tái lập được.
cd ~/l3dgs/third_party/gaussian-splatting-pup || exit 1
echo "REPO gaussian-splatting-pup @ $(git rev-parse HEAD)"
echo
echo "########## utils/general_utils.py ##########"
git diff --unified=2 -- utils/general_utils.py
echo
for m in compress-diff-gaussian-rasterization diff-gaussian-rasterization \
         rasterization_and_pup_fisher simple-knn; do
  echo "########## submodules/$m @ $(cd submodules/$m && git rev-parse --short HEAD) ##########"
  ( cd "submodules/$m" || exit 0
    st=$(git status --porcelain)
    if [ -z "$st" ]; then echo "  (không có thay đổi trong cây làm việc)"; else
      echo "$st" | sed 's/^/  /'
      git diff --unified=2
    fi )
  echo
done
