#!/usr/bin/env bash
# Kiểm môi trường build/chạy có đủ điều kiện không. Chạy từ gốc repo: bash submission/scripts/00_check_env.sh
# Thoát 1 nếu có mục BẮT BUỘC không đạt.
FAIL=0
ok()   { echo "  [ OK ] $*"; }
bad()  { echo "  [FAIL] $*"; FAIL=1; }
warn() { echo "  [warn] $*"; }

echo "== Công cụ"
command -v git >/dev/null && ok "git $(git --version | cut -d' ' -f3)" || bad "thiếu git"
command -v docker >/dev/null && ok "docker $(docker --version | sed 's/Docker version //; s/,.*//')" || bad "thiếu docker"
docker buildx version >/dev/null 2>&1 && ok "BuildKit/buildx có" || warn "không thấy buildx (vẫn build được với DOCKER_BUILDKIT=1)"
docker info >/dev/null 2>&1 && ok "daemon docker truy cập được" || bad "không chạy được 'docker info' (quyền hoặc daemon)"
command -v md5sum >/dev/null && ok "md5sum" || bad "thiếu md5sum"
command -v pigz >/dev/null && ok "pigz (nén nhanh)" || warn "không có pigz: sẽ dùng gzip (chậm hơn nhiều với tệp ~30GB)"
command -v python3 >/dev/null && ok "python3 $(python3 --version | cut -d' ' -f2)" || bad "thiếu python3"

echo "== GPU (cần để chạy thử thật; build image thì không cần)"
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | sed 's/^/        /'
  DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
  [ "${DRV:-0}" -ge 570 ] && ok "driver $DRV >= 570 (RTX 5090 cần R570+)" || warn "driver ${DRV:-?} < 570: máy này không chạy được RTX 5090, chỉ dùng được để build"
  docker info 2>/dev/null | grep -qi nvidia && ok "NVIDIA Container Toolkit (runtime nvidia)" || warn "docker chưa có runtime nvidia: 'docker run --gpus' sẽ không chạy"
else
  warn "không có nvidia-smi: chỉ build/save được, KHÔNG chạy thử được"
fi

echo "== Ổ đĩa (image ~30GB + cache build + docker save ~50GB tạm + tệp .tar.gz ~35GB)"
DOCKER_ROOT=$(docker info -f '{{.DockerRootDir}}' 2>/dev/null || echo /var/lib/docker)
for d in "$DOCKER_ROOT" "$PWD"; do
  FREE=$(df -Pk "$d" 2>/dev/null | awk 'NR==2{printf "%d", $4/1048576}')
  [ -n "$FREE" ] && { [ "$FREE" -ge 150 ] && ok "$d còn ${FREE} GB" || bad "$d chỉ còn ${FREE} GB (cần >= 150 GB trống, tốt nhất 200 GB)"; }
done
echo "== RAM"
RAM=$(awk '/MemTotal/{printf "%d", $2/1048576}' /proc/meminfo); [ "$RAM" -ge 32 ] && ok "RAM ${RAM} GB" || warn "RAM ${RAM} GB (nên >= 32 GB để gộp LoRA/kiểm tra)"

echo "== Mạng (cần khi build/tải model; KHÔNG cần khi chạy bài nộp)"
for u in https://huggingface.co https://pypi.org https://registry-1.docker.io/v2/; do
  curl -sI -m 15 -o /dev/null "$u" && ok "$u" || bad "không tới được $u"
done

echo "== Repo"
[ -f submission/Dockerfile ] && ok "submission/Dockerfile" || bad "chạy từ GỐC repo (không thấy submission/Dockerfile)"
[ -d dataset/train ] && ok "dataset/train (dữ liệu huấn luyện cho training_code/data)" || warn "không thấy dataset/train: cần để đóng gói dữ liệu huấn luyện"
[ -n "${TEAM:-}" ] && ok "TEAM=$TEAM" || bad "chưa đặt TEAM (tên đội viết thường, vd export TEAM=ten_doi)"
echo
[ $FAIL -eq 0 ] && echo "KẾT QUẢ: đủ điều kiện." || echo "KẾT QUẢ: CHƯA đủ điều kiện (xem các dòng FAIL)."
exit $FAIL
