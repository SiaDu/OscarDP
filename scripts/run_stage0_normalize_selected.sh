#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

INPUT_ROOT='/media/sdu/SiyaoDu/datasets/oscar_movie'
REPORT_ROOT='/media/sdu/SiyaoDu/datasets/oscar_movie_standardized'

MOVIE_IDS=(
  tt1065073
  tt0491747
  tt0410097
  tt0180073
)
failed_movies=()

for movie_id in "${MOVIE_IDS[@]}"; do
  echo
  echo "============================================================"
  echo "开始处理: ${movie_id}"
  echo "============================================================"

  if python scripts/stage0_media_normalize.py \
    --input-root "$INPUT_ROOT" \
    --output-root "$REPORT_ROOT/$movie_id" \
    --movie-id "$movie_id" \
    --gpu-hdr \
    --execute; then
    echo "完成处理: ${movie_id}"
  else
    echo "处理失败: ${movie_id}；查看 $REPORT_ROOT/$movie_id/stage0_errors.jsonl"
    failed_movies+=("$movie_id")
  fi
done

echo
if ((${#failed_movies[@]})); then
  echo "处理结束，失败: ${failed_movies[*]}"
  exit 1
fi
echo "全部电影已完成。"
