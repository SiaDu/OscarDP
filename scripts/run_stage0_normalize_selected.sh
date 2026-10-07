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

for movie_id in "${MOVIE_IDS[@]}"; do
  echo
  echo "============================================================"
  echo "开始处理: ${movie_id}"
  echo "============================================================"

  python scripts/stage0_media_normalize.py \
    --input-root "$INPUT_ROOT" \
    --output-root "$REPORT_ROOT/$movie_id" \
    --movie-id "$movie_id" \
    --gpu-hdr \
    --execute

  echo "完成处理: ${movie_id}"
done

echo
echo "全部电影已完成。"
