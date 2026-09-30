#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

INPUT_ROOT='/media/sdu/ERAZER G900/datasets/oscar_movie'

MOVIE_IDS=(
  tt1602620
  tt1907668
  tt1340800
  tt1478338
  tt1655420
  tt1285016
  tt0887912
  tt0929632
  tt1315981
  tt0497465
  tt0918927
  tt1013753
  tt0443680
  tt0469494
  tt0477348
  tt0758758
  tt0775529
  tt0436697
)

for movie_id in "${MOVIE_IDS[@]}"; do
  echo
  echo "============================================================"
  echo "开始处理: ${movie_id}"
  echo "============================================================"

  python scripts/stage0_media_normalize.py \
    --input-root "$INPUT_ROOT" \
    --movie-id "$movie_id" \
    --execute

  echo "完成处理: ${movie_id}"
done

echo
echo "全部电影已完成。"