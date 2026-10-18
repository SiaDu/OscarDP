#!/usr/bin/env bash
# OscarDP Stage 01 runner. Edit MOVIE_IDS or supply IDs as arguments.
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

INPUT_ROOT="/media/sdu/SiyaoDu/datasets/oscar_movie"
OUTPUT_ROOT="/media/sdu/SiyaoDu/datasets/oscar_movie_analysis"
TITLES="data/metadata/movie_titles.json"
WEIGHTS="models/transnetv2/transnetv2-pytorch-weights.pth"
DEVICE="cuda"
LOG_FILE="${OSCARDP_STAGE01_LOG:-/tmp/oscardp_stage01.log}"

# Add the IMDb IDs you want to process, one per line (without #).
# Leave empty and use --all to explicitly run the full title list.
MOVIE_IDS=(
  tt27714581
  tt27847051
  tt30343021
  tt31193180
  tt32536315
  tt32916440
  tt11563598
  tt1262426
  tt14961016
  tt17526714
  tt20215234
  tt20221436
  tt21823606
  tt28479262
  tt28607951
  tt8368368
  tt8999762
  tt1200263
  tt14160834
  tt14230458
  tt14849194
  tt1517268
  tt15398776
  tt17009710
  tt23561236
  tt5302918
  tt5535276
  tt5537002
  tt3704428
  tt11813216
  tt13833688
  tt14444726
  tt1655389
  tt19770238
  tt6710474
  tt9051908
  tt9114286
  tt10095582
  tt10293406
  tt10366460
  tt12536294
  tt12618926
  tt12789558
  tt3581652
)

usage() {
  cat <<'HELP'
Usage:
  ./scripts/run_stage01.sh                  # IDs configured in MOVIE_IDS
  ./scripts/run_stage01.sh tt0318374 tt17009710
  ./scripts/run_stage01.sh --all            # all IDs in movie_titles.json
  ./scripts/run_stage01.sh --limit 5        # first 5 discovered IDs
  ./scripts/run_stage01.sh tt0318374 --qc
  ./scripts/run_stage01.sh --dry-run --limit 5

Options passed to oscardp.shots: --qc, --dry-run, --overwrite,
  --no-progress, --save-all-boundary-frames, --save-raw-predictions,
  --threshold VALUE, --limit NUMBER (batch mode only).
Log: /tmp/oscardp_stage01.log (override with OSCARDP_STAGE01_LOG).
HELP
}

cli_ids=()
extra_args=()
run_all=false
has_limit=false

while (($#)); do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --all)
      run_all=true
      shift
      ;;
    --movie-id)
      if (($# < 2)); then echo 'Missing value for --movie-id' >&2; exit 2; fi
      cli_ids+=("$2")
      shift 2
      ;;
    --limit|--threshold)
      if (($# < 2)); then echo "Missing value for $1" >&2; exit 2; fi
      [[ "$1" == "--limit" ]] && has_limit=true
      extra_args+=("$1" "$2")
      shift 2
      ;;
    --qc|--dry-run|--overwrite|--no-progress|--save-all-boundary-frames|--save-raw-predictions)
      extra_args+=("$1")
      shift
      ;;
    tt[0-9]*)
      cli_ids+=("$1")
      shift
      ;;
    *)
      echo "Unknown option or IMDb ID: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if $run_all && ((${#cli_ids[@]} > 0)); then
  echo 'Do not combine --all with specific IMDb IDs.' >&2
  exit 2
fi

if $run_all; then
  movie_ids=()
elif ((${#cli_ids[@]} > 0)); then
  movie_ids=("${cli_ids[@]}")
else
  movie_ids=("${MOVIE_IDS[@]}")
fi

if ((${#movie_ids[@]} > 0)) && $has_limit; then
  echo 'Use --limit only in batch mode, without specific IMDb IDs.' >&2
  exit 2
fi

if ((${#movie_ids[@]} == 0)) && ! $run_all && ! $has_limit; then
  echo 'No IMDb IDs selected. Edit MOVIE_IDS, pass IDs, or use --all.' >&2
  exit 2
fi

for id in "${movie_ids[@]}"; do
  if [[ ! "$id" =~ ^tt[0-9]+$ ]]; then
    echo "Invalid IMDb ID: $id" >&2
    exit 2
  fi
done

mkdir -p "$(dirname -- "$LOG_FILE")"

run_stage01() {
  printf '\n========== Stage01 run %s ==========\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "Project: $PROJECT_ROOT"
  echo "Log: $LOG_FILE"
  if [[ ! -f "$TITLES" ]]; then echo "Missing titles file: $TITLES" >&2; return 2; fi
  if [[ ! -f "$WEIGHTS" ]]; then echo "Missing weights file: $WEIGHTS" >&2; return 2; fi

  local base=(uv run python -m oscardp.shots process
    --input-root "$INPUT_ROOT"
    --output-root "$OUTPUT_ROOT"
    --titles "$TITLES"
    --weights "$WEIGHTS"
    --device "$DEVICE")

  if ((${#movie_ids[@]} == 0)); then
    echo 'Selection: batch from movie_titles.json'
    "${base[@]}" "${extra_args[@]}"
    return $?
  fi

  local ok=0 failed=0 id
  echo "Selection: ${movie_ids[*]}"
  for id in "${movie_ids[@]}"; do
    printf '\n--- Processing %s ---\n' "$id"
    if "${base[@]}" --movie-id "$id" "${extra_args[@]}"; then
      ((ok += 1))
    else
      echo "FAILED: $id" >&2
      ((failed += 1))
    fi
  done
  printf '\nStage01 selection finished: %d succeeded, %d failed.\n' "$ok" "$failed"
  if ((failed > 0)); then return 1; fi
  return 0
}

# tee -a saves stdout and stderr without hiding non-zero exit statuses.
set +e
run_stage01 2>&1 | tee -a "$LOG_FILE"
run_status=${PIPESTATUS[0]}
set -e
printf 'Stage01 finished with exit code %d. Log: %s\n' "$run_status" "$LOG_FILE"
exit "$run_status"
