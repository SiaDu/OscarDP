#!/usr/bin/env bash
set -uo pipefail

cd /home/sia/OscarDP || exit 1

PY="/home/sia/OscarDP/.venv/bin/python"
INPUT="/mnt/i/datasets/oscar_movie"
OUTPUT="/mnt/i/datasets/oscar_movie_processed"
WEIGHTS="/home/sia/OscarDP/models/transnetv2/transnetv2-pytorch-weights.pth"
EXPECTED_SHA="53f3e734bc191ae1c58ef61121711518c40767013ea32644fa5f1db9dcbb5ae8"
MIN_FREE_BYTES=21474836480

mkdir -p "$OUTPUT"

actual_sha="$(sha256sum "$WEIGHTS" | cut -d' ' -f1)"
if [[ "$actual_sha" != "$EXPECTED_SHA" ]]; then
    echo "ERROR: TransNetV2 weight SHA-256 mismatch" >&2
    echo "Expected: $EXPECTED_SHA" >&2
    echo "Actual:   $actual_sha" >&2
    exit 1
fi

"$PY" - <<'PY'
import sys
import torch

if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA is unavailable; refusing to fall back to CPU")

tensor = torch.zeros(1, device="cuda:0")

print("Python:", sys.executable)
print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
print("Capability:", torch.cuda.get_device_capability(0))
print("Test tensor:", tensor.device)
PY

declare -A VIDEOS=(
    [tt31193180]='/mnt/i/datasets/oscar_movie/tt31193180_Sinners/Sinners (2025) 4K HDR10+流媒体版 杜比全景声DDP5.1 简英特效字幕 .mkv'
    [tt27847051]='/mnt/i/datasets/oscar_movie/tt27847051_TheSecretAgent/The.Secret.Agent.2025.2160p.AMZN.WEB-DL.DDP5.1.H.265.mkv'
    [tt1312221]='/mnt/i/datasets/oscar_movie/tt1312221_Frankenstein/Frankenstein.2025.1080p.NF.WEB-DL.H.264.DDP5.1.Atmos.mkv'
    [tt27714581]='/mnt/i/datasets/oscar_movie/tt27714581_SentimentalValue/Sentimental.Value.2025.MULTi.1080p.WEB-DL.H264.DDP5.1.Atmos-HamiltonShare.mkv'
    [tt14905854]='/mnt/i/datasets/oscar_movie/tt14905854_Hamnet/Hamnet.哈姆奈特.2025.1080p.中英字幕.mp4.mp4'
    [tt18382850]='/mnt/i/datasets/oscar_movie/tt18382850_IfIHadLegsIdKickYou/如果有腿，我会踢你.2025.BD1080P.mp4'
    [tt30343021]='/mnt/i/datasets/oscar_movie/tt30343021_SongSungBlue/Song.Sung.Blue.2025.HDR.2160p.WEB-DL.H.265.Atmos-HamiltonShare.mkv'
    [tt12300742]='/mnt/i/datasets/oscar_movie/tt12300742_Bugonia/【4KUHD】拯救地球.中英双字.Bugonia.2025.BluRay.2160p.TrueHD7.1.HDR.x265.10bit.mkv'
)

ORDER=(
    tt31193180
    tt27847051
    tt1312221
    tt27714581
    tt14905854
    tt18382850
    tt30343021
    tt12300742
)

completed=()
failed=()

for key in "${ORDER[@]}"; do
    video="${VIDEOS[$key]}"

    echo
    echo "============================================================"
    echo "Starting $key"
    echo "Video: $video"
    echo "Time:  $(date '+%F %T %Z')"
    echo "============================================================"

    if [[ ! -f "$video" ]]; then
        echo "ERROR: Source video does not exist: $video" >&2
        failed+=("$key:source_missing")
        continue
    fi

    available_bytes="$(df -PB1 "$OUTPUT" | awk 'NR == 2 {print $4}')"
    if [[ -z "$available_bytes" ]] || (( available_bytes < MIN_FREE_BYTES )); then
        echo "ERROR: Less than 20 GiB free; stopping before $key" >&2
        failed+=("$key:not_started_low_space")
        break
    fi

    if "$PY" -m oscardp.shots process-one \
        --video "$video" \
        --input-root "$INPUT" \
        --output-root "$OUTPUT" \
        --weights "$WEIGHTS" \
        --threshold 0.5 \
        --device cuda \
        --resume
    then
        if "$PY" -m oscardp.shots validate \
            --output-root "$OUTPUT" \
            --movie-key "$key"
        then
            completed+=("$key")
            echo "COMPLETED AND VALIDATED: $key"
        else
            failed+=("$key:validation_failed")
            echo "VALIDATION FAILED: $key" >&2
        fi
    else
        failed+=("$key:processing_failed")
        echo "PROCESSING FAILED: $key" >&2
    fi

    df -h "$OUTPUT"
done

echo
echo "======================== FINAL SUMMARY ========================"
echo "Completed: ${completed[*]:-none}"
echo "Failed:    ${failed[*]:-none}"
echo "Output:    $OUTPUT"
echo "Finished:  $(date '+%F %T %Z')"

if ((${#failed[@]} > 0)); then
    exit 1
fi