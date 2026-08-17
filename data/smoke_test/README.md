# Real-video smoke test

## Source

- Movie key: `tt32536315`
- Source video: `/mnt/g/datasets/oscar_movie/tt32536315_BlueMoon/Blue.Moon.2025.2160p.WEB-DL.SDR.H.265..DDP5.1.Atmos-HamiltonShare.mkv`
- Selected interval: `00:10:00.000` through approximately `00:10:30.000`
- Selection reason: a frame at `00:10:00` was visually checked and contains a
  normally lit bar scene rather than a title card or black frame.

## Sample command

```bash
ffmpeg -y \
  -ss 00:10:00 \
  -i '/mnt/g/datasets/oscar_movie/tt32536315_BlueMoon/Blue.Moon.2025.2160p.WEB-DL.SDR.H.265..DDP5.1.Atmos-HamiltonShare.mkv' \
  -t 30 \
  -map 0:v:0 -an -sn \
  -vf scale=1280:-2 \
  -c:v libx264 -preset veryfast -crf 18 -pix_fmt yuv420p \
  -movflags +faststart \
  data/smoke_test/input/sample_30s.mp4
```

The generated video and everything under `data/smoke_test/output/` are ignored
by Git. Only this provenance record is intended to be committed.
