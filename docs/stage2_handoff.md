# OscarDP Stage 2 handoff

## Frozen release

- Release: `v3_2_1_production_3_final_seven`
- Status: `FROZEN`
- Reviewer: `v3.2.1-production.3-retrieval-v3-validator-v3`
- Complete / terminal / blocked: 7/8 / 8/8 / 1/8
- Pending isolated human ambiguities: 4

The authoritative generated release remains under
`/mnt/g/datasets/oscar_movie_processed/stage2_releases/v3_2_1_production_3_final_seven/`.
It contains the release manifest, validation, pending-ambiguity package and
generated Stage 3 handoff. The 14 IMDb-prefixed QC/manifest delivery copies
remain in `/mnt/g/datasets/oscar_movie_processed/all/`.

`tt30144839` remains `BLOCKED_WITH_EXPLICIT_REASON`: the Song Sung Blue PDF is
already the completed `tt30343021` screenplay and cannot be reused.

## Dataset relocation

Frozen artifacts keep their original `/mnt/i/...` paths and SHA-256 values.
They must not be edited to change mount prefixes. Consumers can explicitly map
the historical mount when invoking the supported commands:

```bash
python -m oscardp.performance_candidates mine \
  --release-manifest /mnt/g/datasets/oscar_movie_processed/stage2_releases/v3_2_1_production_3_final_seven/release_manifest.json \
  --path-map /mnt/i=/mnt/g \
  ...
```

The resolver records both the manifest-declared path and the locally resolved
path, then verifies the existing artifact hash.

## Ownership boundary

`/home/sia/OscarDP` is for source, tests, reusable documentation and policy.
`/mnt/g/datasets/oscar_movie_processed` is for source-data-adjacent generated
outputs, immutable provenance, status registries, Batch artifacts and releases.
Do not move or rewrite frozen manifests, video, subtitles, screenplay PDFs,
shots, keyframes, Stage 1 QC, deterministic Stage 2 outputs or raw Batch files.
