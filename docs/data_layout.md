# OscarDP project and dataset layout

| Location | Owns |
|---|---|
| `/home/sia/OscarDP` | Python source, tests, packaging, reusable policy, documentation and commands. |
| `/mnt/g/datasets/oscar_movie` | Source video and subtitle inputs; never modify in place. |
| `/mnt/g/datasets/oscar_movie_processed` | Generated Stage 1–3 outputs, Stage 2 status/inventory, Batch provenance, frozen gold, release manifests and delivery packages. |

An artifact manifest is evidence, not a reusable project configuration. It may
contain per-run absolute paths, hashes, time stamps, model IDs and Batch IDs;
therefore it remains next to the data it describes. Immutable artifact files
must never be rewritten merely to change a mount path. Use the explicit
`--path-map OLD=NEW` consumer option after a mount relocation.

The canonical, reusable Stage 2 annotation policy is
[`stage2_annotation_policy_v1.md`](stage2_annotation_policy_v1.md). Its dataset
source remains frozen for historical provenance.
