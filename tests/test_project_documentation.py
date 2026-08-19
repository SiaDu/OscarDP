from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_stage2_repository_documents_identify_frozen_policy_source() -> None:
    policy = (ROOT / "docs/stage2_annotation_policy_v1.md").read_text(encoding="utf-8")
    handoff = (ROOT / "docs/stage2_handoff.md").read_text(encoding="utf-8")
    layout = (ROOT / "docs/data_layout.md").read_text(encoding="utf-8")

    assert "0ac78ef566d1e84198528fd706d6d31e241ed6c37444c87aed0e2de4e34b74c3" in policy
    assert "/mnt/g/datasets/oscar_movie_processed" in handoff
    assert "--path-map /mnt/i=/mnt/g" in handoff
    assert "/home/sia/OscarDP" in layout
