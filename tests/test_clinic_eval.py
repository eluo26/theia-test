from pathlib import Path

import pytest

from vision.clinic import match_inventory
from vision.config import load_settings
from vision.eval import GroundTruthItem, load_ground_truth, score_scene
from vision.references import TODO_REFERENCE_SEARCH, reference_note
from vision.schemas import QueryResult


def test_inventory_matches_a_starter_pack():
    settings = load_settings(load_env=False)
    match = match_inventory("Jardiance starter pack", threshold=settings.label_sim)
    assert match is not None
    assert match.name == "Jardiance"
    assert match.count == 2
    assert match.expiry == "2027-04-30"
    assert match.resource_links
    assert match_inventory("not a real medicine", threshold=settings.label_sim) is None


def test_reference_hook_does_not_search(tmp_path: Path):
    assert reference_note(Path("data/references")) is None
    note = reference_note(tmp_path)
    assert note is None
    (tmp_path / "pack.png").write_bytes(b"png")
    assert reference_note(tmp_path) == TODO_REFERENCE_SEARCH


def test_eval_scores_correct_objects_and_angular_error():
    items = [
        GroundTruthItem(query="where is the bottle?", expected_label="blue water bottle", az=10, el=0),
        GroundTruthItem(query="where is the lamp?", expected_label="yellow lamp", az=40, el=5),
    ]
    found = QueryResult(
        status="found",
        object_id="obj_001",
        label="blue water bottle",
        azimuth_deg=12,
        elevation_deg=0,
        range_m=None,
        confidence=0.9,
        frame_file="a.png",
        bbox_px=[1, 2, 3, 4],
        candidates=[],
        metadata={"count": 1, "drug_name": None, "expiry_text": None, "last_seen": None},
        reason="matched",
    )
    missed = QueryResult(
        status="not_found",
        confidence=0.2,
        range_m=None,
        reason="nothing",
    )
    report = score_scene("desk", items, [found, missed], label_sim=85)
    assert report.queries == 2
    assert report.correct == 1
    assert report.correct_object_rate == 0.5
    assert report.mean_angular_error_deg == pytest.approx(2.0, abs=0.05)
    assert report.rows[1].angular_error_deg is None


def test_desk_ground_truth_file_loads():
    items = load_ground_truth(Path("data/test_scenes/desk/ground_truth.json"))
    labels = {item.expected_label for item in items}
    assert {"blue water bottle", "Jardiance", "yellow lamp", "green mug"} <= labels
