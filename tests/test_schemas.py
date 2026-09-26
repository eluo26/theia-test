import pytest
from pydantic import ValidationError

from vision.schemas import IndexResponse, IndexedObject, QueryResult, norm_box_to_pixels


def _object(**overrides):
    payload = {
        "label": "blue water bottle",
        "description": "blue bottle on the desk, next to a lamp",
        "box": [100, 200, 300, 800],
        "count": 1,
        "drug_name": None,
        "expiry_text": None,
    }
    payload.update(overrides)
    return payload


def test_index_response_accepts_a_drug_package():
    parsed = IndexResponse.model_validate(
        {
            "objects": [
                _object(
                    label="Jardiance box",
                    drug_name="Jardiance",
                    expiry_text="EXP 2027-04",
                )
            ]
        }
    )
    assert parsed.objects[0].drug_name == "Jardiance"


def test_box_must_stay_inside_0_to_1000():
    with pytest.raises(ValidationError):
        IndexedObject.model_validate(_object(box=[0, 0, 1001, 10]))


def test_box_must_have_area():
    with pytest.raises(ValidationError):
        IndexedObject.model_validate(_object(box=[50, 50, 50, 80]))


def test_unknown_field_is_rejected():
    with pytest.raises(ValidationError):
        IndexedObject.model_validate(_object(confidence=0.9))


def test_query_result_range_stays_null():
    result = QueryResult.model_validate(
        {
            "status": "found",
            "object_id": "obj_007",
            "label": "blue water bottle",
            "azimuth_deg": 34.2,
            "elevation_deg": -12.5,
            "range_m": None,
            "confidence": 0.86,
            "frame_file": "pan030_tilt-10.jpg",
            "bbox_px": [812, 440, 960, 720],
            "candidates": [],
            "metadata": {
                "count": 1,
                "drug_name": None,
                "expiry_text": None,
                "last_seen": "2026-09-26T15:40:00",
            },
            "reason": "Matched the blue bottle next to the lamp",
        }
    )
    assert result.range_m is None
    payload = result.model_dump()
    payload["range_m"] = 1.5
    with pytest.raises(ValidationError):
        QueryResult.model_validate(payload)


def test_norm_box_maps_center_and_clamps_edges():
    assert norm_box_to_pixels([0, 0, 1000, 1000], 200, 100) == (0, 0, 199, 99)
    assert norm_box_to_pixels([500, 500, 500.1, 500.1], 100, 100)[0] == 50
