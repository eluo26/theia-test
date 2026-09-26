import json
import logging
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from vision.ingest import (
    ingest_directory,
    ingest_video,
    keep_sharpest_per_bucket,
    parse_angles_from_name,
    SampledFrame,
)


def test_filename_angles():
    assert parse_angles_from_name("pan060_tilt-10.jpg") == (60.0, -10.0)
    assert parse_angles_from_name("pan-040_tilt000.png") == (-40.0, 0.0)
    assert parse_angles_from_name("pan000_tilt-015.png") == (0.0, -15.0)
    assert parse_angles_from_name("scene.png") is None


def test_untagged_names_take_pan_steps_and_tagged_names_keep_angles(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    """a.jpg and b.jpg have no integer, so filename order puts them ahead of 10.jpg.

    10.jpg sorts by that integer after the unnumbered names. pan30_tilt-10.jpg
    keeps 30 and -10 and does not take a slot in the 0, 30, 60 sequence.
    """
    for name in ("b.jpg", "a.jpg", "10.jpg", "pan30_tilt-10.jpg"):
        Image.new("RGB", (8, 8), (1, 2, 3)).save(tmp_path / name)
    with caplog.at_level(logging.INFO, logger="vision.ingest"):
        frames = ingest_directory(tmp_path)
    by_name = {frame.source_file: frame for frame in frames}
    assert by_name["a.jpg"].pan_deg == 0
    assert by_name["a.jpg"].tilt_deg == 0
    assert by_name["b.jpg"].pan_deg == 30
    assert by_name["b.jpg"].tilt_deg == 0
    assert by_name["10.jpg"].pan_deg == 60
    assert by_name["10.jpg"].tilt_deg == 0
    assert by_name["pan30_tilt-10.jpg"].pan_deg == 30
    assert by_name["pan30_tilt-10.jpg"].tilt_deg == -10
    logged = {record.getMessage() for record in caplog.records}
    assert "a.jpg assigned pan 0" in logged
    assert "b.jpg assigned pan 30" in logged
    assert "10.jpg assigned pan 60" in logged
    assert not any("pan30_tilt-10.jpg" in message for message in logged)


def test_numbered_untagged_names_sort_by_integer(tmp_path: Path):
    for name in ("10.jpg", "2.jpg"):
        Image.new("RGB", (8, 8), (4, 5, 6)).save(tmp_path / name)
    frames = ingest_directory(tmp_path)
    by_name = {frame.source_file: frame for frame in frames}
    assert by_name["2.jpg"].pan_deg == 0
    assert by_name["10.jpg"].pan_deg == 30
    assert by_name["2.jpg"].tilt_deg == 0
    assert by_name["10.jpg"].tilt_deg == 0


def test_manifest_wins_over_the_filename(tmp_path: Path):
    Image.new("RGB", (16, 12), (10, 20, 30)).save(tmp_path / "pan000_tilt000.png")
    manifest = [
        {"file": "pan000_tilt000.png", "pan": 12.5, "tilt": -3, "timestamp": "2026-09-26T15:40:00"}
    ]
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    frames = ingest_directory(tmp_path)
    assert len(frames) == 1
    assert frames[0].pan_deg == 12.5
    assert frames[0].tilt_deg == -3
    assert frames[0].timestamp == "2026-09-26T15:40:00"
    assert frames[0].image.shape == (12, 16, 3)


def test_sharpest_frame_in_each_bucket_drops_blur():
    flat = np.full((24, 24, 3), 128, np.uint8)
    sharp = flat.copy()
    sharp[4:20, 4:20] = 0
    samples = [
        SampledFrame(0.0, 10.2, 0.0, flat, sharpness=0.0),
        SampledFrame(0.5, 10.4, 0.2, sharp, sharpness=50.0),
        SampledFrame(1.0, 10.3, -0.2, sharp, sharpness=80.0),
        SampledFrame(1.5, 14.0, 0.0, sharp, sharpness=40.0),
    ]
    kept = keep_sharpest_per_bucket(samples, blur_threshold=30)
    assert len(kept) == 2
    by_pan = {round(sample.pan_deg): sample for sample in kept}
    assert by_pan[10].sharpness == 80.0
    assert by_pan[14].sharpness == 40.0


def test_video_csv_interpolates_angles(tmp_path: Path):
    path = tmp_path / "sweep.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    if not writer.isOpened():
        pytest.skip("MJPG video writer is not available")
    for _ in range(10):
        frame = np.full((48, 64, 3), 20, np.uint8)
        frame[8:40, 8:56] = 255
        writer.write(frame)
    writer.release()
    csv_path = tmp_path / "angles.csv"
    csv_path.write_text("timestamp_s,pan,tilt\n0,0,0\n1,20,10\n", encoding="utf-8")
    frames = ingest_video(
        path,
        out_dir=tmp_path / "out",
        sample_every_s=0.5,
        blur_threshold=1,
        sweep=None,
        angles_csv=csv_path,
    )
    by_time = {round(float(frame.timestamp or "0"), 1): frame for frame in frames}
    assert by_time[0.0].pan_deg == pytest.approx(0.0)
    assert by_time[0.0].tilt_deg == pytest.approx(0.0)
    assert by_time[0.5].pan_deg == pytest.approx(10.0)
    assert by_time[0.5].tilt_deg == pytest.approx(5.0)


def test_video_sweep_assigns_pan_and_drops_a_flat_frame(tmp_path: Path):
    path = tmp_path / "sweep.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    if not writer.isOpened():
        pytest.skip("MJPG video writer is not available")
    for index in range(20):
        frame = np.full((48, 64, 3), 180, np.uint8)
        if index != 0:
            frame[8:40, 8:56] = 0
        writer.write(frame)
    writer.release()

    frames = ingest_video(
        path,
        out_dir=tmp_path / "out",
        sample_every_s=0.5,
        blur_threshold=100,
        sweep=(0.0, 20.0, -5.0),
        angles_csv=None,
    )
    assert frames
    assert all(frame.tilt_deg == pytest.approx(-5.0) for frame in frames)
    pans = [frame.pan_deg for frame in frames]
    assert pans == sorted(pans)
    assert min(pans) > 0
    assert all(frame.path.is_file() for frame in frames)
