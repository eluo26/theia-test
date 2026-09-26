import math

import pytest

from vision.config import load_settings
from vision.geometry import (
    angular_distance_deg,
    direction_to_pixel,
    pixel_to_azimuth_elevation,
)


def _fov_settings():
    settings = load_settings(load_env=False)
    return settings.model_copy(
        update={"fx": None, "fy": None, "cx": None, "cy": None, "dist_coeffs": None}
    )


def test_image_center_returns_pan_and_tilt():
    settings = _fov_settings()
    width, height = 640, 480
    for pan, tilt in ((0.0, 0.0), (20.0, -12.0), (-15.0, 8.0), (30.0, -10.0)):
        azimuth, elevation = pixel_to_azimuth_elevation(
            width / 2, height / 2, width, height, pan, tilt, settings
        )
        assert azimuth == pytest.approx(pan, abs=1e-8)
        assert elevation == pytest.approx(tilt, abs=1e-8)


def test_right_edge_at_zero_tilt_is_pan_plus_half_hfov():
    settings = _fov_settings().model_copy(update={"hfov_deg": 60.0, "vfov_deg": 40.0})
    width, height = 640, 480
    pan = 12.0
    # u = W is the mathematical right edge: cx = W/2 and fx = (W/2)/tan(hfov/2).
    azimuth, elevation = pixel_to_azimuth_elevation(
        float(width), height / 2, width, height, pan, 0.0, settings
    )
    assert azimuth == pytest.approx(pan + settings.hfov_deg / 2, abs=1e-6)
    assert elevation == pytest.approx(0.0, abs=1e-6)


def test_pixel_round_trip():
    settings = _fov_settings()
    width, height = 640, 480
    azimuth, elevation = pixel_to_azimuth_elevation(500, 200, width, height, 15, -5, settings)
    u, v, z = direction_to_pixel(azimuth, elevation, 15, -5, width, height, settings)
    assert z > 0
    assert u == pytest.approx(500, abs=1e-4)
    assert v == pytest.approx(200, abs=1e-4)
    assert angular_distance_deg(azimuth, elevation, 15, -5) > 0
    assert angular_distance_deg(10, 0, 10, 0) == pytest.approx(0.0, abs=1e-8)


def test_intrinsics_override_changes_the_ray():
    base = _fov_settings()
    overridden = base.model_copy(update={"fx": 5000.0, "fy": 5000.0, "cx": 320.0, "cy": 240.0})
    width, height = 640, 480
    fov_az, _ = pixel_to_azimuth_elevation(width, height / 2, width, height, 0, 0, base)
    cal_az, _ = pixel_to_azimuth_elevation(width, height / 2, width, height, 0, 0, overridden)
    assert fov_az == pytest.approx(base.hfov_deg / 2, abs=1e-6)
    assert cal_az < fov_az
    assert math.isfinite(cal_az)
