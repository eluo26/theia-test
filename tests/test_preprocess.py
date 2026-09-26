import numpy as np

from vision.config import load_settings
from vision.preprocess import map_norm_box_to_frame, tile_windows, undistort


def test_tile_box_maps_back_to_full_frame_pixels():
    # Tile is 50x40 and starts at (50, 40) inside a 100x80 frame.
    # A full-tile 0-1000 box covers that tile, clamped to the last pixel.
    mapped = map_norm_box_to_frame(
        [0, 0, 1000, 1000],
        tile_width=50,
        tile_height=40,
        offset_x=50,
        offset_y=40,
        frame_width=100,
        frame_height=80,
    )
    assert mapped == [50, 40, 99, 79]

    quarter = map_norm_box_to_frame(
        [0, 0, 500, 1000],
        tile_width=50,
        tile_height=40,
        offset_x=50,
        offset_y=40,
        frame_width=100,
        frame_height=80,
    )
    assert quarter == [50, 40, 75, 79]


def test_overlap_zero_grid_covers_the_frame_without_gaps():
    windows = tile_windows(100, 80, columns=2, rows=2, overlap=0)
    assert windows == [
        (0, 0, 50, 40),
        (50, 0, 50, 40),
        (0, 40, 50, 40),
        (50, 40, 50, 40),
    ]


def test_overlapping_tiles_stay_inside_and_meet_the_far_edge():
    windows = tile_windows(200, 100, columns=2, rows=2, overlap=0.15)
    assert windows[0][0] == 0 and windows[0][1] == 0
    far = windows[-1]
    assert far[0] + far[2] == 200
    assert far[1] + far[3] == 100
    assert far[0] < windows[0][2]


def test_zero_distortion_keeps_the_image_shape():
    settings = load_settings(load_env=False).model_copy(
        update={"fx": 100.0, "fy": 100.0, "cx": 16.0, "cy": 12.0, "dist_coeffs": [0, 0, 0, 0, 0]}
    )
    image = np.zeros((24, 32, 3), dtype=np.uint8)
    image[8:16, 10:22] = (0, 80, 200)
    result = undistort(image, settings)
    assert result.shape == image.shape
    assert int(result[12, 16, 2]) > 100
