import numpy as np

from vision.detector import (
    DetectorHit,
    DetectorUnavailable,
    FallbackDetector,
    NullDetector,
    OwlV2Detector,
    box_iou,
    build_detector,
    refine_box,
)
from vision.config import load_settings


def test_refine_uses_the_detector_only_when_iou_clears_the_threshold():
    grok = [0, 0, 100, 100]
    close = DetectorHit(bbox_px=[10, 10, 90, 90], score=0.8)
    far = DetectorHit(bbox_px=[80, 80, 120, 120], score=0.99)
    box, source, confidence = refine_box(grok, [far, close], refine_iou=0.3)
    assert source == "detector"
    assert box == [10, 10, 90, 90]
    assert confidence == 0.8
    assert box_iou(grok, close.bbox_px) >= 0.3

    box, source, confidence = refine_box(grok, [far], refine_iou=0.3)
    assert source == "grok"
    assert box == grok
    assert confidence < 0.6


def test_constructing_owlv2_does_not_load_weights():
    detector = OwlV2Detector("google/owlv2-base-patch16-ensemble")
    assert detector._model is None
    assert detector._processor is None


def test_build_detector_follows_config_without_downloading():
    settings = load_settings(load_env=False)
    detector = build_detector(settings)
    assert isinstance(detector, OwlV2Detector)
    swapped = settings.model_copy(update={"detector": "grounding_dino"})
    assert type(build_detector(swapped)).__name__ == "GroundingDinoDetector"


def test_missing_weights_fall_back_to_no_hits():
    class Broken:
        def detect(self, image, text):
            raise DetectorUnavailable("weights were not downloaded")

    fallback = FallbackDetector(Broken())
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    assert fallback.detect(image, "mug") == []
    assert fallback.disabled is True
    assert fallback.detect(image, "mug") == []
    assert NullDetector().detect(image, "mug") == []
