
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import pytest


# ==========================================
# 1. MOCK NUMPY / OPENCV
# ==========================================
# These tests cover scalar geometry logic.
# Full image-processing tests need real numpy/cv2.

fake_cv2 = ModuleType("cv2")
fake_numpy = ModuleType("numpy")

fake_numpy.float64 = float


def fake_asarray(value, dtype=None):
    return float(value)


def fake_polyval(coefficients, x):
    result = 0.0
    for coefficient in coefficients:
        result = result * x + coefficient
    return result


def fake_clip(value, low, high):
    return max(low, min(high, value))


fake_numpy.asarray = fake_asarray
fake_numpy.polyval = fake_polyval
fake_numpy.clip = fake_clip


# Load the actual LaneDetector source without src/__init__.py
module_path = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "lane_detector.py"
)

spec = importlib.util.spec_from_file_location(
    "lane_detector_under_test",
    module_path
)

with patch.dict(sys.modules, {
    "cv2": fake_cv2,
    "numpy": fake_numpy,
}):
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

LaneDetector = module.LaneDetector


# ==========================================
# 2. TEST DATA
# ==========================================

FRAME_SHAPE = (600, 1000, 3)

# Synthetic straight lane: x from 300 to 700.
LANE_GEOMETRY = {
    "valid": True,
    "left_coefficients": [0.0, 0.0, 300.0],
    "right_coefficients": [0.0, 0.0, 700.0],
}


# ==========================================
# GROUP A: INITIALIZATION AND VALIDATION
# ==========================================

# TEST 1
def test_initial_state():
    detector = LaneDetector(history_len=5)

    assert len(detector.left_history) == 0
    assert len(detector.right_history) == 0
    assert detector.left_history.maxlen == 5
    assert detector.last_lane_width is None


# TEST 2
def test_fit_curve_none():
    detector = LaneDetector()

    assert detector._fit_curve(
        None, frame_height=600
    ) is None


# TEST 3
def test_fit_curve_insufficient_segments():
    detector = LaneDetector()

    segments = [(100, 500, 200, 300)]

    assert detector._fit_curve(
        segments, frame_height=600
    ) is None


# TEST 4
def test_smooth_history_empty():
    assert LaneDetector._smooth_history([]) is None


# TESTS 5-6
@pytest.mark.parametrize(
    "left,right",
    [
        (None, [0.0, 0.0, 700.0]),
        ([0.0, 0.0, 300.0], None),
    ]
)
def test_lane_geometry_missing_side(left, right):
    detector = LaneDetector()

    assert detector._lane_geometry(
        left, right, 600, 1000
    ) is None


# ==========================================
# GROUP B: COORDINATE NORMALIZATION
# ==========================================

# TESTS 7-9
@pytest.mark.parametrize(
    "y,expected",
    [
        (0, -1.0),
        (300, 0.0),
        (600, 1.0),
    ]
)
def test_normalize_y(y, expected):
    result = LaneDetector._normalize_y(y, 600)

    assert result == pytest.approx(expected)


# ==========================================
# GROUP C: CURVE EVALUATION
# ==========================================

# TEST 10
def test_evaluate_constant_curve():
    coeff = [0.0, 0.0, 300.0]

    result = LaneDetector._evaluate_curve(
        coeff, y=450, frame_height=600
    )

    assert result == pytest.approx(300.0)


# TEST 11
def test_evaluate_linear_curve():
    coeff = [0.0, 100.0, 500.0]

    result = LaneDetector._evaluate_curve(
        coeff, y=600, frame_height=600
    )

    assert result == pytest.approx(600.0)


# ==========================================
# GROUP D: BOUNDING BOX IN LANE
# ==========================================

# TEST 12
def test_bbox_invalid_geometry():
    result = LaneDetector.is_bbox_in_lane(
        bbox=(400, 200, 600, 500),
        lane_geometry={"valid": False},
        frame_shape=FRAME_SHAPE
    )

    assert result is False


# TEST 13
def test_bbox_missing_coefficients():
    result = LaneDetector.is_bbox_in_lane(
        bbox=(400, 200, 600, 500),
        lane_geometry={
            "valid": True,
            "left_coefficients": None,
            "right_coefficients": None
        },
        frame_shape=FRAME_SHAPE
    )

    assert result is False


# TEST 14
def test_bbox_inside_lane():
    result = LaneDetector.is_bbox_in_lane(
        bbox=(400, 200, 600, 500),
        lane_geometry=LANE_GEOMETRY,
        frame_shape=FRAME_SHAPE
    )

    assert result is True


# TESTS 15-16
@pytest.mark.parametrize(
    "bbox",
    [
        (50, 200, 200, 500),
        (800, 200, 950, 500),
    ]
)
def test_bbox_outside_lane(bbox):
    result = LaneDetector.is_bbox_in_lane(
        bbox=bbox,
        lane_geometry=LANE_GEOMETRY,
        frame_shape=FRAME_SHAPE
    )

    assert result is False


# TEST 17
def test_bbox_partial_overlap():
    result = LaneDetector.is_bbox_in_lane(
        bbox=(250, 200, 450, 500),
        lane_geometry=LANE_GEOMETRY,
        frame_shape=FRAME_SHAPE
    )

    assert result is True


# TEST 18
def test_bbox_insufficient_overlap():
    result = LaneDetector.is_bbox_in_lane(
        bbox=(100, 200, 900, 500),
        lane_geometry=LANE_GEOMETRY,
        frame_shape=FRAME_SHAPE,
        min_overlap=0.60
    )

    assert result is False


# TEST 19
def test_bbox_reversed_lane_boundaries():
    invalid_geometry = {
        "valid": True,
        "left_coefficients": [0.0, 0.0, 700.0],
        "right_coefficients": [0.0, 0.0, 300.0],
    }

    result = LaneDetector.is_bbox_in_lane(
        bbox=(400, 200, 600, 500),
        lane_geometry=invalid_geometry,
        frame_shape=FRAME_SHAPE
    )

    assert result is False
