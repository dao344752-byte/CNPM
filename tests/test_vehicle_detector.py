
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import pytest


# ==========================================
# 1. MOCK DEPENDENCIES
# ==========================================
# Avoid importing heavy packages in CI.

fake_cv2 = ModuleType("cv2")

fake_numpy = ModuleType("numpy")
fake_numpy.ndarray = object

fake_ultralytics = ModuleType("ultralytics")
fake_ultralytics.YOLO = object

module_path = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "vehicle_detector.py"
)

spec = importlib.util.spec_from_file_location(
    "vehicle_detector_under_test",
    module_path
)

with patch.dict(sys.modules, {
    "cv2": fake_cv2,
    "numpy": fake_numpy,
    "ultralytics": fake_ultralytics,
}):
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

VehicleDetector = module.VehicleDetector


# ==========================================
# 2. MOCK YOLO RESULTS
# ==========================================

def make_box(class_id, confidence, bbox):
    return SimpleNamespace(
        cls=[class_id],
        conf=[confidence],
        xyxy=[bbox]
    )


def make_detector(boxes):
    model = Mock(
        return_value=[SimpleNamespace(boxes=boxes)]
    )
    return VehicleDetector(model), model


def make_frame():
    # Fake frame with H=600, W=1000
    return SimpleNamespace(shape=(600, 1000, 3))


# ==========================================
# GROUP A: THREAT CLASSIFICATION
# ==========================================

# TEST 1: Close vehicle in lane
def test_danger_when_vehicle_close():
    result = VehicleDetector._threat(
        "car", 5.0, True
    )
    assert result == "DANGER"


# TEST 2: Vehicle at warning distance
def test_warning_when_vehicle_near():
    result = VehicleDetector._threat(
        "car", 12.0, True
    )
    assert result == "WARNING"


# TEST 3: Distant vehicle
def test_safe_when_vehicle_far():
    result = VehicleDetector._threat(
        "car", 25.0, True
    )
    assert result == "SAFE"


# TEST 4: Vehicle outside current lane
def test_safe_when_outside_lane():
    result = VehicleDetector._threat(
        "car", 4.0, False
    )
    assert result == "SAFE"


# TEST 5: Traffic signs are not threats
def test_traffic_sign_is_safe():
    result = VehicleDetector._threat(
        "stop sign", 3.0, True
    )
    assert result == "SAFE"


# ==========================================
# GROUP B: IN-LANE DETECTION
# ==========================================

# TEST 6: Close vehicle at image center
def test_center_vehicle_in_lane():
    result = VehicleDetector._in_lane(
        cx=500,
        cy=450,
        w=1000,
        h=600,
        dist=5.0,
        name="car"
    )
    assert result is True


# TEST 7: Distant centered vehicle
def test_distant_center_vehicle_in_lane():
    result = VehicleDetector._in_lane(
        cx=500,
        cy=400,
        w=1000,
        h=600,
        dist=25.0,
        name="car"
    )
    assert result is True


# TEST 8: Vehicle far from image center
def test_vehicle_outside_lane():
    result = VehicleDetector._in_lane(
        cx=800,
        cy=450,
        w=1000,
        h=600,
        dist=5.0,
        name="car"
    )
    assert result is False


# TEST 9: Detection above horizon
def test_vehicle_above_horizon():
    result = VehicleDetector._in_lane(
        cx=500,
        cy=100,
        w=1000,
        h=600,
        dist=10.0,
        name="car"
    )
    assert result is False


# TEST 10: Traffic light is excluded
def test_traffic_light_not_in_lane():
    result = VehicleDetector._in_lane(
        cx=500,
        cy=450,
        w=1000,
        h=600,
        dist=4.0,
        name="traffic light"
    )
    assert result is False


# ==========================================
# GROUP C: YOLO RESULT PROCESSING
# ==========================================

# TEST 11: Detections sorted by distance
def test_detect_sorted_by_distance():
    boxes = [
        make_box(2, 0.90, (450, 350, 550, 480)),
        make_box(7, 0.95, (400, 350, 600, 500)),
    ]

    detector, model = make_detector(boxes)
    results = detector.detect(make_frame())

    assert len(results) == 2
    assert results[0]["class"] == "truck"
    assert results[1]["class"] == "car"

    assert (
        results[0]["distance"]
        < results[1]["distance"]
    )

    model.assert_called_once()


# TEST 12: Invalid detections are discarded
def test_detect_filters_invalid_boxes():
    boxes = [
        make_box(99, 0.90, (450, 350, 550, 480)),
        make_box(2, 0.90, (490, 350, 496, 360)),
        make_box(2, 0.90, (100, 350, 900, 500)),
    ]

    detector, _ = make_detector(boxes)
    results = detector.detect(make_frame())

    assert results == []


# TEST 13: Detection metadata is correct
def test_detect_returns_correct_metadata():
    boxes = [
        make_box(2, 0.90, (450, 350, 550, 480))
    ]

    detector, model = make_detector(boxes)
    results = detector.detect(make_frame())

    assert len(results) == 1

    detection = results[0]

    assert detection["class"] == "car"
    assert detection["bbox"] == (450, 350, 550, 480)
    assert detection["confidence"] == pytest.approx(0.90)
    assert detection["center"] == (500, 415)
    assert detection["distance"] == pytest.approx(9.0)
    assert detection["in_lane"] is True
    assert detection["threat"] == "WARNING"
    assert detection["side"] == "right"
    assert detection["relative_x"] == pytest.approx(0.0)
    assert detection["relative_y"] == pytest.approx(
        1.0 - 415 / 600
    )

    model.assert_called_once_with(
        pytest.ANY if hasattr(pytest, "ANY") else model.call_args.args[0],
        conf=0.40,
        verbose=False,
        device="cpu",
    )
