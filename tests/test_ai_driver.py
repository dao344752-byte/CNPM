
import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest


# ==========================================
# 1. MOCK DEPENDENCIES
# ==========================================

fake_carla = ModuleType("carla")
fake_carla.Vehicle = object
fake_carla.Map = object
fake_carla.LaneType = SimpleNamespace(Driving=1)
fake_carla.TrafficLightState = SimpleNamespace(
    Red=0, Yellow=1, Green=2
)
fake_carla.VehicleControl = SimpleNamespace

fake_numpy = ModuleType("numpy")
fake_numpy.clip = lambda value, low, high: max(
    low, min(high, value)
)

fake_src = ModuleType("src")
fake_src.__path__ = []


class DummyRoadAnalyzer:
    def __init__(self, *args, **kwargs):
        pass


class DummyOvertakingPlanner:
    def __init__(self, *args, **kwargs):
        pass


class DummyLaneChangeController:
    def __init__(self, *args, **kwargs):
        self.active = False
        self.state = "NORMAL"
        self.target_lane_id = None
        self.target_path = []
        self.progress_index = 0


def fake_module(name, class_name, cls):
    mod = ModuleType(name)
    setattr(mod, class_name, cls)
    return mod


dependencies = {
    "carla": fake_carla,
    "numpy": fake_numpy,
    "src": fake_src,
    "src.road_analyzer": fake_module(
        "src.road_analyzer",
        "RoadAnalyzer",
        DummyRoadAnalyzer
    ),
    "src.overtaking_planner": fake_module(
        "src.overtaking_planner",
        "OvertakingPlanner",
        DummyOvertakingPlanner
    ),
    "src.lane_change_controller": fake_module(
        "src.lane_change_controller",
        "LaneChangeController",
        DummyLaneChangeController
    ),
}

module_path = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "ai_driver.py"
)

spec = importlib.util.spec_from_file_location(
    "ai_driver_under_test",
    module_path
)

with patch.dict(sys.modules, dependencies):
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

AIDriver = module.AIDriver


def make_driver(speed=0.0, target_speed=35.0):
    driver = AIDriver(
        vehicle=SimpleNamespace(),
        carla_map=SimpleNamespace(),
        target_speed=target_speed
    )
    driver.speed = speed
    return driver


def driving_decision(
    driver,
    light="GREEN",
    dangers=None,
    warnings=None,
    vehicles=None
):
    return driver._make_driving_decision(
        light_state=light,
        dangers=dangers or [],
        warnings=warnings or [],
        overtaking_decision={},
        same_lane_vehicles=vehicles or [],
        adjacent_traffic={}
    )


# ==========================================
# GROUP A: INPUT VALIDATION
# ==========================================

# TEST 1
def test_valid_number():
    assert AIDriver._valid_number(10.0)
    assert not AIDriver._valid_number("abc")
    assert not AIDriver._valid_number(float("nan"))
    assert not AIDriver._valid_number(float("inf"))


# TEST 2
def test_safe_distance():
    assert AIDriver._safe_distance(
        {"distance": "12.5"}
    ) == 12.5

    assert math.isinf(
        AIDriver._safe_distance({"distance": -2})
    )

    assert math.isinf(
        AIDriver._safe_distance(None)
    )


# TEST 3
def test_detection_class_normalization():
    result = AIDriver._get_detection_class({
        "class": " CAR "
    })

    assert result == "car"


# TEST 4
def test_vehicle_detection():
    assert AIDriver._is_vehicle_detection({
        "class": "truck"
    })

    assert not AIDriver._is_vehicle_detection({
        "class": "stop sign"
    })


# TEST 5
def test_ego_lane_flags():
    assert AIDriver._is_in_ego_lane({
        "in_lane": True
    })

    assert AIDriver._is_in_ego_lane({
        "same_lane": "yes"
    })

    assert not AIDriver._is_in_ego_lane({
        "in_lane": False
    })


# ==========================================
# GROUP B: SAFETY CALCULATIONS
# ==========================================

# TEST 6
def test_ttc_when_closing():
    ttc = AIDriver._get_ttc(
        distance=20.0,
        closing_speed=5.0
    )

    assert ttc == pytest.approx(4.0)


# TEST 7
def test_ttc_when_not_closing():
    assert math.isinf(
        AIDriver._get_ttc(20.0, 0.0)
    )

    assert AIDriver._get_ttc(None, 5.0) is None


# TEST 8
def test_emergency_braking_distance():
    driver = make_driver(speed=36.0)

    result = driver._get_emergency_braking_distance()

    # 36 km/h = 10 m/s
    expected = (
        10.0 * 0.8
        + 10.0 ** 2 / (2.0 * 6.0)
        + 4.0
    )

    assert result == pytest.approx(expected)


# TEST 9
def test_following_distance():
    driver = make_driver(speed=36.0)

    # 10 m/s * 2 seconds = 20 metres
    assert driver._get_following_distance() == pytest.approx(
        20.0
    )


# ==========================================
# GROUP C: DRIVING DECISIONS
# ==========================================

# TEST 10
def test_red_light_emergency_stop():
    driver = make_driver(speed=30.0)

    throttle, brake, action, reason = driving_decision(
        driver, light="RED"
    )

    assert throttle == 0.0
    assert brake == 1.0
    assert action == "STOP — RED LIGHT"


# TEST 11
def test_stop_sign_braking():
    driver = make_driver(speed=15.0)
    driver.stop_sign_state = "STOPPING"

    throttle, brake, action, reason = driving_decision(
        driver
    )

    assert throttle == 0.0
    assert brake == pytest.approx(0.8)
    assert action == "STOP — SIGN"


# TEST 12
def test_yellow_light_slowdown():
    driver = make_driver(speed=30.0)

    throttle, brake, action, reason = driving_decision(
        driver, light="YELLOW"
    )

    assert throttle == 0.0
    assert brake == pytest.approx(0.4)
    assert action == "SLOW — YELLOW"


# TEST 13
def test_emergency_brake_for_close_obstacle():
    driver = make_driver(speed=36.0)

    dangers = [{
        "class": "car",
        "distance": 3.0,
        "in_lane": True,
        "threat": "DANGER"
    }]

    throttle, brake, action, reason = driving_decision(
        driver,
        dangers=dangers
    )

    assert throttle == 0.0
    assert brake == 1.0
    assert action == "EMERGENCY BRAKE"


# TEST 14
def test_accelerate_on_clear_road():
    driver = make_driver(
        speed=10.0,
        target_speed=35.0
    )

    throttle, brake, action, reason = driving_decision(
        driver
    )

    assert throttle > 0.0
    assert brake == 0.0
    assert action == "ACCELERATE"


# TEST 15
def test_cruise_at_target_speed():
    driver = make_driver(
        speed=35.0,
        target_speed=35.0
    )

    throttle, brake, action, reason = driving_decision(
        driver
    )

    assert throttle == pytest.approx(0.25)
    assert brake == 0.0
    assert action == "CRUISE"


# ==========================================
# GROUP D: SAFETY OVERRIDE
# ==========================================

# TEST 16
def test_safety_override_critical_front_gap():
    driver = make_driver(speed=36.0)

    traffic = {
        "observed": True,
        "front_distance": 2.0,
        "front_ttc": 0.5
    }

    throttle, brake, action, reason, overridden = (
        driver._apply_safety_override(
            0.3,
            0.0,
            "CRUISE",
            "Road clear",
            traffic
        )
    )

    assert throttle == 0.0
    assert brake == 1.0
    assert overridden is True
    assert action == "SAFETY — EMERGENCY BRAKE"


# TEST 17
def test_safety_override_warning_ttc():
    driver = make_driver(speed=0.0)

    traffic = {
        "observed": True,
        "front_distance": None,
        "front_ttc": 2.0
    }

    throttle, brake, action, reason, overridden = (
        driver._apply_safety_override(
            0.3,
            0.0,
            "CRUISE",
            "Road clear",
            traffic
        )
    )

    assert throttle == 0.0
    assert brake >= 0.4
    assert overridden is False
    assert action == "SAFETY — FRONT TTC"


# ==========================================
# GROUP E: URGENCY CLASSIFICATION
# ==========================================

# TEST 18
def test_action_urgency():
    assert AIDriver._get_urgency(
        "EMERGENCY BRAKE"
    ) == "CRITICAL"

    assert AIDriver._get_urgency(
        "STOP — SIGN"
    ) == "HIGH"

    assert AIDriver._get_urgency(
        "CAUTION"
    ) == "MEDIUM"

    assert AIDriver._get_urgency(
        "CRUISE"
    ) == "LOW"
