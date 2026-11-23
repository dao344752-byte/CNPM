
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest


# ==========================================
# 1. MOCK CARLA
# ==========================================

fake_carla = ModuleType("carla")
fake_carla.LaneType = SimpleNamespace(Driving=1)
fake_carla.Location = SimpleNamespace

module_path = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "lane_change_controller.py"
)

spec = importlib.util.spec_from_file_location(
    "lane_change_controller_under_test",
    module_path
)

with patch.dict(sys.modules, {"carla": fake_carla}):
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

LaneChangeController = module.LaneChangeController


# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def location(x=0.0, y=0.0, z=0.0):
    return SimpleNamespace(x=x, y=y, z=z)


def forward(x=1.0, y=0.0):
    return SimpleNamespace(x=x, y=y, z=0.0)


def waypoint(
    x=0.0,
    y=0.0,
    lane_id=1,
    road_id=1,
    direction=(1.0, 0.0),
    lane_type=1,
):
    loc = location(x, y)
    vec = forward(*direction)

    transform = SimpleNamespace(
        location=loc,
        get_forward_vector=lambda: vec
    )

    return SimpleNamespace(
        transform=transform,
        lane_id=lane_id,
        road_id=road_id,
        section_id=0,
        lane_type=lane_type,
        lane_width=3.5,
        next=lambda distance: []
    )


def make_controller(**kwargs):
    loc = location()
    vec = forward()

    transform = SimpleNamespace(
        location=loc,
        get_forward_vector=lambda: vec
    )

    vehicle = SimpleNamespace(
        get_location=lambda: loc,
        get_transform=lambda: transform
    )

    return LaneChangeController(
        vehicle=vehicle,
        carla_map=None,
        **kwargs
    )


# ==========================================
# GROUP A: INITIALIZATION AND GEOMETRY
# ==========================================

# TEST 1
def test_initial_state():
    controller = make_controller()

    assert controller.state == "NORMAL"
    assert controller.active is False
    assert controller.reference_path == []
    assert controller.step_count == 0


# TEST 2
def test_clamp_value():
    assert LaneChangeController._clamp(
        2.0, -1.0, 1.0
    ) == 1.0

    assert LaneChangeController._clamp(
        -2.0, -1.0, 1.0
    ) == -1.0

    assert LaneChangeController._clamp(
        0.5, -1.0, 1.0
    ) == 0.5


# TEST 3
def test_distance_2d():
    result = LaneChangeController._distance_2d(
        location(0, 0),
        location(3, 4)
    )

    assert result == pytest.approx(5.0)


# TEST 4
def test_direction_dot():
    assert LaneChangeController._direction_dot(
        forward(1, 0),
        forward(-1, 0)
    ) == pytest.approx(-1.0)


# ==========================================
# GROUP B: START LANE CHANGE
# ==========================================

# TEST 5
def test_start_rejects_none():
    controller = make_controller()

    assert controller.start(None) is False
    assert controller.active is False


# TEST 6
def test_start_rejects_when_active():
    controller = make_controller()
    controller.active = True

    assert controller.start(waypoint()) is False


# TEST 7
def test_start_rejects_opposite_direction(
    monkeypatch
):
    controller = make_controller()

    current = waypoint(direction=(1, 0))
    target = waypoint(
        lane_id=2,
        direction=(-1, 0)
    )

    monkeypatch.setattr(
        controller,
        "_get_current_waypoint",
        lambda: current
    )

    assert controller.start(target) is False
    assert controller.active is False


# TEST 8
def test_start_rejects_different_road(
    monkeypatch
):
    controller = make_controller()

    current = waypoint(road_id=1)
    target = waypoint(
        lane_id=2,
        road_id=2
    )

    monkeypatch.setattr(
        controller,
        "_get_current_waypoint",
        lambda: current
    )

    assert controller.start(target) is False


# ==========================================
# GROUP C: WAYPOINT AND PATH
# ==========================================

# TEST 9
def test_select_preferred_lane():
    controller = make_controller()

    start_wp = waypoint(lane_id=1)
    candidate_a = waypoint(
        x=2, lane_id=2
    )
    candidate_b = waypoint(
        x=2, lane_id=3
    )

    start_wp.next = lambda distance: [
        candidate_a,
        candidate_b
    ]

    selected = controller._select_lane_continuation(
        start_wp,
        distance=2.0,
        preferred_lane_id=3
    )

    assert selected is candidate_b


# TEST 10
def test_select_lane_no_candidates():
    controller = make_controller()

    start_wp = waypoint()
    start_wp.next = lambda distance: []

    selected = controller._select_lane_continuation(
        start_wp,
        distance=2.0
    )

    assert selected is None


# TEST 11
def test_resample_path_midpoint():
    controller = make_controller()

    path = [
        waypoint(x=0.0, y=0.0),
        waypoint(x=10.0, y=0.0)
    ]

    result = controller._resample_path(
        path,
        distances=[0.0, 5.0, 10.0]
    )

    assert len(result) == 3
    assert result[0].x == pytest.approx(0.0)
    assert result[1].x == pytest.approx(5.0)
    assert result[2].x == pytest.approx(10.0)


# ==========================================
# GROUP D: STEERING CONTROL
# ==========================================

# TEST 12
def test_steering_straight(monkeypatch):
    controller = make_controller()

    monkeypatch.setattr(
        controller,
        "_get_lookahead_target",
        lambda: location(10.0, 0.0)
    )

    steer = controller._calculate_steering()

    assert steer == pytest.approx(0.0)


# TEST 13
def test_steering_left(monkeypatch):
    controller = make_controller()

    monkeypatch.setattr(
        controller,
        "_get_lookahead_target",
        lambda: location(10.0, 5.0)
    )

    steer = controller._calculate_steering()

    assert steer > 0.0
    assert steer <= controller.max_steer


# ==========================================
# GROUP E: STATE MANAGEMENT
# ==========================================

# TEST 14
def test_cancel_resets_state():
    controller = make_controller()

    controller.active = True
    controller.state = "LANE_CHANGE"
    controller.step_count = 10
    controller.target_lane_id = 2
    controller.reference_path = [
        location(0, 0),
        location(1, 1)
    ]

    controller.cancel()

    assert controller.active is False
    assert controller.state == "NORMAL"
    assert controller.step_count == 0
    assert controller.target_lane_id is None
    assert controller.reference_path == []


# TEST 15
def test_run_step_when_inactive():
    controller = make_controller()

    assert controller.run_step() is None


# TEST 16
def test_run_step_timeout(monkeypatch):
    controller = make_controller(max_steps=1)

    controller.active = True
    controller.state = "LANE_CHANGE"

    monkeypatch.setattr(
        controller,
        "_find_nearest_reference_index",
        lambda: 0
    )

    monkeypatch.setattr(
        controller,
        "_calculate_steering",
        lambda: 0.2
    )

    monkeypatch.setattr(
        controller,
        "_is_completed",
        lambda: False
    )

    monkeypatch.setattr(
        controller,
        "_is_on_target_lane",
        lambda: False
    )

    result = controller.run_step()

    assert result["failed"] is True
    assert result["completed"] is False
    assert result["state"] == "FAILED"
    assert controller.active is False
