
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest


# Load RoadAnalyzer with a lightweight CARLA stub.
# Do not execute src/__init__.py.
fake_carla = SimpleNamespace(
    LaneType=SimpleNamespace(Driving=1)
)

module_path = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "road_analyzer.py"
)

spec = importlib.util.spec_from_file_location(
    "road_analyzer_under_test",
    module_path
)

with patch.dict(sys.modules, {"carla": fake_carla}):
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

RoadAnalyzer = module.RoadAnalyzer


def make_waypoint(x=1.0, y=0.0, road_id=1):
    """Create a simple CARLA waypoint substitute."""
    forward = SimpleNamespace(x=x, y=y, z=0.0)

    transform = SimpleNamespace(
        get_forward_vector=lambda: forward
    )

    return SimpleNamespace(
        transform=transform,
        road_id=road_id,
        section_id=0,
        lane_id=1,
        lane_type=1,
        lane_change=0
    )


# TEST 1 - Two-way road
def test_two_way_road():
    result = RoadAnalyzer.classify_road({
        "same_direction": 1,
        "opposite_direction": 1
    })
    assert result == "TWO_WAY"


# TEST 2 - One-way road with multiple lanes
def test_one_way_road():
    result = RoadAnalyzer.classify_road({
        "same_direction": 3,
        "opposite_direction": 0
    })
    assert result == "ONE_WAY"


# TEST 3 - Unknown road structure
def test_unknown_road():
    result = RoadAnalyzer.classify_road({
        "same_direction": 1,
        "opposite_direction": 0
    })
    assert result == "UNKNOWN"


# TEST 4 - Same travel direction
def test_same_direction():
    analyzer = RoadAnalyzer(carla_map=None)

    result = analyzer._direction_type(
        make_waypoint(1.0, 0.0),
        make_waypoint(1.0, 0.0)
    )

    assert result == "SAME_DIRECTION"


# TEST 5 - Opposite travel direction
def test_opposite_direction():
    analyzer = RoadAnalyzer(carla_map=None)

    result = analyzer._direction_type(
        make_waypoint(1.0, 0.0),
        make_waypoint(-1.0, 0.0)
    )

    assert result == "OPPOSITE_DIRECTION"


# TEST 6 - Perpendicular direction
def test_unknown_direction():
    analyzer = RoadAnalyzer(carla_map=None)

    result = analyzer._direction_type(
        make_waypoint(1.0, 0.0),
        make_waypoint(0.0, 1.0)
    )

    assert result == "UNKNOWN"


# TEST 7 - Right lane change allowed
def test_right_lane_change_allowed():
    waypoint = make_waypoint()
    waypoint.lane_change = 1

    assert RoadAnalyzer._can_change(
        waypoint, "right"
    ) is True


# TEST 8 - Left lane change allowed
def test_left_lane_change_allowed():
    waypoint = make_waypoint()
    waypoint.lane_change = 2

    assert RoadAnalyzer._can_change(
        waypoint, "left"
    ) is True


# TEST 9 - Lane change prohibited
def test_lane_change_prohibited():
    waypoint = make_waypoint()
    waypoint.lane_change = 0

    assert RoadAnalyzer._can_change(
        waypoint, "left"
    ) is False

    assert RoadAnalyzer._can_change(
        waypoint, "right"
    ) is False
