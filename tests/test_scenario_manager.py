
import importlib.util
import random
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import pytest


# ==========================================
# 1. MOCK CARLA AND LOAD SOURCE
# ==========================================

fake_carla = ModuleType("carla")
fake_carla.Client = object
fake_carla.World = object
fake_carla.Vehicle = object
fake_carla.LaneType = SimpleNamespace(Driving=1)

source = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "scenario_manager.py"
)

spec = importlib.util.spec_from_file_location(
    "scenario_manager_under_test",
    source
)

with patch.dict(sys.modules, {"carla": fake_carla}):
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

ScenarioManager = module.ScenarioManager


# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def loc(x=0, y=0, z=0):
    return SimpleNamespace(x=x, y=y, z=z)


def vec(x=1, y=0, z=0):
    return SimpleNamespace(x=x, y=y, z=z)


def waypoint(
    lane_id=1,
    road_id=1,
    direction=(1, 0, 0),
    x=0,
    y=0,
    z=0,
):
    direction_vector = vec(*direction)

    return SimpleNamespace(
        lane_id=lane_id,
        road_id=road_id,
        lane_type=1,
        transform=SimpleNamespace(
            location=loc(x, y, z),
            get_forward_vector=lambda: direction_vector
        ),
        next=Mock(return_value=[]),
        previous=Mock(return_value=[]),
        get_left_lane=Mock(return_value=None),
        get_right_lane=Mock(return_value=None)
    )


def make_manager():
    # Avoid connecting to a CARLA server during unit tests.
    manager = ScenarioManager.__new__(ScenarioManager)

    manager.random = random.Random(42)
    manager.traffic_manager = Mock()
    manager.traffic_manager_port = 8000
    manager.spawned_vehicles = []
    manager.scenario_name = None
    manager.ego_vehicle = Mock()
    manager.ego_vehicle.id = 1
    manager.ego_vehicle.get_location.return_value = loc()
    manager.carla_map = Mock()
    manager.world = Mock()

    return manager


# ==========================================
# GROUP A: BASIC GEOMETRY AND CONFIGURATION
# ==========================================

# TEST 1
def test_distance_3d():
    result = ScenarioManager._distance(
        loc(0, 0, 0),
        loc(3, 4, 12)
    )
    assert result == pytest.approx(13.0)


# TEST 2
def test_direction_dot():
    a = waypoint(direction=(1, 0, 0))
    b = waypoint(direction=(-1, 0, 0))

    assert ScenarioManager._direction_dot(
        a, b
    ) == pytest.approx(-1.0)


# TEST 3
def test_same_direction():
    manager = make_manager()

    a = waypoint(direction=(1, 0, 0))
    b = waypoint(direction=(1, 0, 0))

    assert manager._is_same_direction(a, b) is True


# TEST 4
def test_opposite_direction_rejected():
    manager = make_manager()

    a = waypoint(direction=(1, 0, 0))
    b = waypoint(direction=(-1, 0, 0))

    assert manager._is_same_direction(a, b) is False


# TEST 5
def test_available_scenarios():
    manager = make_manager()

    scenarios = manager.get_available_scenarios()

    assert "slow_vehicle_ahead" in scenarios
    assert "safe_overtake" in scenarios
    assert "rear_vehicle_blocking" in scenarios
    assert "both_lanes_blocked" in scenarios
    assert "mixed" in scenarios


# TEST 6
def test_invalid_scenario_rejected():
    manager = make_manager()

    with pytest.raises(ValueError, match="Unknown scenario"):
        manager.setup_scenario("invalid_scenario")


# ==========================================
# GROUP B: VEHICLE BLUEPRINTS
# ==========================================

# TEST 7
def test_filter_vehicle_blueprints():
    manager = make_manager()

    blueprints = [
        SimpleNamespace(id="vehicle.tesla.model3"),
        SimpleNamespace(id="vehicle.audi.a2"),
        SimpleNamespace(id="vehicle.bh.crossbike"),
        SimpleNamespace(id="vehicle.harley.low_rider"),
    ]

    manager.blueprint_library = Mock()
    manager.blueprint_library.filter.return_value = blueprints

    result = manager._get_vehicle_blueprints()

    assert len(result) == 2
    assert result[0].id == "vehicle.tesla.model3"
    assert result[1].id == "vehicle.audi.a2"


# TEST 8
def test_no_suitable_blueprints():
    manager = make_manager()

    manager.blueprint_library = Mock()
    manager.blueprint_library.filter.return_value = [
        SimpleNamespace(id="vehicle.bh.crossbike")
    ]

    with pytest.raises(RuntimeError):
        manager._get_vehicle_blueprints()


# ==========================================
# GROUP C: WAYPOINTS AND ADJACENT LANES
# ==========================================

# TEST 9
def test_missing_ego_waypoint():
    manager = make_manager()
    manager.carla_map.get_waypoint.return_value = None

    with pytest.raises(RuntimeError):
        manager._get_ego_waypoint()


# TEST 10
def test_select_left_adjacent_lane():
    manager = make_manager()

    current = waypoint(lane_id=1)
    left = waypoint(lane_id=2)

    current.get_left_lane.return_value = left

    side, selected = manager._get_adjacent_lane(current)

    assert side == "left"
    assert selected is left


# TEST 11
def test_reject_opposite_adjacent_lane():
    manager = make_manager()

    current = waypoint(lane_id=1)
    left = waypoint(
        lane_id=2,
        direction=(-1, 0, 0)
    )

    current.get_left_lane.return_value = left

    side, selected = manager._get_adjacent_lane(current)

    assert side is None
    assert selected is None


# TEST 12
def test_advance_zero_distance():
    manager = make_manager()
    current = waypoint()

    result = manager._advance_waypoint(current, 0.0)

    assert result is current
    current.next.assert_not_called()
    current.previous.assert_not_called()


# TEST 13
def test_advance_prefers_same_lane():
    manager = make_manager()

    current = waypoint(lane_id=1)
    same_lane = waypoint(lane_id=1)
    other_lane = waypoint(lane_id=2)

    current.next.return_value = [
        other_lane, same_lane
    ]

    selected = manager._advance_waypoint(
        current, 10.0
    )

    assert selected is same_lane


# TEST 14
def test_advance_no_candidates():
    manager = make_manager()
    current = waypoint()

    current.next.return_value = []

    assert manager._advance_waypoint(
        current, 10.0
    ) is None


# TEST 15
def test_position_front_and_rear():
    manager = make_manager()
    current = waypoint()

    front = waypoint(x=10)
    rear = waypoint(x=-10)

    current.next.return_value = [front]
    current.previous.return_value = [rear]

    assert manager._get_position_waypoint(
        current, "front", 10.0
    ) is front

    assert manager._get_position_waypoint(
        current, "rear", 10.0
    ) is rear


# ==========================================
# GROUP D: NPC SPEED AND SPAWN POSITIONS
# ==========================================

# TEST 16
def test_set_npc_speed():
    manager = make_manager()

    vehicle = Mock()
    vehicle.get_speed_limit.return_value = 50.0

    manager._set_npc_speed(vehicle, 30.0)

    manager.traffic_manager.vehicle_percentage_speed_difference.assert_called_once_with(
        vehicle, pytest.approx(40.0)
    )


# TEST 17
def test_npc_speed_clamped():
    manager = make_manager()

    vehicle = Mock()
    vehicle.get_speed_limit.return_value = 50.0

    manager._set_npc_speed(vehicle, 100.0)

    # Speed is limited to 80 km/h in the source.
    manager.traffic_manager.vehicle_percentage_speed_difference.assert_called_once_with(
        vehicle, pytest.approx(-60.0)
    )


# TEST 18
def test_distributed_spawn_points():
    manager = make_manager()

    points = [
        SimpleNamespace(location=loc(0, 0)),
        SimpleNamespace(location=loc(20, 0)),
        SimpleNamespace(location=loc(40, 0)),
    ]

    selected = manager._select_distributed_spawn_points(
        points, 2
    )

    assert len(selected) == 2

    distance = manager._distance(
        selected[0].location,
        selected[1].location
    )

    assert distance >= manager.MIN_MAP_DISTRIBUTION_DISTANCE


# ==========================================
# GROUP E: CLEANUP AND STATE
# ==========================================

# TEST 19
def test_cleanup_vehicles():
    manager = make_manager()

    vehicle = Mock()
    vehicle.is_alive = True

    manager.spawned_vehicles = [vehicle]
    manager.scenario_name = "safe_overtake"

    manager.cleanup()

    vehicle.set_autopilot.assert_called_once_with(
        False, 8000
    )
    vehicle.destroy.assert_called_once()

    assert manager.spawned_vehicles == []
    assert manager.scenario_name is None


# TEST 20
def test_get_spawned_vehicles_returns_copy():
    manager = make_manager()

    vehicle = Mock()
    manager.spawned_vehicles = [vehicle]

    result = manager.get_spawned_vehicles()
    result.clear()

    assert len(manager.spawned_vehicles) == 1
    assert result == []
