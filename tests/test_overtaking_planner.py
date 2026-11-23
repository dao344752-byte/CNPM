
from types import SimpleNamespace

from src.overtaking_planner import OvertakingPlanner


def make_waypoint():
    """Tạo waypoint giả lập, không cần CARLA."""
    forward = SimpleNamespace(x=1.0, y=0.0, z=0.0)
    transform = SimpleNamespace(
        get_forward_vector=lambda: forward
    )
    return SimpleNamespace(transform=transform)


# TEST 1: Không phát hiện phương tiện
def test_no_front_vehicle():
    planner = OvertakingPlanner()

    result = planner.find_front_vehicle([])

    assert result is None


# TEST 2: Tìm xe gần nhất trong làn
def test_find_nearest_front_vehicle():
    planner = OvertakingPlanner()

    detections = [
        {"class": "car", "distance": 20, "in_lane": True},
        {"class": "truck", "distance": 12, "in_lane": True},
        {"class": "car", "distance": 5, "in_lane": False},
    ]

    result = planner.find_front_vehicle(detections)

    assert result["class"] == "truck"
    assert result["distance"] == 12


# TEST 3: Làn bên cạnh đủ an toàn
def test_safe_adjacent_lane():
    planner = OvertakingPlanner()

    traffic = {
        "observed": True,
        "front_distance": 30,
        "rear_distance": 25,
        "front_ttc": 8,
        "rear_ttc": 7,
    }

    assert planner.is_lane_safe(traffic) is True


# TEST 4: Có xe quá gần phía sau
def test_unsafe_rear_distance():
    planner = OvertakingPlanner()

    traffic = {
        "observed": True,
        "front_distance": 30,
        "rear_distance": 5,
        "front_ttc": 8,
        "rear_ttc": 7,
    }

    assert planner.is_lane_safe(traffic) is False


# TEST 5: TTC phía sau quá thấp
def test_unsafe_rear_ttc():
    planner = OvertakingPlanner()

    traffic = {
        "observed": True,
        "front_distance": 30,
        "rear_distance": 25,
        "front_ttc": 8,
        "rear_ttc": 1.0,
    }

    assert planner.is_lane_safe(traffic) is False


# TEST 6: Không có thông tin giao thông
def test_unobserved_lane_is_not_safe():
    planner = OvertakingPlanner()

    assert planner.is_lane_safe({"observed": False}) is False


# TEST 7: Không có xe trước -> Không vượt
def test_no_vehicle_no_overtaking():
    planner = OvertakingPlanner()

    result = planner.plan(
        road_info={"road_type": "TWO_WAY"},
        detections=[],
        ego_speed=30,
        target_speed=35,
    )

    assert result["should_overtake"] is False
    assert result["reason"] == (
        "NO_VEHICLE_CONFIRMED_IN_CURRENT_LANE"
    )


# TEST 8: Xe chậm trước mặt, làn trái an toàn
def test_safe_overtaking():
    planner = OvertakingPlanner()

    road_info = {
        "road_type": "TWO_WAY",
        "current_waypoint": make_waypoint(),
        "left_lane": make_waypoint(),
        "right_lane": None,
        "can_change_left": True,
        "can_change_right": False,
    }

    detections = [
        {
            "class": "car",
            "distance": 15,
            "in_lane": True,
            "speed_kmh": 10,
        }
    ]

    adjacent_traffic = {
        "left": {
            "observed": True,
            "front_distance": 40,
            "rear_distance": 35,
            "front_ttc": 8,
            "rear_ttc": 7,
        }
    }

    result = planner.plan(
        road_info=road_info,
        detections=detections,
        adjacent_traffic=adjacent_traffic,
        ego_speed=30,
        target_speed=35,
    )

    assert result["should_overtake"] is True
    assert result["direction"] == "left"

