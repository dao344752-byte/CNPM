import math


class OvertakingPlanner:
    VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "motorbike", "bicycle", "bike"}

    def __init__(
        self,
        obstacle_distance=30.0,
        min_front_distance=8.0,
        overtake_distance=20.0,
        safe_front_distance=12.0,
        safe_rear_distance=10.0,
        min_stopped_front_distance=3.5,
        min_rear_ttc=2.5,
        min_front_ttc=2.0,
        min_relative_speed=2.0
    ):
        self.obstacle_distance = float(obstacle_distance)
        self.min_front_distance = float(min_front_distance)
        self.overtake_distance = float(overtake_distance)
        self.safe_front_distance = float(safe_front_distance)
        self.safe_rear_distance = float(safe_rear_distance)
        self.min_stopped_front_distance = float(min_stopped_front_distance)
        self.min_rear_ttc = float(min_rear_ttc)
        self.min_front_ttc = float(min_front_ttc)
        self.min_relative_speed = float(min_relative_speed)

    @staticmethod
    def _number(value, default=None):
        try:
            value = float(value)
            return value if math.isfinite(value) else default
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _as_bool(value):
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            return value.strip().lower() in {"true", "1", "yes", "y"}
        return False

    @classmethod
    def _vehicle_speed(cls, detection):
        for key in ("speed_kmh", "speed", "velocity"):
            value = cls._number(detection.get(key))
            if value is not None:
                return value
        return None

    @classmethod
    def _belongs_to_current_lane(cls, detection, detections_are_same_lane):
        if "in_lane" in detection:
            return cls._as_bool(detection.get("in_lane"))
        if "same_lane" in detection:
            return cls._as_bool(detection.get("same_lane"))
        return bool(detections_are_same_lane)

    def find_front_vehicle(self, detections, detections_are_same_lane=False):
        candidates = []

        for detection in detections or []:
            if not isinstance(detection, dict):
                continue

            name = str(detection.get("class", "")).lower().strip()
            if name not in self.VEHICLE_CLASSES:
                continue

            if not self._belongs_to_current_lane(detection, detections_are_same_lane):
                continue

            distance = self._number(detection.get("distance"))
            if distance is None or distance <= 0.0 or distance > self.obstacle_distance:
                continue

            candidates.append((distance, detection))

        if not candidates:
            return None

        return min(candidates, key=lambda item: item[0])[1]

    def is_lane_safe(self, traffic_info):
        if not isinstance(traffic_info, dict) or not traffic_info.get("observed", False):
            return False

        front_distance = self._number(traffic_info.get("front_distance"))
        rear_distance = self._number(traffic_info.get("rear_distance"))
        front_ttc = self._number(traffic_info.get("front_ttc"))
        rear_ttc = self._number(traffic_info.get("rear_ttc"))

        if front_distance is not None and front_distance < self.safe_front_distance:
            return False
        if rear_distance is not None and rear_distance < self.safe_rear_distance:
            return False
        if front_ttc is not None and front_ttc < self.min_front_ttc:
            return False
        if rear_ttc is not None and rear_ttc < self.min_rear_ttc:
            return False

        return True

    @staticmethod
    def is_same_direction(current_wp, target_wp):
        if current_wp is None or target_wp is None:
            return False

        v1 = current_wp.transform.get_forward_vector()
        v2 = target_wp.transform.get_forward_vector()
        dot = v1.x * v2.x + v1.y * v2.y + v1.z * v2.z
        return dot > 0.8

    def evaluate_side(self, side, road_info, adjacent_traffic):
        target_lane = road_info.get(side + "_lane")
        if target_lane is None:
            return False, "NO_ADJACENT_LANE", float("-inf")

        if not road_info.get("can_change_" + side, False):
            return False, "LANE_CHANGE_NOT_ALLOWED", float("-inf")

        current_wp = road_info.get("current_waypoint")
        if not self.is_same_direction(current_wp, target_lane):
            return False, "OPPOSITE_DIRECTION", float("-inf")

        traffic_info = adjacent_traffic.get(side)
        if not self.is_lane_safe(traffic_info):
            return False, "ADJACENT_LANE_NOT_SAFE", float("-inf")

        front_distance = self._number(traffic_info.get("front_distance"), 50.0)
        rear_distance = self._number(traffic_info.get("rear_distance"), 50.0)
        front_ttc = self._number(traffic_info.get("front_ttc"), 10.0)
        rear_ttc = self._number(traffic_info.get("rear_ttc"), 10.0)

        score = (
            min(front_distance, 50.0) * 0.30 +
            min(rear_distance, 50.0) * 0.30 +
            min(front_ttc, 10.0) +
            min(rear_ttc, 10.0)
        )

        return True, "SAFE", score

    @staticmethod
    def _result(should_overtake, reason, **kwargs):
        result = {
            "should_overtake": should_overtake,
            "direction": None,
            "target_lane": None,
            "reason": reason
        }
        result.update(kwargs)
        return result

    def plan(
        self,
        road_info,
        detections,
        adjacent_traffic=None,
        ego_speed=0.0,
        target_speed=35.0,
        detections_are_same_lane=False
    ):
        if adjacent_traffic is None:
            adjacent_traffic = {}

        if road_info is None:
            return self._result(False, "ROAD_INFORMATION_UNAVAILABLE")

        front_vehicle = self.find_front_vehicle(detections, detections_are_same_lane)
        if front_vehicle is None:
            return self._result(False, "NO_VEHICLE_CONFIRMED_IN_CURRENT_LANE")

        distance = self._number(front_vehicle.get("distance"))
        if distance is None:
            return self._result(False, "INVALID_FRONT_DISTANCE")

        ego_speed = max(0.0, self._number(ego_speed, 0.0))
        target_speed = max(0.0, self._number(target_speed, 35.0))
        front_speed = self._vehicle_speed(front_vehicle)

        if distance > self.overtake_distance:
            return self._result(False, "FRONT_VEHICLE_NOT_BLOCKING", obstacle_distance=distance)

        if ego_speed <= 2.0 and distance < self.min_stopped_front_distance:
            return self._result(False, "STOPPED_TOO_CLOSE_TO_FRONT_VEHICLE", obstacle_distance=distance)

        if ego_speed > 2.0 and distance < self.min_front_distance:
            return self._result(False, "TOO_CLOSE_TO_OVERTAKE", obstacle_distance=distance)

        if road_info.get("road_type") == "UNKNOWN":
            return self._result(False, "UNKNOWN_ROAD_TYPE", obstacle_distance=distance)

        if front_speed is not None:
            relative_speed = ego_speed - max(0.0, front_speed)
            blocked = relative_speed >= self.min_relative_speed or (front_speed <= 1.0 and distance <= self.overtake_distance)
        else:
            blocked = ego_speed < target_speed - 5.0

        if not blocked:
            return self._result(False, "FRONT_VEHICLE_NOT_SLOWER", obstacle_distance=distance)

        safe_lanes = []
        for side in ("left", "right"):
            safe, reason, score = self.evaluate_side(side, road_info, adjacent_traffic)
            if safe:
                safe_lanes.append({
                    "side": side,
                    "score": score,
                    "target_lane": road_info.get(side + "_lane")
                })

        if not safe_lanes:
            return self._result(
                False,
                "NO_SAFE_ADJACENT_LANE",
                obstacle_distance=distance,
                obstacle_class=front_vehicle.get("class")
            )

        selected = max(safe_lanes, key=lambda item: item["score"])
        return {
            "should_overtake": True,
            "direction": selected["side"],
            "target_lane": selected["target_lane"],
            "obstacle_distance": distance,
            "obstacle_class": front_vehicle.get("class"),
            "front_vehicle_speed": front_speed,
            "ego_speed": ego_speed,
            "reason": "SLOW_FRONT_VEHICLE_CONFIRMED_IN_CURRENT_LANE"
        }