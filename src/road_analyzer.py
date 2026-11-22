import carla
import math


class RoadAnalyzer:
    def __init__(self, carla_map, sample_resolution=2.0):
        self.map = carla_map
        self.sample_resolution = max(1.0, float(sample_resolution))

    @staticmethod
    def _dot(v1, v2):
        return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z

    @staticmethod
    def _distance(loc1, loc2):
        return loc1.distance(loc2)

    @staticmethod
    def _is_driving(waypoint):
        return waypoint is not None and waypoint.lane_type == carla.LaneType.Driving

    @staticmethod
    def _lane_key(waypoint):
        return waypoint.road_id, waypoint.section_id, waypoint.lane_id

    def _same_direction(self, wp1, wp2, threshold=0.8):
        if wp1 is None or wp2 is None:
            return False

        if wp1.road_id != wp2.road_id:
            return False

        forward1 = wp1.transform.get_forward_vector()
        forward2 = wp2.transform.get_forward_vector()

        return self._dot(forward1, forward2) >= threshold

    def _direction_type(self, current_waypoint, candidate_waypoint):
        if current_waypoint is None or candidate_waypoint is None:
            return "UNKNOWN"

        forward1 = current_waypoint.transform.get_forward_vector()
        forward2 = candidate_waypoint.transform.get_forward_vector()
        dot = self._dot(forward1, forward2)

        if dot >= 0.8:
            return "SAME_DIRECTION"

        if dot <= -0.8:
            return "OPPOSITE_DIRECTION"

        return "UNKNOWN"

    def get_current_waypoint(self, location):
        return self.map.get_waypoint(location, project_to_road=True, lane_type=carla.LaneType.Driving)

    def _get_adjacent_waypoint(self, waypoint, side):
        if not self._is_driving(waypoint):
            return None

        try:
            adjacent = waypoint.get_left_lane() if side == "left" else waypoint.get_right_lane()
        except (RuntimeError, AttributeError):
            return None

        if not self._is_driving(adjacent):
            return None

        return adjacent

    def get_adjacent_lanes(self, waypoint):
        result = {
            "left": None,
            "right": None,
            "left_direction": "UNKNOWN",
            "right_direction": "UNKNOWN"
        }

        if not self._is_driving(waypoint):
            return result

        for side in ("left", "right"):
            adjacent = self._get_adjacent_waypoint(waypoint, side)

            if adjacent is None:
                continue

            direction = self._direction_type(waypoint, adjacent)
            result[f"{side}_direction"] = direction

            if direction == "SAME_DIRECTION":
                result[side] = adjacent

        return result

    @staticmethod
    def _can_change(waypoint, side):
        if waypoint is None:
            return False

        try:
            permission = int(waypoint.lane_change)
        except (TypeError, ValueError, AttributeError):
            return False

        if side == "right":
            return bool(permission & 1)

        if side == "left":
            return bool(permission & 2)

        return False

    def _can_change_along_lane(self, waypoint, side, distance=10.0):
        if not self._is_driving(waypoint):
            return False

        if not self._can_change(waypoint, side):
            return False

        current = waypoint
        travelled = 0.0
        step = self.sample_resolution

        while travelled < distance:
            try:
                next_waypoints = current.next(step)
            except (RuntimeError, AttributeError):
                return False

            if not next_waypoints:
                return False

            next_wp = min(next_waypoints, key=lambda wp: self._distance(wp.transform.location, current.transform.location))

            if not self._is_driving(next_wp):
                return False

            if next_wp.road_id != waypoint.road_id:
                return False

            if next_wp.lane_id != waypoint.lane_id:
                return False

            if not self._can_change(next_wp, side):
                return False

            travelled += self._distance(current.transform.location, next_wp.transform.location)
            current = next_wp

        return True

    def _walk_lanes(self, waypoint, side, max_lanes=12):
        lanes = []
        current = waypoint
        visited = {self._lane_key(waypoint)}

        for _ in range(max_lanes):
            adjacent = self._get_adjacent_waypoint(current, side)

            if adjacent is None:
                break

            key = self._lane_key(adjacent)

            if key in visited:
                break

            visited.add(key)
            lanes.append(adjacent)
            current = adjacent

        return lanes

    def count_lanes(self, current_waypoint, search_radius=20.0):
        if not self._is_driving(current_waypoint):
            return {
                "same_direction": 0,
                "opposite_direction": 0,
                "same_direction_lane_ids": [],
                "opposite_direction_lane_ids": []
            }

        same_direction = {current_waypoint.lane_id}
        opposite_direction = set()

        for side in ("left", "right"):
            lanes = self._walk_lanes(current_waypoint, side)

            for waypoint in lanes:
                direction = self._direction_type(current_waypoint, waypoint)

                if direction == "SAME_DIRECTION":
                    same_direction.add(waypoint.lane_id)

                elif direction == "OPPOSITE_DIRECTION":
                    opposite_direction.add(waypoint.lane_id)

        return {
            "same_direction": len(same_direction),
            "opposite_direction": len(opposite_direction),
            "same_direction_lane_ids": sorted(same_direction),
            "opposite_direction_lane_ids": sorted(opposite_direction)
        }

    @staticmethod
    def classify_road(lane_info):
        same = lane_info["same_direction"]
        opposite = lane_info["opposite_direction"]

        if opposite > 0:
            return "TWO_WAY"

        if same > 1:
            return "ONE_WAY"

        return "UNKNOWN"

    def analyze(self, location):
        current_wp = self.get_current_waypoint(location)

        if current_wp is None:
            return None

        adjacent = self.get_adjacent_lanes(current_wp)
        lane_info = self.count_lanes(current_wp)
        road_type = self.classify_road(lane_info)

        can_change_left = self._can_change_along_lane(current_wp, "left")
        can_change_right = self._can_change_along_lane(current_wp, "right")

        return {
            "road_id": current_wp.road_id,
            "section_id": current_wp.section_id,
            "current_lane_id": current_wp.lane_id,
            "road_type": road_type,
            "same_direction_lanes": lane_info["same_direction"],
            "opposite_direction_lanes": lane_info["opposite_direction"],
            "same_direction_lane_ids": lane_info["same_direction_lane_ids"],
            "opposite_direction_lane_ids": lane_info["opposite_direction_lane_ids"],
            "left_lane": adjacent["left"],
            "right_lane": adjacent["right"],
            "left_direction": adjacent["left_direction"],
            "right_direction": adjacent["right_direction"],
            "can_change_left": can_change_left,
            "can_change_right": can_change_right,
            "current_waypoint": current_wp
        }