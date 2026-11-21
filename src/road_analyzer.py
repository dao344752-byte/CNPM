
import carla
import math


class RoadAnalyzer:
    """
    Analyze road and lane topology using CARLA Map.

    Main responsibilities:
        - Find current driving lane.
        - Count lanes in the same/opposite direction.
        - Detect adjacent lanes.
        - Check lane-change permissions.
        - Estimate road direction type.
    """

    def __init__(self, carla_map, sample_resolution=2.0):
        self.map = carla_map
        self.sample_resolution = sample_resolution

        # Cache generated map waypoints.
        self.map_waypoints = self.map.generate_waypoints(
            self.sample_resolution
        )

    # --------------------------------------------------
    # Basic waypoint utilities
    # --------------------------------------------------

    @staticmethod
    def _get_direction_vector(waypoint):
        return waypoint.transform.get_forward_vector()

    @staticmethod
    def _dot_product(v1, v2):
        return (
            v1.x * v2.x +
            v1.y * v2.y +
            v1.z * v2.z
        )

    def _same_direction(self, wp1, wp2):
        """
        Compare the heading of two lanes.

        Returns:
            True: same direction
            False: opposite direction
        """

        v1 = self._get_direction_vector(wp1)
        v2 = self._get_direction_vector(wp2)

        dot = self._dot_product(v1, v2)

        return dot > 0.8

    # --------------------------------------------------
    # Find current waypoint
    # --------------------------------------------------

    def get_current_waypoint(self, location):
        waypoint = self.map.get_waypoint(
            location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )

        return waypoint

    # --------------------------------------------------
    # Find adjacent lanes
    # --------------------------------------------------

    def get_adjacent_lanes(self, waypoint):

        left_lane = waypoint.get_left_lane()
        right_lane = waypoint.get_right_lane()

        if left_lane is not None:
            if left_lane.lane_type != carla.LaneType.Driving:
                left_lane = None

        if right_lane is not None:
            if right_lane.lane_type != carla.LaneType.Driving:
                right_lane = None

        return {
            "left": left_lane,
            "right": right_lane
        }

    # --------------------------------------------------
    # Lane-change permission
    # --------------------------------------------------

    @staticmethod
    def _can_change(waypoint, side):

        if waypoint is None:
            return False

        permission = waypoint.lane_change

        try:
            permission = int(permission)
        except (TypeError, ValueError):
            permission = 0

        # CARLA LaneChange:
        # NONE  = 0
        # Right = 1
        # Left  = 2
        # Both  = 3

        if side == "right":
            return bool(permission & 1)

        if side == "left":
            return bool(permission & 2)

        return False

    # --------------------------------------------------
    # Analyze lanes in current road section
    # --------------------------------------------------

    def count_lanes(self, current_waypoint, search_radius=20.0):

        same_direction = set()
        opposite_direction = set()

        current_location = current_waypoint.transform.location

        for wp in self.map_waypoints:

            # Only consider the same road and section.
            if wp.road_id != current_waypoint.road_id:
                continue

            if wp.section_id != current_waypoint.section_id:
                continue

            distance = wp.transform.location.distance(
                current_location
            )

            if distance > search_radius:
                continue

            if self._same_direction(current_waypoint, wp):
                same_direction.add(wp.lane_id)
            else:
                opposite_direction.add(wp.lane_id)

        # Always include the current lane.
        same_direction.add(current_waypoint.lane_id)

        return {
            "same_direction": len(same_direction),
            "opposite_direction": len(opposite_direction),
            "same_direction_lane_ids": sorted(same_direction),
            "opposite_direction_lane_ids": sorted(opposite_direction)
        }

    # --------------------------------------------------
    # Classify road direction
    # --------------------------------------------------

    @staticmethod
    def classify_road(lane_info):

        same = lane_info["same_direction"]
        opposite = lane_info["opposite_direction"]

        if opposite > 0:
            return "TWO_WAY"

        if same > 1:
            return "ONE_WAY"

        return "UNKNOWN"

    # --------------------------------------------------
    # Main analysis function
    # --------------------------------------------------

    def analyze(self, location):

        current_wp = self.get_current_waypoint(location)

        if current_wp is None:
            return None

        adjacent = self.get_adjacent_lanes(current_wp)

        lane_info = self.count_lanes(current_wp)

        road_type = self.classify_road(lane_info)

        result = {
            "road_id": current_wp.road_id,
            "section_id": current_wp.section_id,

            "current_lane_id": current_wp.lane_id,

            "road_type": road_type,

            "same_direction_lanes":
                lane_info["same_direction"],

            "opposite_direction_lanes":
                lane_info["opposite_direction"],

            "left_lane": adjacent["left"],

            "right_lane": adjacent["right"],

            "can_change_left":
                self._can_change(current_wp, "left"),

            "can_change_right":
                self._can_change(current_wp, "right"),

            "current_waypoint": current_wp
        }

        return result
