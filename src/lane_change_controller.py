
import math
import carla


class LaneChangeController:

    def __init__(
        self,
        vehicle,
        carla_map,
        transition_distance=12.0,
        lookahead_distance=5.0,
        waypoint_spacing=2.0,
        wheelbase=2.8,
        max_steering_angle=0.60,
        max_steer=0.65,
        completion_tolerance=1.0,
        max_steps=400
    ):
        self.vehicle = vehicle
        self.map = carla_map

        self.transition_distance = max(5.0, float(transition_distance))
        self.lookahead_distance = max(2.0, float(lookahead_distance))
        self.waypoint_spacing = max(0.5, float(waypoint_spacing))

        self.wheelbase = float(wheelbase)
        self.max_steering_angle = float(max_steering_angle)
        self.max_steer = float(max_steer)

        self.completion_tolerance = float(completion_tolerance)
        self.max_steps = int(max_steps)

        self.state = "NORMAL"
        self.active = False

        self.target_lane_id = None
        self.target_road_id = None
        self.target_waypoint = None

        self.reference_path = []
        self.target_path = []
        self.source_path = []

        self.progress_index = 0
        self.completion_count = 0
        self.step_count = 0

        self.last_steer = 0.0
        self.last_lateral_error = 0.0

    @staticmethod
    def _distance_2d(a, b):
        return math.hypot(a.x - b.x, a.y - b.y)

    @staticmethod
    def _direction_dot(forward_a, forward_b):
        return (
            forward_a.x * forward_b.x +
            forward_a.y * forward_b.y
        )

    @staticmethod
    def _clamp(value, low, high):
        return max(low, min(high, value))

    def _get_current_waypoint(self):
        return self.map.get_waypoint(
            self.vehicle.get_location(),
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )

    def _select_lane_continuation(
        self,
        waypoint,
        distance,
        preferred_lane_id=None,
        previous_forward=None
    ):
        candidates = waypoint.next(distance)

        if not candidates:
            return None

        if previous_forward is None:
            previous_forward = waypoint.transform.get_forward_vector()

        best_candidate = None
        best_score = -float("inf")

        for candidate in candidates:
            if candidate.lane_type != carla.LaneType.Driving:
                continue

            candidate_forward = candidate.transform.get_forward_vector()

            direction_dot = self._direction_dot(
                previous_forward,
                candidate_forward
            )

            if direction_dot < 0.7:
                continue

            score = direction_dot * 3.0

            if preferred_lane_id is not None:
                if candidate.lane_id == preferred_lane_id:
                    score += 2.0

            if candidate.road_id == waypoint.road_id:
                score += 0.5

            if candidate.section_id == waypoint.section_id:
                score += 0.25

            if score > best_score:
                best_score = score
                best_candidate = candidate

        return best_candidate

    def _build_lane_path(self, start_waypoint, number_of_points):
        path = [start_waypoint]
        current_wp = start_waypoint

        preferred_lane_id = start_waypoint.lane_id

        for _ in range(number_of_points - 1):
            next_wp = self._select_lane_continuation(
                current_wp,
                self.waypoint_spacing,
                preferred_lane_id=preferred_lane_id,
                previous_forward=current_wp.transform.get_forward_vector()
            )

            if next_wp is None:
                break

            path.append(next_wp)
            current_wp = next_wp

            preferred_lane_id = next_wp.lane_id

        return path

    def _path_length(self, path):
        total = 0.0

        for i in range(1, len(path)):
            total += self._distance_2d(
                path[i - 1].transform.location,
                path[i].transform.location
            )

        return total

    def _resample_path(self, path, distances):
        if len(path) < 2:
            return []

        cumulative = [0.0]

        for i in range(1, len(path)):
            cumulative.append(
                cumulative[-1] + self._distance_2d(
                    path[i - 1].transform.location,
                    path[i].transform.location
                )
            )

        result = []

        for distance in distances:
            if distance > cumulative[-1]:
                return []

            index = 0

            while index < len(cumulative) - 2 and cumulative[index + 1] < distance:
                index += 1

            start_distance = cumulative[index]
            end_distance = cumulative[index + 1]

            segment_length = end_distance - start_distance

            ratio = (
                0.0 if segment_length < 1e-6
                else (distance - start_distance) / segment_length
            )

            start = path[index].transform.location
            end = path[index + 1].transform.location

            location = carla.Location(
                x=start.x + ratio * (end.x - start.x),
                y=start.y + ratio * (end.y - start.y),
                z=start.z + ratio * (end.z - start.z)
            )

            result.append(location)

        return result

    def _build_reference_trajectory(self, current_waypoint, target_waypoint):
        total_distance = (
            self.transition_distance +
            self.lookahead_distance * 2.0
        )

        number_of_points = max(
            4,
            int(math.ceil(total_distance / self.waypoint_spacing)) + 1
        )

        source_path = self._build_lane_path(
            current_waypoint,
            number_of_points
        )

        target_path = self._build_lane_path(
            target_waypoint,
            number_of_points
        )

        if len(source_path) < 3 or len(target_path) < 3:
            return False

        source_distances = [0.0]

        for i in range(1, len(source_path)):
            source_distances.append(
                source_distances[-1] + self._distance_2d(
                    source_path[i - 1].transform.location,
                    source_path[i].transform.location
                )
            )

        target_locations = self._resample_path(
            target_path,
            source_distances
        )

        if len(target_locations) != len(source_path):
            return False

        self.source_path = source_path
        self.target_path = target_path
        self.reference_path = []

        for i, source_wp in enumerate(source_path):
            source_loc = source_wp.transform.location
            target_loc = target_locations[i]

            progress = source_distances[i]

            t = self._clamp(
                progress / self.transition_distance,
                0.0,
                1.0
            )

            alpha = 3.0 * t ** 2 - 2.0 * t ** 3

            self.reference_path.append(
                carla.Location(
                    x=source_loc.x + alpha * (target_loc.x - source_loc.x),
                    y=source_loc.y + alpha * (target_loc.y - source_loc.y),
                    z=source_loc.z + alpha * (target_loc.z - source_loc.z)
                )
            )

        self.progress_index = 0
        self.completion_count = 0
        self.step_count = 0

        return True

    def start(self, target_waypoint):
        if self.active or target_waypoint is None:
            return False

        current_waypoint = self._get_current_waypoint()

        if current_waypoint is None:
            return False

        if target_waypoint.lane_type != carla.LaneType.Driving:
            return False

        current_forward = current_waypoint.transform.get_forward_vector()
        target_forward = target_waypoint.transform.get_forward_vector()

        direction_dot = self._direction_dot(
            current_forward,
            target_forward
        )

        if direction_dot < 0.9:
            return False

        if current_waypoint.road_id != target_waypoint.road_id:
            return False

        success = self._build_reference_trajectory(
            current_waypoint,
            target_waypoint
        )

        if not success:
            self._clear_trajectory()
            return False

        self.target_lane_id = target_waypoint.lane_id
        self.target_road_id = target_waypoint.road_id
        self.target_waypoint = target_waypoint

        self.state = "LANE_CHANGE"
        self.active = True

        self.last_steer = 0.0
        self.last_lateral_error = 0.0

        return True

    def _find_nearest_reference_index(self):
        if not self.reference_path:
            return 0

        vehicle_location = self.vehicle.get_location()

        start_index = max(0, self.progress_index - 1)
        end_index = min(
            len(self.reference_path),
            self.progress_index + 9
        )

        best_index = start_index
        best_distance = float("inf")

        for i in range(start_index, end_index):
            distance = self._distance_2d(
                vehicle_location,
                self.reference_path[i]
            )

            if distance < best_distance:
                best_distance = distance
                best_index = i

        self.progress_index = max(
            self.progress_index,
            best_index
        )

        return self.progress_index

    def _get_lookahead_target(self):
        if not self.reference_path:
            return None

        nearest_index = self._find_nearest_reference_index()

        vehicle_location = self.vehicle.get_location()

        accumulated_distance = 0.0
        previous_location = vehicle_location

        for i in range(nearest_index, len(self.reference_path)):
            current_location = self.reference_path[i]

            accumulated_distance += self._distance_2d(
                previous_location,
                current_location
            )

            if accumulated_distance >= self.lookahead_distance:
                return current_location

            previous_location = current_location

        return self.reference_path[-1] if nearest_index < len(self.reference_path) else None

    def _calculate_steering(self):
        target = self._get_lookahead_target()

        if target is None:
            return 0.0

        transform = self.vehicle.get_transform()
        vehicle_location = transform.location
        forward = transform.get_forward_vector()

        dx = target.x - vehicle_location.x
        dy = target.y - vehicle_location.y

        local_x = dx * forward.x + dy * forward.y
        local_y = -dx * forward.y + dy * forward.x

        lookahead = math.hypot(local_x, local_y)

        if lookahead < 0.05:
            return 0.0

        alpha = math.atan2(local_y, local_x)

        curvature = 2.0 * math.sin(alpha) / lookahead

        steering_angle = math.atan(
            self.wheelbase * curvature
        )

        steer = steering_angle / max(
            self.max_steering_angle,
            1e-3
        )

        return float(
            self._clamp(steer, -self.max_steer, self.max_steer)
        )

    def _get_target_path_waypoint(self):
        if not self.target_path:
            return None

        vehicle_location = self.vehicle.get_location()

        start = max(0, self.progress_index - 3)
        end = min(len(self.target_path), self.progress_index + 5)

        candidates = self.target_path[start:end]

        if not candidates:
            candidates = self.target_path

        return min(
            candidates,
            key=lambda wp: self._distance_2d(
                vehicle_location,
                wp.transform.location
            )
        )

    def get_target_lane_waypoint(self):
        if not self.target_path:
            return self.target_waypoint
        index = min(max(self.progress_index, 0), len(self.target_path) - 1)
        return self.target_path[index]

    def _is_on_target_lane(self):
        current_wp = self._get_current_waypoint()
        target_wp = self._get_target_path_waypoint()

        if current_wp is None or target_wp is None:
            return False

        if current_wp.lane_type != carla.LaneType.Driving:
            return False

        if current_wp.road_id != target_wp.road_id:
            return False

        if current_wp.lane_id != target_wp.lane_id:
            return False

        distance = self._distance_2d(
            self.vehicle.get_location(),
            target_wp.transform.location
        )

        tolerance = max(
            self.completion_tolerance,
            target_wp.lane_width * 0.45
        )

        return distance <= tolerance

    def _has_passed_transition_end(self):
        if not self.reference_path:
            return False

        final_location = self.reference_path[-1]
        previous_location = self.reference_path[-2]

        direction_x = final_location.x - previous_location.x
        direction_y = final_location.y - previous_location.y

        length = math.hypot(direction_x, direction_y)

        if length < 1e-6:
            return False

        direction_x /= length
        direction_y /= length

        vehicle_location = self.vehicle.get_location()

        passed_distance = (
            (vehicle_location.x - final_location.x) * direction_x +
            (vehicle_location.y - final_location.y) * direction_y
        )

        return passed_distance >= 0.0

    def _is_completed(self):
        if not self.target_path or not self.reference_path:
            return False

        if not self._is_on_target_lane():
            self.completion_count = 0
            return False

        final_index = len(self.reference_path) - 1

        enough_progress = (
            self.progress_index >= final_index - 3 or
            self._has_passed_transition_end()
        )

        if not enough_progress:
            self.completion_count = 0
            return False

        vehicle_forward = self.vehicle.get_transform().get_forward_vector()
        target_wp = self._get_target_path_waypoint()

        if target_wp is None:
            self.completion_count = 0
            return False

        target_forward = target_wp.transform.get_forward_vector()

        direction_dot = self._direction_dot(
            vehicle_forward,
            target_forward
        )

        if direction_dot < 0.85:
            self.completion_count = 0
            return False

        self.completion_count += 1

        return self.completion_count >= 3

    def run_step(self):
        if not self.active or self.state != "LANE_CHANGE":
            return None

        self.step_count += 1

        self._find_nearest_reference_index()

        steer = self._calculate_steering()

        self.last_steer = steer

        completed = self._is_completed()

        if completed:
            self.active = False
            self.state = "LANE_FOLLOW"

            return {
                "steer": steer,
                "completed": True,
                "failed": False,
                "state": self.state
            }

        if self.step_count >= self.max_steps:
            self.active = False

            if self._is_on_target_lane():
                self.state = "LANE_FOLLOW"

                return {
                    "steer": steer,
                    "completed": True,
                    "failed": False,
                    "state": self.state
                }

            self.state = "FAILED"

            return {
                "steer": steer,
                "completed": False,
                "failed": True,
                "state": self.state
            }

        return {
            "steer": steer,
            "completed": False,
            "failed": False,
            "state": self.state
        }

    def get_follow_waypoint(self, distance=8.0):
        if self.state != "LANE_FOLLOW":
            return None

        current_wp = self._get_current_waypoint()

        if current_wp is None:
            return None

        vehicle_forward = self.vehicle.get_transform().get_forward_vector()

        candidates = current_wp.next(distance)

        if not candidates:
            return None

        valid_candidates = []

        for candidate in candidates:
            if candidate.lane_type != carla.LaneType.Driving:
                continue

            candidate_forward = candidate.transform.get_forward_vector()

            direction_dot = self._direction_dot(
                vehicle_forward,
                candidate_forward
            )

            if direction_dot < 0.7:
                continue

            if candidate.road_id != current_wp.road_id:
                continue

            valid_candidates.append(candidate)

        if not valid_candidates:
            return None

        return max(
            valid_candidates,
            key=lambda wp: self._direction_dot(
                vehicle_forward,
                wp.transform.get_forward_vector()
            )
        )

    def _clear_trajectory(self):
        self.reference_path = []
        self.target_path = []
        self.source_path = []

        self.progress_index = 0
        self.completion_count = 0
        self.step_count = 0

    def cancel(self):
        self.active = False
        self.state = "NORMAL"

        self.target_lane_id = None
        self.target_road_id = None
        self.target_waypoint = None

        self._clear_trajectory()

        self.last_steer = 0.0
        self.last_lateral_error = 0.0
