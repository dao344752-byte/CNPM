
import math
import carla
import numpy as np

from src.road_analyzer import RoadAnalyzer
from src.overtaking_planner import OvertakingPlanner
from src.lane_change_controller import LaneChangeController


class AIDriver:
    VEHICLE_CLASSES = {'car', 'truck', 'bus', 'motorcycle', 'motorbike', 'bicycle', 'bike'}

    def __init__(self, vehicle: carla.Vehicle, carla_map: carla.Map, target_speed: float = 35.0):
        self.vehicle = vehicle
        self.map = carla_map
        self.target_speed = max(0.0, float(target_speed))
        self.speed = 0.0
        self.log = []

        self.steer_kp = 1.2
        self.prev_steer = 0.0
        self.waypoint_lookahead = 8.0

        self.emergency_deceleration = 6.0
        self.emergency_safety_margin = 4.0
        self.reaction_time = 0.8
        self.caution_multiplier = 1.5
        self.danger_multiplier = 1.8

        self.follow_time_gap = 2.0
        self.minimum_follow_distance = 7.0

        self.lane_change_cooldown = 100
        self.lane_change_cooldown_timer = 0
        self.lane_change_front_min = 18.0
        self.lane_change_rear_min = 22.0
        self.lane_change_front_ttc_min = 4.0
        self.lane_change_rear_ttc_min = 4.5
        self.lane_change_speed_margin = 0.8

        self.stop_sign_state = 'NONE'
        self.stop_sign_timer = 0
        self.stop_sign_cooldown = 0
        self.stop_sign_last_seen_frame = -1
        self.stop_sign_distance = float('inf')
        self.stop_sign_memory_frames = 30
        self.stop_sign_estimated_dt = 0.05
        self.stop_sign_stop_frames = 40

        self.safety_enabled = True
        self.safety_front_margin = 5.0
        self.safety_rear_margin = 7.0
        self.safety_ttc_critical = 1.2
        self.safety_ttc_warning = 2.5
        self.safety_min_gap = 3.0
        self.safety_target_gap = 5.0
        self.safety_brake_min = 0.35
        self.safety_brake_max = 1.0

        self.lane_change_min_speed = 5.0
        self.lane_change_acceleration = 0.35
        self.lane_change_cruise_throttle = 0.20
        self.lane_change_target_front_gap = 2.5
        self.lane_change_target_rear_gap = 3.0
        self.lane_change_target_ttc = 0.8

        self.road_analyzer = RoadAnalyzer(carla_map)
        self.overtaking_planner = OvertakingPlanner()
        self.lane_change_controller = LaneChangeController(vehicle, carla_map)

        self.overtaking_direction = None
        self.last_overtaking_reason = 'NOT_STARTED'
        self.last_lane_change_result = None
        self.last_road_info = None
        self.last_adjacent_traffic = {}

        self._current_frame_num = None
        self._target_lane_traffic_cache = None
        self._target_lane_traffic_cache_frame = None
        self._actor_cache_frame = None
        self._actor_cache = []
        self._waypoint_cache = {}

    @staticmethod
    def _normalize_angle(angle):
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    @staticmethod
    def _valid_number(value):
        try:
            return math.isfinite(float(value))
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _safe_distance(detection, default=float('inf')):
        if not isinstance(detection, dict):
            return default
        try:
            value = float(detection.get('distance', default))
            return value if math.isfinite(value) and value >= 0.0 else default
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _get_detection_class(detection):
        return str(detection.get('class', '')).lower().strip() if isinstance(detection, dict) else ''

    @staticmethod
    def _is_vehicle_detection(detection):
        return AIDriver._get_detection_class(detection) in AIDriver.VEHICLE_CLASSES

    @staticmethod
    def _is_in_ego_lane(detection):
        if not isinstance(detection, dict):
            return False
        value = detection.get('in_lane', detection.get('same_lane'))
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return math.isfinite(float(value)) and value != 0
        if isinstance(value, str):
            return value.strip().lower() in ('true', '1', 'yes', 'same', 'ego')
        return False

    @staticmethod
    def _empty_traffic():
        return {
            'observed': False,
            'front_distance': None,
            'rear_distance': None,
            'front_speed': None,
            'rear_speed': None,
            'front_relative_speed': None,
            'rear_relative_speed': None,
            'front_ttc': None,
            'rear_ttc': None
        }

    @staticmethod
    def _get_ttc(distance, closing_speed):
        if distance is None or closing_speed is None:
            return None
        if closing_speed <= 0.1:
            return float('inf')
        return max(0.0, distance / closing_speed)

    def _get_actor_list(self):
        if self._actor_cache_frame == self._current_frame_num:
            return self._actor_cache
        try:
            self._actor_cache = list(self.vehicle.get_world().get_actors().filter('vehicle.*'))
        except (RuntimeError, AttributeError):
            self._actor_cache = []
        self._actor_cache_frame = self._current_frame_num
        self._waypoint_cache = {}
        return self._actor_cache

    def _get_actor_waypoint(self, actor):
        if actor.id in self._waypoint_cache:
            return self._waypoint_cache[actor.id]
        try:
            waypoint = self.map.get_waypoint(
                actor.get_location(),
                project_to_road=True,
                lane_type=carla.LaneType.Driving
            )
        except (RuntimeError, AttributeError, TypeError):
            waypoint = None
        self._waypoint_cache[actor.id] = waypoint
        return waypoint

    def _get_actor_half_length(self, actor, direction):
        try:
            transform = actor.get_transform()
            forward = transform.get_forward_vector()
            right = transform.get_right_vector()
            extent = actor.bounding_box.extent

            forward_projection = abs(
                forward.x * direction.x +
                forward.y * direction.y +
                forward.z * direction.z
            )
            right_projection = abs(
                right.x * direction.x +
                right.y * direction.y +
                right.z * direction.z
            )

            return extent.x * forward_projection + extent.y * right_projection
        except (RuntimeError, AttributeError, TypeError):
            return 2.5

    def _get_actor_lateral_extent(self, actor, lane_right):
        try:
            transform = actor.get_transform()
            forward = transform.get_forward_vector()
            right = transform.get_right_vector()
            extent = actor.bounding_box.extent

            forward_projection = abs(
                forward.x * lane_right.x +
                forward.y * lane_right.y +
                forward.z * lane_right.z
            )
            right_projection = abs(
                right.x * lane_right.x +
                right.y * lane_right.y +
                right.z * lane_right.z
            )

            return (
                extent.x * forward_projection +
                extent.y * right_projection
            )
        except (RuntimeError, AttributeError, TypeError):
            return 1.0

    def _actor_overlaps_lane(self, actor, waypoint, margin=0.2):
        try:
            actor_location = actor.get_location()
            lane_location = waypoint.transform.location
            lane_right = waypoint.transform.get_right_vector()

            dx = actor_location.x - lane_location.x
            dy = actor_location.y - lane_location.y
            dz = actor_location.z - lane_location.z

            lateral_distance = abs(
                dx * lane_right.x +
                dy * lane_right.y +
                dz * lane_right.z
            )

            actor_extent = self._get_actor_lateral_extent(actor, lane_right)
            lane_half_width = waypoint.lane_width * 0.5

            return lateral_distance <= lane_half_width + actor_extent + margin

        except (RuntimeError, AttributeError, TypeError, ValueError):
            return True

    def _get_vehicle_gap(self, actor, direction, longitudinal_distance):
        ego_half = self._get_actor_half_length(self.vehicle, direction)
        actor_half = self._get_actor_half_length(actor, direction)
        return max(0.0, abs(longitudinal_distance) - ego_half - actor_half)

    def _get_emergency_braking_distance(self):
        speed_ms = max(0.0, self.speed / 3.6)
        return (
            speed_ms * self.reaction_time +
            speed_ms ** 2 / (2.0 * max(self.emergency_deceleration, 0.1)) +
            self.emergency_safety_margin
        )

    def _get_caution_distance(self):
        return self._get_emergency_braking_distance() * self.caution_multiplier

    def _get_following_distance(self):
        return max(self.minimum_follow_distance, self.speed / 3.6 * self.follow_time_gap)

    def _get_current_waypoint(self):
        try:
            return self.map.get_waypoint(
                self.vehicle.get_location(),
                project_to_road=True,
                lane_type=carla.LaneType.Driving
            )
        except (RuntimeError, AttributeError, TypeError):
            return None

    def _select_next_waypoint(self, waypoint, distance):
        if waypoint is None:
            return None
        try:
            candidates = waypoint.next(distance)
            if not candidates:
                return None

            forward = self.vehicle.get_transform().get_forward_vector()
            best_waypoint = None
            best_score = -float('inf')

            for candidate in candidates:
                if candidate.lane_type != carla.LaneType.Driving:
                    continue

                candidate_forward = candidate.transform.get_forward_vector()
                dot = forward.x * candidate_forward.x + forward.y * candidate_forward.y

                if dot < 0.0:
                    continue

                score = (
                    dot +
                    (0.3 if candidate.road_id == waypoint.road_id else 0.0) +
                    (0.5 if candidate.lane_id == waypoint.lane_id else 0.0)
                )

                if score > best_score:
                    best_score = score
                    best_waypoint = candidate

            return best_waypoint
        except (RuntimeError, AttributeError, TypeError):
            return None

    def _get_waypoint_steer(self):
        transform = self.vehicle.get_transform()
        current_wp = self._get_current_waypoint()

        if current_wp is None:
            return 0.0, False

        target_wp = None

        if self.lane_change_controller.state == 'LANE_FOLLOW':
            try:
                target_wp = self.lane_change_controller.get_follow_waypoint(self.waypoint_lookahead)
            except (AttributeError, RuntimeError, TypeError):
                target_wp = None

        if target_wp is None:
            target_wp = self._select_next_waypoint(current_wp, self.waypoint_lookahead)

        if target_wp is None:
            return 0.0, False

        location = transform.location
        target = target_wp.transform.location
        target_angle = math.atan2(target.y - location.y, target.x - location.x)
        forward = transform.get_forward_vector()
        forward_angle = math.atan2(forward.y, forward.x)
        angle_diff = self._normalize_angle(target_angle - forward_angle)
        steer = float(np.clip(self.steer_kp * angle_diff, -1.0, 1.0))

        return steer, current_wp.lane_type == carla.LaneType.Driving

    def _check_traffic_light(self):
        try:
            if not self.vehicle.is_at_traffic_light():
                return 'UNKNOWN'

            light = self.vehicle.get_traffic_light()

            if light is None:
                return 'UNKNOWN'

            state = light.get_state()

            if state == carla.TrafficLightState.Red:
                return 'RED'
            if state == carla.TrafficLightState.Yellow:
                return 'YELLOW'
            if state == carla.TrafficLightState.Green:
                return 'GREEN'

        except (RuntimeError, AttributeError):
            return 'UNKNOWN'

        return 'UNKNOWN'

    def _get_lane_traffic(self, target_wp):
        result = self._empty_traffic()
        result['observed'] = target_wp is not None

        if target_wp is None:
            return result

        ego_location = self.vehicle.get_location()
        lane_forward = target_wp.transform.get_forward_vector()
        ego_velocity = self.vehicle.get_velocity()

        ego_speed = (
            ego_velocity.x * lane_forward.x +
            ego_velocity.y * lane_forward.y +
            ego_velocity.z * lane_forward.z
        )

        for actor in self._get_actor_list():
            try:
                if actor.id == self.vehicle.id or not actor.is_alive:
                    continue

                actor_wp = self._get_actor_waypoint(actor)

                if (
                    actor_wp is None or
                    actor_wp.road_id != target_wp.road_id or
                    actor_wp.lane_id != target_wp.lane_id
                ):
                    continue

                if not self._actor_overlaps_lane(actor, actor_wp):
                    continue

                actor_forward = actor_wp.transform.get_forward_vector()

                direction_dot = (
                    lane_forward.x * actor_forward.x +
                    lane_forward.y * actor_forward.y +
                    lane_forward.z * actor_forward.z
                )

                if direction_dot < 0.8:
                    continue

                actor_location = actor.get_location()

                dx = actor_location.x - ego_location.x
                dy = actor_location.y - ego_location.y
                dz = actor_location.z - ego_location.z

                longitudinal = (
                    dx * lane_forward.x +
                    dy * lane_forward.y +
                    dz * lane_forward.z
                )

                gap = self._get_vehicle_gap(actor, lane_forward, longitudinal)

                actor_velocity = actor.get_velocity()

                actor_speed = (
                    actor_velocity.x * lane_forward.x +
                    actor_velocity.y * lane_forward.y +
                    actor_velocity.z * lane_forward.z
                )

                if longitudinal > 0.0:
                    if result['front_distance'] is None or gap < result['front_distance']:
                        closing = ego_speed - actor_speed

                        result.update({
                            'front_distance': float(gap),
                            'front_speed': float(actor_speed),
                            'front_relative_speed': float(closing),
                            'front_ttc': self._get_ttc(gap, closing)
                        })

                else:
                    if result['rear_distance'] is None or gap < result['rear_distance']:
                        closing = actor_speed - ego_speed

                        result.update({
                            'rear_distance': float(gap),
                            'rear_speed': float(actor_speed),
                            'rear_relative_speed': float(closing),
                            'rear_ttc': self._get_ttc(gap, closing)
                        })

            except (RuntimeError, AttributeError, TypeError, ValueError):
                continue

        return result

    def _get_adjacent_traffic(self, road_info):
        result = {
            'left': self._empty_traffic(),
            'right': self._empty_traffic()
        }

        if not isinstance(road_info, dict):
            return result

        for side in ('left', 'right'):
            result[side] = self._get_lane_traffic(road_info.get(side + '_lane'))

        return result

    def _get_current_lane_traffic(self, road_info):
        if not isinstance(road_info, dict):
            return self._empty_traffic()

        current_wp = road_info.get('current_waypoint') or self._get_current_waypoint()
        return self._get_lane_traffic(current_wp)

    def _is_adjacent_lane_safe(self, side, adjacent_traffic):
        if side not in ('left', 'right') or not isinstance(adjacent_traffic, dict):
            return False

        traffic = adjacent_traffic.get(side)

        if not isinstance(traffic, dict) or not traffic.get('observed', False):
            return False

        speed_margin = max(0.0, self.speed / 3.6 * self.lane_change_speed_margin)
        front_min = self.lane_change_front_min + speed_margin
        rear_min = self.lane_change_rear_min + speed_margin

        front_distance = traffic.get('front_distance')
        rear_distance = traffic.get('rear_distance')
        front_ttc = traffic.get('front_ttc')
        rear_ttc = traffic.get('rear_ttc')

        if front_distance is not None and front_distance < front_min:
            return False
        if rear_distance is not None and rear_distance < rear_min:
            return False
        if front_ttc is not None and front_ttc < self.lane_change_front_ttc_min:
            return False
        if rear_ttc is not None and rear_ttc < self.lane_change_rear_ttc_min:
            return False

        return True

    def _get_target_lane_traffic(self):
        if (
            self._target_lane_traffic_cache_frame == self._current_frame_num and
            self._target_lane_traffic_cache is not None
        ):
            return self._target_lane_traffic_cache

        controller = self.lane_change_controller
        target_path = getattr(controller, 'target_path', None)

        if not target_path:
            result = self._empty_traffic()
        else:
            index = min(max(int(getattr(controller, 'progress_index', 0)), 0), len(target_path) - 1)
            target_wp = target_path[index]
            result = self._get_lane_traffic(target_wp)

        self._target_lane_traffic_cache = result
        self._target_lane_traffic_cache_frame = self._current_frame_num

        return result

    def _is_target_lane_critical(self):
        traffic = self._get_target_lane_traffic()

        if not traffic.get('observed', False):
            return True

        front_distance = traffic.get('front_distance')
        rear_distance = traffic.get('rear_distance')
        front_ttc = traffic.get('front_ttc')
        rear_ttc = traffic.get('rear_ttc')

        if front_distance is not None and front_distance < self.lane_change_target_front_gap:
            return True
        if front_ttc is not None and front_ttc < self.lane_change_target_ttc:
            return True
        if (
            rear_distance is not None and
            rear_distance < self.lane_change_target_rear_gap and
            rear_ttc is not None and
            rear_ttc < self.lane_change_target_ttc
        ):
            return True
        if rear_ttc is not None and rear_ttc < self.lane_change_target_ttc:
            return True

        return False

    def _analyze_overtaking(self, detections):
        try:
            road_info = self.road_analyzer.analyze(self.vehicle.get_location())
        except (RuntimeError, AttributeError, TypeError):
            road_info = None

        if road_info is None:
            return {
                'road_info': None,
                'adjacent_traffic': {
                    'left': self._empty_traffic(),
                    'right': self._empty_traffic()
                },
                'decision': {
                    'should_overtake': False,
                    'direction': None,
                    'target_lane': None,
                    'reason': 'ROAD_INFORMATION_UNAVAILABLE'
                }
            }

        adjacent_traffic = self._get_adjacent_traffic(road_info)

        try:
            decision = self.overtaking_planner.plan(
                road_info,
                detections,
                adjacent_traffic,
                ego_speed=self.speed,
                target_speed=self.target_speed,
                detections_are_same_lane=True
            )
        except (RuntimeError, AttributeError, TypeError, ValueError):
            decision = None

        if not isinstance(decision, dict):
            decision = {
                'should_overtake': False,
                'direction': None,
                'target_lane': None,
                'reason': 'INVALID_OVERTAKING_DECISION'
            }

        return {
            'road_info': road_info,
            'adjacent_traffic': adjacent_traffic,
            'decision': decision
        }

    def _start_overtaking(self, decision, adjacent_traffic):
        controller = self.lane_change_controller

        if controller.active or self.lane_change_cooldown_timer > 0:
            return False
        if self.speed < self.lane_change_min_speed:
            return False
        if not isinstance(decision, dict) or not decision.get('should_overtake', False):
            return False

        target_lane = decision.get('target_lane')
        direction = decision.get('direction')

        if target_lane is None or direction not in ('left', 'right'):
            return False
        if not self._is_adjacent_lane_safe(direction, adjacent_traffic):
            return False

        try:
            started = controller.start(target_lane)
        except (RuntimeError, AttributeError, TypeError):
            started = False

        if not started:
            return False

        self.overtaking_direction = direction
        self.last_overtaking_reason = decision.get('reason', 'SAFE_TO_OVERTAKE')
        self._target_lane_traffic_cache = None
        self._target_lane_traffic_cache_frame = None

        return True

    def _update_lane_change(self):
        controller = self.lane_change_controller

        if not controller.active:
            return None

        try:
            result = controller.run_step()
        except (RuntimeError, AttributeError, TypeError):
            return None

        if not isinstance(result, dict):
            return None

        if result.get('completed', False):
            self.last_overtaking_reason = 'LANE_CHANGE_COMPLETED'
            self.lane_change_cooldown_timer = self.lane_change_cooldown
            self.overtaking_direction = None
            self._target_lane_traffic_cache = None
            self._target_lane_traffic_cache_frame = None

        elif result.get('failed', False):
            self.last_overtaking_reason = 'LANE_CHANGE_FAILED'
            self.lane_change_cooldown_timer = self.lane_change_cooldown
            self.overtaking_direction = None
            self._target_lane_traffic_cache = None
            self._target_lane_traffic_cache_frame = None

        return result

    def _update_stop_sign(self, stop_signs, frame_num):
        if self.stop_sign_cooldown > 0:
            self.stop_sign_cooldown -= 1

        if self.stop_sign_state == 'STOPPING':
            if self.speed < 2.0:
                self.stop_sign_state = 'STOPPED'
                self.stop_sign_timer = self.stop_sign_stop_frames
            return

        if self.stop_sign_state == 'STOPPED':
            self.stop_sign_timer = max(0, self.stop_sign_timer - 1)
            if self.stop_sign_timer == 0:
                self.stop_sign_state = 'DONE'
            return

        if self.stop_sign_state == 'DONE':
            if self.speed > 15.0:
                self.stop_sign_state = 'NONE'
                self.stop_sign_cooldown = 100
                self.stop_sign_distance = float('inf')
                self.stop_sign_last_seen_frame = -1
            return

        valid_signs = [d for d in stop_signs if self._safe_distance(d) < 25.0]

        if valid_signs:
            closest = min(valid_signs, key=self._safe_distance)
            self.stop_sign_distance = self._safe_distance(closest)
            self.stop_sign_last_seen_frame = frame_num

            if self.stop_sign_state == 'NONE' and self.stop_sign_cooldown == 0:
                self.stop_sign_state = 'APPROACHING'

        elif self.stop_sign_state == 'APPROACHING':
            elapsed = frame_num - self.stop_sign_last_seen_frame

            if elapsed <= self.stop_sign_memory_frames:
                self.stop_sign_distance = max(
                    0.0,
                    self.stop_sign_distance - self.speed / 3.6 * self.stop_sign_estimated_dt
                )
            else:
                self.stop_sign_state = 'NONE'
                self.stop_sign_distance = float('inf')
                self.stop_sign_last_seen_frame = -1
                return

        if self.stop_sign_state == 'APPROACHING' and self.stop_sign_distance <= 12.0:
            self.stop_sign_state = 'STOPPING'

    def _get_closest(self, detections):
        valid = [
            d for d in detections
            if isinstance(d, dict) and math.isfinite(self._safe_distance(d))
        ]
        return min(valid, key=self._safe_distance, default=None)

    def _get_same_object_danger(self, dangers, vehicles):
        vehicle_dangers = [d for d in dangers if self._is_vehicle_detection(d)]

        if not vehicle_dangers or not vehicles:
            return None

        closest_vehicle = self._get_closest(vehicles)

        if closest_vehicle is None:
            return None

        vehicle_distance = self._safe_distance(closest_vehicle)

        matching = [
            d for d in vehicle_dangers
            if abs(self._safe_distance(d) - vehicle_distance) <= max(5.0, vehicle_distance * 0.25)
        ]

        return min(matching, key=self._safe_distance, default=None)

    def _make_driving_decision(self, light_state, dangers, warnings, overtaking_decision, same_lane_vehicles, adjacent_traffic):
        emergency_distance = self._get_emergency_braking_distance()
        caution_distance = self._get_caution_distance()

        closest_danger = self._get_closest(dangers)
        closest_warning = self._get_closest(warnings)
        closest_vehicle = self._get_closest(same_lane_vehicles)

        danger_distance = self._safe_distance(closest_danger)
        warning_distance = self._safe_distance(closest_warning)
        vehicle_distance = self._safe_distance(closest_vehicle)

        lane_change_active = self.lane_change_controller.active

        if light_state == 'RED':
            return 0.0, 1.0, 'STOP — RED LIGHT', 'Red traffic light — stopping'

        if self.stop_sign_state == 'STOPPING':
            return 0.0, 0.8, 'STOP — SIGN', 'Stopping at stop sign'

        if self.stop_sign_state == 'STOPPED':
            return 0.0, 1.0, 'WAITING — STOP', f'Stopped at sign ({self.stop_sign_timer / 20.0:.1f}s left)'

        if self.stop_sign_state == 'APPROACHING':
            return 0.0, 0.4, 'SLOW — STOP SIGN', f'Stop sign ahead at approximately {self.stop_sign_distance:.1f}m'

        if light_state == 'YELLOW':
            return 0.0, 0.4, 'SLOW — YELLOW', 'Yellow light — slowing down'

        if lane_change_active:
            target_traffic = self._get_target_lane_traffic()

            if not target_traffic.get('observed', False):
                return 0.0, 0.5, 'OVERTAKING — CHECK TARGET LANE', 'Target lane traffic information unavailable'

            if self._is_target_lane_critical():
                return 0.0, 1.0, 'OVERTAKING — TARGET LANE EMERGENCY', 'Critical collision risk in target lane'

            if danger_distance < 4.0:
                return 0.0, 1.0, 'OVERTAKING — EMERGENCY', f'Immediate obstacle risk at {danger_distance:.1f}m'

            if self.speed < 5.0:
                return self.lane_change_acceleration, 0.0, 'OVERTAKING — ACCELERATING', 'Accelerating during lane change'

            if self.speed < self.target_speed - 5.0:
                return 0.30, 0.0, 'OVERTAKING — ACCELERATING', 'Building speed during lane change'

            return self.lane_change_cruise_throttle, 0.0, 'OVERTAKING — LANE CHANGE', f'Continuing lane change to {self.overtaking_direction}'

        if danger_distance <= emergency_distance:
            return 0.0, 1.0, 'EMERGENCY BRAKE', f'Emergency obstacle: {self._get_detection_class(closest_danger)} at {danger_distance:.1f}m'

        if vehicle_distance <= emergency_distance:
            if closest_vehicle is not None and self._start_overtaking(overtaking_decision, adjacent_traffic):
                return 0.30, 0.0, 'OVERTAKING — START', f'Overtaking vehicle on {self.overtaking_direction}'

            return 0.0, 1.0, 'EMERGENCY BRAKE', f'Vehicle too close in ego lane at {vehicle_distance:.1f}m'

        if closest_danger is not None and danger_distance <= emergency_distance * self.danger_multiplier:
            return 0.0, 0.4, 'CAUTION — REDUCE SPEED', f'Danger approaching at {danger_distance:.1f}m'

        if closest_warning is not None and warning_distance <= caution_distance:
            if warning_distance < 10.0:
                return 0.0, 0.5, 'BRAKE — WARNING', f'Warning: {self._get_detection_class(closest_warning)} at {warning_distance:.1f}m'
            return 0.08, 0.0, 'CAUTION', f'Warning: {self._get_detection_class(closest_warning)} at {warning_distance:.1f}m'

        if closest_vehicle is not None and self._start_overtaking(overtaking_decision, adjacent_traffic):
            return 0.30, 0.0, 'OVERTAKING — START', f'Overtaking {self._get_detection_class(closest_vehicle)} on {self.overtaking_direction}'

        if closest_vehicle is not None:
            desired_distance = self._get_following_distance()

            if vehicle_distance <= desired_distance:
                return 0.0, 0.35, 'FOLLOWING — REDUCE SPEED', f'Vehicle ahead at {vehicle_distance:.1f}m'

            if vehicle_distance <= desired_distance * 1.5:
                return 0.12, 0.0, 'FOLLOWING — MAINTAIN', f'Maintaining safe gap at {vehicle_distance:.1f}m'

            if self.speed < self.target_speed - 5.0:
                return 0.30, 0.0, 'FOLLOWING — ACCELERATE', f'Following vehicle at {vehicle_distance:.1f}m'

            return 0.15, 0.0, 'FOLLOWING — CAUTION', f'Vehicle ahead at {vehicle_distance:.1f}m'

        if self.stop_sign_state == 'DONE':
            return 0.30, 0.0, 'DEPARTING STOP', 'Leaving stop sign — accelerating'

        if closest_warning is not None:
            return 0.12, 0.0, 'CAUTION', f'Monitoring warning at {warning_distance:.1f}m'

        if self.speed < self.target_speed - 5.0:
            return 0.35, 0.0, 'ACCELERATE', f'Speeding up to {self.target_speed:.0f} km/h'

        if self.speed > self.target_speed + 5.0:
            return 0.0, 0.2, 'SLOW DOWN', f'Reducing from {self.speed:.0f} km/h'

        return 0.25, 0.0, 'CRUISE', f'Cruising at {self.speed:.0f} km/h — road clear'

    def _apply_safety_override(self, throttle, brake, action, reason, current_traffic):
        if not self.safety_enabled:
            return throttle, brake, action, reason, False

        critical = False
        safety_reason = None
        lane_change_active = self.lane_change_controller.active

        if isinstance(current_traffic, dict) and current_traffic.get('observed', False):
            front_distance = current_traffic.get('front_distance')
            front_ttc = current_traffic.get('front_ttc')

            if lane_change_active:
                if front_distance is not None and front_distance < 3.0:
                    critical = True
                    safety_reason = f'Immediate source lane collision risk: {front_distance:.1f}m'
                if front_ttc is not None and front_ttc < 0.7:
                    critical = True
                    safety_reason = f'Immediate source lane TTC: {front_ttc:.2f}s'
            else:
                emergency_distance = self._get_emergency_braking_distance()

                if front_distance is not None and front_distance <= max(self.safety_min_gap, emergency_distance):
                    critical = True
                    safety_reason = f'CARLA front collision risk at {front_distance:.1f}m'
                elif front_distance is not None and front_distance <= emergency_distance * self.caution_multiplier:
                    brake = max(brake, self.safety_brake_min)
                    throttle = 0.0
                    action = 'SAFETY — FRONT GAP'
                    safety_reason = f'Insufficient front gap: {front_distance:.1f}m'

                if front_ttc is not None and front_ttc < self.safety_ttc_critical:
                    critical = True
                    safety_reason = f'Critical front TTC: {front_ttc:.2f}s'
                elif front_ttc is not None and front_ttc < self.safety_ttc_warning:
                    brake = max(brake, 0.4)
                    throttle = 0.0
                    action = 'SAFETY — FRONT TTC'
                    safety_reason = f'Low front TTC: {front_ttc:.2f}s'

        if lane_change_active:
            target_traffic = self._get_target_lane_traffic()

            if not target_traffic.get('observed', False):
                critical = True
                safety_reason = 'Target lane traffic information unavailable'
            else:
                front_distance = target_traffic.get('front_distance')
                rear_distance = target_traffic.get('rear_distance')
                front_ttc = target_traffic.get('front_ttc')
                rear_ttc = target_traffic.get('rear_ttc')

                if front_distance is not None and front_distance < self.lane_change_target_front_gap:
                    critical = True
                    safety_reason = f'Target lane front gap: {front_distance:.1f}m'

                if (
                    rear_distance is not None and
                    rear_distance < self.lane_change_target_rear_gap and
                    rear_ttc is not None and
                    rear_ttc < self.lane_change_target_ttc
                ):
                    critical = True
                    safety_reason = f'Target lane rear gap: {rear_distance:.1f}m'

                if front_ttc is not None and front_ttc < self.lane_change_target_ttc:
                    critical = True
                    safety_reason = f'Target lane front TTC: {front_ttc:.2f}s'

                if rear_ttc is not None and rear_ttc < self.lane_change_target_ttc:
                    critical = True
                    safety_reason = f'Target lane rear TTC: {rear_ttc:.2f}s'

        if critical:
            return 0.0, self.safety_brake_max, 'SAFETY — EMERGENCY BRAKE', safety_reason or 'Critical collision risk', True

        if safety_reason is not None:
            return 0.0, brake, action, safety_reason, False

        return throttle, brake, action, reason, False

    def _get_lane_change_steer(self, wp_steer, lane_change_result, lane_steer, lanes_found):
        if isinstance(lane_change_result, dict) and self._valid_number(lane_change_result.get('steer')):
            return float(np.clip(float(lane_change_result['steer']), -1.0, 1.0))

        if self.lane_change_controller.state == 'LANE_FOLLOW' and lanes_found >= 3 and abs(lane_steer) < 0.5:
            return float(np.clip(0.85 * wp_steer + 0.15 * lane_steer, -1.0, 1.0))

        return wp_steer

    def _smooth_steer(self, steer, lane_change_active):
        alpha = 0.9 if lane_change_active else 0.85
        steer = float(np.clip(steer, -1.0, 1.0))
        smoothed = alpha * steer + (1.0 - alpha) * self.prev_steer
        self.prev_steer = float(np.clip(smoothed, -1.0, 1.0))
        return self.prev_steer

    @staticmethod
    def _get_urgency(action):
        if 'EMERGENCY' in action or 'RED' in action or 'CRITICAL' in action:
            return 'CRITICAL'
        if 'BRAKE' in action or 'STOP' in action:
            return 'HIGH'
        if 'SAFETY' in action or 'CAUTION' in action or 'YELLOW' in action or 'SLOW' in action:
            return 'MEDIUM'
        if 'OVERTAKING' in action:
            return 'MEDIUM'
        return 'LOW'

    def decide_and_control(self, detections: list[dict], lane_data: dict, frame_num: int) -> dict:
        self._current_frame_num = frame_num
        self._target_lane_traffic_cache = None
        self._target_lane_traffic_cache_frame = None

        velocity = self.vehicle.get_velocity()
        self.speed = 3.6 * math.sqrt(
            velocity.x ** 2 +
            velocity.y ** 2 +
            velocity.z ** 2
        )

        if self.lane_change_cooldown_timer > 0:
            self.lane_change_cooldown_timer -= 1

        detections = [d for d in (detections or []) if isinstance(d, dict)]
        lane_data = lane_data if isinstance(lane_data, dict) else {}

        wp_steer, on_road = self._get_waypoint_steer()

        try:
            lane_steer = float(np.clip(float(lane_data.get('steering', 0.0)), -1.0, 1.0))
            if not math.isfinite(lane_steer):
                lane_steer = 0.0
        except (TypeError, ValueError):
            lane_steer = 0.0

        try:
            lanes_found = max(0, int(lane_data.get('lines_found', 0) or 0))
        except (TypeError, ValueError):
            lanes_found = 0

        light_state = self._check_traffic_light()

        same_lane_detections = [d for d in detections if self._is_in_ego_lane(d)]
        other_lane_detections = [d for d in detections if not self._is_in_ego_lane(d)]

        same_lane_dangers = [
            d for d in same_lane_detections
            if str(d.get('threat', '')).upper() == 'DANGER'
        ]
        same_lane_warnings = [
            d for d in same_lane_detections
            if str(d.get('threat', '')).upper() == 'WARNING'
        ]
        same_lane_vehicles = [
            d for d in same_lane_detections
            if self._is_vehicle_detection(d)
        ]

        same_lane_dangers.sort(key=self._safe_distance)
        same_lane_warnings.sort(key=self._safe_distance)
        same_lane_vehicles.sort(key=self._safe_distance)

        stop_signs = [
            d for d in detections
            if self._get_detection_class(d) == 'stop sign'
        ]
        self._update_stop_sign(stop_signs, frame_num)

        overtaking_info = self._analyze_overtaking(same_lane_detections)
        road_info = overtaking_info['road_info']
        adjacent_traffic = overtaking_info['adjacent_traffic']
        overtaking_decision = overtaking_info['decision']

        self.last_road_info = road_info
        self.last_adjacent_traffic = adjacent_traffic

        current_traffic = self._get_current_lane_traffic(road_info)

        lane_change_result = self._update_lane_change()
        self.last_lane_change_result = lane_change_result

        throttle, brake, action, reason = self._make_driving_decision(
            light_state,
            same_lane_dangers,
            same_lane_warnings,
            overtaking_decision,
            same_lane_vehicles,
            adjacent_traffic
        )

        throttle, brake, action, reason, safety_override = self._apply_safety_override(
            throttle,
            brake,
            action,
            reason,
            current_traffic
        )

        steer = self._get_lane_change_steer(
            wp_steer,
            lane_change_result,
            lane_steer,
            lanes_found
        )

        lane_change_active = self.lane_change_controller.active or lane_change_result is not None
        steer = self._smooth_steer(steer, lane_change_active)

        control = carla.VehicleControl()
        control.throttle = float(np.clip(throttle, 0.0, 1.0))
        control.brake = float(np.clip(brake, 0.0, 1.0))
        control.steer = float(np.clip(steer, -1.0, 1.0))
        control.hand_brake = False

        if control.brake > 0.0:
            control.throttle = 0.0

        self.vehicle.apply_control(control)

        controller = self.lane_change_controller

        decision = {
            'action': action,
            'reason': reason,
            'urgency': self._get_urgency(action),
            'speed': self.speed,
            'throttle': control.throttle,
            'brake': control.brake,
            'steer': control.steer,
            'steering_cv': lane_steer,
            'wp_steer': wp_steer,
            'n_detections': len(detections),
            'n_dangers': len(same_lane_dangers),
            'n_warnings': len(same_lane_warnings),
            'n_same_lane_vehicles': len(same_lane_vehicles),
            'n_other_lane_detections': len(other_lane_detections),
            'light': light_state,
            'on_road': on_road,
            'stop_sign': self.stop_sign_state,
            'stop_sign_distance': self.stop_sign_distance if math.isfinite(self.stop_sign_distance) else None,
            'frame': frame_num,
            'road_type': road_info.get('road_type', 'UNKNOWN') if isinstance(road_info, dict) else 'UNKNOWN',
            'current_lane_id': road_info.get('current_lane_id') if isinstance(road_info, dict) else None,
            'overtaking_active': controller.active,
            'overtaking_direction': self.overtaking_direction,
            'overtaking_reason': overtaking_decision.get('reason', 'UNKNOWN') if isinstance(overtaking_decision, dict) else 'UNKNOWN',
            'adjacent_traffic': adjacent_traffic,
            'current_lane_traffic': current_traffic,
            'target_lane_traffic': self._get_target_lane_traffic() if controller.active else None,
            'lane_change_target_observed': bool(
                adjacent_traffic.get(self.overtaking_direction, {}).get('observed', False)
            ) if self.overtaking_direction in ('left', 'right') else False,
            'lane_change_state': controller.state,
            'target_lane_id': controller.target_lane_id,
            'lane_follow_active': controller.state == 'LANE_FOLLOW',
            'lane_change_completed': lane_change_result.get('completed', False) if isinstance(lane_change_result, dict) else False,
            'lane_change_failed': lane_change_result.get('failed', False) if isinstance(lane_change_result, dict) else False,
            'safety_override': safety_override,
            'emergency_distance': self._get_emergency_braking_distance(),
            'caution_distance': self._get_caution_distance(),
            'closest_danger_distance': self._safe_distance(same_lane_dangers[0]) if same_lane_dangers else None,
            'closest_vehicle_distance': self._safe_distance(same_lane_vehicles[0]) if same_lane_vehicles else None,
            'closest_warning_distance': self._safe_distance(same_lane_warnings[0]) if same_lane_warnings else None
        }

        self.log.append(decision)

        if frame_num % 20 == 0:
            target_traffic = decision['target_lane_traffic']
            target_front = target_traffic.get('front_distance') if isinstance(target_traffic, dict) else None
            target_rear = target_traffic.get('rear_distance') if isinstance(target_traffic, dict) else None

            print(
                f"[CONTROL] Frame={frame_num} | "
                f"Speed={self.speed:.1f} km/h | "
                f"Throttle={control.throttle:.2f} | "
                f"Brake={control.brake:.2f} | "
                f"Steer={control.steer:.2f} | "
                f"LaneChange={controller.active} | "
                f"State={controller.state} | "
                f"TargetFront={target_front} | "
                f"TargetRear={target_rear} | "
                f"Safety={safety_override} | "
                f"Action={action} | Reason={reason}"
            )

        return decision
