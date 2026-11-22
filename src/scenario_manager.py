
import math
import random
import carla


class ScenarioManager:

    SCENARIOS = {
        "slow_vehicle_ahead": {
            "description": "Slow vehicles ahead of the ego vehicle.",
            "vehicles": [{"position": "front", "distance": 15.0, "speed": 15.0}],
        },
        "safe_overtake": {
            "description": "A slow vehicle ahead with an available adjacent lane.",
            "vehicles": [{"position": "front", "distance": 18.0, "speed": 15.0}],
        },
        "rear_vehicle_blocking": {
            "description": "A slow vehicle ahead and a fast vehicle approaching from behind.",
            "vehicles": [
                {"position": "front", "distance": 18.0, "speed": 15.0},
                {"position": "adjacent_rear", "distance": 20.0, "speed": 35.0},
            ],
        },
        "both_lanes_blocked": {
            "description": "Vehicles occupy the ego lane and adjacent lane.",
            "vehicles": [
                {"position": "front", "distance": 18.0, "speed": 15.0},
                {"position": "adjacent_front", "distance": 24.0, "speed": 25.0},
                {"position": "adjacent_rear", "distance": 20.0, "speed": 30.0},
            ],
        },
        "mixed": {
            "description": "Mixed traffic conditions around the ego vehicle.",
            "vehicles": [
                {"position": "front", "distance": 18.0, "speed": 15.0},
                {"position": "adjacent_front", "distance": 28.0, "speed": 25.0},
                {"position": "adjacent_rear", "distance": 22.0, "speed": 30.0},
            ],
        },
    }

    BACKGROUND_VEHICLE_COUNTS = {
        "slow_vehicle_ahead": 100,
        "safe_overtake": 110,
        "rear_vehicle_blocking": 120,
        "both_lanes_blocked": 130,
        "mixed": 150,
    }

    EGO_SPAWN_CLEARANCE = 12.0
    MIN_SAME_LANE_GAP = 12.0
    MIN_OTHER_LANE_DISTANCE = 6.0
    MIN_MAP_DISTRIBUTION_DISTANCE = 8.0

    SPAWN_OFFSETS = [0.0, 4.0, -4.0, 7.0, -7.0]

    BACKGROUND_MIN_SPEED = 25.0
    BACKGROUND_MAX_SPEED = 45.0

    DEFAULT_TM_PORT = 8000
    NPC_LANE_CHANGE_PERCENTAGE = 0.0
    TM_LEADING_DISTANCE = 5.0

    EXCLUDED_BLUEPRINTS = (
        "isetta", "motorcycle", "motorbike", "harley", "kawasaki",
        "vespa", "yamaha", "bicycle", "crossbike", "diamondback",
    )

    def __init__(self, client: carla.Client, world: carla.World, ego_vehicle: carla.Vehicle, traffic_manager_port: int = 8000, seed: int = 42):
        self.client = client
        self.world = world
        self.ego_vehicle = ego_vehicle
        self.carla_map = self.world.get_map()
        self.traffic_manager_port = traffic_manager_port
        self.traffic_manager = self.client.get_trafficmanager(traffic_manager_port)
        self.random = random.Random(seed)
        self.spawned_vehicles = []
        self.scenario_name = None
        self.blueprint_library = self.world.get_blueprint_library()
        self.vehicle_blueprints = self._get_vehicle_blueprints()

        world_settings = self.world.get_settings()
        self.synchronous_mode = bool(world_settings.synchronous_mode)

        self.traffic_manager.set_synchronous_mode(self.synchronous_mode)
        self.traffic_manager.set_global_distance_to_leading_vehicle(self.TM_LEADING_DISTANCE)

        print("[ScenarioManager] Initialized.")
        print(f"[ScenarioManager] Vehicle blueprints: {len(self.vehicle_blueprints)}")
        print(f"[ScenarioManager] Traffic Manager port: {traffic_manager_port}")
        print(f"[ScenarioManager] World synchronous mode: {self.synchronous_mode}")
        print(f"[ScenarioManager] NPC lane changing: {self.NPC_LANE_CHANGE_PERCENTAGE}%")

    def _get_vehicle_blueprints(self):
        blueprints = self.blueprint_library.filter("vehicle.*")
        result = [bp for bp in blueprints if not any(name in bp.id.lower() for name in self.EXCLUDED_BLUEPRINTS)]

        if not result:
            raise RuntimeError("No suitable vehicle blueprints found.")

        print(f"[ScenarioManager] Available vehicle blueprints: {len(result)}")
        print(f"[ScenarioManager] Excluded vehicle blueprints: {len(blueprints) - len(result)}")
        return result

    def _random_vehicle_blueprint(self):
        selected = self.random.choice(self.vehicle_blueprints)
        bp = self.blueprint_library.find(selected.id)

        if bp.has_attribute("color"):
            colors = bp.get_attribute("color").recommended_values
            if colors:
                bp.set_attribute("color", self.random.choice(colors))

        return bp

    @staticmethod
    def _distance(location_a, location_b):
        dx = location_a.x - location_b.x
        dy = location_a.y - location_b.y
        dz = location_a.z - location_b.z
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    @staticmethod
    def _direction_dot(wp_a, wp_b):
        a = wp_a.transform.get_forward_vector()
        b = wp_b.transform.get_forward_vector()
        return a.x * b.x + a.y * b.y + a.z * b.z

    def _get_ego_waypoint(self):
        waypoint = self.carla_map.get_waypoint(self.ego_vehicle.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving)

        if waypoint is None:
            raise RuntimeError("Cannot find a driving waypoint for the ego vehicle.")

        return waypoint

    def _is_same_direction(self, wp_a, wp_b, threshold=0.8):
        return self._direction_dot(wp_a, wp_b) >= threshold

    def _get_adjacent_lane(self, ego_waypoint):
        for side, waypoint in (("left", ego_waypoint.get_left_lane()), ("right", ego_waypoint.get_right_lane())):
            if waypoint is None or waypoint.lane_type != carla.LaneType.Driving:
                continue
            if self._is_same_direction(ego_waypoint, waypoint):
                return side, waypoint

        return None, None

    def _advance_waypoint(self, waypoint, distance):
        if waypoint is None:
            return None

        if abs(distance) < 1e-6:
            return waypoint

        candidates = waypoint.next(distance) if distance > 0 else waypoint.previous(abs(distance))

        if not candidates:
            return None

        same_lane = [candidate for candidate in candidates if candidate.road_id == waypoint.road_id and candidate.lane_id == waypoint.lane_id]

        if same_lane:
            candidates = same_lane

        return max(candidates, key=lambda candidate: self._direction_dot(waypoint, candidate))

    def _get_position_waypoint(self, ego_waypoint, position, distance, adjacent_waypoint=None):
        if position == "front":
            return self._advance_waypoint(ego_waypoint, distance)

        if position == "rear":
            return self._advance_waypoint(ego_waypoint, -distance)

        if adjacent_waypoint is None:
            return None

        if position == "adjacent_front":
            return self._advance_waypoint(adjacent_waypoint, distance)

        if position == "adjacent_rear":
            return self._advance_waypoint(adjacent_waypoint, -distance)

        return None

    def _set_npc_speed(self, vehicle, target_speed_kmh):
        speed_limit = vehicle.get_speed_limit()
        if speed_limit <= 0:
            speed_limit = 50.0

        target_speed_kmh = max(0.0, min(float(target_speed_kmh), 80.0))
        percentage = ((speed_limit - target_speed_kmh) / speed_limit) * 100.0
        percentage = max(-100.0, min(100.0, percentage))

        self.traffic_manager.vehicle_percentage_speed_difference(vehicle, percentage)

    def _activate_npc(self, vehicle, target_speed_kmh, background=False):
        vehicle.set_autopilot(True, self.traffic_manager_port)
        self._set_npc_speed(vehicle, target_speed_kmh)
        self.traffic_manager.auto_lane_change(vehicle, False)
        self.traffic_manager.random_left_lanechange_percentage(vehicle, self.NPC_LANE_CHANGE_PERCENTAGE)
        self.traffic_manager.random_right_lanechange_percentage(vehicle, self.NPC_LANE_CHANGE_PERCENTAGE)

        try:
            self.traffic_manager.distance_to_leading_vehicle(vehicle, self.TM_LEADING_DISTANCE)
        except (AttributeError, RuntimeError):
            pass

    def _get_world_vehicles(self):
        try:
            return list(self.world.get_actors().filter("vehicle.*"))
        except RuntimeError:
            return []

    def _is_scenario_spawn_clear(self, candidate_waypoint):
        candidate_location = candidate_waypoint.transform.location

        if self._distance(candidate_location, self.ego_vehicle.get_location()) < 10.0:
            return False

        for vehicle in self._get_world_vehicles():
            if not vehicle.is_alive or vehicle.id == self.ego_vehicle.id:
                continue
            if self._distance(candidate_location, vehicle.get_location()) < 8.0:
                return False

        return True

    def _is_background_spawn_clear(self, transform, candidate_waypoint):
        candidate_location = transform.location

        if self._distance(candidate_location, self.ego_vehicle.get_location()) < self.EGO_SPAWN_CLEARANCE:
            return False

        for vehicle in self._get_world_vehicles():
            if not vehicle.is_alive or vehicle.id == self.ego_vehicle.id:
                continue

            npc_location = vehicle.get_location()
            npc_waypoint = self.carla_map.get_waypoint(npc_location, project_to_road=True, lane_type=carla.LaneType.Driving)

            if npc_waypoint is None:
                if self._distance(candidate_location, npc_location) < self.MIN_OTHER_LANE_DISTANCE:
                    return False
                continue

            same_lane = candidate_waypoint.road_id == npc_waypoint.road_id and candidate_waypoint.lane_id == npc_waypoint.lane_id

            if same_lane:
                gap = abs(candidate_waypoint.s - npc_waypoint.s)
                if gap < self.MIN_SAME_LANE_GAP:
                    return False
            elif self._distance(candidate_location, npc_location) < self.MIN_OTHER_LANE_DISTANCE:
                return False

        return True

    def _spawn_npc(self, waypoint, target_speed_kmh):
        if waypoint is None:
            return None

        for offset in self.SPAWN_OFFSETS:
            candidate = self._advance_waypoint(waypoint, offset)

            if candidate is None or not self._is_scenario_spawn_clear(candidate):
                continue

            transform = candidate.transform
            transform.location.z += 0.3
            vehicle = self.world.try_spawn_actor(self._random_vehicle_blueprint(), transform)

            if vehicle is None:
                continue

            try:
                self._activate_npc(vehicle, target_speed_kmh, background=False)
                self.spawned_vehicles.append(vehicle)
                print(f"[ScenarioManager] Scenario NPC | {vehicle.type_id} | road={candidate.road_id} | lane={candidate.lane_id} | target={target_speed_kmh:.1f} km/h")
                return vehicle
            except Exception:
                vehicle.destroy()
                raise

        print(f"[ScenarioManager] WARNING: Scenario NPC spawn failed | lane={waypoint.lane_id}")
        return None

    def _select_distributed_spawn_points(self, spawn_points, number_of_vehicles):
        candidates = list(spawn_points)
        self.random.shuffle(candidates)
        selected = []
        remaining = []

        for transform in candidates:
            if len(selected) >= number_of_vehicles:
                break

            if all(self._distance(transform.location, chosen.location) >= self.MIN_MAP_DISTRIBUTION_DISTANCE for chosen in selected):
                selected.append(transform)
            else:
                remaining.append(transform)

        if len(selected) < number_of_vehicles:
            for transform in remaining:
                if len(selected) >= number_of_vehicles:
                    break
                selected.append(transform)

        return selected

    def _spawn_map_traffic(self, number_of_vehicles):
        spawn_points = list(self.carla_map.get_spawn_points())

        if not spawn_points:
            print("[ScenarioManager] WARNING: No map spawn points available.")
            return 0, 0

        selected_points = self._select_distributed_spawn_points(spawn_points, number_of_vehicles)
        spawned = 0
        failed = 0
        attempts = 0
        reasons = {"invalid_waypoint": 0, "occupied": 0, "spawn_failed": 0, "activation_failed": 0}

        print("\n" + "=" * 65)
        print("[ScenarioManager] MAP-WIDE BACKGROUND TRAFFIC")
        print(f"[ScenarioManager] Available spawn points: {len(spawn_points)}")
        print(f"[ScenarioManager] Selected spawn points: {len(selected_points)}")
        print(f"[ScenarioManager] Requested NPCs: {number_of_vehicles}")
        print("=" * 65)

        for transform in selected_points:
            if spawned >= number_of_vehicles:
                break

            attempts += 1
            waypoint = self.carla_map.get_waypoint(transform.location, project_to_road=True, lane_type=carla.LaneType.Driving)

            if waypoint is None:
                failed += 1
                reasons["invalid_waypoint"] += 1
                continue

            if not self._is_background_spawn_clear(transform, waypoint):
                failed += 1
                reasons["occupied"] += 1
                continue

            vehicle = self.world.try_spawn_actor(self._random_vehicle_blueprint(), transform)

            if vehicle is None:
                failed += 1
                reasons["spawn_failed"] += 1
                continue

            try:
                target_speed = self.random.uniform(self.BACKGROUND_MIN_SPEED, self.BACKGROUND_MAX_SPEED)
                self._activate_npc(vehicle, target_speed, background=True)
                self.spawned_vehicles.append(vehicle)
                spawned += 1

                print(f"[ScenarioManager] VEHICLE {spawned}/{number_of_vehicles} | {vehicle.type_id} | road={waypoint.road_id} | lane={waypoint.lane_id} | target={target_speed:.1f} km/h")

            except Exception:
                reasons["activation_failed"] += 1
                failed += 1

                try:
                    vehicle.destroy()
                except Exception:
                    pass

        print("\n" + "=" * 65)
        print("[ScenarioManager] BACKGROUND SPAWN SUMMARY")
        print(f"[ScenarioManager] Requested: {number_of_vehicles}")
        print(f"[ScenarioManager] Spawned: {spawned}")
        print(f"[ScenarioManager] Failed: {failed}")
        print(f"[ScenarioManager] Attempts: {attempts}")
        print(f"[ScenarioManager] Invalid waypoint: {reasons['invalid_waypoint']}")
        print(f"[ScenarioManager] Occupied positions: {reasons['occupied']}")
        print(f"[ScenarioManager] CARLA spawn failures: {reasons['spawn_failed']}")
        print(f"[ScenarioManager] Activation failures: {reasons['activation_failed']}")
        print("=" * 65)

        return spawned, failed

    def cleanup(self):
        count = len(self.spawned_vehicles)

        for vehicle in self.spawned_vehicles:
            try:
                if vehicle.is_alive:
                    vehicle.set_autopilot(False, self.traffic_manager_port)
                    vehicle.destroy()
            except Exception:
                pass

        self.spawned_vehicles.clear()
        self.scenario_name = None

        if count:
            print(f"[ScenarioManager] Removed {count} NPC vehicles.")

    def setup_scenario(self, scenario_name="slow_vehicle_ahead"):
        if scenario_name not in self.SCENARIOS:
            available = ", ".join(self.SCENARIOS.keys())
            raise ValueError(f"Unknown scenario '{scenario_name}'. Available scenarios: {available}")

        self.cleanup()
        self.scenario_name = scenario_name
        config = self.SCENARIOS[scenario_name]
        ego_waypoint = self._get_ego_waypoint()
        adjacent_side, adjacent_waypoint = self._get_adjacent_lane(ego_waypoint)

        print("\n" + "=" * 65)
        print(f"[ScenarioManager] Scenario: {scenario_name}")
        print(f"[ScenarioManager] {config['description']}")
        print(f"[ScenarioManager] Ego road: {ego_waypoint.road_id}")
        print(f"[ScenarioManager] Ego lane: {ego_waypoint.lane_id}")

        if adjacent_waypoint is not None:
            print(f"[ScenarioManager] Adjacent lane: {adjacent_side} | road={adjacent_waypoint.road_id} | lane={adjacent_waypoint.lane_id}")
        else:
            print("[ScenarioManager] No same-direction adjacent lane found.")

        print("=" * 65)

        scenario_spawned = 0
        scenario_failed = 0

        for index, spec in enumerate(config["vehicles"], start=1):
            waypoint = self._get_position_waypoint(ego_waypoint, spec["position"], spec["distance"], adjacent_waypoint)

            if waypoint is None:
                print(f"[ScenarioManager] Cannot resolve NPC {index} at {spec['position']}.")
                scenario_failed += 1
                continue

            vehicle = self._spawn_npc(waypoint, spec["speed"])

            if vehicle is None:
                scenario_failed += 1
            else:
                scenario_spawned += 1

        background_requested = self.BACKGROUND_VEHICLE_COUNTS.get(scenario_name, 30)
        background_spawned, background_failed = self._spawn_map_traffic(background_requested)

        result = {
            "scenario": scenario_name,
            "description": config["description"],
            "scenario_requested": len(config["vehicles"]),
            "scenario_spawned": scenario_spawned,
            "scenario_failed": scenario_failed,
            "background_requested": background_requested,
            "background_spawned": background_spawned,
            "background_failed": background_failed,
            "requested": len(config["vehicles"]) + background_requested,
            "spawned": len(self.spawned_vehicles),
            "failed": scenario_failed + background_failed,
            "vehicles": list(self.spawned_vehicles),
            "adjacent_side": adjacent_side,
            "adjacent_lane_id": adjacent_waypoint.lane_id if adjacent_waypoint is not None else None,
        }

        print("\n" + "=" * 65)
        print("[ScenarioManager] FINAL RESULT")
        print(f"[ScenarioManager] Scenario NPCs: {scenario_spawned}/{len(config['vehicles'])}")
        print(f"[ScenarioManager] Background NPCs: {background_spawned}/{background_requested}")
        print(f"[ScenarioManager] Total NPCs: {result['spawned']}/{result['requested']}")
        print("=" * 65)

        return result

    def get_spawned_vehicles(self):
        return list(self.spawned_vehicles)

    def get_available_scenarios(self):
        return {name: config["description"] for name, config in self.SCENARIOS.items()}
