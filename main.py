import argparse
import os
import random
import sys
import time
import traceback

import cv2
import numpy as np

try:
    import carla
except ImportError:
    sys.exit("[ERROR] CARLA Python API not found.\nCheck your CARLA installation and Python environment.")

try:
    from ultralytics import YOLO
except ImportError:
    sys.exit("[ERROR] ultralytics not found.\nRun: pip install ultralytics")

from src.lane_detector import LaneDetector
from src.vehicle_detector import VehicleDetector
from src.ai_driver import AIDriver
from src.cockpit_renderer import CockpitRenderer
from src.scenario_manager import ScenarioManager


def parse_args():
    parser = argparse.ArgumentParser(description="Vision-only autonomous driving in CARLA")

    parser.add_argument("--host", default="localhost", help="CARLA server host")
    parser.add_argument("--port", default=2000, type=int)
    parser.add_argument("--frames", default=2000, type=int, help="Number of simulation frames")
    parser.add_argument("--fps", default=20, type=int, help="Simulation FPS")
    parser.add_argument("--speed", default=35.0, type=float, help="Ego target cruise speed in km/h")
    parser.add_argument("--output", default="outputs/autonomous_driving_carla.mp4")
    parser.add_argument("--yolo-model", default="yolov8n.pt")
    parser.add_argument(
        "--scenario",
        default="slow_vehicle_ahead",
        choices=["slow_vehicle_ahead", "safe_overtake", "rear_vehicle_blocking", "both_lanes_blocked", "mixed"],
        help="Traffic scenario"
    )
    parser.add_argument("--no-traffic", action="store_true", help="Disable NPC traffic")
    parser.add_argument("--no-pygame", action="store_true", help="Run without live display")

    return parser.parse_args()


def connect_to_carla(host: str, port: int, timeout: float = 30.0):
    client = carla.Client(host, port)
    client.set_timeout(timeout)
    world = client.get_world()
    return client, world


def configure_world(world, fps):
    if fps <= 0:
        raise ValueError("FPS must be greater than zero.")

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / fps
    world.apply_settings(settings)

    world.set_weather(carla.WeatherParameters(
        cloudiness=10.0,
        precipitation=0.0,
        sun_altitude_angle=60.0,
        fog_density=0.0,
        wetness=0.0
    ))

    print(f"[CARLA] Synchronous mode enabled | FPS={fps} | Delta={1.0 / fps:.3f}s")


def spawn_ego_vehicle(world, blueprint_library):
    blueprint = None

    for bp_filter in ("vehicle.tesla.model3", "vehicle.audi.a2", "vehicle.*"):
        blueprints = blueprint_library.filter(bp_filter)

        if blueprints:
            blueprint = blueprints[0]
            break

    if blueprint is None:
        raise RuntimeError("No ego vehicle blueprint found.")

    spawn_points = list(world.get_map().get_spawn_points())
    random.shuffle(spawn_points)

    for transform in spawn_points:
        vehicle = world.try_spawn_actor(blueprint, transform)

        if vehicle is None:
            continue

        vehicle.set_autopilot(False)
        print(f"[EGO] Spawned {vehicle.type_id} | Autopilot=False")
        return vehicle

    raise RuntimeError("Could not spawn ego vehicle at any map spawn point.")


def attach_camera(world, vehicle, blueprint_library, width=960, height=540):
    cam_bp = blueprint_library.find("sensor.camera.rgb")
    settings = world.get_settings()
    fixed_delta = settings.fixed_delta_seconds or 0.05

    cam_bp.set_attribute("image_size_x", str(width))
    cam_bp.set_attribute("image_size_y", str(height))
    cam_bp.set_attribute("fov", "100")
    cam_bp.set_attribute("sensor_tick", str(fixed_delta))

    transform = carla.Transform(
        carla.Location(x=1.8, z=1.4),
        carla.Rotation(pitch=-5)
    )

    camera = world.spawn_actor(cam_bp, transform, attach_to=vehicle)

    latest_frame = {"image": None, "frame": -1}

    def camera_callback(image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        latest_frame["image"] = array[:, :, :3].copy()
        latest_frame["frame"] = image.frame

    camera.listen(camera_callback)

    print(f"[CAMERA] {width}x{height} | FOV=100 | Sensor tick={fixed_delta:.3f}s")

    return camera, latest_frame


def wait_for_camera_frame(latest_frame, target_frame, timeout=2.0):
    start_time = time.monotonic()

    while time.monotonic() - start_time < timeout:
        if latest_frame["image"] is not None and latest_frame["frame"] >= target_frame:
            return latest_frame["image"]

        time.sleep(0.001)

    return latest_frame["image"]


def setup_traffic(client, world, ego_vehicle, args):
    if args.no_traffic:
        print("[TRAFFIC] NPC traffic disabled.")
        return None, []

    scenario_manager = ScenarioManager(
        client=client,
        world=world,
        ego_vehicle=ego_vehicle,
        traffic_manager_port=8000,
        seed=42,
    )

    try:
        result = scenario_manager.setup_scenario(args.scenario)
        vehicles = scenario_manager.get_spawned_vehicles()

        print("\n" + "=" * 65)
        print("[TRAFFIC] SCENARIO READY")
        print(f"[TRAFFIC] Scenario: {result['scenario']}")
        print(f"[TRAFFIC] Scenario NPCs: {result['scenario_spawned']}")
        print(f"[TRAFFIC] Background NPCs: {result['background_spawned']}")
        print(f"[TRAFFIC] Total NPCs: {result['spawned']}")
        print(f"[TRAFFIC] Failed spawns: {result['failed']}")
        print("=" * 65)

        return scenario_manager, vehicles

    except Exception:
        scenario_manager.cleanup()
        raise


def initialize_ai(ego_vehicle, carla_map, args):
    print("[AI] Loading YOLO model...")

    yolo_model = YOLO(args.yolo_model)
    lane_detector = LaneDetector()
    vehicle_detector = VehicleDetector(yolo_model)
    ai_driver = AIDriver(ego_vehicle, carla_map, target_speed=args.speed)
    cockpit = CockpitRenderer(960, 540)

    print("[AI] Modules ready | LaneDetector | VehicleDetector | AIDriver")

    return lane_detector, vehicle_detector, ai_driver, cockpit


def initialize_video(args, cockpit):
    output_directory = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_directory, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, float(args.fps), (cockpit.cw, cockpit.ch))

    if not writer.isOpened():
        raise RuntimeError(f"Could not open video output: {args.output}")

    return writer


def initialize_pygame(args):
    if args.no_pygame:
        return None, None, None

    try:
        import pygame

        pygame.init()
        screen = pygame.display.set_mode((1280, 540))
        pygame.display.set_caption("Autonomous Driving - CARLA")
        clock = pygame.time.Clock()

        return pygame, screen, clock

    except Exception as error:
        print(f"[WARNING] Pygame unavailable: {error}")
        return None, None, None


def process_pygame_events(pygame, screen):
    if pygame is None or screen is None:
        return True

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False

        if event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
            return False

    return True


def render_pygame(pygame, screen, clock, cockpit_frame, fps):
    if pygame is None or screen is None:
        return

    rgb = cv2.cvtColor(cockpit_frame, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (1280, 540))
    surface = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))

    screen.blit(surface, (0, 0))
    pygame.display.flip()
    clock.tick(fps)


def run_simulation(world, latest_frame, lane_detector, vehicle_detector, ai_driver, cockpit, video_writer, pygame, screen, clock, args):
    running = True
    processed_frames = 0
    start_time = time.time()

    print("\n" + "=" * 65)
    print(f"[SIMULATION] Running {args.frames} frames | Target FPS={args.fps}")
    print("[SIMULATION] Press ESC or Q to stop.")
    print("=" * 65)

    for frame_num in range(1, args.frames + 1):
        running = process_pygame_events(pygame, screen)

        if not running:
            print("[SIMULATION] Stopped by user.")
            break

        snapshot = world.tick()
        frame = wait_for_camera_frame(latest_frame, snapshot)

        if frame is None:
            print(f"[WARNING] Camera frame unavailable at simulation frame {frame_num}.")
            continue

        lane_data = lane_detector.detect(frame)
        detections = vehicle_detector.detect(frame)
        decision = ai_driver.decide_and_control(detections, lane_data, frame_num)

        cam_vis = cv2.addWeighted(frame, 1.0, lane_data["overlay"], 0.35, 0)
        cam_vis = vehicle_detector.draw(cam_vis, detections)
        cockpit_frame = cockpit.render(cam_vis, detections, lane_data, decision)

        video_writer.write(cockpit_frame)

        render_pygame(pygame, screen, clock, cockpit_frame, args.fps)

        processed_frames += 1

        if frame_num % 100 == 0:
            elapsed = max(time.time() - start_time, 0.001)
            actual_fps = processed_frames / elapsed

            print(
                f"[{frame_num:4d}/{args.frames}] "
                f"{actual_fps:.1f} FPS | "
                f"{decision['speed']:.1f} km/h | "
                f"{decision['action']}"
            )

    elapsed = time.time() - start_time

    return processed_frames, elapsed


def cleanup(world, camera=None, ego_vehicle=None, scenario_manager=None, video_writer=None, pygame=None):
    print("\n[CLEANUP] Releasing resources...")

    if camera is not None:
        try:
            camera.stop()
        except Exception:
            pass

        try:
            if camera.is_alive:
                camera.destroy()
        except Exception:
            pass

    if scenario_manager is not None:
        try:
            scenario_manager.cleanup()
        except Exception as error:
            print(f"[CLEANUP] ScenarioManager: {error}")

        try:
            scenario_manager.traffic_manager.set_synchronous_mode(False)
        except Exception:
            pass

    if ego_vehicle is not None:
        try:
            if ego_vehicle.is_alive:
                ego_vehicle.destroy()
        except Exception:
            pass

    if video_writer is not None:
        try:
            video_writer.release()
        except Exception:
            pass

    if pygame is not None:
        try:
            pygame.quit()
        except Exception:
            pass

    if world is not None:
        try:
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)
        except Exception as error:
            print(f"[CLEANUP] World settings: {error}")

    print("[CLEANUP] Finished.")


def main():
    args = parse_args()

    if args.frames <= 0:
        raise ValueError("--frames must be greater than zero.")

    if args.fps <= 0:
        raise ValueError("--fps must be greater than zero.")

    print("=" * 65)
    print("  AUTONOMOUS DRIVING - VISION-ONLY CARLA")
    print("=" * 65)

    client = None
    world = None
    ego_vehicle = None
    camera = None
    scenario_manager = None
    video_writer = None
    pygame = None

    processed_frames = 0
    elapsed = 0.0

    try:
        client, world = connect_to_carla(args.host, args.port)
        configure_world(world, args.fps)

        carla_map = world.get_map()
        blueprint_library = world.get_blueprint_library()

        print(f"[CARLA] Connected | Map={carla_map.name}")

        ego_vehicle = spawn_ego_vehicle(world, blueprint_library)

        camera, latest_frame = attach_camera(world, ego_vehicle, blueprint_library)

        for _ in range(10):
            world.tick()

        time.sleep(0.2)

        print("[CARLA] Ego and camera ready.")

        scenario_manager, traffic_vehicles = setup_traffic(client, world, ego_vehicle, args)

        lane_detector, vehicle_detector, ai_driver, cockpit = initialize_ai(ego_vehicle, carla_map, args)

        video_writer = initialize_video(args, cockpit)

        pygame, screen, clock = initialize_pygame(args)

        processed_frames, elapsed = run_simulation(
            world=world,
            latest_frame=latest_frame,
            lane_detector=lane_detector,
            vehicle_detector=vehicle_detector,
            ai_driver=ai_driver,
            cockpit=cockpit,
            video_writer=video_writer,
            pygame=pygame,
            screen=screen,
            clock=clock,
            args=args,
        )

    except KeyboardInterrupt:
        print("\n[MAIN] Interrupted by user.")

    except Exception as error:
        print(f"\n[ERROR] {error}")
        traceback.print_exc()

    finally:
        cleanup(
            world=world,
            camera=camera,
            ego_vehicle=ego_vehicle,
            scenario_manager=scenario_manager,
            video_writer=video_writer,
            pygame=pygame,
        )

    print("\n" + "=" * 65)
    print("  SIMULATION FINISHED")
    print("=" * 65)

    print(f"Processed frames: {processed_frames}")
    print(f"Elapsed time: {elapsed:.2f}s")

    if os.path.exists(args.output):
        file_size = os.path.getsize(args.output) / 1_048_576
        print(f"Output video: {args.output}")
        print(f"Video size: {file_size:.2f} MB")

    print("=" * 65)


if __name__ == "__main__":
    main()