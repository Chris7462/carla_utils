#!/usr/bin/env python3
"""Free-camera viewer for a remote CARLA server (run on the laptop, via SSH tunnel).

Spawns its own RGB camera actor and moves that camera only. It never touches the
world spectator and never changes world settings, so it cannot interfere with
whatever the agent is doing on the server.

Free mode:
    W / S          forward / backward
    A / D          left / right
    E / Q          up / down
    Shift          move faster
    Right mouse    hold and drag to look around

Follow mode:
    F              toggle follow (targets role_name hero/ego, else the newest vehicle)
    N              switch to the next vehicle while following
    Right mouse    hold and drag to orbit around the vehicle
    Mouse wheel    zoom in / out
    W/A/S/D/E/Q    leave follow mode and keep the current camera position

Esc quits.
"""
import argparse
import math
import threading
import time

import numpy as np
import pygame
import carla

TARGET_HEIGHT = 1.5      # aim point above the vehicle origin, meters
DEFAULT_DISTANCE = 8.0   # chase distance, meters
DEFAULT_ORBIT_PITCH = -15.0


def parse_args():
    p = argparse.ArgumentParser(description="Free-camera viewer for a remote CARLA server")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fov", type=float, default=90.0)
    p.add_argument("--speed", type=float, default=10.0, help="base speed in m/s")
    return p.parse_args()


def list_vehicles(world):
    return sorted(world.get_actors().filter("vehicle.*"), key=lambda a: a.id)


def pick_default(vehicles):
    for v in vehicles:
        if v.attributes.get("role_name") in ("hero", "ego"):
            return v
    return vehicles[-1] if vehicles else None  # highest id = most recently spawned


def next_vehicle(vehicles, current):
    if not vehicles:
        return None
    ids = [v.id for v in vehicles]
    if current is not None and current.id in ids:
        return vehicles[(ids.index(current.id) + 1) % len(vehicles)]
    return vehicles[0]


def describe(v):
    role = v.attributes.get("role_name", "")
    return "%s (id %d%s)" % (v.type_id, v.id, ", " + role if role else "")


def main():
    args = parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()

    if world.get_settings().synchronous_mode:
        print("WARNING: the world is in synchronous mode. Camera frames only arrive "
              "when something ticks the world. This viewer does not tick or change "
              "settings, so the picture may stay black until the agent ticks.")

    bp = world.get_blueprint_library().find("sensor.camera.rgb")
    bp.set_attribute("image_size_x", str(args.width))
    bp.set_attribute("image_size_y", str(args.height))
    bp.set_attribute("fov", str(args.fov))

    # Start where the spectator currently is (read only; the spectator is not modified).
    start = world.get_spectator().get_transform()
    pos = [start.location.x, start.location.y, start.location.z]
    pitch, yaw = start.rotation.pitch, start.rotation.yaw

    camera = world.spawn_actor(bp, start)

    latest = {"frame": None}
    lock = threading.Lock()

    def on_image(image):
        arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
        rgb = np.ascontiguousarray(arr[:, :, 2::-1])  # BGRA -> RGB
        with lock:
            latest["frame"] = rgb

    camera.listen(on_image)

    pygame.init()
    screen = pygame.display.set_mode((args.width, args.height))
    pygame.display.set_caption("CARLA remote viewer")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    sensitivity = 0.15
    dragging = False
    last_sent = None
    t0 = time.time()
    warned_no_frames = False

    follow_target = None
    follow_dist = DEFAULT_DISTANCE
    orbit_yaw = 0.0
    orbit_pitch = DEFAULT_ORBIT_PITCH
    status = ("", 0.0)  # (text, expire time)

    def set_status(text, seconds=3.0):
        nonlocal status
        status = (text, time.time() + seconds)
        print(text)

    def start_follow(target):
        nonlocal follow_target, follow_dist, orbit_yaw, orbit_pitch
        follow_target = target
        follow_dist = DEFAULT_DISTANCE
        orbit_yaw = 0.0
        orbit_pitch = DEFAULT_ORBIT_PITCH
        set_status("Following " + describe(target))

    try:
        running = True
        while running:
            dt = clock.tick(60) / 1000.0

            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
                elif e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_ESCAPE:
                        running = False
                    elif e.key == pygame.K_f:
                        if follow_target is not None:
                            follow_target = None
                            set_status("Follow off (free camera)")
                        else:
                            target = pick_default(list_vehicles(world))
                            if target is None:
                                set_status("No vehicles in the world to follow")
                            else:
                                start_follow(target)
                    elif e.key == pygame.K_n and follow_target is not None:
                        target = next_vehicle(list_vehicles(world), follow_target)
                        if target is None:
                            set_status("No vehicles in the world to follow")
                            follow_target = None
                        else:
                            start_follow(target)
                elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 3:
                    dragging = True
                elif e.type == pygame.MOUSEBUTTONUP and e.button == 3:
                    dragging = False
                elif e.type == pygame.MOUSEWHEEL and follow_target is not None:
                    follow_dist = max(2.0, min(60.0, follow_dist - e.y * 1.0))
                elif e.type == pygame.MOUSEMOTION and dragging:
                    dx, dy = e.rel
                    if follow_target is not None:
                        orbit_yaw += dx * sensitivity
                        orbit_pitch = max(-80.0, min(80.0, orbit_pitch - dy * sensitivity))
                    else:
                        yaw += dx * sensitivity
                        pitch = max(-89.0, min(89.0, pitch - dy * sensitivity))

            keys = pygame.key.get_pressed()
            f = int(keys[pygame.K_w]) - int(keys[pygame.K_s])
            r = int(keys[pygame.K_d]) - int(keys[pygame.K_a])
            u = int(keys[pygame.K_e]) - int(keys[pygame.K_q])

            # Any movement key leaves follow mode; pos/yaw/pitch already hold the
            # last follow camera pose, so free flight continues from there.
            if follow_target is not None and (f or r or u):
                follow_target = None
                set_status("Follow off (free camera)")

            if follow_target is None:
                speed = args.speed * (4.0 if (keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]) else 1.0)
                yr, pr = math.radians(yaw), math.radians(pitch)
                forward = (math.cos(pr) * math.cos(yr), math.cos(pr) * math.sin(yr), math.sin(pr))
                right = (-math.sin(yr), math.cos(yr), 0.0)
                for i in range(3):
                    pos[i] += (forward[i] * f + right[i] * r) * speed * dt
                pos[2] += u * speed * dt
            else:
                try:
                    tf = follow_target.get_transform()
                except RuntimeError:
                    set_status("Followed vehicle no longer exists; free camera")
                    follow_target = None
                else:
                    cy = math.radians(tf.rotation.yaw + orbit_yaw)
                    cp = math.radians(orbit_pitch)
                    d = (math.cos(cp) * math.cos(cy), math.cos(cp) * math.sin(cy), math.sin(cp))
                    aim = (tf.location.x, tf.location.y, tf.location.z + TARGET_HEIGHT)
                    pos = [aim[i] - d[i] * follow_dist for i in range(3)]
                    yaw = tf.rotation.yaw + orbit_yaw
                    pitch = orbit_pitch

            state = (tuple(pos), pitch, yaw)
            if state != last_sent:
                camera.set_transform(carla.Transform(
                    carla.Location(x=pos[0], y=pos[1], z=pos[2]),
                    carla.Rotation(pitch=pitch, yaw=yaw, roll=0.0)))
                last_sent = state

            with lock:
                frame = latest["frame"]

            if frame is not None:
                screen.blit(pygame.surfarray.make_surface(frame.swapaxes(0, 1)), (0, 0))
            else:
                screen.fill((0, 0, 0))
                msg = font.render("Waiting for camera frames...", True, (200, 200, 200))
                screen.blit(msg, (20, 20))
                if not warned_no_frames and time.time() - t0 > 5.0:
                    print("No frames after 5 s. If the world is in synchronous mode, "
                          "something must tick it.")
                    warned_no_frames = True

            mode = "FOLLOW %s" % describe(follow_target) if follow_target is not None else "FREE"
            hud = "%s | x=%.1f y=%.1f z=%.1f  yaw=%.0f pitch=%.0f  %.0f fps" % (
                mode, pos[0], pos[1], pos[2], yaw, pitch, clock.get_fps())
            screen.blit(font.render(hud, True, (255, 255, 0)), (10, args.height - 28))
            if time.time() < status[1]:
                screen.blit(font.render(status[0], True, (255, 255, 255)), (10, 10))
            pygame.display.flip()
    finally:
        try:
            camera.stop()
            camera.destroy()
        except RuntimeError as exc:
            print("Camera cleanup failed:", exc)
        pygame.quit()


if __name__ == "__main__":
    main()
