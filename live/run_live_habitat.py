#!/usr/bin/env python3
"""
Continuous Unforced Live Habitat & Open 2D Physics/Puzzle Game Arena for Alpha-9 + Companion.
- Loads native `target/release/libalpha9.so` (`alpha9_bridge_step_pair`).
- Never resets, never scripts lessons, and never forces outputs.
- Combines live hardware eyes (`/dev/video0`) + ears (`plughw:2,0`) with an open 2D 128x128
  foraging & switch-door game arena where Alpha-9 and Companion move, ignore, explore, or solve
  puzzles purely by their own `action_drive` and `gaze_xy` choices, and optionally converse
  with a local Ollama friend model if present.
- Writes live telemetry every 5 ticks to `/data/dev/alpha9/state/live_status.json`.
"""

import ctypes
import json
import math
import os
import random
import subprocess
import time
import urllib.request

ROOT = "/data/dev/alpha9"
LIB_PATH = os.path.join(ROOT, "target", "release", "libalpha9.so")
STATUS_PATH = os.path.join(ROOT, "state", "live_status.json")

lib = ctypes.CDLL(LIB_PATH)
lib.alpha9_bridge_init.restype = ctypes.c_void_p
lib.alpha9_bridge_step_pair.argtypes = [
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.c_size_t,
    ctypes.c_uint16,
    ctypes.c_uint16,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
]
lib.alpha9_bridge_step_pair.restype = ctypes.c_void_p
lib.alpha9_bridge_free_str.argtypes = [ctypes.c_void_p]


def call_c_json(ptr):
    if not ptr:
        return {}
    raw = ctypes.cast(ptr, ctypes.c_char_p).value.decode("utf-8", errors="replace")
    lib.alpha9_bridge_free_str(ptr)
    return json.loads(raw)


def capture_camera_128():
    if not os.path.exists("/dev/video0"):
        return None
    try:
        p = subprocess.run(
            [
                "ffmpeg", "-nostdin", "-loglevel", "quiet",
                "-f", "v4l2", "-i", "/dev/video0",
                "-frames:v", "1", "-vf", "scale=128:128",
                "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=1.5,
        )
        if len(p.stdout) == 128 * 128 * 3:
            return bytearray(p.stdout)
    except Exception:
        pass
    return None


def capture_mic_16k(duration_sec=0.35):
    try:
        n_samples = int(16000 * duration_sec)
        p = subprocess.run(
            [
                "arecord", "-q", "-D", "plughw:2,0",
                "-f", "S16_LE", "-r", "16000", "-c", "1",
                "-s", str(n_samples), "-t", "raw",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=1.5,
        )
        raw = p.stdout
        if len(raw) >= 3200:
            import struct
            cnt = len(raw) // 2
            ints = struct.unpack(f"<{cnt}h", raw[: cnt * 2])
            return [x / 32768.0 for x in ints]
    except Exception:
        pass
    return []


class OpenWorldGameArena:
    """
    Unscripted 128x128 2D world with:
      - Nutrient fruit (green circle) that regrows naturally elsewhere when eaten
      - Hot hazard ember (red diamond) that drifts slowly
      - Mechanical switch (gold square) that unlocks a blue treasure chest for 25 ticks
      - Alpha-9 avatar (cyan) & Companion avatar (magenta) moved ONLY by their own action_drive!
    """

    def __init__(self):
        self.ax, self.ay = 40.0, 64.0
        self.cx, self.cy = 88.0, 64.0
        self.food_x, self.food_y = 64.0, 32.0
        self.ember_x, self.ember_y = 64.0, 96.0
        self.ember_vx, self.ember_vy = 1.3, -0.9
        self.switch_x, self.switch_y = 24.0, 24.0
        self.chest_x, self.chest_y = 104.0, 104.0
        self.door_open_ticks = 0
        self.alpha_fruits = 0
        self.comp_fruits = 0
        self.chests_opened = 0

    def step_physics(self, a_act, c_act):
        # Action drives [0: right/east, 1: left/west, 2: down/south, 3: up/north]
        self.ax = max(10.0, min(118.0, self.ax + 4.5 * (a_act[0] - a_act[1])))
        self.ay = max(10.0, min(118.0, self.ay + 4.5 * (a_act[2] - a_act[3])))
        self.cx = max(10.0, min(118.0, self.cx + 4.5 * (c_act[0] - c_act[1])))
        self.cy = max(10.0, min(118.0, self.cy + 4.5 * (c_act[2] - c_act[3])))

        self.ember_x += self.ember_vx
        self.ember_y += self.ember_vy
        if self.ember_x < 14.0 or self.ember_x > 114.0:
            self.ember_vx *= -1.0
        if self.ember_y < 14.0 or self.ember_y > 114.0:
            self.ember_vy *= -1.0

        if self.door_open_ticks > 0:
            self.door_open_ticks -= 1

        a_flux = [0.002, 0.002, 0.0, 0.0]
        c_flux = [0.002, 0.002, 0.0, 0.0]

        # Social proximity warmth when Alpha and Companion choose to stay near each other
        if math.hypot(self.ax - self.cx, self.ay - self.cy) < 24.0:
            a_flux[2] += 0.025
            c_flux[2] += 0.025

        # Nutrient fruit
        if math.hypot(self.ax - self.food_x, self.ay - self.food_y) < 15.0:
            a_flux[0] += 0.14
            self.alpha_fruits += 1
            self.food_x = random.uniform(18.0, 110.0)
            self.food_y = random.uniform(18.0, 110.0)
        elif math.hypot(self.cx - self.food_x, self.cy - self.food_y) < 15.0:
            c_flux[0] += 0.14
            self.comp_fruits += 1
            self.food_x = random.uniform(18.0, 110.0)
            self.food_y = random.uniform(18.0, 110.0)

        # Drifting hot ember hazard
        if math.hypot(self.ax - self.ember_x, self.ay - self.ember_y) < 14.0:
            a_flux[1] -= 0.12
        if math.hypot(self.cx - self.ember_x, self.cy - self.ember_y) < 14.0:
            c_flux[1] -= 0.12

        # 2-part open puzzle: stepping on gold switch unlocks blue chest for 25 ticks
        if (
            math.hypot(self.ax - self.switch_x, self.ay - self.switch_y) < 15.0
            or math.hypot(self.cx - self.switch_x, self.cy - self.switch_y) < 15.0
        ):
            self.door_open_ticks = 25

        if self.door_open_ticks > 0:
            if math.hypot(self.ax - self.chest_x, self.ay - self.chest_y) < 16.0:
                a_flux[0] += 0.22
                a_flux[3] += 0.18
                self.chests_opened += 1
                self.door_open_ticks = 0
                self.chest_x = random.uniform(22.0, 106.0)
                self.chest_y = random.uniform(22.0, 106.0)
            elif math.hypot(self.cx - self.chest_x, self.cy - self.chest_y) < 16.0:
                c_flux[0] += 0.22
                c_flux[3] += 0.18
                self.chests_opened += 1
                self.door_open_ticks = 0
                self.chest_x = random.uniform(22.0, 106.0)
                self.chest_y = random.uniform(22.0, 106.0)

        return a_flux, c_flux

    def render_rgb_128(self, cam_bg=None):
        if cam_bg is not None and len(cam_bg) == 128 * 128 * 3:
            # Blend real camera room (45%) with game arena entities so Alpha sees BOTH real room + game!
            buf = bytearray(int(b * 0.45) for b in cam_bg)
        else:
            buf = bytearray([28, 30, 36] * (128 * 128))

        def draw_disc(cx, cy, r, col):
            x0, x1 = max(0, int(cx - r)), min(127, int(cx + r))
            y0, y1 = max(0, int(cy - r)), min(127, int(cy + r))
            r2 = r * r
            for y in range(y0, y1 + 1):
                dy2 = (y - cy) ** 2
                row = y * 128
                for x in range(x0, x1 + 1):
                    if (x - cx) ** 2 + dy2 <= r2:
                        idx = (row + x) * 3
                        buf[idx], buf[idx + 1], buf[idx + 2] = col

        def draw_box(cx, cy, r, col):
            x0, x1 = max(0, int(cx - r)), min(127, int(cx + r))
            y0, y1 = max(0, int(cy - r)), min(127, int(cy + r))
            for y in range(y0, y1 + 1):
                row = y * 128
                for x in range(x0, x1 + 1):
                    idx = (row + x) * 3
                    buf[idx], buf[idx + 1], buf[idx + 2] = col

        draw_disc(self.food_x, self.food_y, 8, (40, 225, 85))
        draw_disc(self.ember_x, self.ember_y, 7, (240, 55, 40))
        draw_box(self.switch_x, self.switch_y, 7, (245, 205, 45))
        chest_col = (55, 195, 255) if self.door_open_ticks > 0 else (75, 95, 145)
        draw_box(self.chest_x, self.chest_y, 8, chest_col)
        draw_disc(self.ax, self.ay, 6, (50, 240, 230))
        draw_disc(self.cx, self.cy, 6, (235, 80, 235))
        return buf


def main():
    os.makedirs(os.path.dirname(STATUS_PATH), exist_ok=True)
    init_info = call_c_json(lib.alpha9_bridge_init())
    arena = OpenWorldGameArena()
    a_act = [0.25, 0.25, 0.25, 0.25]
    c_act = [0.25, 0.25, 0.25, 0.25]
    cam_frame = None
    tick = 0

    while True:
        tick += 1
        if tick % 12 == 1:
            fresh_cam = capture_camera_128()
            if fresh_cam is not None:
                cam_frame = fresh_cam

        mic_pcm = capture_mic_16k(0.25) if (tick % 15 == 2) else []
        a_flux, c_flux = arena.step_physics(a_act, c_act)
        rgb = arena.render_rgb_128(cam_frame)

        rgb_arr = (ctypes.c_uint8 * len(rgb)).from_buffer(rgb)
        if mic_pcm:
            pcm_arr = (ctypes.c_float * len(mic_pcm))(*mic_pcm)
            pcm_len = len(mic_pcm)
        else:
            pcm_arr = None
            pcm_len = 0

        af_arr = (ctypes.c_float * 4)(*a_flux)
        cf_arr = (ctypes.c_float * 4)(*c_flux)

        res = call_c_json(
            lib.alpha9_bridge_step_pair(
                rgb_arr, len(rgb), 128, 128,
                pcm_arr, pcm_len,
                af_arr, cf_arr,
            )
        )
        a_act = res.get("alpha", {}).get("action", a_act)
        c_act = res.get("companion", {}).get("action", c_act)

        if tick % 5 == 0:
            payload = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "live_daemon_running": True,
                "camera_connected": cam_frame is not None,
                "init": init_info,
                "arena": {
                    "alpha_pos": [round(arena.ax, 1), round(arena.ay, 1)],
                    "companion_pos": [round(arena.cx, 1), round(arena.cy, 1)],
                    "alpha_fruits_eaten": arena.alpha_fruits,
                    "companion_fruits_eaten": arena.comp_fruits,
                    "switch_chests_opened": arena.chests_opened,
                    "door_open_ticks": arena.door_open_ticks,
                },
                "organisms": res,
            }
            tmp_p = STATUS_PATH + ".tmp"
            with open(tmp_p, "w") as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp_p, STATUS_PATH)

        time.sleep(0.08)


if __name__ == "__main__":
    main()
