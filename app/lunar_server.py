"""
LunarLander DAgger - Phase 1 Server
FastAPI + Socket.IO + Gymnasium environment with fixed-rate game loop.
"""

import os
import struct
import io
import asyncio
import logging
from pathlib import Path

import numpy as np
import gymnasium as gym
import socketio
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from PIL import Image


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("lunar")


# Binary protocol header: 8 floats (state) + 1 uint8 (action) + 1 float (reward)
#                        + 1 uint32 (step) + 1 uint32 (episode) + 1 uint8 (done)
# Total: 32 + 1 + 4 + 4 + 4 + 1 = 46 bytes, followed by JPEG data
HEADER_FORMAT = "<8f B f I I B"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)


class LunarEnvironment:
    """Wraps Gymnasium LunarLander-v3 with frame capture and episode recording."""

    def __init__(self, episodes_dir: str = "episodes"):
        self.episodes_dir = Path(episodes_dir)
        self.episodes_dir.mkdir(parents=True, exist_ok=True)

        self.env = gym.make("LunarLander-v3", render_mode="rgb_array")
        self.episode_num = 0
        self.step_num = 0
        self.total_reward = 0.0
        self.state = None
        self.done = True

        # Episode recording buffers
        self._states = []
        self._actions = []
        self._rewards = []

    def reset(self):
        """Reset the environment and start a new episode."""
        self.episode_num += 1
        self.step_num = 0
        self.total_reward = 0.0
        self.done = False

        self._states = []
        self._actions = []
        self._rewards = []

        self.state, _ = self.env.reset()
        return self.state

    def step(self, action: int):
        """Advance one step. Returns (state, reward, done, frame_jpeg)."""
        if self.done:
            self.reset()

        prev_state = self.state.copy()
        self.state, reward, terminated, truncated, _ = self.env.step(action)
        self.done = terminated or truncated
        self.step_num += 1
        self.total_reward += reward

        # Record transition
        self._states.append(prev_state)
        self._actions.append(action)
        self._rewards.append(reward)

        # Capture frame as JPEG
        frame_rgb = self.env.render()
        frame_jpeg = self._encode_jpeg(frame_rgb)

        if self.done:
            self._save_episode()

        return self.state, reward, self.done, frame_jpeg

    def get_initial_frame(self):
        """Capture the current frame without stepping."""
        frame_rgb = self.env.render()
        return self._encode_jpeg(frame_rgb)

    def _encode_jpeg(self, frame_rgb: np.ndarray, quality: int = 80) -> bytes:
        img = Image.fromarray(frame_rgb)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        return buf.getvalue()

    def _save_episode(self):
        """Save the completed episode to disk."""
        if not self._states:
            return

        filename = self.episodes_dir / f"episode_{self.episode_num:04d}.npz"
        np.savez_compressed(
            filename,
            states=np.array(self._states, dtype=np.float32),
            actions=np.array(self._actions, dtype=np.uint8),
            rewards=np.array(self._rewards, dtype=np.float32),
        )
        log.info(
            f"Episode {self.episode_num} saved: "
            f"{len(self._states)} steps, "
            f"total reward {self.total_reward:.2f} -> {filename}"
        )

    def close(self):
        self.env.close()


class LunarServer:
    """FastAPI + Socket.IO server for DAgger Phase 1."""

    TICK_RATE = 20  # FPS for the game loop
    TICK_INTERVAL = 1.0 / TICK_RATE

    def __init__(self, host: str = "0.0.0.0", port: int = 8500, episodes_dir: str = "episodes"):
        self.host = host
        self.port = port

        # Socket.IO
        self.sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")
        self.app = FastAPI()
        self.asgi = socketio.ASGIApp(self.sio, self.app)

        # Environment
        self.env = LunarEnvironment(episodes_dir=episodes_dir)

        # Client tracking
        self.client_sid = None

        # Current held actions (empty = noop) and round-robin index
        self.held_actions = []
        self.action_index = 0

        # Game loop control
        self.running = False
        self._loop_task = None

        self._setup_routes()
        self._setup_socketio()

    def _setup_routes(self):
        web_dir = Path(__file__).parent / "web"

        @self.app.get("/")
        async def index():
            return FileResponse(web_dir / "lunar_client.html")

        self.app.mount("/static", StaticFiles(directory=web_dir), name="static")

    def _setup_socketio(self):

        @self.sio.on("connect")
        async def on_connect(sid, environ):
            log.info(f"Client connected: {sid}")
            self.client_sid = sid
            self.held_actions = []
            self.action_index = 0

            # Start a fresh episode and send the initial frame
            state = self.env.reset()
            frame_jpeg = self.env.get_initial_frame()
            packet = self._pack_frame(state, 0, 0.0, 0, self.env.episode_num, False, frame_jpeg)
            await self.sio.emit("frame", packet, to=sid)

            log.info(f"Episode {self.env.episode_num} started, awaiting human input")

        @self.sio.on("disconnect")
        async def on_disconnect(sid):
            log.info(f"Client disconnected: {sid}")
            if self.client_sid == sid:
                self.client_sid = None
                self._stop_loop()

        @self.sio.on("actions")
        async def on_actions(sid, data):
            if isinstance(data, list):
                self.held_actions = [a for a in data if 0 <= a <= 3]
                self.action_index = 0
            else:
                self.held_actions = []

            # Start the game loop on first input
            if not self.running:
                self._start_loop()

        @self.sio.on("reset")
        async def on_reset(sid):
            """Manual reset requested by client."""
            self._stop_loop()
            self.held_actions = []
            self.action_index = 0

            state = self.env.reset()
            frame_jpeg = self.env.get_initial_frame()
            packet = self._pack_frame(state, 0, 0.0, 0, self.env.episode_num, False, frame_jpeg)
            await self.sio.emit("frame", packet, to=sid)
            log.info(f"Manual reset -> Episode {self.env.episode_num}")

    def _start_loop(self):
        """Start the fixed-rate game loop."""
        if self.running:
            return
        self.running = True
        self._loop_task = asyncio.ensure_future(self._game_loop())
        log.info("Game loop started")

    def _stop_loop(self):
        """Stop the game loop."""
        self.running = False
        if self._loop_task:
            self._loop_task.cancel()
            self._loop_task = None
        log.info("Game loop stopped")

    async def _game_loop(self):
        """Fixed-rate loop that steps the environment and emits frames."""
        try:
            while self.running and self.client_sid:
                tick_start = asyncio.get_event_loop().time()

                # Round-robin through held actions, or noop if none
                if self.held_actions:
                    action = self.held_actions[self.action_index % len(self.held_actions)]
                    self.action_index += 1
                else:
                    action = 0
                state, reward, done, frame_jpeg = self.env.step(action)

                packet = self._pack_frame(
                    state, action, reward,
                    self.env.step_num, self.env.episode_num,
                    done, frame_jpeg
                )
                await self.sio.emit("frame", packet, to=self.client_sid)

                if done:
                    await self.sio.emit("episode_end", {
                        "episode": self.env.episode_num,
                        "steps": self.env.step_num,
                        "total_reward": round(self.env.total_reward, 2),
                    }, to=self.client_sid)

                    # Auto-reset: start new episode
                    state = self.env.reset()
                    frame_jpeg = self.env.get_initial_frame()
                    packet = self._pack_frame(
                        state, 0, 0.0, 0, self.env.episode_num, False, frame_jpeg
                    )
                    await self.sio.emit("frame", packet, to=self.client_sid)
                    log.info(f"Episode {self.env.episode_num} started")

                # Maintain fixed tick rate
                elapsed = asyncio.get_event_loop().time() - tick_start
                sleep_time = self.TICK_INTERVAL - elapsed
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            log.error(f"Game loop error: {e}")
        finally:
            self.running = False

    def _pack_frame(self, state, action, reward, step, episode, done, jpeg_data):
        """Pack state + metadata + JPEG into a single binary message."""
        header = struct.pack(
            HEADER_FORMAT,
            *state.tolist(),
            action,
            reward,
            step,
            episode,
            int(done),
        )
        return header + jpeg_data

    def run(self):
        log.info(f"Starting LunarLander DAgger server on {self.host}:{self.port}")
        uvicorn.run(self.asgi, host=self.host, port=self.port, log_level="info")


if __name__ == "__main__":
    server = LunarServer()
    server.run()
