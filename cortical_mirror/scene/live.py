"""The scene graph, live in a browser: watch what it explains, and change it.

A source (the sprite world, a video or a camera) feeds the graph frame by frame.
The page shows what is seen beside what the graph explains, with every node drawn
over it. Clicking a node deletes it and dragging moves it. Those edits persist,
following the node, while the graph goes on tracking it: the "explained" view
is the world as it would look if it were changed.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources

import cv2
import numpy as np

from .graph import SceneGraph, _as_gray
from .world import SpriteWorld

VIEWS = ("seen", "explained")
SIZE = (256, 192)
DUPLICATE_OFFSET = (36, 0)


class SpriteSource:
    """The sprite world with a screen, played forever, a new scene every 600 frames."""

    def __init__(self, seed: int = 0, scene_frames: int = 600):
        self.seed, self.scene_frames, self.frame = seed, scene_frames, 0
        self._world()

    def _world(self) -> None:
        self.world = SpriteWorld(
            seed=self.seed, objects=3, occluders=1, occluder_width=40, radius=(9, 14)
        )

    def read(self) -> np.ndarray:
        if self.frame and self.frame % self.scene_frames == 0:
            self.seed += 1
            self._world()
        self.frame += 1
        self.world.step()
        return self.world.render()


class VideoSource:
    """A camera index, video file or stream URL, looped; frames as gray at SIZE."""

    def __init__(self, source: str):
        self.source = int(source) if source.isdigit() else source
        self.capture = cv2.VideoCapture(self.source)
        if not self.capture.isOpened():
            raise SystemExit(f"cannot open video source {source!r}")

    def read(self) -> np.ndarray:
        ok, frame = self.capture.read()
        if not ok:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.capture.read()
            if not ok:
                raise SystemExit("the video source stopped")
        return _as_gray(frame, *SIZE)


def _jpeg(image: np.ndarray) -> bytes:
    gray = np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8)
    big = cv2.resize(gray, (SIZE[0] * 2, SIZE[1] * 2), interpolation=cv2.INTER_NEAREST)
    return cv2.imencode(".jpg", big, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()


class SceneLive:
    def __init__(self, source, graph: SceneGraph | None = None, fps: float = 25.0):
        self.source = source
        self.graph = graph or SceneGraph(*SIZE)
        self.fps = fps
        self.running = True
        self.error: str | None = None
        self._condition = threading.Condition()
        self._lock = threading.Lock()
        self._frames = {view: (0, b"") for view in VIEWS}
        self.deleted: set[int] = set()
        self.moved: dict[int, list[float]] = {}
        self.duplicated: set[int] = set()
        self.explained = np.zeros(SIZE[::-1], np.float32)
        self._scene: dict = {"nodes": [], "edges": []}

    # Edits ------------------------------------------------------------------

    def control(self, command: dict) -> dict:
        with self._lock:
            if command.get("reset"):
                self.deleted.clear()
                self.moved.clear()
                self.duplicated.clear()
            if "delete" in command:
                self.deleted.add(int(command["delete"]))
            if "restore" in command:
                self.deleted.discard(int(command["restore"]))
            if "move" in command:
                dx, dy = (float(v) for v in command.get("by", (0, 0)))
                old = self.moved.get(int(command["move"]), [0.0, 0.0])
                self.moved[int(command["move"])] = [old[0] + dx, old[1] + dy]
            if "duplicate" in command:
                self.duplicated.add(int(command["duplicate"]))
        return {"ok": True, "edits": self._edits()}

    def _edits(self) -> dict:
        return {
            "deleted": sorted(self.deleted),
            "moved": {str(k): v for k, v in sorted(self.moved.items())},
            "duplicated": sorted(self.duplicated),
        }

    # Frames -----------------------------------------------------------------

    def step(self) -> None:
        image = self.source.read()
        self.graph.observe(image)
        with self._lock:
            ids = {n.id for n in self.graph.nodes}
            # The picture shows what is explained in view; what is believed but
            # hidden is drawn on the page as a ring, not painted over what hides it.
            hidden = {n.id for n in self.graph.nodes if n.state != "seen"}
            self.explained = self.graph.edit(
                move={k: tuple(v) for k, v in self.moved.items() if k in ids},
                delete=(self.deleted & ids) | hidden,
                duplicate={k: DUPLICATE_OFFSET for k in self.duplicated if k in ids},
            )
            scene = self.graph.to_json()
            scene["edits"] = self._edits()
            scene["size"] = list(SIZE)
            self._scene = scene
        self._publish({"seen": _jpeg(image), "explained": _jpeg(self.explained)})

    def _publish(self, jpegs: dict[str, bytes]) -> None:
        with self._condition:
            for view, jpeg in jpegs.items():
                version = self._frames[view][0] + 1
                self._frames[view] = (version, jpeg)
            self._condition.notify_all()

    def latest(self, view: str) -> tuple[int, bytes]:
        with self._condition:
            return self._frames[view]

    def wait_for_change(self, view: str, version: int, timeout: float = 1.0):
        with self._condition:
            self._condition.wait_for(
                lambda: self._frames[view][0] != version or not self.running, timeout
            )
            return self._frames[view]

    def scene(self) -> dict:
        with self._lock:
            return json.loads(json.dumps({**self._scene, "edits": self._edits()}))

    def run(self) -> None:
        period = 1.0 / self.fps
        try:
            while self.running:
                started = time.perf_counter()
                self.step()
                time.sleep(max(0.0, period - (time.perf_counter() - started)))
        except BaseException as error:  # noqa: BLE001 - shown on the page and re-raised
            self.error = f"{type(error).__name__}: {error}"
            self.running = False
            with self._condition:
                self._condition.notify_all()


PAGE = resources.files("cortical_mirror").joinpath("static/scene.html")


class Handler(BaseHTTPRequestHandler):
    def _bytes(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, value) -> None:
        self._bytes(200, "application/json", json.dumps(value).encode())

    def do_GET(self):
        state: SceneLive = self.server.state
        path = self.path.split("?")[0]
        if path == "/":
            self._bytes(200, "text/html; charset=utf-8", PAGE.read_bytes())
            return
        if path == "/scene.json":
            self._json({**state.scene(), "error": state.error})
            return
        view = path.removeprefix("/stream/").removesuffix(".mjpg")
        if path.startswith("/stream/") and view in VIEWS:
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            version = -1
            try:
                while state.running:
                    version, jpeg = state.wait_for_change(view, version)
                    if jpeg:
                        self.wfile.write(
                            b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                        )
                        self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return
            return
        self._bytes(404, "text/plain; charset=utf-8", b"not found")

    def do_POST(self):
        if self.path.split("?")[0] != "/edit":
            self._bytes(404, "text/plain; charset=utf-8", b"not found")
            return
        length = min(int(self.headers.get("Content-Length") or 0), 4096)
        try:
            command = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(command, dict):
                raise ValueError("expected a JSON object")
            self._json(self.server.state.control(command))
        except (ValueError, TypeError) as error:
            self._bytes(400, "text/plain; charset=utf-8", str(error).encode())

    def log_message(self, *_):
        return


def serve(state: SceneLive, host: str, port: int) -> None:
    server = ThreadingHTTPServer((host, port), Handler)
    server.state = state
    server.daemon_threads = True
    thread = threading.Thread(target=state.run, name="scene-source", daemon=True)
    thread.start()
    print(f"Cortical Mirror scene graph: http://{host}:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        state.running = False
        with state._condition:
            state._condition.notify_all()
        thread.join(timeout=5)
        server.server_close()
    if state.error:
        raise RuntimeError(state.error)
