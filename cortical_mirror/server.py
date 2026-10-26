"""Headless live mirror served as MJPEG views plus small JSON metrics and controls."""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

from .engines import EnginePrediction, PredictiveEngine

INDEX_HTML = (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")
VIEWS = ("camera", "prediction", "error", "overlay")
# BGR. Blue is the model's expectation, violet the pretrained motion expert, pewter
# persistence; the page legend uses the same colours.
EXPERT_COLOURS = {
    "copy": (173, 163, 154),
    "cortical": (255, 184, 143),
    "dmvfn": (255, 139, 213),
}


def error_view(due: np.ndarray, actual: np.ndarray) -> np.ndarray:
    error = np.mean(np.abs(due.astype(np.float32) - actual.astype(np.float32)), axis=2)
    error = np.clip(error / 64.0, 0.0, 1.0)
    return cv2.applyColorMap(np.rint(error * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)


def trust_view(prediction: np.ndarray, routing: np.ndarray, names: tuple[str, ...]) -> np.ndarray:
    """Tint each region by the predictor the router trusts there."""

    palette = np.array([EXPERT_COLOURS.get(name, (255, 255, 255)) for name in names], np.float32)
    colour = routing @ palette
    gray = cv2.cvtColor(cv2.cvtColor(prediction, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    return np.clip(0.45 * gray + 0.55 * colour, 0, 255).astype(np.uint8)


def flow_view(prediction: np.ndarray, flow: np.ndarray, full_scale: float = 6.0) -> np.ndarray:
    """Hue shows where the model expects things to move, brightness how fast."""

    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)
    hsv = np.zeros((*flow.shape[:2], 3), np.uint8)
    hsv[..., 0] = np.rint(angle / 2).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.rint(np.clip(magnitude / full_scale, 0, 1) * 255).astype(np.uint8)
    colour = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(np.float32)
    gray = cv2.cvtColor(cv2.cvtColor(prediction, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    return np.clip(0.35 * gray + colour, 0, 255).astype(np.uint8)


def overlay_view(engine: PredictiveEngine, prediction: np.ndarray) -> tuple[np.ndarray, str]:
    routing = getattr(engine, "routing", None)
    if routing is not None:
        return trust_view(prediction, routing, engine.names), "trust"
    flow = getattr(engine, "last_flow", None)
    if flow is not None:
        return flow_view(prediction, flow), "motion"
    return prediction, "none"


def find_learner(engine: PredictiveEngine):
    return getattr(engine, "learner", None)


class LiveState:
    def __init__(
        self,
        engine: PredictiveEngine,
        source: str,
        engine_name: str,
        loop: bool,
        max_frames: int,
        horizon: int = 1,
    ):
        self.engine = engine
        self.source = int(source) if source.isdigit() else source
        self.engine_name = engine_name
        self.loop = loop
        self.max_frames = max_frames
        self.running = True
        self.error = ""
        self.eyes_closed = False
        self.horizon = max(1, int(horizon))
        self._learning_before_dream = True
        self._condition = threading.Condition()
        self._views: dict[str, bytes] = {}
        self._version = 0
        self._history: deque[dict] = deque(maxlen=1800)
        self._metrics = {
            "engine": engine_name,
            "scored": 0,
            "model_mae": 0.0,
            "copy_mae": 0.0,
            "beat_copy": 0.0,
            "inference_ms": 0.0,
            "inference_ms_p95": 0.0,
            "frames_read": 0,
            "fps": 0.0,
            "eyes_closed": False,
            "dream_frames": 0,
            "horizon": self.horizon,
            "overlay": "none",
            "experts": list(getattr(engine, "names", ())),
            "learner": None,
            "error": "",
        }

    def latest(self, view: str) -> tuple[int, bytes]:
        with self._condition:
            return self._version, self._views.get(view, b"")

    def wait_for_change(self, view: str, version: int, timeout: float = 1.0):
        with self._condition:
            self._condition.wait_for(lambda: self._version != version or not self.running, timeout)
            return self._version, self._views.get(view, b"")

    def metrics(self) -> dict:
        with self._condition:
            return dict(self._metrics)

    def history(self) -> list[dict]:
        with self._condition:
            return list(self._history)

    def control(self, command: dict) -> dict:
        learner = find_learner(self.engine)
        if "eyes_closed" in command:
            closed = bool(command["eyes_closed"])
            if closed and not self.eyes_closed and learner is not None:
                # Never learn from its own dreams.
                self._learning_before_dream = not learner.paused
                learner.paused = True
            opening = not closed and self.eyes_closed
            # Stop dreaming before learning resumes, so no imagined frame is learned.
            self.eyes_closed = closed
            if opening and learner is not None:
                learner.paused = not self._learning_before_dream
        if "learning" in command and learner is not None and not self.eyes_closed:
            learner.paused = not bool(command["learning"])
        if command.get("reset_learning") and learner is not None:
            learner.reset()
            with self._condition:
                self._history.clear()
        if "horizon" in command:
            self.horizon = int(np.clip(int(command["horizon"]), 1, 8))
        return {
            "eyes_closed": self.eyes_closed,
            "horizon": self.horizon,
            "learning": learner is not None and not learner.paused,
        }

    def _publish(self, views: dict[str, bytes], metrics: dict) -> None:
        with self._condition:
            if views:
                self._views = views
            self._metrics = metrics
            self._version += 1
            self._condition.notify_all()

    def _learner_metrics(self) -> dict | None:
        learner = find_learner(self.engine)
        if learner is None:
            return None
        stats = learner.stats()
        return {
            "steps": stats.steps,
            "frames_seen": stats.frames_seen,
            "paused": stats.paused,
            "rollbacks": stats.rollbacks,
            "error": stats.error,
            "replay_mae": None if np.isnan(stats.prediction_mae) else stats.prediction_mae,
            "replay_copy_mae": None if np.isnan(stats.copy_mae) else stats.copy_mae,
        }

    def run(self) -> None:
        capture = cv2.VideoCapture(self.source)
        if not capture.isOpened():
            self.error = f"cannot open source {self.source!r}"
            self.running = False
            self._publish({}, self.metrics() | {"error": self.error})
            return
        is_file = isinstance(self.source, str) and not self.source.isdigit()
        fps = capture.get(cv2.CAP_PROP_FPS) if is_file else 30.0
        interval = 1.0 / (fps if fps > 0 else 30.0)
        self.engine.reset()
        pending: EnginePrediction | None = None
        previous_actual = None
        scored = wins = dream_frames = 0
        model_error = copy_error = 0.0
        latencies: deque[float] = deque(maxlen=256)
        frame_times: deque[float] = deque(maxlen=60)
        bucket = {"model": [], "copy": [], "cortical": []}
        bucket_start = time.perf_counter()
        started = bucket_start
        frame_count = 0
        next_tick = time.perf_counter()
        try:
            while self.running and (not self.max_frames or frame_count < self.max_frames):
                ok, frame = capture.read()
                if not ok:
                    if is_file and self.loop:
                        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    break
                frame_times.append(time.perf_counter())
                actual = cv2.resize(
                    frame, (self.engine.width, self.engine.height), interpolation=cv2.INTER_AREA
                )
                dreaming = self.eyes_closed and pending is not None
                dream_frames = dream_frames + 1 if dreaming else 0
                # With eyes closed the model sees only its own last prediction.
                current = self.engine.predict(pending.image if dreaming else actual)
                latencies.append(current.inference_ms)
                if pending is not None:
                    model_mae = float(
                        np.mean(np.abs(pending.image.astype(np.float32) - actual)) / 255.0
                    )
                    copy_mae = float(
                        np.mean(np.abs(previous_actual.astype(np.float32) - actual)) / 255.0
                    )
                    model_error += model_mae
                    copy_error += copy_mae
                    wins += int(model_mae < copy_mae)
                    scored += 1
                    bucket["model"].append(model_mae)
                    bucket["copy"].append(copy_mae)
                    experts = getattr(self.engine, "last_expert_errors", {})
                    bucket["cortical"].append(experts.get("cortical", model_mae))
                    error = error_view(pending.image, actual)
                else:
                    error = np.zeros_like(actual)
                shown = current.image
                if self.horizon > 1 and hasattr(self.engine, "rollout"):
                    shown = self.engine.rollout(current.image, self.horizon - 1)
                overlay, overlay_kind = overlay_view(self.engine, shown)
                images = {"camera": actual, "prediction": shown, "error": error, "overlay": overlay}
                views = {}
                for name, image in images.items():
                    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])
                    if not ok:
                        raise RuntimeError("JPEG encoding failed")
                    views[name] = encoded.tobytes()
                now = time.perf_counter()
                if now - bucket_start >= 1.0 and bucket["model"]:
                    learner = self._learner_metrics()
                    with self._condition:
                        self._history.append(
                            {
                                "t": round(now - started, 1),
                                "model": float(np.mean(bucket["model"])),
                                "copy": float(np.mean(bucket["copy"])),
                                "cortical": float(np.mean(bucket["cortical"])),
                                "dreaming": self.eyes_closed,
                                "steps": learner["steps"] if learner else 0,
                            }
                        )
                    bucket = {key: [] for key in bucket}
                    bucket_start = now
                span = frame_times[-1] - frame_times[0] if len(frame_times) > 1 else 0.0
                self._publish(
                    views,
                    {
                        "engine": self.engine_name,
                        "scored": scored,
                        "model_mae": model_error / max(scored, 1),
                        "copy_mae": copy_error / max(scored, 1),
                        "beat_copy": wins / max(scored, 1),
                        "inference_ms": float(np.mean(latencies)),
                        "inference_ms_p95": float(np.percentile(latencies, 95)),
                        "frames_read": frame_count + 1,
                        "fps": (len(frame_times) - 1) / span if span else 0.0,
                        "eyes_closed": self.eyes_closed,
                        "dream_frames": dream_frames,
                        "horizon": self.horizon,
                        "overlay": overlay_kind,
                        "experts": list(getattr(self.engine, "names", ())),
                        "learner": self._learner_metrics(),
                        "error": "",
                    },
                )
                previous_actual = actual
                pending = current
                frame_count += 1
                next_tick += interval
                if is_file:
                    time.sleep(max(0.0, next_tick - time.perf_counter()))
        except Exception as error:
            self.error = f"{type(error).__name__}: {error}"
            self._publish({}, self.metrics() | {"error": self.error})
        finally:
            capture.release()
            self.running = False
            with self._condition:
                self._condition.notify_all()


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
        state: LiveState = self.server.state
        path = self.path.split("?")[0]
        if path == "/":
            self._bytes(200, "text/html; charset=utf-8", INDEX_HTML.encode())
            return
        if path == "/health":
            self._json({"ok": state.running, "error": state.error})
            return
        if path == "/metrics.json":
            self._json(state.metrics())
            return
        if path == "/history.json":
            self._json(state.history())
            return
        if path == "/state.json":
            # One poll for the page, so streams and polls fit the browser's
            # six-connection limit per host.
            self._json({"metrics": state.metrics(), "history": state.history()})
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
                    if not jpeg:
                        continue
                    self.wfile.write(
                        b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                    )
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return
            return
        self._bytes(404, "text/plain; charset=utf-8", b"not found")

    def do_POST(self):
        if self.path.split("?")[0] != "/control":
            self._bytes(404, "text/plain; charset=utf-8", b"not found")
            return
        length = min(int(self.headers.get("Content-Length") or 0), 4096)
        try:
            command = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(command, dict):
                raise ValueError("expected a JSON object")
        except ValueError as error:
            self._bytes(400, "text/plain; charset=utf-8", str(error).encode())
            return
        self._json(self.server.state.control(command))

    def log_message(self, *_):
        return


def serve(
    engine: PredictiveEngine,
    source: str,
    engine_name: str,
    host: str,
    port: int,
    loop: bool,
    max_frames: int,
    horizon: int = 1,
) -> None:
    state = LiveState(engine, source, engine_name, loop, max_frames, horizon)
    server = ThreadingHTTPServer((host, port), Handler)
    server.state = state
    server.daemon_threads = True
    capture_thread = threading.Thread(target=state.run, name="cortical-capture", daemon=True)
    capture_thread.start()
    print(f"Cortical Mirror: http://{host}:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        state.running = False
        with state._condition:
            state._condition.notify_all()
        capture_thread.join(timeout=5)
        server.server_close()
        learner = find_learner(engine)
        if learner is not None:
            learner.stop()
    if state.error:
        raise RuntimeError(state.error)
