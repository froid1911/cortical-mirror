"""Headless live demo served as MJPEG plus small JSON metrics."""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

from .engines import EnginePrediction, PredictiveEngine
from .evaluate import comparison_view, label

INDEX_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Cortical Mirror</title><style>
:root{color-scheme:dark;--bg:#0c1015;--panel:#151b23;--text:#e7edf5;--muted:#91a0b3;--accent:#70e1b0}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 system-ui,sans-serif}
main{max-width:1200px;margin:auto;padding:28px 20px}header{display:flex;align-items:baseline;justify-content:space-between;gap:16px;flex-wrap:wrap}
h1{font-size:28px;letter-spacing:-.03em;margin:0}#engine{color:var(--accent);font-weight:600}
.stream{margin-top:20px;background:#000;border:1px solid #2a3542;border-radius:12px;overflow:hidden}
img{display:block;width:100%;height:auto}.metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(145px,1fr));gap:12px;margin-top:16px}
.card{background:var(--panel);border:1px solid #25303c;border-radius:10px;padding:14px}.name{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.08em}
.value{font-size:24px;font-variant-numeric:tabular-nums;margin-top:4px}.hint{color:var(--muted);margin-top:16px;max-width:780px}
#error{color:#ff8d8d}footer{margin-top:18px;color:var(--muted);font-size:13px}
</style></head><body><main><header><div><h1>Cortical Mirror</h1><div class="hint">The future is generated from frames already observed. The lower-left panel is the forecast issued for the frame arriving now.</div></div><div id="engine">connecting…</div></header>
<div class="stream"><img src="/stream.mjpg" alt="Live camera, next-frame prediction, prediction due now, and prediction error"></div>
<section class="metrics">
<div class="card"><div class="name">Prediction MAE</div><div class="value" id="mae">—</div></div>
<div class="card"><div class="name">Persistence MAE</div><div class="value" id="copy">—</div></div>
<div class="card"><div class="name">Frames beating copy</div><div class="value" id="wins">—</div></div>
<div class="card"><div class="name">Inference</div><div class="value" id="latency">—</div></div>
<div class="card"><div class="name">Scored frames</div><div class="value" id="frames">0</div></div>
</section><div id="error"></div><footer>MAE is normalized pixel error; lower is better. Timing includes host transfer and synchronization.</footer>
</main><script>
async function update(){try{const r=await fetch('/metrics.json',{cache:'no-store'});const m=await r.json();
const text=(id,v)=>document.getElementById(id).textContent=v;
text('engine',m.engine);text('mae',m.model_mae.toFixed(4));text('copy',m.copy_mae.toFixed(4));
text('wins',(100*m.beat_copy).toFixed(1)+'%');text('latency',m.inference_ms.toFixed(1)+' ms');text('frames',m.scored);
text('error',m.error||'');}catch(e){document.getElementById('error').textContent=String(e)}}
setInterval(update,500);update();
</script></body></html>"""


class LiveState:
    def __init__(
        self, engine: PredictiveEngine, source: str, engine_name: str, loop: bool, max_frames: int
    ):
        self.engine = engine
        self.source = int(source) if source.isdigit() else source
        self.engine_name = engine_name
        self.loop = loop
        self.max_frames = max_frames
        self.running = True
        self.error = ""
        self._condition = threading.Condition()
        self._jpeg = b""
        self._version = 0
        self._metrics = {
            "engine": engine_name,
            "scored": 0,
            "model_mae": 0.0,
            "copy_mae": 0.0,
            "beat_copy": 0.0,
            "inference_ms": 0.0,
            "inference_ms_p95": 0.0,
            "frames_read": 0,
            "error": "",
        }

    def latest(self) -> tuple[int, bytes]:
        with self._condition:
            return self._version, self._jpeg

    def wait_for_change(self, version: int, timeout: float = 1.0) -> tuple[int, bytes]:
        with self._condition:
            self._condition.wait_for(lambda: self._version != version or not self.running, timeout)
            return self._version, self._jpeg

    def metrics(self) -> dict:
        with self._condition:
            return dict(self._metrics)

    def _publish(self, jpeg: bytes, metrics: dict) -> None:
        with self._condition:
            self._jpeg = jpeg
            self._metrics = metrics
            self._version += 1
            self._condition.notify_all()

    def run(self) -> None:
        capture = cv2.VideoCapture(self.source)
        if not capture.isOpened():
            self.error = f"cannot open source {self.source!r}"
            self.running = False
            self._publish(b"", self.metrics() | {"error": self.error})
            return
        is_file = isinstance(self.source, str) and not self.source.isdigit()
        fps = capture.get(cv2.CAP_PROP_FPS) if is_file else 30.0
        interval = 1.0 / (fps if fps > 0 else 30.0)
        self.engine.reset()
        pending: EnginePrediction | None = None
        previous_actual = None
        scored = wins = 0
        model_error = copy_error = 0.0
        latencies = []
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
                actual = cv2.resize(
                    frame, (self.engine.width, self.engine.height), interpolation=cv2.INTER_AREA
                )
                current = self.engine.predict(actual)
                latencies.append(current.inference_ms)
                latencies = latencies[-256:]
                if pending is not None:
                    difference = pending.image.astype(np.float32) - actual.astype(np.float32)
                    copy_difference = previous_actual.astype(np.float32) - actual.astype(np.float32)
                    model_mae = float(np.mean(np.abs(difference)) / 255.0)
                    copy_mae = float(np.mean(np.abs(copy_difference)) / 255.0)
                    model_error += model_mae
                    copy_error += copy_mae
                    wins += int(model_mae < copy_mae)
                    scored += 1
                    view = comparison_view(actual, pending, previous_actual, current)
                else:
                    blank = np.zeros_like(actual)
                    view = np.vstack(
                        (
                            np.hstack(
                                (
                                    label(actual, "camera now"),
                                    label(current.image, "predicted next"),
                                )
                            ),
                            np.hstack(
                                (
                                    label(blank, "waiting for due prediction"),
                                    label(blank, "waiting for error"),
                                )
                            ),
                        )
                    )
                ok, encoded = cv2.imencode(".jpg", view, [cv2.IMWRITE_JPEG_QUALITY, 90])
                if not ok:
                    raise RuntimeError("JPEG encoding failed")
                self._publish(
                    encoded.tobytes(),
                    {
                        "engine": self.engine_name,
                        "scored": scored,
                        "model_mae": model_error / max(scored, 1),
                        "copy_mae": copy_error / max(scored, 1),
                        "beat_copy": wins / max(scored, 1),
                        "inference_ms": float(np.mean(latencies)),
                        "inference_ms_p95": float(np.percentile(latencies, 95)),
                        "frames_read": frame_count + 1,
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
            self._publish(self.latest()[1], self.metrics() | {"error": self.error})
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

    def do_GET(self):
        state: LiveState = self.server.state
        path = self.path.split("?")[0]
        if path == "/":
            self._bytes(200, "text/html; charset=utf-8", INDEX_HTML.encode())
            return
        if path == "/health":
            self._bytes(
                200,
                "application/json",
                json.dumps({"ok": state.running, "error": state.error}).encode(),
            )
            return
        if path == "/metrics.json":
            self._bytes(200, "application/json", json.dumps(state.metrics()).encode())
            return
        if path == "/stream.mjpg":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            version = -1
            try:
                while state.running:
                    version, jpeg = state.wait_for_change(version)
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
) -> None:
    state = LiveState(engine, source, engine_name, loop, max_frames)
    capture_thread = threading.Thread(target=state.run, name="cortical-capture", daemon=True)
    capture_thread.start()
    server = ThreadingHTTPServer((host, port), Handler)
    server.state = state
    server.daemon_threads = True
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
    if state.error:
        raise RuntimeError(state.error)
