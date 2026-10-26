import cv2
import numpy as np

from cortical_mirror.engines import CopyEngine, EnginePrediction
from cortical_mirror.evaluate import comparison_view, evaluate_video


def test_evaluation_pairs_prediction_with_its_due_frame(tmp_path):
    path = tmp_path / "sequence.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), 10, (64, 48))
    assert writer.isOpened()
    for value in (10, 20, 30, 40):
        writer.write(np.full((48, 64, 3), value, dtype=np.uint8))
    writer.release()

    result = evaluate_video(
        CopyEngine(width=64, height=48), path, output=tmp_path / "out", max_frames=4, samples=4
    )
    assert result["frames_read"] == 4
    assert result["model"]["frames"] == 3
    assert result["beat_copy"] == 0.0
    for index, (actual_value, predicted_value) in enumerate(
        ((20, 10), (30, 20), (40, 30)), start=1
    ):
        actual = cv2.imread(str(tmp_path / "out" / f"{index:03d}_actual.png"))
        predicted = cv2.imread(str(tmp_path / "out" / f"{index:03d}_pred.png"))
        assert abs(float(actual.mean()) - actual_value) <= 1.0
        assert abs(float(predicted.mean()) - predicted_value) <= 1.0


def test_live_view_contains_both_next_and_due_predictions():
    actual = np.zeros((48, 64, 3), dtype=np.uint8)
    due = EnginePrediction(np.full_like(actual, 10), actual, None, 0.0)
    future = EnginePrediction(np.full_like(actual, 20), actual, None, 0.0)
    view = comparison_view(actual, due, actual.copy(), future)
    assert view[32, 69, 0] == 20
    assert view[80, 5, 0] == 10


def test_live_server_paces_and_loops_only_real_files(tmp_path, monkeypatch):
    import cortical_mirror.server as server
    from cortical_mirror.engines import CopyEngine

    opened = []

    class FakeCapture:
        def __init__(self, source):
            opened.append(source)

        def isOpened(self):
            return True

        def get(self, prop):
            return 25.0

        def read(self):
            return False, None

        def set(self, *args):
            raise AssertionError("a live stream must not be rewound")

        def release(self):
            pass

    monkeypatch.setattr(server.cv2, "VideoCapture", FakeCapture)
    state = server.LiveState(CopyEngine(64, 48), "tcp://127.0.0.1:9000", "copy", True, 0)
    state.run()  # ends at the first missing frame instead of looping
    assert opened == ["tcp://127.0.0.1:9000"]
    assert not state.error
