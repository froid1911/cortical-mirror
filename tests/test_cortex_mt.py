import torch

from cortical_mirror.cortex.mt import MotionLevel
from tests.cortex_fixtures import texture


def moving(frames: int, dx: int, dy: int, channels: int = 4, seed: int = 0):
    """Sparse non-negative 'V1 activity' that moves by (dx, dy) positions per frame."""

    base = texture(48 + 2 * frames * abs(dy), 64 + 2 * frames * abs(dx), seed)
    base = torch.cat([base.roll(c * 5, -1) for c in range(channels)], 1)
    activity = (base - 0.6).clamp(min=0)
    out = []
    for t in range(frames):
        y0 = frames * abs(dy) - dy * t
        x0 = frames * abs(dx) - dx * t
        out.append(activity[..., y0 : y0 + 48, x0 : x0 + 64])
    return out


def train(level: MotionLevel, sequences, repeats: int = 3):
    for _ in range(repeats):
        for frames in sequences:
            for before, after in zip(frames, frames[1:], strict=False):
                level.step(before, after, learn=True)


def test_kernels_stay_non_negative_and_sum_to_one():
    level = MotionLevel(types=8, reach=3, seed=0)
    train(level, [moving(8, 1, 0)])
    assert (level.kernels >= 0).all()
    assert torch.allclose(level.kernels.flatten(1).sum(1), torch.ones(8), atol=1e-5)
    assert not level.kernels.requires_grad


def test_responsibilities_sum_to_one_at_every_place():
    level = MotionLevel(types=8, reach=3, seed=0)
    frames = moving(3, 1, 0)
    r = level.responsibilities(frames[0], frames[1])
    assert r.shape[1] == 8
    assert torch.allclose(r.sum(1), torch.ones_like(r[:, 0]), atol=1e-5)


def test_a_newborn_knows_no_motion():
    level = MotionLevel(types=8, reach=3, seed=0)
    assert level.displacements().norm(dim=1).max() < 0.5


def test_experience_of_motion_teaches_a_type_to_predict_it():
    level = MotionLevel(types=8, reach=3, seed=0)
    train(
        level,
        [moving(10, 2, 0, seed=s) for s in range(3)]
        + [moving(10, -2, 0, seed=s) for s in range(3, 6)],
    )
    frames = moving(4, 2, 0, seed=9)
    r = level.responsibilities(frames[1], frames[2])
    winner = int(r.mean((0, 2, 3)).argmax())
    dx, dy = level.displacements()[winner].tolist()
    assert abs(dx - 2) < 0.6 and abs(dy) < 0.6
    velocity = level.velocity(r)[0, :, 8:-8, 8:-8].mean((1, 2))
    assert velocity[0] > 1.0
    opposite = moving(4, -2, 0, seed=9)
    back = level.velocity(level.responsibilities(opposite[1], opposite[2]))[0, :, 8:-8, 8:-8]
    assert back.mean((1, 2))[0] < -1.0
