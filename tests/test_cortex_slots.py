import numpy as np
import torch

from cortical_mirror.cortex.grouping import AssociationField
from cortical_mirror.cortex.slots import MotionSlots, components

H, W = 24, 32


def square(top: int, left: int, size: int = 6) -> np.ndarray:
    mask = np.zeros((H, W), bool)
    mask[top : top + size, left : left + size] = True
    return mask


def scene(masks, velocities):
    """V1 code with type 0 on objects and type 1 on the background, and MT's
    velocity (retinal px per frame) on the objects."""

    activity = torch.zeros(2, H, W)
    velocity = torch.zeros(2, H, W)
    background = np.ones((H, W), bool)
    for mask, (vx, vy) in zip(masks, velocities, strict=True):
        m = torch.from_numpy(mask)
        activity[0][m] = 1.0
        velocity[0][m], velocity[1][m] = vx, vy
        background &= ~mask
    activity[1][torch.from_numpy(background)] = 1.0
    return activity, velocity


def iou(a: np.ndarray, b: np.ndarray) -> float:
    return float((a & b).sum() / max((a | b).sum(), 1))


def test_components_finds_four_connected_regions():
    mask = np.zeros((5, 5), bool)
    mask[0, 0:2] = True
    mask[3:5, 3:5] = True
    mask[1, 2] = True  # touches the first region only diagonally
    labels = components(mask)
    assert labels.max() == 3
    assert labels[0, 0] == labels[0, 1] != labels[3, 3]


def test_a_still_scene_has_no_objects():
    slots = MotionSlots(stride=2)
    activity, velocity = scene([], [])
    for _ in range(5):
        labels = slots.step(velocity, activity)
    assert labels.max() == 0


def test_a_moving_square_gets_one_slot():
    slots = MotionSlots(stride=2)
    # 2 retinal px per frame = 1 V1 position per frame, to the right.
    for t in range(8):
        mask = square(8, 4 + t)
        labels = slots.step(*reversed(scene([mask], [(2.0, 0.0)])))
    found = labels.numpy()
    assert len(np.unique(found[mask])) == 1 and found[mask][0] > 0
    assert iou(found > 0, mask) > 0.8


def test_two_squares_moving_apart_get_two_slots():
    slots = MotionSlots(stride=2)
    for t in range(8):
        left, right = square(8, 6 - t // 2), square(8, 13 + t // 2)
        labels = slots.step(*reversed(scene([left, right], [(-1.0, 0.0), (1.0, 0.0)])))
    found = labels.numpy()
    a, b = np.bincount(found[left]).argmax(), np.bincount(found[right]).argmax()
    assert a > 0 and b > 0 and a != b


def test_links_hold_an_object_through_a_still_moment():
    field = AssociationField(types=2, radius=2)
    field.links[0, 0] = 1.0  # object features belong with object features
    field.links[1, 1] = 1.0
    field.links[0, 1] = field.links[1, 0] = -1.0
    scores = {}
    for beta in (0.0, 2.0):
        slots = MotionSlots(stride=2, field=field, beta=beta)
        for t in range(8):
            mask = square(8, 4 + t)
            slots.step(*reversed(scene([mask], [(2.0, 0.0)])))
        for _ in range(30):
            labels = slots.step(*reversed(scene([mask], [(0.0, 0.0)])))
        scores[beta] = iou(labels.numpy() > 0, mask)
    assert scores[2.0] > 0.8
    assert scores[2.0] > scores[0.0]
