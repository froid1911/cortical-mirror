import numpy as np

from cortical_mirror.cortex.object_files import ObjectFile

H, W = 96, 128


def world(t: int, x0: float = 30.0, y0: float = 40.0, vx: float = 1.0, vy: float = 0.5):
    """A textured disc of radius 8 sliding over a different texture; returns the
    image, the disc's centre and its true mask."""

    rng = np.random.default_rng(1)
    back = rng.uniform(0.1, 0.6, (H, W)).astype(np.float32)
    patch = np.random.default_rng(2).uniform(0.5, 0.9, (17, 17)).astype(np.float32)
    yy, xx = np.mgrid[-8:9, -8:9]
    disc = np.hypot(xx, yy) <= 8
    x, y = x0 + vx * t, y0 + vy * t
    image, mask = back.copy(), np.zeros((H, W), bool)
    top, left = int(round(y)) - 8, int(round(x)) - 8
    image[top : top + 17, left : left + 17][disc] = patch[disc]
    mask[top : top + 17, left : left + 17] = disc
    return image, (x, y), mask


def iou(a, b):
    return (a & b).sum() / max((a | b).sum(), 1)


def test_a_file_learns_its_object_s_shape_from_what_moves_with_it():
    image, (x, y), mask = world(0)
    file = ObjectFile(x, y, radius=12)
    for t in range(60):
        image, _, mask = world(t)
        file.follow(image, velocity=(1.0, 0.5))
    assert iou(file.mask(H, W), mask) > 0.8


def test_a_file_corrects_a_wrong_velocity_by_its_own_prediction():
    image, (x, y), mask = world(0)
    file = ObjectFile(x, y, radius=12)
    offsets = []
    for t in range(60):
        image, (x, y), mask = world(t)
        # The motion it is told is off by half a pixel per frame: 30 px in all.
        file.follow(image, velocity=(1.5, 0.5))
        offsets.append(file.x - x)
    # The file's frame need not sit on the centre, but it stays with the object.
    assert abs(offsets[-1] - offsets[30]) <= 1.0
    assert iou(file.mask(H, W), mask) > 0.7


def test_a_young_file_has_no_shape_yet():
    image, (x, y), _ = world(0)
    file = ObjectFile(x, y, radius=12, mature=20)
    for t in range(5):
        file.follow(world(t)[0], velocity=(1.0, 0.5))
    assert not file.ready


def test_a_file_knows_when_its_appearance_no_longer_fits():
    image, (x, y), _ = world(0)
    file = ObjectFile(x, y, radius=12)
    for t in range(40):
        file.follow(world(t)[0], velocity=(1.0, 0.5))
    assert file.confident
    # The object vanishes: the file's prediction fails.
    empty = world(40)[0].copy()
    empty[:] = np.random.default_rng(5).uniform(0.1, 0.6, empty.shape)
    file.follow(empty, velocity=(1.0, 0.5))
    assert not file.confident


def test_a_lost_file_finds_its_object_again_near_where_it_is_told():
    image, (x, y), _ = world(0)
    file = ObjectFile(x, y, radius=12)
    for t in range(40):
        file.follow(world(t)[0], velocity=(1.0, 0.5))
    # It is told a wrong velocity for a few frames and loses the object.
    for t in range(40, 46):
        file.follow(world(t)[0], velocity=(-2.0, 0.5))
    image, (x, y), mask = world(46)
    file.follow(image, velocity=(1.0, 0.5))
    file.relocate(image, x + 2, y - 2, search=8)
    assert file.confident
    assert iou(file.mask(H, W), mask) > 0.7


def test_a_file_that_explains_only_background_is_not_sure():
    image, (x, y), _ = world(0)
    back = world(0)[0].copy()
    back[:] = np.random.default_rng(1).uniform(0.1, 0.6, back.shape)  # world()'s background
    # A file that sits on still background from the start.
    file = ObjectFile(100, 70, radius=12)
    for _ in range(40):
        file.follow(back, velocity=(0.0, 0.0), background=back)
    assert not file.confident
