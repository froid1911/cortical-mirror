import numpy as np

from cortical_mirror.cortex.learned_scene import LearnedScene

H, W, BLOCK = 32, 48, 4


def scene_with(labels: np.ndarray) -> LearnedScene:
    """A scene without a cortex: labels on the grid are given directly."""

    return LearnedScene(cortex=None, slots=None, width=W, height=H)


def background() -> np.ndarray:
    yy, xx = np.mgrid[0:H, 0:W]
    return (0.3 + 0.2 * np.sin(xx / 3.0) * np.cos(yy / 4.0)).astype(np.float32)


def test_the_background_is_remembered_where_the_background_slot_owns_it():
    scene = scene_with(None)
    labels = np.zeros((H // BLOCK, W // BLOCK), np.int64)
    scene.remember(background(), labels)
    assert np.allclose(scene.memory, background())
    assert (scene.seen > 0).all()


def test_deleting_an_object_shows_the_remembered_background():
    scene = scene_with(None)
    empty = np.zeros((H // BLOCK, W // BLOCK), np.int64)
    scene.remember(background(), empty)
    # Later an object sits on the left; the background behind it was seen before.
    frame = background().copy()
    frame[8:16, 8:16] = 0.9
    labels = empty.copy()
    labels[2:4, 2:4] = 1
    scene.remember(frame, labels)
    edited = scene.edit(frame, labels, delete={1})
    assert np.allclose(edited[8:16, 8:16], background()[8:16, 8:16])
    assert np.allclose(edited[20:, 30:], frame[20:, 30:])


def test_never_seen_background_is_inpainted_not_left_as_the_object():
    scene = scene_with(None)
    frame = background().copy()
    frame[8:16, 8:16] = 0.9
    labels = np.zeros((H // BLOCK, W // BLOCK), np.int64)
    labels[2:4, 2:4] = 1
    scene.remember(frame, labels)
    edited = scene.edit(frame, labels, delete={1})
    assert edited[10:14, 10:14].max() < 0.8


def test_moving_an_object_pastes_it_at_the_new_place():
    scene = scene_with(None)
    empty = np.zeros((H // BLOCK, W // BLOCK), np.int64)
    scene.remember(background(), empty)
    frame = background().copy()
    frame[8:16, 8:16] = 0.9
    labels = empty.copy()
    labels[2:4, 2:4] = 1
    edited = scene.edit(frame, labels, move={1: (20, 4)})
    assert np.allclose(edited[12:20, 28:36], 0.9)
    assert np.allclose(edited[8:12, 8:16], background()[8:12, 8:16])


def test_pixel_masks_follow_the_grid_labels():
    scene = scene_with(None)
    labels = np.zeros((H // BLOCK, W // BLOCK), np.int64)
    labels[1, 2] = 3
    masks = scene.masks(labels)
    assert set(masks) == {3}
    assert masks[3].sum() == BLOCK * BLOCK and masks[3][4:8, 8:12].all()
