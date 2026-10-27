import numpy as np

from cortical_mirror.scene.world import SpriteWorld


def test_same_seed_gives_the_same_frames_and_another_seed_does_not():
    a, b, c = SpriteWorld(seed=1), SpriteWorld(seed=1), SpriteWorld(seed=2)
    for _ in range(5):
        a.step(), b.step(), c.step()
    assert np.array_equal(a.render(), b.render())
    assert not np.array_equal(a.render(), c.render())


def test_frames_are_gray_images_of_the_world_size():
    world = SpriteWorld(seed=0, width=256, height=192)
    image = world.render()
    assert image.shape == (192, 256) and image.dtype == np.float32
    assert 0.0 <= image.min() and image.max() <= 1.0
    frame = world.frame()
    assert frame.shape == (192, 256, 3) and frame.dtype == np.uint8


def test_objects_move_and_stay_inside_the_world():
    world = SpriteWorld(seed=3, objects=3)
    start = [(o.x, o.y) for o in world.objects]
    for _ in range(300):
        world.step()
        for o in world.objects:
            assert 0 <= o.x <= world.width and 0 <= o.y <= world.height
    assert [(o.x, o.y) for o in world.objects] != start


def test_truth_gives_each_object_its_visible_mask_with_nearer_ones_cutting_it():
    world = SpriteWorld(seed=0, objects=2)
    back, front = sorted(world.objects, key=lambda o: o.depth)
    front.x, front.y = back.x, back.y
    truth = {t["id"]: t for t in world.truth()}
    assert truth[back.id]["visible"].sum() < truth[back.id]["full"].sum()
    assert truth[front.id]["visible"].sum() == truth[front.id]["full"].sum()


def test_an_occluder_hides_what_passes_behind_it():
    world = SpriteWorld(seed=0, objects=1, occluders=1)
    bar, thing = world.occluders[0], world.objects[0]
    thing.x, thing.y = bar.x, bar.y
    assert world.truth()[0]["visible"].sum() < world.truth()[0]["full"].sum()


def test_deleting_an_object_reveals_the_background_and_moving_it_draws_it_elsewhere():
    world = SpriteWorld(seed=4, objects=1)
    thing = world.objects[0]
    full = world.truth()[0]["full"]
    gone = world.render(deleted={thing.id})
    assert np.allclose(gone[full], world.background_view()[full])
    moved = world.render(moved={thing.id: (20, 0)})
    assert not np.allclose(moved, world.render())
    shifted = world.render(moved={thing.id: (20, 0)})
    assert np.array_equal(moved, shifted)
    assert (thing.x, thing.y) == (world.objects[0].x, world.objects[0].y)


def test_a_wide_screen_hides_a_small_object_completely():
    world = SpriteWorld(seed=0, objects=1, occluders=1, occluder_width=44, radius=(7, 9))
    screen, thing = world.occluders[0], world.objects[0]
    thing.x, thing.y = screen.x, screen.y
    assert world.truth()[0]["full"].sum() > 0
    assert world.truth()[0]["visible"].sum() == 0
