import numpy as np

from cortical_mirror.scene.eye import look
from cortical_mirror.scene.motion import PartMotion
from cortical_mirror.scene.world import SpriteWorld


def frames(count: int, pan: float = 2.0, seed: int = 0):
    world = SpriteWorld(seed=seed, width=640, height=192, objects=1, speed=(0.0, 0.0))
    thing = world.objects[0]
    thing.x, thing.y, thing.vx = 300.0, 96.0, -1.5
    out = []
    for t in range(count):
        world.step()
        out.append((look(world.render(), (128 + pan * t, 96), (256, 192)), world, thing, t))
    return out


def test_parts_moving_with_the_camera_predict_a_pan_better_than_repeating():
    motion, expected, repeated, previous = PartMotion(), [], [], None
    for image, *_ in frames(40):
        if previous is not None and motion.ready:
            expected.append(np.mean((motion.predict(previous) - image) ** 2))
            repeated.append(np.mean((previous - image) ** 2))
        if previous is not None:
            motion.observe(previous, image, parts={})
        previous = image
    assert np.mean(expected) < 0.5 * np.mean(repeated)


def test_an_object_moving_against_the_pan_is_predicted_by_its_own_part():
    motion, previous = PartMotion(), None
    errors = {"with_part": [], "background_only": []}
    background_only = PartMotion()
    for image, _world, thing, t in frames(40):
        x = int(round(thing.x - 2.0 * t))  # the object's place in the view
        mask = np.zeros(image.shape, bool)
        mask[max(0, 96 - 16) : 96 + 16, max(0, x - 16) : x + 16] = True
        if previous is not None and motion.ready and mask.any():
            region = mask
            errors["with_part"].append(np.mean((motion.predict(previous) - image)[region] ** 2))
            errors["background_only"].append(
                np.mean((background_only.predict(previous) - image)[region] ** 2)
            )
        if previous is not None:
            motion.observe(previous, image, parts={0: mask})
            background_only.observe(previous, image, parts={})
        previous = image
    assert np.mean(errors["with_part"]) < np.mean(errors["background_only"])


def test_the_video_graph_predicts_from_part_motion_by_default():
    from cortical_mirror.scene.video import VideoGraph

    assert VideoGraph().motion is not None
    assert VideoGraph(motion=None).motion is None


def test_part_motion_can_use_a_given_flow():
    a = np.zeros((40, 40), np.float32)
    a[10:20, 10:20] = 1.0
    b = np.roll(a, 3, axis=1)
    flow = np.zeros((40, 40, 2), np.float32)
    flow[..., 0] = 3.0
    motion = PartMotion(strength=1.0, history=1)
    motion.observe(a, b, parts={}, flow=flow)
    predicted = motion.predict(b)
    future = np.roll(b, 3, axis=1)
    assert np.abs(predicted - future).mean() < np.abs(b - future).mean()


def test_a_video_graph_with_silenced_learned_motion_predicts_no_motion():
    from cortical_mirror.cortex.mt import MotionCortex
    from cortical_mirror.scene.video import VideoGraph

    cortex = MotionCortex(types=4, seed=0, settle=5)
    cortex.silenced = True
    graph = VideoGraph(motion_source="learned", mt=cortex)
    images = [image for image, *_ in frames(8)]
    for image in images[:-1]:
        graph.observe(image)
    assert np.allclose(graph.expect(), images[-2], atol=1e-4)
