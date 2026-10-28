import numpy as np
import pytest

from cortical_mirror.scene.front import CortexFront, LuminanceFront
from cortical_mirror.scene.world import SpriteWorld


@pytest.fixture(scope="module")
def front():
    return CortexFront()  # an untrained V1 is enough to test the plumbing


def test_luminance_front_is_the_image_itself():
    image = SpriteWorld(seed=0).render()
    assert np.array_equal(LuminanceFront().code(image), image)


def test_cortex_code_has_the_graphs_size_and_range(front):
    code = front.code(SpriteWorld(seed=0).render())
    assert code.shape == (192, 256) and code.dtype == np.float32
    assert 0.0 <= code.min() and code.max() <= 1.0


def test_cortex_code_barely_changes_when_the_light_does(front):
    image = SpriteWorld(seed=0, objects=2).render()
    brighter = np.clip(1.3 * image + 0.08, 0, 1)
    code_change = np.abs(front.code(brighter) - front.code(image)).mean()
    light_change = np.abs(brighter - image).mean()
    assert code_change < 0.25 * light_change
