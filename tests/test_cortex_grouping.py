import numpy as np
import torch

from cortical_mirror.cortex.grouping import AssociationField, adjusted_rand_index


def blob_code(types: int = 4, h: int = 16, w: int = 24):
    """Two blobs: the left one made of type 0, the right one of type 1."""

    a = torch.zeros(types, h, w)
    a[0, 4:10, 3:9] = 1.0
    a[1, 4:10, 14:20] = 1.0
    return a


def test_a_newborn_field_links_nothing_and_finds_no_groups():
    field = AssociationField(types=4, radius=2)
    a = blob_code()
    assert field.affinity(a).abs().max() == 0
    assert (field.groups(a) == 0).all()


def test_features_that_move_together_become_linked_and_those_that_do_not_become_apart():
    field = AssociationField(types=4, radius=2, rate=1.0)
    a = blob_code()
    velocity = torch.zeros(2, 16, 24)
    velocity[0, :, :12] = 2.0  # the left blob moves right
    velocity[0, :, 12:] = -2.0  # the right blob moves left
    for _ in range(20):
        field.learn(a, velocity)
    centre = field.radius * (2 * field.radius + 1) + field.radius
    assert field.links[0, 0].sum() > 0 and field.links[1, 1].sum() > 0
    assert field.links[0, 1].sum() <= 0
    assert field.links.shape[-1] == (2 * field.radius + 1) ** 2 and centre >= 0


def test_two_separately_linked_blobs_are_two_groups():
    field = AssociationField(types=4, radius=2)
    field.links[0, 0] = 1.0
    field.links[1, 1] = 1.0
    labels = field.groups(blob_code(), minimum=6)
    left, right = labels[4:10, 3:9], labels[4:10, 14:20]
    assert (left > 0).all() and (right > 0).all()
    assert len(torch.unique(left)) == 1 and len(torch.unique(right)) == 1
    assert int(left[0, 0]) != int(right[0, 0])
    assert (labels[0] == 0).all()


def test_adjusted_rand_index_is_one_for_matching_labels_and_near_zero_for_random():
    truth = np.repeat(np.arange(4), 50)
    assert abs(adjusted_rand_index(truth, (truth + 1) % 4) - 1.0) < 1e-9
    rng = np.random.default_rng(0)
    assert abs(adjusted_rand_index(truth, rng.integers(0, 4, truth.size))) < 0.05
