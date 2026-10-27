import numpy as np

from cortical_mirror.scene import phenomena
from cortical_mirror.scene.world import SpriteWorld


def square(x, y, size=10, shape=(40, 60)):
    mask = np.zeros(shape, bool)
    mask[y : y + size, x : x + size] = True
    return mask


def test_matching_pairs_each_object_with_at_most_one_node_best_first():
    truth = [square(0, 0), square(30, 20)]
    nodes = [square(31, 20), square(0, 0), square(2, 0)]
    pairs = phenomena.match(truth, nodes)
    assert [(t, n) for t, n, _ in pairs] == [(0, 1), (1, 0)]
    assert pairs[0][2] == 1.0 and 0.5 < pairs[1][2] < 1.0


def test_an_object_without_a_node_scores_zero():
    assert phenomena.mean_iou([square(0, 0), square(30, 20)], [square(0, 0)]) == 0.5


def test_the_motion_baseline_finds_a_moving_square_and_ignores_stillness():
    before, after = np.full((60, 80), 0.5, np.float32), np.full((60, 80), 0.5, np.float32)
    after[20:40, 20:40] = 0.9
    masks = phenomena.motion_masks(after, before)
    assert len(masks) == 1 and masks[0][25, 25]
    assert phenomena.motion_masks(before, before) == []


def test_bitmap_delete_fills_the_hole_from_its_surroundings():
    frame = np.full((60, 80), 0.4, np.float32)
    frame[20:30, 20:30] = 0.9
    mask = square(20, 20, 10, (60, 80))
    deleted = phenomena.bitmap_edit(frame, mask, delete=True)
    assert np.allclose(deleted[mask], 0.4, atol=0.02)
    moved = phenomena.bitmap_edit(frame, mask, move=(30, 0))
    assert np.allclose(moved[20:30, 50:60], 0.9) and np.allclose(moved[mask], 0.4, atol=0.02)


def test_discovery_reports_iou_and_count_accuracy_per_condition():
    report = phenomena.object_discovery(seeds=[1000], frames=60, sample_from=40, every=10)
    for condition in ("learning", "frozen", "motion_baseline"):
        assert 0.0 <= report[condition]["mean_iou"] <= 1.0
        assert 0.0 <= report[condition]["count_accuracy"] <= 1.0
    assert report["samples"] == 2


def test_manipulation_scores_graph_and_bitmap_against_the_edited_world():
    report = phenomena.manipulation(seeds=[2001], frames=80)
    assert report["objects"] == 3
    for edit in ("delete", "move"):
        assert report[edit]["edits"] == report["matched"]
        if report["matched"]:
            assert report[edit]["graph_mse"] >= 0.0 and report[edit]["bitmap_mse"] >= 0.0


def test_the_move_offset_points_back_toward_the_middle():
    world = SpriteWorld(seed=0, objects=1)
    world.objects[0].x = 200.0
    assert phenomena.move_offset(world, world.objects[0]) == (-30, 0)
    world.objects[0].x = 40.0
    assert phenomena.move_offset(world, world.objects[0]) == (30, 0)


def test_an_occlusion_trial_hides_its_object_completely_and_lets_it_come_back():
    world = phenomena.occlusion_world(3000)
    seen = []
    for _ in range(120):
        world.step()
        seen.append(int(world.truth()[0]["visible"].sum()))
    hidden = seen.index(0)
    assert 0 < hidden and any(v > 0 for v in seen[hidden:])


def test_the_magic_world_removes_an_object_once_it_is_completely_hidden():
    world = phenomena.occlusion_world(3000)
    for _ in range(120):
        world.step()
        phenomena.vanish_hidden(world)
    assert world.objects == []


def test_training_persistence_reports_where_it_went():
    tau, trajectory = phenomena.train_persistence(seeds=[100, 101], frames=80)
    assert tau >= 1.0 and len(trajectory) == 2


def test_permanence_keeps_a_hidden_object_only_with_a_long_persistence():
    patient = phenomena.object_permanence(60.0, seeds=[3000])
    forgetful = phenomena.object_permanence(1.0, seeds=[3000])
    assert patient["kept"] == 1.0 and forgetful["kept"] == 0.0
    for report in (patient, forgetful):
        assert report["surprise_vanish"] >= 0 and report["surprise_reappear"] >= 0


def test_training_the_motor_map_returns_one_shift_per_command():
    motor = phenomena.train_motor(seeds=[400], frames=60)
    assert motor.shape == (9, 2)


def test_a_true_motor_map_sees_its_own_saccades_as_still_and_a_zero_map_does_not():
    from cortical_mirror.cortex.eye import COMMANDS
    from cortical_mirror.scene.eye import SACCADE

    true = np.array(COMMANDS, float) * SACCADE
    adult = phenomena.self_and_world(true, seeds=[4000])
    newborn = phenomena.self_and_world(np.zeros((9, 2)), seeds=[4000])
    assert adult["commanded"] <= 0.1 and adult["uncommanded"] >= 0.8
    assert newborn["commanded"] >= 0.8
    assert adult["prediction_ratio"] < newborn["prediction_ratio"]


def test_the_attention_world_has_fresh_noise_in_one_place_and_a_late_newcomer():
    trial = phenomena.AttentionTrial(5000, novel_at=5)
    first, second = trial.frame(0), trial.frame(1)
    x, y = trial.noise
    patch = (slice(y - 48, y + 48), slice(x - 48, x + 48))
    assert not np.allclose(first[patch], second[patch])
    assert len(trial.world.objects) == 3
    for t in range(2, 7):
        trial.frame(t, gaze=(192, 144))
    assert len(trial.world.objects) == 4


def test_attention_measures_report_noise_capture_world_error_and_novelty():
    report = phenomena.attention_trials("random", seeds=[5000], frames=60, novel_at=30)
    assert 0.0 <= report["noise_capture"] <= 1.0
    assert report["world_error"] > 0.0
    assert 0 <= report["time_to_novelty"] <= 30


def test_training_attention_returns_one_weight_per_feature():
    from cortical_mirror.scene.attention import FEATURES

    weights = phenomena.train_attention(seeds=[500], frames=60)
    assert weights.shape == (len(FEATURES),)
