from argparse import Namespace
from pathlib import Path

from cortical_mirror import cli


def demo_args(**overrides):
    values = {"assets": Path("/assets"), "device": "cpu", "pretrained": False, "source": "0"}
    return Namespace(**(values | overrides))


def test_demo_starts_from_scratch_without_any_pretrained_model(monkeypatch):
    fetched = []
    monkeypatch.setattr(cli, "fetch", lambda name, folder: fetched.append(name) or folder / name)
    monkeypatch.setattr(cli, "camera_available", lambda source: True)
    args = demo_args()
    cli.demo_arguments(args)
    assert args.from_scratch and args.learn
    assert args.experts == "copy,cortical"
    assert fetched == []


def test_pretrained_demo_on_cpu_turns_learning_off_and_skips_dmvfn(monkeypatch):
    fetched = []
    monkeypatch.setattr(cli, "fetch", lambda name, folder: fetched.append(name) or folder / name)
    monkeypatch.setattr(cli, "camera_available", lambda source: False)
    args = demo_args(pretrained=True)
    cli.demo_arguments(args)
    assert not args.from_scratch and not args.learn
    assert args.experts == "copy,cortical"
    assert fetched == ["cortical.pt", "sample.mp4"]
    assert args.source.endswith("sample.mp4")


def test_twins_command_pairs_a_fresh_newborn_with_the_saved_adult(tmp_path):
    from cortical_mirror.cortex.engine import CortexEngine

    adult = CortexEngine(seed=3)
    adult.hierarchy.age_frames = 12345
    adult.save(tmp_path / "adult.pt")
    twin = cli.twins_engine(Namespace(adult=tmp_path / "adult.pt", device="cpu", frozen_adult=True))
    assert twin.newborn.age_frames == 0 and twin.newborn.learn
    assert twin.adult.age_frames == 12345 and not twin.adult.learn


def test_twins_without_an_adult_downloads_the_adult_of_the_chosen_phase(monkeypatch):
    fetched = []
    monkeypatch.setattr(cli, "fetch", lambda name, folder: fetched.append(name) or folder / name)
    monkeypatch.setattr(cli, "camera_available", lambda source: True)
    args = Namespace(adult=None, phase=3, assets=Path("/assets"), source="0")
    cli.twins_arguments(args)
    assert fetched == ["cortex-adult-p3.pt"]
    assert args.adult == Path("/assets/cortex-adult-p3.pt")


def test_twins_with_an_adult_and_no_camera_plays_the_sample_clip(monkeypatch):
    fetched = []
    monkeypatch.setattr(cli, "fetch", lambda name, folder: fetched.append(name) or folder / name)
    monkeypatch.setattr(cli, "camera_available", lambda source: False)
    args = Namespace(adult=Path("mine.pt"), phase=1, assets=Path("/assets"), source="0")
    cli.twins_arguments(args)
    assert fetched == ["sample.mp4"]
    assert args.adult == Path("mine.pt") and args.source.endswith("sample.mp4")


def test_every_adult_phase_has_a_download():
    for phase in (1, 2, 3):
        assert f"cortex-adult-p{phase}.pt" in cli.DEMO_ASSETS


def test_a_failed_adult_download_says_to_pass_an_adult(monkeypatch, tmp_path):
    import pytest

    def offline(url):
        raise OSError("offline")

    monkeypatch.setattr(cli.urllib.request, "urlopen", offline)
    with pytest.raises(SystemExit, match="--adult"):
        cli.fetch("cortex-adult-p1.pt", tmp_path)


def test_the_scene_command_downloads_the_adult_and_plays_the_sprite_world(monkeypatch):
    fetched = []
    monkeypatch.setattr(cli, "fetch", lambda name, folder: fetched.append(name) or folder / name)
    args = Namespace(adult=None, newborn=False, assets=Path("/assets"), source="world")
    cli.scene_arguments(args)
    assert fetched == ["scene-adult.pt"] and args.adult == Path("/assets/scene-adult.pt")


def test_a_newborn_scene_needs_no_download(monkeypatch):
    fetched = []
    monkeypatch.setattr(cli, "fetch", lambda name, folder: fetched.append(name) or folder / name)
    args = Namespace(adult=None, newborn=True, assets=Path("/assets"), source="world")
    cli.scene_arguments(args)
    assert fetched == [] and args.adult is None


def test_the_scene_graph_takes_the_adults_persistence(tmp_path):
    from cortical_mirror.scene.graph import NEWBORN_PERSISTENCE, SceneGraph

    SceneGraph(persistence=99.0).save(tmp_path / "adult.pt")
    graph = cli.scene_graph(Namespace(adult=tmp_path / "adult.pt", newborn=False))
    assert graph.persistence == 99.0
    assert cli.scene_graph(Namespace(adult=None, newborn=True)).persistence == NEWBORN_PERSISTENCE


def test_the_scene_adult_has_a_download():
    assert "scene-adult.pt" in cli.DEMO_ASSETS


def test_a_camera_scene_downloads_the_learned_motion_cells(monkeypatch):
    fetched = []
    monkeypatch.setattr(cli, "fetch", lambda name, folder: fetched.append(name) or folder / name)
    args = Namespace(
        adult=None, newborn=False, assets=Path("/assets"), source="0", motion="learned", mt=None
    )
    cli.scene_arguments(args)
    assert fetched == ["scene-adult.pt", "mt-adult.pt"]
    assert args.mt == Path("/assets/mt-adult.pt")


def test_engineered_flow_needs_no_motion_cells(monkeypatch):
    fetched = []
    monkeypatch.setattr(cli, "fetch", lambda name, folder: fetched.append(name) or folder / name)
    args = Namespace(
        adult=None, newborn=True, assets=Path("/assets"), source="0", motion="flow", mt=None
    )
    cli.scene_arguments(args)
    assert fetched == []
