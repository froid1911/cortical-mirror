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
