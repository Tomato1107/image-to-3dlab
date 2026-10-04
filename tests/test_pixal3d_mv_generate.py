"""Tests for the Pixal3D multiview wrapper.

The whole mode stands or falls on two contracts: the cameras written beside the run are
exactly the canonical turntable rig the mv weights were trained on, and matting never
re-crops the frame -- transforms.json describes the framing as given, so a crop would
point every camera at the wrong place.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "pixal3d_mv_generate.py"


def _load():
    spec = importlib.util.spec_from_file_location("pixal3d_mv_generate", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mv = _load()

DISTANCE = 3.1192049980163574


def test_the_staged_cameras_are_the_canonical_turntable_rig():
    meta = mv.transforms_json()
    assert meta["mesh_scale"] == 1.0
    assert meta["camera_angle_x"] == pytest.approx(0.3490658503988659)
    frames = meta["frames"]
    assert [frame["file_path"] for frame in frames] == list(mv.VIEW_NAMES)
    rotations = []
    for frame in frames:
        matrix = frame["transform_matrix"]
        translations = [abs(matrix[row][3]) for row in range(3)]
        assert max(translations) == pytest.approx(DISTANCE)
        rotations.append(tuple(tuple(row[:3]) for row in matrix[:3]))
    # front, right, back and left are four distinct orientations.
    assert len(set(rotations)) == 4


def _photo(path: Path, colour) -> Path:
    Image.new("RGB", (20, 12), colour).save(path)
    return path


def test_stage_views_mattes_each_image_keeping_the_frame(tmp_path):
    images = [_photo(tmp_path / f"in{i}.png", (10 * i, 0, 0)) for i in range(4)]
    calls = []

    def fake_matte(path: Path) -> Image.Image:
        calls.append(path)
        cut = np.zeros((12, 20, 4), dtype=np.uint8)
        cut[:, 10:, :3] = 200
        cut[:, 10:, 3] = 255
        return Image.fromarray(cut)

    views = mv.stage_views(images, tmp_path / "views", matte_fn=fake_matte)
    assert calls == images
    meta = json.loads((views / "transforms.json").read_text())
    assert [frame["file_path"] for frame in meta["frames"]] == list(mv.VIEW_NAMES)
    kept = np.asarray(Image.open(views / mv.VIEW_NAMES[0]))
    assert kept.shape == (12, 20, 4)
    assert (kept[:, :10, 3] == 0).all() and (kept[:, 10:, 3] == 255).all()


def test_a_pre_matted_image_goes_in_without_being_matted_again(tmp_path):
    rgba = np.zeros((12, 20, 4), dtype=np.uint8)
    rgba[:, 8:, 3] = 255
    rgba[:, 8:, :3] = 120
    images = []
    for index in range(4):
        path = tmp_path / f"pre{index}.png"
        Image.fromarray(rgba).save(path)
        images.append(path)

    def forbidden(path):  # pragma: no cover - must not run
        raise AssertionError(f"matted an already-matted image: {path}")

    views = mv.stage_views(images, tmp_path / "views", matte_fn=forbidden)
    kept = np.asarray(Image.open(views / mv.VIEW_NAMES[2]))
    assert kept.shape == (12, 20, 4)
    assert (kept[:, :8, 3] == 0).all() and (kept[:, 8:, 3] == 255).all()


def test_build_command_runs_the_mv_weights_on_the_views_dir():
    command = mv.build_command(
        Path("views"), Path("out.glb"), 1024, 42,
        models=Path("models"), cli=Path("trellis-cli"), gss=10.0,
    )
    assert command[command.index("--views") + 1] == "views"
    assert command[command.index("--pixal3d-weights") + 1] == "mv"
    assert command[command.index("--res") + 1] == "1024"
    assert command[command.index("--gss") + 1] == "10.0"
    assert command[-1] == "out.glb"


def test_exactly_four_views_are_required():
    with pytest.raises(SystemExit):
        mv.main(["a.png", "b.png", "c.png", "out.glb"])


def test_brighten_lifts_a_dark_subject_to_the_target():
    rgba = np.zeros((10, 10, 4), dtype=np.uint8)
    rgba[:, 5:, :3] = 50
    rgba[:, 5:, 3] = 255
    lifted, gain = mv.brighten_to_target(rgba, 110.0)
    assert gain == pytest.approx(110.0 / 50.0)
    assert lifted[..., :3][lifted[..., 3] > 128].mean() == pytest.approx(110.0, rel=0.01)
    assert (lifted[..., 3] == rgba[..., 3]).all()  # alpha untouched


def test_brighten_gain_is_clamped_and_never_darkens():
    rgba = np.zeros((6, 6, 4), dtype=np.uint8)
    rgba[..., :3] = 10
    rgba[..., 3] = 255
    lifted, gain = mv.brighten_to_target(rgba, 110.0, max_gain=2.0)
    assert gain == 2.0 and lifted[..., :3].mean() == pytest.approx(20.0, rel=0.01)
    bright = np.full((6, 6, 4), 200, dtype=np.uint8)
    bright[..., 3] = 255
    same, gain = mv.brighten_to_target(bright, 110.0)
    assert gain == 1.0 and (same[..., :3] == 200).all()


def test_stage_views_brightens_after_matting(tmp_path):
    images = [_photo(tmp_path / f"in{i}.png", (10 * i, 0, 0)) for i in range(4)]

    def dark_matte(path):
        cut = np.zeros((12, 20, 4), dtype=np.uint8)
        cut[:, 10:, :3] = 50
        cut[:, 10:, 3] = 255
        return Image.fromarray(cut)

    views = mv.stage_views(images, tmp_path / "views", matte_fn=dark_matte, brighten=110.0)
    kept = np.asarray(Image.open(views / mv.VIEW_NAMES[0]))
    subject = kept[..., :3][kept[..., 3] > 128]
    assert subject.mean() == pytest.approx(110.0, rel=0.01)
