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
