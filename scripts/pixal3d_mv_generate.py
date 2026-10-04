#!/usr/bin/env python3
"""End-to-end Pixal3D multiview generation: four views -> one textured GLB.

    python scripts/pixal3d_mv_generate.py front.png right.png back.png left.png output.glb

The four images must show one object from the canonical turntable rig -- front, right,
back, left at elevation 0 -- centred and at the same scale in every frame. trellis-cli
runs the camera-aware *_mv.gguf flow weights on them (--views), with the canonical rig
cameras (azimuth 0/90/180/270, FOV 20 degrees, distance 3.119, mesh_scale 1.0) written to
<output>.views/transforms.json, so the run can be replayed and Pixel Match can project
each photo back onto the model.

Unlike --sv-image, --views does no matting and refuses a fully opaque frame, so every
image is cut out here first -- keeping the original framing, because the cameras describe
the framing as given. Re-cropping to the alpha bounding box would point every camera at
the wrong place, which is exactly what trellis-cli's own --bg-only does, so there is no
CLI fallback: matting needs rembg (scripts/bootstrap_matte.py for the lite model).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from image_to_3dlab.host import executable
from image_to_3dlab.matte import cut_out, is_matted, matte_model
from image_to_3dlab.provenance import sha256_file

PIXAL3D_ROOT = REPO / "vendor" / "pixal3d-cpp"
CLI = executable(PIXAL3D_ROOT / "build", "trellis-cli")
MODELS = PIXAL3D_ROOT / "models" / "pixal3d-mv"

VIEW_NAMES = (
    "view00_azim000.png",
    "view01_azim090.png",
    "view02_azim180.png",
    "view03_azim270.png",
)

# The canonical rig from TencentARC/Pixal3D assets/mv_images/example/transforms.json:
# FOV 20 degrees, mesh_scale 1.0, camera distance 3.119 at elevation 0.
RIG_FOV = 0.3490658503988659
RIG_DISTANCE = 3.1192049980163574
_R = RIG_DISTANCE
RIG_MATRICES = (
    ((1.0, 0.0, 0.0, 0.0), (0.0, 0.0, -1.0, -_R), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
    ((0.0, 0.0, 1.0, _R), (1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
    ((-1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 1.0, _R), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
    ((0.0, 0.0, -1.0, -_R), (-1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
)

# Same default as the single-view wrapper: the CLI's 7.5 can drop thin parts.
DEFAULT_GSS = 10.0

# Turntable photos and video frames are often exposed for a bright backdrop, leaving
# a dark subject that conditions the flows badly (2026-10-02: a mecha turnaround at
# subject mean 49-58/255). stage_views lifts each view to this subject mean by default.
BRIGHTEN_TARGET = 110.0
BRIGHTEN_MAX_GAIN = 2.5


def brighten_to_target(rgba: np.ndarray, target: float,
                       max_gain: float = BRIGHTEN_MAX_GAIN) -> tuple[np.ndarray, float]:
    """Lift an RGBA image's subject mean towards `target`; returns (image, gain).

    Only the opaque subject is measured and scaled; the alpha channel is untouched.
    Gain never goes below 1 (brightening only) and is clamped at `max_gain`, so an
    extremely dark frame is improved without being blown out.
    """
    mask = rgba[..., 3] > 128
    if not mask.any():
        return rgba, 1.0
    mean = float(rgba[..., :3][mask].mean())
    gain = float(np.clip(target / max(mean, 1e-6), 1.0, max_gain))
    out = rgba.copy()
    out[..., :3] = np.clip(np.rint(rgba[..., :3] * gain), 0, 255).astype(np.uint8)
    return out, gain

LICENSE_NAME = "MIT (code + flow weights); DINOv3 License (bundled encoder)"
LICENSE_URL = "https://huggingface.co/raven38/pixal3d-q8_0-v1"


def transforms_json(names: tuple[str, ...] = VIEW_NAMES) -> dict[str, object]:
    """The canonical turntable rig as a transforms.json document for `names`."""
    return {
        "camera_angle_x": RIG_FOV,
        "mesh_scale": 1.0,
        "frames": [
            {
                "file_path": name,
                "name": name.removeprefix("view0").removesuffix(".png"),
                "transform_matrix": [list(row) for row in matrix],
            }
            for name, matrix in zip(names, RIG_MATRICES)
        ],
    }


def _default_matte(image: Path):
    from PIL import Image

    with Image.open(image) as opened:
        cut, _model = cut_out(opened.convert("RGB"))
    return cut


def stage_views(images, directory: Path, matte_fn=None, brighten: float | None = None) -> Path:
    """Write the four views and their transforms.json into `directory`.

    Every frame keeps the framing it arrived with: a pre-matted image goes in untouched,
    anything else is cut out at its own size. `matte_fn(path) -> PIL RGBA image` exists so
    tests can stand in for rembg. `brighten` lifts each view's subject mean luminance
    towards that target (see brighten_to_target).
    """
    from PIL import Image

    matte_fn = matte_fn or _default_matte
    directory.mkdir(parents=True, exist_ok=True)
    for image, name in zip(images, VIEW_NAMES):
        with Image.open(image) as opened:
            if is_matted(opened):
                cut = opened.convert("RGBA")
            else:
                cut = matte_fn(image)
        if brighten:
            lifted, _gain = brighten_to_target(np.asarray(cut), brighten)
            cut = Image.fromarray(lifted)
        cut.save(directory / name)
    (directory / "transforms.json").write_text(
        json.dumps(transforms_json(), indent=2) + "\n", encoding="utf-8"
    )
    return directory


def build_command(views: Path, output: Path, res: int, seed: int,
                  models: Path = MODELS, cli: Path = CLI,
                  gss: float = DEFAULT_GSS) -> list[str]:
    """The `trellis-cli --views` invocation; mv weights, exactly four frames."""
    return [
        str(cli),
        "--views", str(views),
        "--models", str(models),
        "--seed", str(seed),
        "--res", str(res),
        "--pixal3d-weights", "mv",
        "--gss", str(gss),
        str(output),
    ]


def manifest(images, output: Path, *, res: int, seed: int, gss: float,
             seconds: float, remover: str | None, brighten: float | None = None) -> dict[str, object]:
    """The run's provenance record, written beside the GLB as `<output>.json`."""
    return {
        "schema_version": 1,
        "backend": "pixal3d-mv",
        "inputs": [
            {"path": str(image), "sha256": sha256_file(image)} for image in images
        ],
        "output": {"path": str(output), "sha256": sha256_file(output)},
        "parameters": {
            "res": res, "seed": seed, "gss": gss, "brighten": brighten,
            "views": list(VIEW_NAMES), "fov": RIG_FOV, "rig_distance": RIG_DISTANCE,
        },
        "license": {"name": LICENSE_NAME, "url": LICENSE_URL},
        "components": [{
            "component": f"rembg/{remover}",
            "purpose": "background removal",
            "license": "MIT (BiRefNet)" if "birefnet" in str(remover) else "MIT code / Apache-2.0 U-2-Net",
        }] if remover else [],
        "timings_seconds": {"total": round(seconds, 1)},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("images", nargs=4, type=Path,
                        metavar="VIEW",
                        help="front, right, back and left views, in this order")
    parser.add_argument("output", type=Path)
    parser.add_argument("--res", type=int, choices=(1024, 1536), default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gss", type=float, default=DEFAULT_GSS)
    parser.add_argument("--brighten", type=float, default=BRIGHTEN_TARGET, metavar="LUX",
                        help=f"lift each view's subject mean luminance to LUX "
                             f"(default {BRIGHTEN_TARGET}); 0 disables")
    parser.add_argument("--models", type=Path, default=MODELS)
    parser.add_argument("--cli", type=Path, default=CLI)
    args = parser.parse_args(argv)

    for image in args.images:
        if not image.is_file():
            raise SystemExit(f"not found: {image}")
    if not args.cli.is_file():
        raise SystemExit(f"trellis-cli not found: {args.cli}")
    if not args.models.is_dir() or not list(args.models.glob("*_mv.gguf")):
        raise SystemExit(
            f"Pixal3D mv weights not found in {args.models}. Download the "
            f"raven38/pixal3d-q8_0-v1 set (about 8.4 GB) into that directory."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output = args.output.resolve()
    views = args.output.with_suffix(".views")

    started = time.time()
    needs_matte = True
    try:
        import rembg  # noqa: F401
    except ModuleNotFoundError:
        from PIL import Image
        needs_matte = any(
            not is_matted(Image.open(image)) for image in args.images
        )
        if needs_matte:
            raise SystemExit(
                "rembg is required to matte multiview input (trellis-cli --views does no "
                "matting, and its --bg-only re-crops, which would break the cameras). "
                "pip install rembg, or pass four pre-matted RGBA images."
            )
    remover = matte_model() if needs_matte else None
    if remover:
        print(f"[pixal3d-mv] matting with {remover}", flush=True)
    brighten = args.brighten if args.brighten > 0 else None
    if brighten:
        print(f"[pixal3d-mv] brightening views towards subject mean {brighten:.0f}", flush=True)
    stage_views(args.images, views, brighten=brighten)
    print(f"[pixal3d-mv] staged {views} (canonical rig: 4 views, FOV 20 deg)", flush=True)

    command = build_command(views, args.output, args.res, args.seed,
                            args.models, args.cli, args.gss)
    print(f"[pixal3d-mv] res={args.res} seed={args.seed} gss={args.gss}", flush=True)
    runtime_root = args.cli.expanduser().resolve().parent.parent
    process = subprocess.Popen(
        command, cwd=str(runtime_root), stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    assert process.stdout is not None
    for raw in process.stdout:
        line = raw.rstrip("\n")
        if line.startswith("ggml_metal") or "loaded kernel" in line:
            continue
        print(line, flush=True)
    code = process.wait()
    if code != 0:
        raise SystemExit(f"trellis-cli exited with code {code}")
    if not args.output.is_file():
        raise SystemExit(f"trellis-cli exited 0 without writing {args.output}")

    seconds = time.time() - started
    record = manifest(args.images, args.output, res=args.res, seed=args.seed,
                      gss=args.gss, seconds=seconds, remover=remover, brighten=brighten)
    record_path = args.output.with_name(f"{args.output.stem}.json")
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    size = args.output.stat().st_size / 1048576
    print(f"[pixal3d-mv] done in {seconds:.0f}s -> {args.output} "
          f"({size:.1f} MB); manifest {record_path.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
