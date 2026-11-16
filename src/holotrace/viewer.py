"""A movie of an eye looking at a hologram: moving across, then refocusing.

    holotrace view NAME [--pupil P] [--focus DEPTH]

reads the plate's light, lit by the reference beam again, saved by a
scene (out/NAME/replay.npy; see solid.render), and writes out/NAME/
movie.mp4 (with ffmpeg; speckle makes GIFs huge). First the eye moves
from left to right and back, focused at DEPTH (um behind the plate;
default 200), so that what is nearer and
farther shifts against each other (parallax), and what hides what
changes; then, looking straight on, it refocuses from in front of the
plate to well behind it. Its pupil's numerical aperture is P (default
0.03: sharper than a real eye's at reading distance, about 0.01, which
at this size would blur the scene and hardly see depth).
"""

import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .gpu import GpuEye
from .magnifier import CONE, PLATE, WAVELENGTH

HALF = 900.0  # um either way of the middle
SHRINK = 4


def _option(name, default):
    if name in sys.argv:
        return float(sys.argv[sys.argv.index(name) + 1])
    return default


def _frame(view, scale, text):
    n = view.shape[0]
    m = int(HALF / PLATE["pitch"])
    patch = view[n // 2 - m : n // 2 + m, n // 2 - m : n // 2 + m]
    k = patch.shape[0] // SHRINK
    patch = patch[: k * SHRINK, : k * SHRINK].reshape(k, SHRINK, k, SHRINK).mean(axis=(1, 3))
    patch = np.sqrt(patch[::-1] / scale)
    img = Image.fromarray((np.clip(patch, 0, 1) * 255).astype(np.uint8))
    ImageDraw.Draw(img).text((6, 6), text, fill=255)
    return img


def main():
    args = [a for a in sys.argv[2:] if not a.startswith("--")]
    name = args[0] if args else "benchy"
    pupil = _option("--pupil", 0.03)
    focus = _option("--focus", 200.0)
    out = Path("out") / name
    start = time.time()
    eye = GpuEye(np.load(out / "replay.npy"), PLATE["pitch"], WAVELENGTH)

    # Across, as far as the light reaches past the pupil, and back.
    reach = CONE - pupil
    across = np.concatenate([np.linspace(-reach, reach, 24), np.linspace(reach, -reach, 24)])
    shots = [((sx, 0.0), focus, f"looking from {sx:+.2f} rad, focused {focus:.0f} um behind") for sx in across]
    for depth in np.linspace(-400, 900, 27):
        where = "in front of" if depth < 0 else "behind"
        shots.append(((0.0, 0.0), depth, f"straight on, focused {abs(depth):.0f} um {where} the plate"))

    first = eye.view((0.0, 0.0), pupil, focus)
    m = int(HALF / PLATE["pitch"])
    n = first.shape[0]
    scale = np.percentile(first[n // 2 - m : n // 2 + m, n // 2 - m : n // 2 + m], 99.5)
    with tempfile.TemporaryDirectory() as frames:
        for i, (d, z, text) in enumerate(shots):
            _frame(eye.view(d, pupil, z), scale, text).save(f"{frames}/{i:04d}.png")
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-framerate", "8", "-i", f"{frames}/%04d.png",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(out / "movie.mp4")],
            check=True,
        )
    print(f"{len(shots)} frames in {time.time() - start:.1f} s; wrote {out}/movie.mp4")
