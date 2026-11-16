"""Looking at holograms: a movie of an eye moving and refocusing, and a
sheet comparing scenes and light.

    holotrace view NAME [--pupil P] [--focus DEPTH] [--light laser|led]

reads the plate's light, lit by the reference beam again, saved by a
scene (out/NAME/replay.npy; see solid.render), and writes out/NAME/
movie-<light>.mp4 (with ffmpeg; speckle makes GIFs huge). First the eye
moves from left to right and back, then from below to above and back,
focused at DEPTH (um behind the plate; default 200), so that what is
nearer and farther shifts against each other (parallax, both ways: a
hologram has it up and down as well as side to side), and what hides
what changes; then, looking straight on, it refocuses from in front of
the plate to well behind it.
Its pupil's numerical aperture is P (default 0.03: sharper than a real
eye's at reading distance, about 0.01, which at this size would blur
the scene and hardly see depth).

    holotrace compare NAME...

writes out/compare.png: each scene seen straight on, by laser light and
by an LED's.

The light is a laser's, of one wavelength from one point, or an LED's
(see Look): a few wavelengths, from a source with some size, each
replaying the plate a little differently, their brightness added. That
smooths speckle, the more so the farther from the plate, and blurs by
as much.
"""

import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cupy as cp
import numpy as np
from PIL import Image, ImageDraw

from .gpu import GpuEye
from .magnifier import CONE, PLATE, REFERENCE, WAVELENGTH, sheet

# An LED's light, as replays averaged: wavelengths about the middle, in
# um, and directions its light arrives from across the plate (radians:
# a source a few mm across, some metres off).
LED = [(0.0, 0.0), (-0.008, 0.0), (0.008, 0.0), (0.0, -0.001), (0.0, 0.001)]
LASER = [(0.0, 0.0)]


class Look:
    """An eye looking at the plate's light `replay` (lit by the reference
    beam it was made for), lit instead by `light`: replays of it, for each
    wavelength offset and tilt, each its brightness added."""

    def __init__(self, replay, light):
        n = replay.shape[0]
        pitch = PLATE["pitch"]
        self.half = 0.44 * n * pitch
        self.size = n
        c = cp.asarray((np.arange(n) - n / 2 + 0.5) * pitch, cp.float32)
        x, y = c[None, :], c[:, None]
        sine = REFERENCE / math.sqrt(1 + REFERENCE**2)
        field = cp.asarray(replay, cp.complex64)
        self.eyes = []
        for offset, tilt in light:
            lam = WAVELENGTH + offset
            # The beam this wavelength and direction brings, against the one
            # the replay was lit by.
            turn = (2 * math.pi / lam - 2 * math.pi / WAVELENGTH) * sine * y + 2 * math.pi / lam * tilt * x
            self.eyes.append(GpuEye(field * cp.exp(1j * turn).astype(cp.complex64), pitch, lam))

    def view(self, direction, pupil, depth):
        return sum(e.view(direction, pupil, depth) for e in self.eyes) / len(self.eyes)

    def image(self, view, scale, text, width=900):
        """The middle of a view as an image `width` pixels across, brightened
        (square root), `scale` the brightness shown as white."""
        m = int(self.half / PLATE["pitch"])
        n = view.shape[0]
        patch = view[n // 2 - m : n // 2 + m, n // 2 - m : n // 2 + m]
        k = patch.shape[0] // width
        patch = patch[: width * k, : width * k].reshape(width, k, width, k).mean(axis=(1, 3))
        img = Image.fromarray((np.clip(np.sqrt(patch[::-1] / scale), 0, 1) * 255).astype(np.uint8))
        ImageDraw.Draw(img).text((6, 6), text, fill=255)
        return img

    def scale(self, view):
        """The brightness to show as white: near the brightest in the middle."""
        m = int(self.half / PLATE["pitch"])
        n = view.shape[0]
        return np.percentile(view[n // 2 - m : n // 2 + m, n // 2 - m : n // 2 + m], 99.5)


def _option(name, default, kind=float):
    if name in sys.argv:
        return kind(sys.argv[sys.argv.index(name) + 1])
    return default


def _names():
    """The command's arguments that are not options or their values."""
    return [a for i, a in enumerate(sys.argv[2:], 2) if not a.startswith("--") and not sys.argv[i - 1].startswith("--")]


LIGHTS = {"laser": LASER, "led": LED}


def main():
    names = _names()
    name = names[0] if names else "benchy"
    pupil = _option("--pupil", 0.03)
    focus = _option("--focus", 200.0)
    light = _option("--light", "laser", str)
    out = Path("out") / name
    start = time.time()
    look = Look(np.load(out / "replay.npy"), LIGHTS[light])

    # Across and back, then up and back, as far as the light reaches past
    # the pupil.
    reach = CONE - pupil
    sweep = np.concatenate([np.linspace(-reach, reach, 24), np.linspace(reach, -reach, 24)])
    shots = [((s, 0.0), focus, f"{light}, from {s:+.2f} rad side to side, focused {focus:.0f} um behind") for s in sweep]
    shots += [((0.0, s), focus, f"{light}, from {s:+.2f} rad up and down, focused {focus:.0f} um behind") for s in sweep]
    for depth in np.linspace(-400, 900, 27) * look.size / PLATE["size"]:
        where = "in front of" if depth < 0 else "behind"
        shots.append(((0.0, 0.0), depth, f"{light}, straight on, focused {abs(depth):.0f} um {where} the plate"))

    scale = look.scale(look.view((0.0, 0.0), pupil, focus))
    movie = out / f"movie-{light}.mp4"
    with tempfile.TemporaryDirectory() as frames:
        for i, (d, z, text) in enumerate(shots):
            look.image(look.view(d, pupil, z), scale, text).save(f"{frames}/{i:04d}.png")
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-framerate", "8", "-i", f"{frames}/%04d.png",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(movie)],
            check=True,
        )
    print(f"{len(shots)} frames in {time.time() - start:.1f} s; wrote {movie}")


def compare():
    """Each scene named, straight on, focused on the plate, by laser and
    by LED: a row each."""
    pupil = _option("--pupil", 0.03)
    tiles = []
    for name in _names():
        replay = np.load(Path("out") / name / "replay.npy")
        for light, samples in LIGHTS.items():
            look = Look(replay, samples)
            view = look.view((0.0, 0.0), pupil, 0.0)
            tiles.append((f"{name}, {light}", look.image(view, look.scale(view), "", width=600)))
            del look
            cp.get_default_memory_pool().free_all_blocks()
    sheet(tiles, columns=2).save("out/compare.png")
    print("wrote out/compare.png")
