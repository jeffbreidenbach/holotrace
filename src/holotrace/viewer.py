"""Looking at holograms: a movie of an eye moving and refocusing, and a
sheet comparing scenes and light.

    holotrace view NAME [--pupil P] [--focus DEPTH] [--light laser|led]

reads the spectrum of the plate's light, lit by the reference beam
again, saved by a scene (out/NAME/spectrum.npy; see solid.render), and
writes out/NAME/
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

from .magnifier import CONE, PLATE, REFERENCE, WAVELENGTH, sheet

# An LED's light, as replays averaged: wavelengths about the middle, in
# um, and directions its light arrives from across the plate (radians:
# a source a few mm across, some metres off).
LED = [(0.0, 0.0), (-0.008, 0.0), (0.008, 0.0), (0.0, -0.001), (0.0, 0.001)]
LASER = [(0.0, 0.0)]


class Look:
    """An eye looking at a plate's light, lit by `light` (LASER or LED):
    replays at each wavelength offset and tilt, their brightness added.

    The eye takes in only the light leaving within its pupil's small cone
    of directions: a small patch of the light's spectrum (its plane
    waves). So the spectrum of the plate's light, as lit by the reference
    beam it was made for, is worked out once, and each view from just the
    patch its pupil takes in: about 1000 x 1000 for a 1 cm plate, however
    many pixels the plate has. A different wavelength or direction of the
    light shifts the spectrum, so the patch is taken from a little
    elsewhere. Each view covers the whole plate, at about half the eye's
    resolution."""

    def __init__(self, spectrum, light=LASER):
        self.spectrum = spectrum
        self.light = light
        self.n = spectrum.shape[0]
        self.width = self.n * PLATE["pitch"]

    @classmethod
    def from_replay(cls, replay, light=LASER):
        """From the plate's light, lit by the reference beam again; the
        array may be reused."""
        import scipy.fft

        return cls(scipy.fft.fft2(replay, workers=-1, overwrite_x=True).astype(np.complex64, copy=False), light)

    @classmethod
    def load(cls, path, light=LASER):
        return cls(np.load(path, mmap_mode="r"), light)

    def save(self, path):
        np.save(path, self.spectrum)

    def view(self, direction, pupil, depth):
        """The brightness across the plane `depth` behind the plate, seen
        along `direction` (sines) through a pupil of numerical aperture
        `pupil`: an array covering the whole plate."""
        sine = REFERENCE / math.sqrt(1 + REFERENCE**2)
        total = 0
        for offset, tilt in self.light:
            lam = WAVELENGTH + offset
            # How far this light shifts the spectrum (cycles per um).
            shift = np.array([tilt / lam, (1 / lam - 1 / WAVELENGTH) * sine])
            middle = np.round((np.array(direction) / lam - shift) * self.width).astype(int)
            m = int(math.ceil(pupil / lam * self.width)) + 1
            kx = np.arange(middle[0] - m, middle[0] + m + 1)
            ky = np.arange(middle[1] - m, middle[1] + m + 1)
            patch = cp.asarray(self.spectrum[np.ix_(ky % self.n, kx % self.n)])
            sx = cp.asarray(((kx / self.width + shift[0]) * lam)[None, :], cp.float32)
            sy = cp.asarray(((ky / self.width + shift[1]) * lam)[:, None], cp.float32)
            seen = (sx - direction[0]) ** 2 + (sy - direction[1]) ** 2 < pupil**2
            cos = cp.sqrt(cp.maximum(0, 1 - sx * sx - sy * sy))
            patch = patch * seen * cp.exp((-2j * math.pi / lam) * cos * depth).astype(cp.complex64)
            # Twice as many samples as the patch, so the view is smooth.
            size = 1 << int(math.ceil(math.log2(4 * m + 2)))
            padded = cp.zeros((size, size), cp.complex64)
            c = size // 2
            padded[c - m : c + m + 1, c - m : c + m + 1] = patch
            total = total + cp.abs(cp.fft.ifft2(cp.fft.ifftshift(padded))) ** 2
        return cp.asnumpy(total / len(self.light))

    def _middle(self, view):
        m = int(0.44 * view.shape[0])
        c = view.shape[0] // 2
        return view[c - m : c + m, c - m : c + m]

    def image(self, view, scale, text, width=900):
        """The middle of a view (all but the plate's edges) as an image
        `width` pixels across, brightened (square root), `scale` the
        brightness shown as white."""
        patch = self._middle(view)
        img = Image.fromarray((np.clip(np.sqrt(patch[::-1] / scale), 0, 1) * 255).astype(np.uint8))
        img = img.resize((width, width), Image.BILINEAR)
        ImageDraw.Draw(img).text((6, 6), text, fill=255)
        return img

    def scale(self, view):
        """The brightness to show as white: near the brightest in the middle."""
        return np.percentile(self._middle(view), 99.5)


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
    look = Look.load(out / "spectrum.npy", LIGHTS[light])

    # Across and back, then up and back, as far as the light reaches past
    # the pupil.
    reach = CONE - pupil
    sweep = np.concatenate([np.linspace(-reach, reach, 24), np.linspace(reach, -reach, 24)])
    shots = [((s, 0.0), focus, f"{light}, from {s:+.2f} rad side to side, focused {focus:.0f} um behind") for s in sweep]
    shots += [((0.0, s), focus, f"{light}, from {s:+.2f} rad up and down, focused {focus:.0f} um behind") for s in sweep]
    for depth in np.linspace(-400, 900, 27) * look.n / PLATE["size"]:
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
        for light, samples in LIGHTS.items():
            look = Look.load(Path("out") / name / "spectrum.npy", samples)
            view = look.view((0.0, 0.0), pupil, 0.0)
            tiles.append((f"{name}, {light}", look.image(view, look.scale(view), "", width=600)))
    sheet(tiles, columns=2).save("out/compare.png")
    print("wrote out/compare.png")
