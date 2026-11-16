"""Looking at holograms: a movie of an eye moving and refocusing, and a
sheet comparing scenes and light.

    holotrace view NAME [--pupil P] [--focus DEPTH] [--light laser|led | --source R]
                        [--eye MM [--pupil-mm MM]]

reads the spectrum of the plate's light, as the hologram sends it out,
saved by a scene (out/NAME/spectrum.npy; see solid.render), and
writes out/NAME/
movie-<light>.mp4 (with ffmpeg; speckle makes GIFs huge). First the eye
moves from left to right and back, then from below to above and back,
focused at DEPTH (um behind the plate; default 200), so that what is
nearer and farther shifts against each other (parallax, both ways: a
hologram has it up and down as well as side to side), and what hides
what changes.
Its pupil's numerical aperture is P (default 0.03: sharper than a real
eye's at reading distance, about 0.01, which at this size would blur
the scene and hardly see depth). With --eye, the eye is instead a real
one MM millimetres away, its pupil --pupil-mm across (default 4), seeing
each part of the plate from its own direction (see Look.eye_view); it
moves as far as it can and still see the whole plate.

    holotrace compare NAME...

writes out/compare.png: each scene seen straight on, by laser light and
by an LED's.

The light is a laser's, of one wavelength from one point, or an LED's
(see Look): a few wavelengths, from a source with some size, each
replaying the plate a little differently, their brightness added. That
smooths speckle, the more so the farther from the plate, and blurs by
as much. With --source, the light is instead a lamp R radians across as
seen from the plate (see lamp): each part of it replays the plate from
its own direction, so the eye's pupil takes in a different part of the
plate's light, and so a different speckle, from each. Speckle averages
away once the lamp is wider than the pupil seen from the scene (twice
its numerical aperture), while the image blurs only by the lamp's width
times each point's distance from the plate: little, for a scene at the
plate. A lamp 0.02 rad across (2 cm, 1 m off) roughly halves the speckle
a real eye sees.
"""

import json
import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cupy as cp
import cupyx.scipy.fft as cufft
import numpy as np
from PIL import Image, ImageDraw

from .magnifier import CONE, REFERENCE, WAVELENGTH, sheet

# An LED's light, as replays averaged: wavelengths about the middle, in
# um, and directions its light arrives from across the plate (radians:
# a source a few mm across, some metres off).
LED = [(0.0, 0.0), (-0.008, 0.0), (0.008, 0.0), (0.0, -0.001), (0.0, 0.001)]
LASER = [(0.0, 0.0)]


def lamp(across, count=7, offsets=(-0.008, 0.0, 0.008)):
    """A source `across` radians wide as seen from the plate (a disc),
    at wavelengths `offsets` from the middle: replays from directions on
    a `count` x `count` grid within the disc, each direction across and
    up (radians), at each wavelength."""
    t = (np.arange(count) - (count - 1) / 2) / count * across
    tilts = [(x, y) for y in t for x in t if x * x + y * y <= (across / 2) ** 2] or [(0.0, 0.0)]
    return [(offset, tilt) for offset in offsets for tilt in tilts]


class Look:
    """An eye looking at a plate's light, lit by `light` (LASER or LED):
    replays at each wavelength offset and tilt, their brightness added.

    The eye takes in only the light leaving within its pupil's small cone
    of directions: a small patch of the light's spectrum (its plane
    waves). So the spectrum of the plate's light (the scene's light, as
    the hologram sends it out lit by the reference beam it was made for),
    sampled `pitch` um apart, is worked out once, and each view from just
    the patch its pupil takes in: a few hundred square for a 1 cm plate,
    however many pixels the plate has. Another wavelength or direction of
    light shifts the spectrum (as the hologram's fringes, made for the
    reference beam, bend it differently), so the patch is taken from a
    little elsewhere. Each view covers the whole plate, at about half the
    eye's resolution."""

    def __init__(self, spectrum, pitch, light=LASER, cone=CONE, reference=None):
        self.spectrum = spectrum
        self.light = light
        # Kept whole, or (see planes.py) a block of columns at a time.
        self.n = spectrum.shape[-2]
        self.width = self.n * pitch
        self.cone = cone
        # The reference beam's angle, as a sine.
        self.reference = REFERENCE / math.sqrt(1 + REFERENCE**2) if reference is None else reference

    @staticmethod
    def save(path, spectrum, optics):
        """Saves a spectrum (unless it is None: already in the file), and
        beside it how far apart its samples are and the optics it was
        made with (see solid.Optics)."""
        if spectrum is not None:
            np.save(path, spectrum)
        info = {"pitch": optics.kept_pitch, "cone": optics.cone, "reference": optics.reference}
        Path(path).with_suffix(".json").write_text(json.dumps(info))

    @classmethod
    def load(cls, path, light=LASER):
        info = json.loads(Path(path).with_suffix(".json").read_text())
        return cls(np.load(path, mmap_mode="r"), info["pitch"], light, info.get("cone", CONE), info.get("reference"))

    def view(self, direction, pupil, depth):
        """The brightness across the plane `depth` behind the plate, seen
        along `direction` (sines) through a pupil of numerical aperture
        `pupil`: an array covering the whole plate."""
        sine = self.reference
        total = 0
        # Every wavelength's view the same size, as big as the shortest's
        # patch needs, so that they can be added.
        shortest = WAVELENGTH + min(offset for offset, _ in self.light)
        widest = int(math.ceil(pupil / shortest * self.width)) + 1
        size = cufft.next_fast_len(4 * widest + 2)
        for offset, tilt in self.light:
            lam = WAVELENGTH + offset
            # How far this light shifts the spectrum (cycles per um): its
            # tilt across, and up (as a change of the reference's angle),
            # and its wavelength.
            tx, ty = tilt if isinstance(tilt, tuple) else (tilt, 0.0)
            up = ty * math.sqrt(1 - sine * sine) / lam
            shift = np.array([tx / lam, up + (1 / lam - 1 / WAVELENGTH) * sine])
            middle = np.round((np.array(direction) / lam - shift) * self.width).astype(int)
            m = int(math.ceil(pupil / lam * self.width)) + 1
            kx = np.arange(middle[0] - m, middle[0] + m + 1)
            ky = np.arange(middle[1] - m, middle[1] + m + 1)
            patch = cp.asarray(self._patch(ky % self.n, kx % self.n))
            sx = cp.asarray(((kx / self.width + shift[0]) * lam)[None, :], cp.float32)
            sy = cp.asarray(((ky / self.width + shift[1]) * lam)[:, None], cp.float32)
            seen = (sx - direction[0]) ** 2 + (sy - direction[1]) ** 2 < pupil**2
            cos = cp.sqrt(cp.maximum(0, 1 - sx * sx - sy * sy))
            patch = patch * seen * cp.exp((-2j * math.pi / lam) * cos * depth).astype(cp.complex64)
            # Twice as many samples as the patch, so the view is smooth.
            padded = cp.zeros((size, size), cp.complex64)
            c = size // 2
            padded[c - m : c + m + 1, c - m : c + m + 1] = patch
            total = total + cp.abs(cufft.ifft2(cp.fft.ifftshift(padded), overwrite_x=True)) ** 2
            del padded, patch
        cp.get_default_memory_pool().free_all_blocks()
        return cp.asnumpy(total / len(self.light))

    def _patch(self, ky, kx):
        """The spectrum at rows `ky` and columns `kx`."""
        s = self.spectrum
        if s.ndim == 2:
            return s[np.ix_(ky, kx)]
        block = s.shape[2]
        return s[(kx // block)[None, :], ky[:, None], (kx % block)[None, :]]

    def eye_view(self, eye, pupil, depth, grid=9):
        """What an eye at `eye` (x, y, and its distance in front of the
        plate, in um) sees through a pupil of radius `pupil` (um), focused
        `depth` behind the plate: an array covering the whole plate.

        Up close, different parts of the plate are seen from different
        directions: the eye sees each point of the plane in focus along
        the line from it to the eye. So views along the directions to a
        `grid` x `grid` lattice of points across the plate are blended,
        each part of the image from the directions of the points around
        it (bilinearly). The pupil's numerical aperture is its radius over
        its distance."""
        ex, ey, distance = eye
        aperture = pupil / distance
        nodes = np.linspace(-self.width / 2, self.width / 2, grid)
        views = {}
        for j, qy in enumerate(nodes):
            for i, qx in enumerate(nodes):
                v = np.array([ex - qx, ey - qy, distance + depth])
                views[i, j] = self.view(tuple(v[:2] / np.linalg.norm(v)), aperture, depth)
        size = views[0, 0].shape[0]
        # Where each pixel is among the nodes (the view's pixel j is j /
        # size of the way across the plate).
        f = np.arange(size) / size * (grid - 1)
        i0 = np.minimum(f.astype(int), grid - 2)
        a = f - i0
        out = np.zeros((size, size))
        for di in (0, 1):
            for dj in (0, 1):
                wx = a if di else 1 - a
                wy = a if dj else 1 - a
                for j in range(grid - 1):
                    rows = i0 == j
                    for i in range(grid - 1):
                        cols = i0 == i
                        block = views[i + di, j + dj][np.ix_(rows, cols)]
                        out[np.ix_(rows, cols)] += block * wy[rows][:, None] * wx[cols][None, :]
        return out

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
    focus = _option("--focus", 200.0)
    light = _option("--light", "laser", str)
    samples = LIGHTS[light]
    if "--source" in sys.argv:
        across = _option("--source", 0.02)
        samples = lamp(across)
        light = f"lamp{across:g}rad"
    distance = _option("--eye", 0.0)  # mm; 0 for an eye infinitely far
    out = Path("out") / name
    start = time.time()
    look = Look.load(out / "spectrum.npy", samples)
    cone = look.cone

    if distance:
        # A real eye, `distance` mm away, its pupil `--pupil-mm` across,
        # moving across and up and down, as far as the plate's light
        # reaches it from every part of the plate.
        d = distance * 1000
        radius = _option("--pupil-mm", 4.0) * 500
        reach = (cone - radius / d - look.width / 2 / d) * d
        sweep = np.concatenate([np.linspace(-reach, reach, 24), np.linspace(reach, -reach, 24)])
        what = f"{light}, eye {distance:g} mm away"
        shots = [((s, 0.0, d), focus, f"{what}, {s / 1000:+.1f} mm to the side, focused {focus:.0f} um behind") for s in sweep]
        shots += [((0.0, s, d), focus, f"{what}, {s / 1000:+.1f} mm up, focused {focus:.0f} um behind") for s in sweep]
        see = lambda where, z: look.eye_view(where, radius, z)
        movie = out / f"movie-{light}-eye{distance:g}mm.mp4"
    else:
        # An eye far away, looking along directions across and up and down,
        # as far as the light reaches past the pupil.
        pupil = _option("--pupil", 0.03)
        reach = cone - pupil
        sweep = np.concatenate([np.linspace(-reach, reach, 24), np.linspace(reach, -reach, 24)])
        shots = [((s, 0.0), focus, f"{light}, from {s:+.2f} rad side to side, focused {focus:.0f} um behind") for s in sweep]
        shots += [((0.0, s), focus, f"{light}, from {s:+.2f} rad up and down, focused {focus:.0f} um behind") for s in sweep]
        see = lambda where, z: look.view(where, pupil, z)
        movie = out / f"movie-{light}.mp4"

    scale = look.scale(see(shots[len(sweep) // 2][0], focus))
    with tempfile.TemporaryDirectory() as frames:
        for i, (where, z, text) in enumerate(shots):
            look.image(see(where, z), scale, text).save(f"{frames}/{i:04d}.png")
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
