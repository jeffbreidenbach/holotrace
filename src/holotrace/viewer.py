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
one MM millimeters away, its pupil --pupil-mm across (default 4), seeing
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
plate. A lamp 0.02 rad across (2 cm, 1 m off) cuts the speckle a real
eye sees to about 40% of a laser's.
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
from .solid import results

# An LED's light, as replays averaged: wavelengths about the middle, in
# um, and directions its light arrives from across the plate (radians:
# a source a few mm across, some meters off).
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


def _runs(k, block=None):
    """Where in `k` (whole numbers) each run of successive numbers starts
    and ends, a run ending too where the numbers pass a multiple of
    `block`, if given."""
    k = np.asarray(k)
    cut = np.diff(k) != 1
    if block:
        cut |= k[1:] % block == 0
    edges = np.concatenate([[0], np.flatnonzero(cut) + 1, [len(k)]])
    return list(zip(edges[:-1], edges[1:]))


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
        # Stored in blocks of columns, on disk: copied a run of rows and
        # of columns within a block at a time, which is many times faster
        # than picking the samples out one by one.
        block = s.shape[2]
        out = np.empty((len(ky), len(kx)), s.dtype)
        for r0, r1 in _runs(ky):
            for c0, c1 in _runs(kx, block):
                b, c = divmod(int(kx[c0]), block)
                out[r0:r1, c0:c1] = s[b, int(ky[r0]) : int(ky[r0]) + r1 - r0, c : c + c1 - c0]
        return out

    def eye_view(self, eye, pupil, depth, grid=None):
        """What an eye at `eye` (x, y, and its distance in front of the
        plate, in um) sees through a pupil of radius `pupil` (um), focused
        `depth` behind the plate: an array covering the whole plate.

        Up close, the eye sees each point of the plane in focus along the
        line from it to the eye: through its pupil pass the plane waves
        about that direction, a different one for every point. Light
        headed for the middle of the pupil from all over the plane is a
        wave closing in on it, a sphere; so the light in the plane is
        multiplied by the opposite of that sphere's phase, which brings
        the waves the pupil takes in, wherever they are from, to about
        straight on, and then one filter, the pupil's, serves for every
        point. (This is what the eye's lens does.)

        The plane is done in `grid` x `grid` cells (as many as keep each
        far wider than the eye's blur, up to 5), since near one point the
        light the pupil takes in from thereabouts is within a small patch
        of the spectrum: each cell's light, and a border's, is made from
        the patch about the direction from its middle to the eye, wide
        enough for the whole cell and for every direction of the lamp;
        multiplied by the sphere's phase; and filtered, once for each
        direction of the lamp, whose tilt moves the pupil's place in the
        spectrum (and, with the focus off the plate, shifts the image).
        The pupil is a disc facing the cell, so seen from the side it
        takes in a narrower range of directions that way."""
        ex, ey, distance = eye
        n, width, sine = self.n, self.width, self.reference
        far = distance + depth
        if grid is None:
            blur = WAVELENGTH * far / pupil
            grid = int(max(1, min(5, width / (64 * blur))))
        centers = (np.arange(grid) + 0.5) / grid * width - width / 2
        reach = width / grid * 0.625  # half a cell and its border, an eighth of a cell

        def toward(x, y):
            """The eye's distance from (x, y) in the plane in focus, and
            its direction from there (sines)."""
            r = np.sqrt((ex - x) ** 2 + (ey - y) ** 2 + far**2)
            return r, (ex - x) / r, (ey - y) / r

        by_offset = {}
        for offset, tilt in self.light:
            by_offset.setdefault(offset, []).append(tilt if isinstance(tilt, tuple) else (tilt, 0.0))
        shortest = WAVELENGTH + min(by_offset)
        widest = max(math.hypot(*t) for tilts in by_offset.values() for t in tilts)
        # How far directions to the eye differ within a cell and its
        # border, and the widest the pupil looks from any cell.
        spread = aperture = 0.0
        for qy in centers:
            for qx in centers:
                r, dx, dy = toward(qx, qy)
                aperture = max(aperture, pupil / r)
                for cx in (qx - reach, qx + reach):
                    for cy in (qy - reach, qy + reach):
                        _, ax, ay = toward(cx, cy)
                        spread = max(spread, math.hypot(ax - dx, ay - dy))
        m = int(math.ceil((aperture + spread + widest) / shortest * width)) + 2
        m_spread = int(math.ceil(spread / shortest * width)) + 1
        # The plane's samples: enough for the patch, and for what the
        # sphere's phase moves it by; a whole number to a cell's eighth.
        size = -(-(2 * (m + m_spread) + 2) // (8 * grid)) * 8 * grid
        while cufft.next_fast_len(size) != size:
            size += 8 * grid
        cell, border = size // grid, size // (8 * grid)
        crop = cell + 2 * border
        across = width * crop / size  # a cell and its border, um
        m_pupil = int(math.ceil(aperture / shortest * across)) + 1
        # The image: a whole number of pixels to a cell's eighth, twice
        # as many samples as its light's patch so that it is smooth.
        pixels = -(-(4 * m_pupil + 2) // 10) * 8
        while cufft.next_fast_len(pixels * 5 // 4) != pixels * 5 // 4:
            pixels += 8
        image = pixels * 5 // 4
        out = np.zeros((grid * pixels, grid * pixels))
        kept = width / n
        count = len(self.light)
        for iy, qy in enumerate(centers):
            for ix, qx in enumerate(centers):
                r, dx, dy = toward(qx, qy)
                # The cell's samples (and beyond the plate's edges, the
                # other side's: the plate's light repeats), where they
                # are, and how far from the eye.
                jx = ix * cell - border + np.arange(crop)
                jy = iy * cell - border + np.arange(crop)
                x = cp.asarray((jx * n / size - n / 2 + 0.5) * kept)[None, :]
                y = cp.asarray((jy * n / size - n / 2 + 0.5) * kept)[:, None]
                away = cp.sqrt((ex - x) ** 2 + (ey - y) ** 2 + far**2)
                total = 0
                for offset, tilts in by_offset.items():
                    lam = WAVELENGTH + offset
                    # How far this wavelength shifts the spectrum, up.
                    up = (1 / lam - 1 / WAVELENGTH) * sine
                    mx = int(round(dx / lam * width))
                    my = int(round((dy / lam - up) * width))
                    kx = np.arange(mx - m, mx + m + 1)
                    ky = np.arange(my - m, my + m + 1)
                    patch = cp.asarray(self._patch(ky % n, kx % n))
                    sx = cp.asarray((kx / width * lam)[None, :], cp.float32)
                    sy = cp.asarray(((ky / width + up) * lam)[:, None], cp.float32)
                    cos = cp.sqrt(cp.maximum(0, 1 - sx * sx - sy * sy))
                    patch = patch * cp.exp((-2j * math.pi / lam) * cos * depth).astype(cp.complex64)
                    padded = cp.zeros((size, size), cp.complex64)
                    c = size // 2
                    padded[c - m : c + m + 1, c - m : c + m + 1] = patch
                    del patch
                    plane = cufft.ifft2(cp.fft.ifftshift(padded), overwrite_x=True)
                    del padded
                    here = plane[cp.asarray(jy % size)[:, None], cp.asarray(jx % size)[None, :]]
                    del plane
                    # The phase the patch's middle took off, put back, and
                    # the sphere's taken off: worked out in double
                    # precision (they are millions of radians, nearly
                    # canceling).
                    turns = (cp.asarray((mx * jx) % size)[None, :] / size + cp.asarray((my * jy) % size)[:, None] / size
                             + up * y + away / lam)
                    here = here * cp.exp(2j * math.pi * (turns % 1.0)).astype(cp.complex64)
                    del turns
                    spectrum = cufft.fft2(here, overwrite_x=True)
                    del here
                    slant = 1 - dx * dx - dy * dy
                    for tx, ty in tilts:
                        # The lamp's tilt, as a shift of the spectrum, and
                        # (in sines) of every direction.
                        shift = (tx / lam, ty * math.sqrt(1 - sine * sine) / lam)
                        ux = int(round(-shift[0] * across)) + np.arange(-m_pupil, m_pupil + 1)
                        uy = int(round(-shift[1] * across)) + np.arange(-m_pupil, m_pupil + 1)
                        part = spectrum[cp.asarray(uy % crop)[:, None], cp.asarray(ux % crop)[None, :]]
                        gx, gy = (ux / across)[None, :], (uy / across)[:, None]
                        # Directions from the pupil's middle, and whether
                        # they pass the pupil, a disc facing this way.
                        ax, ay = (gx + shift[0]) * lam, (gy + shift[1]) * lam
                        # (Its edge is shaded over one sample of the
                        # spectrum, as the share of that sample inside.)
                        off = np.sqrt(ax * ax + ay * ay + (dx * ax + dy * ay) ** 2 / slant)
                        seen = np.clip((pupil / r - off) / (lam / across) + 0.5, 0, 1)
                        # With the focus off the plate, the tilt shifts
                        # what is seen (it turns about the plate).
                        sx_, sy_ = shift[0] * lam, shift[1] * lam
                        along = dx * sx_ + dy * sy_
                        move_x = sx_ / math.sqrt(slant) + dx * along / slant**1.5
                        move_y = sy_ / math.sqrt(slant) + dy * along / slant**1.5
                        factor = seen * np.exp(2j * math.pi * depth * (gx * move_x + gy * move_y))
                        padded = cp.zeros((image, image), cp.complex64)
                        c = image // 2
                        padded[c - m_pupil : c + m_pupil + 1, c - m_pupil : c + m_pupil + 1] = part * cp.asarray(
                            factor.astype(np.complex64)
                        )
                        total = total + cp.abs(cufft.ifft2(cp.fft.ifftshift(padded), overwrite_x=True)) ** 2
                    del spectrum
                # The cell's image and its border's: faded in across the
                # border and out across the next cell's, so that where
                # two cells' images overlap (they are of the same thing,
                # and differ only in the faint far reaches of the pupil's
                # blur, cut off at each one's edge) one gives way to the
                # other smoothly. Past the plate's edges it is the other
                # side's, as the light is.
                edge = pixels // 8
                seen = cp.asnumpy(total) / count
                at = (np.arange(image) + 0.5) / image
                fade = np.clip(np.minimum(at, 1 - at) * image / (2 * edge), 0, 1)
                # As bright as from each place, not from the cell's
                # middle: the pupil looks smaller from farther off.
                px = (at * across + qx - across / 2)[None, :]
                py = (at * across + qy - across / 2)[:, None]
                seen *= r**2 / toward(px, py)[0] ** 2 * fade[None, :] * fade[:, None]
                rows = (iy * pixels - edge + np.arange(image)) % (grid * pixels)
                cols = (ix * pixels - edge + np.arange(image)) % (grid * pixels)
                np.add.at(out, (rows[:, None], cols[None, :]), seen)
        cp.get_default_memory_pool().free_all_blocks()
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
    out = results() / name
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
            look = Look.load(results() / name / "spectrum.npy", samples)
            view = look.view((0.0, 0.0), pupil, 0.0)
            tiles.append((f"{name}, {light}", look.image(view, look.scale(view), "", width=600)))
    sheet(tiles, columns=2).save("out/compare.png")
    print("wrote out/compare.png")
