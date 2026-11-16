"""Step 4: less speckle, by choosing each point's phase.

The solid scene of step 3 (solid.py), but each point's phase, so far
random, is chosen to make the eye's views look like the scene without
speckle. Only the points' brightness is set by the scene; their phases
are free, and with random ones, neighbouring points' light interferes at
random, which is speckle.

What is optimized is meant to be the plate as it would be printed:

- Binary. The plate is chrome on glass, clear or opaque. Its fringes are
  binary (encode): each fringe clear where cos(theta) > sqrt(1 - a^2),
  theta the phase of the scene's light against the reference beam's and
  a its strength (scaled to 1 at its brightest): wide fringes where the
  light is strong. Lit by the reference beam, the light it sends the eye
  is then the scene's, a / pi as strong; the rest goes elsewhere.
- Lit as it will be: by a laser, or an LED, whose light spreads over a
  range of wavelengths (LED): each replays the plate a little
  differently, and the eye sees their brightness added.
- Seen by a real eye: through a pupil of numerical aperture 0.01 (about
  a 4 mm pupil at 20 cm), from a grid of places, focused at two depths.

The phases are improved by gradient descent (Adam). The views of the
binary plate depend on the phases through the binarization, which has
no useful slope; the gradient treats the replay as what the encoding
aims at, the scene's light itself, scaled (a straight-through
estimate). The target views are the eye's views of the scene's light
averaged over many random phases: speckle averaged away.

The views of the plate as printed, with random and optimized phases,
against the target, go to out/speckle, with how far each is from the
target. `holotrace speckle --check` checks the gradient instead, on a
few points, against small changes of the phases.
"""

import math
import sys
import time
from pathlib import Path

import cupy as cp
import numpy as np

from .gpu import GpuPlate, SeenPoints
from .magnifier import CONE, PLATE, REFERENCE, WAVELENGTH, picture, sheet
from . import solid

PUPIL = 0.01
# A real eye this close sees the whole scene in focus at once (its depth
# of field, wavelength / PUPIL^2, is about 6 mm), so it looks at one depth,
# the middle of the scene, from places across the whole viewing zone: a
# grid of them a pupil apart, so that their pupils between them take in
# all the light the plate sends out (its directions within CONE).
DEPTH = 200.0
GRID = np.arange(-0.10, 0.1001, 2 * PUPIL)
TRAIN = [((sx, sy), DEPTH) for sy in GRID for sx in GRID]
# Views not trained on: halfway between places on the grid.
TEST = [((sx + PUPIL, sy + PUPIL), DEPTH) for sx, sy in [(-0.06, -0.06), (-0.02, 0.04), (0.02, -0.02), (0.04, 0.06),
                                                         (0.06, -0.08), (-0.08, 0.02), (0.0, 0.0), (-0.04, -0.02)]]
# Each step trains on this many of the grid's views, picked at random.
BATCH = 8
# The LED's wavelengths, about its middle, in um; a laser has one.
LED = (-0.008, 0.0, 0.008)
LASER = (0.0,)
TARGET_REALIZATIONS = 32
ITERATIONS = 200
RATE = 0.1
# The views are compared within this far of the middle (um), averaged
# over BIN x BIN pixels: still much finer than the eye resolves.
HALF = 800.0
BIN = 8

# Points closer to the plate than this are left out: summed straight onto
# it, their light would land on a pixel or two.
NEAREST = 2.0

OUT = Path("out/speckle")


class Scene:
    """The solid scene's points and their brightness maps, on the GPU."""

    def __init__(self, limit=None):
        parts = [solid.ball(), solid.wall()]
        points, normals, albedo, gloss = (np.concatenate(a) for a in zip(*parts))
        maps = solid.maps(solid.scene(), points, normals, albedo, gloss)
        alive = (maps.max(axis=(1, 2)) > 0) & (np.abs(points[:, 2]) >= NEAREST)
        self.points = points[alive][:limit]
        self.maps = cp.asarray(maps[alive][:limit])
        self._seen = SeenPoints(**PLATE, points=self.points, maps=self.maps, cone=CONE)

    def __len__(self):
        return len(self.points)

    def light(self, phases):
        """The scene's light on the plate (on the GPU), each point starting
        with its phase."""
        return self._seen.light(phases)

    def reverse(self, grad):
        """For each point, its light on the plate without its own phase,
        conjugated, times `grad`, summed: how the error changes with its
        light (see seen_reverse_many in gpu.py)."""
        return self._seen.reverse(grad)


class Optics:
    """The reference beam, the binary plate, and the eye looking at it,
    for each of `offsets` from the wavelength, on the GPU."""

    def __init__(self, offsets, encode=True):
        n, pitch = PLATE["size"], PLATE["pitch"]
        self.encode = encode
        self.offsets = offsets
        y = cp.asarray(((np.arange(n) - n / 2 + 0.5) * pitch)[:, None], cp.float32)
        sine = REFERENCE / math.sqrt(1 + REFERENCE**2)
        self.ref = [cp.exp(1j * (2 * math.pi / (WAVELENGTH + d)) * sine * y).astype(cp.complex64) for d in offsets]
        freq = cp.fft.fftfreq(n, pitch).astype(cp.float32)
        self.fx, self.fy = freq[None, :], freq[:, None]
        m = int(HALF / pitch)
        self.crop = (slice(n // 2 - m, n // 2 + m), slice(n // 2 - m, n // 2 + m))

    def plate(self, light):
        """What the plate sends out, lit by each wavelength's reference:
        the binary fringes for `light` times the beam (or with encode
        off, the light itself, as if perfectly recorded)."""
        if not self.encode:
            return [light for _ in self.offsets]
        size = cp.abs(light)
        brightest = float(cp.percentile(size[::8, ::8], 99.5))
        a = cp.minimum(size / brightest, 1)
        theta = cp.angle(light * cp.conj(self.ref[self.offsets.index(0.0)]))
        clear = (cp.cos(theta) > cp.sqrt(1 - a * a)).astype(cp.float32)
        return [clear * r for r in self.ref]

    def _filter(self, i, spot, depth):
        """The eye's pupil, at `spot`, and the step back to `depth`, for
        wavelength i, applied to plane waves."""
        lam = WAVELENGTH + self.offsets[i]
        sx, sy = self.fx * lam, self.fy * lam
        pupil = (sx - spot[0]) ** 2 + (sy - spot[1]) ** 2 < PUPIL**2
        cos = cp.sqrt(cp.maximum(0, 1 - sx * sx - sy * sy))
        return pupil * cp.exp(-2j * math.pi / lam * cos * depth).astype(cp.complex64)

    def views(self, light, views, keep=False):
        """The eye's views of the plate made for `light`: brightness,
        averaged over wavelengths, within HALF of the middle, in BIN x BIN
        blocks. With keep,
        also the plate's spectra, to run them backwards."""
        out = []
        spectra = [cp.fft.fft2(p) for p in self.plate(light)]
        for spot, depth in views:
            total = 0
            for i, spectrum in enumerate(spectra):
                field = cp.fft.ifft2(spectrum * self._filter(i, spot, depth))[self.crop]
                total = total + cp.abs(field) ** 2
            m = total.shape[0] // BIN
            out.append(total.reshape(m, BIN, m, BIN).mean(axis=(1, 3)) / len(spectra))
        return (out, spectra) if keep else out

    def back(self, grads, spectra, views):
        """Runs the views' error backwards to the plate: `grads` the error's
        slope against each view's brightness. Returns its slope against
        the scene's light on the plate (conjugated), taking each
        wavelength's replay as the light times its beam over the middle
        wavelength's (what the encoding aims at). Each view's light is
        worked out again from the spectra, which takes less memory than
        keeping it."""
        n = PLATE["size"]
        total = cp.zeros((n, n), cp.complex64)
        for i, spectrum in enumerate(spectra):
            acc = cp.zeros((n, n), cp.complex64)
            for g, (spot, depth) in zip(grads, views):
                f = self._filter(i, spot, depth)
                field = cp.fft.ifft2(spectrum * f)[self.crop]
                full = cp.zeros((n, n), cp.complex64)
                spread = cp.repeat(cp.repeat(g, BIN, axis=0), BIN, axis=1) / BIN**2
                full[self.crop] = spread * field / len(spectra)
                acc += cp.fft.fft2(full) * cp.conj(f)
            back = cp.fft.ifft2(acc)
            if self.encode:
                back = back * cp.conj(self.ref[i]) * self.ref[self.offsets.index(0.0)]
            total += back
        return total


def loss(views, targets):
    """How far the views are from the targets, each scaled to fit best,
    relative to the target; and the slope of each against its view."""
    total, grads = 0.0, []
    for v, t in zip(views, targets):
        scale = float(cp.sum(v * t) / cp.sum(v * v))
        norm = float(cp.sum(t * t))
        diff = scale * v - t
        total += float(cp.sum(diff * diff)) / norm
        grads.append(2 * scale * diff / norm)
    return total / len(views), [g / len(views) for g in grads]


def gradient(scene, optics, phases, targets, views):
    """The loss for `phases`, and its slope against each phase."""
    light = scene.light(phases)
    seen, spectra = optics.views(light, views, keep=True)
    value, grads = loss(seen, targets)
    back = optics.back(grads, spectra, views)
    g = scene.reverse(back)
    # d/dphase of a point's light e^(i phase) a: i e^(i phase) a.
    return value, -2 * np.imag(np.conj(g) * np.exp(1j * phases))


def targets_for(scene, views, realizations, rng):
    """The eye's views of the scene's light, perfectly recorded, by laser,
    averaged over random phases: what it looks like without speckle."""
    optics = Optics(LASER, encode=False)
    total = None
    for _ in range(realizations):
        seen = optics.views(scene.light(rng.uniform(0, 2 * math.pi, len(scene))), views)
        total = seen if total is None else [a + b for a, b in zip(total, seen)]
    return [t / realizations for t in total]


def check():
    """Checks the all-at-once sums against summing point by point, then
    the gradient against small changes of a few phases, for a few hundred
    points, recorded perfectly, by laser."""
    rng = np.random.default_rng(2)
    scene = Scene(limit=300)
    phases = rng.uniform(0, 2 * math.pi, len(scene))
    plate = GpuPlate(**PLATE)
    for i, (point, phase) in enumerate(zip(scene.points, phases)):
        plate.add_seen(point, phase, CONE, scene.maps, i)
    batched = scene.light(phases)
    off = float(cp.linalg.norm(batched - plate.light) / cp.linalg.norm(plate.light))
    grad = (cp.random.standard_normal(batched.shape) + 1j * cp.random.standard_normal(batched.shape)).astype(cp.complex64)
    out = cp.zeros(2 * len(scene), cp.float32)
    for i, point in enumerate(scene.points):
        plate.seen_reverse(grad, point, CONE, scene.maps, i, out)
    single = cp.asnumpy(out[0::2] + 1j * out[1::2])
    many = scene.reverse(grad)
    print(f"all at once against point by point: light {off:.1e}, reverse {np.linalg.norm(many - single) / np.linalg.norm(single):.1e}")
    views = TRAIN[:2]
    targets = targets_for(scene, views, 4, rng)
    optics = Optics(LASER, encode=False)
    phases = rng.uniform(0, 2 * math.pi, len(scene))
    value, slope = gradient(scene, optics, phases, targets, views)
    for i in rng.choice(len(scene), 5, replace=False):
        step = 1e-2
        up, down = phases.copy(), phases.copy()
        up[i] += step
        down[i] -= step
        f = lambda p: loss(optics.views(scene.light(p), views), targets)[0]
        numeric = (f(up) - f(down)) / (2 * step)
        print(f"point {i}: gradient {slope[i]:+.3e}, by small changes {numeric:+.3e}")


def report(name, optics, scene, phases, targets):
    """The views of the plate made with `phases`, and how far each set
    is from its targets."""
    seen = optics.views(scene.light(phases), TRAIN + TEST)
    train = loss(seen[: len(TRAIN)], targets[: len(TRAIN)])[0]
    test = loss(seen[len(TRAIN) :], targets[len(TRAIN) :])[0]
    print(f"{name}: error {train:.3f} on the views trained on, {test:.3f} on the others", flush=True)
    return seen


def main():
    if "--check" in sys.argv:
        return check()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    start = time.time()
    scene = Scene()
    print(f"{len(scene)} points, their maps by Mitsuba: {time.time() - start:.0f} s", flush=True)

    start = time.time()
    targets = targets_for(scene, TRAIN + TEST, TARGET_REALIZATIONS, rng)
    print(f"targets, {TARGET_REALIZATIONS} random phases averaged: {time.time() - start:.0f} s", flush=True)

    optics = Optics(LED)
    phases = rng.uniform(0, 2 * math.pi, len(scene))
    before = report("random phases", optics, scene, phases, targets)

    # Adam.
    m = np.zeros(len(scene))
    v = np.zeros(len(scene))
    start = time.time()
    for step in range(1, ITERATIONS + 1):
        pick = rng.choice(len(TRAIN), BATCH, replace=False)
        value, slope = gradient(scene, optics, phases, [targets[i] for i in pick], [TRAIN[i] for i in pick])
        m = 0.9 * m + 0.1 * slope
        v = 0.999 * v + 0.001 * slope**2
        phases -= RATE * (m / (1 - 0.9**step)) / (np.sqrt(v / (1 - 0.999**step)) + 1e-12)
        if step % 10 == 0 or step == 1:
            print(f"  step {step}: error {value:.3f} ({(time.time() - start) / step:.1f} s a step)", flush=True)
    after = report("optimized phases", optics, scene, phases, targets)

    tiles = []
    middle = min(range(len(TRAIN)), key=lambda i: np.hypot(*TRAIN[i][0]))
    labels = ["straight on (trained)", "from the lower left (trained)", "between (not trained)"]
    picks = [middle, 0, len(TRAIN) + 6]
    for name, views in [("target", targets), ("random phases", before), ("optimized", after)]:
        for label, k in zip(labels, picks):
            tiles.append((f"{name}: {label}", _picture(views[k])))
    sheet(tiles, columns=3).save(OUT / "views.png")
    print(f"wrote {OUT}/views.png")


def _picture(view):
    """A view as an image, as picture() makes them."""
    v = cp.asnumpy(view)
    return picture(v, PLATE["pitch"] * BIN, half=HALF - 1, shrink=1)
