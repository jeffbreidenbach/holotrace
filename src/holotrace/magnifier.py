"""Step 1: a hologram of a magnifying glass.

A grid of dots, 40 um apart, lies 2 mm behind a small glass lens, inside
its focal length, so that through it they look magnified, as a virtual
image farther back. The lens is 0.3 mm behind a plate about 2 mm across,
which records the hologram. An eye looks at the hologram from far away.

If the hologram reproduces the light exactly, then through the recorded
lens the eye should see the dots magnified, sharp only when it focuses
on the virtual image, and shifting against the lens's rim as it moves;
around the lens, the dots should look as they are, sharp at their own
depth. The run checks how deep the virtual image is and how magnified,
against what tracing rays through the lens predicts, and writes the
eye's views to out/magnifier.
"""

import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .eye import Eye
from .lens import Lens
from .plate import Plate

# Lengths in micrometres.
WAVELENGTH = 0.633  # a helium-neon laser's red
PLATE = dict(size=4096, pitch=0.5, wavelength=WAVELENGTH)
LENS = Lens(z=-300, radius=3000, aperture=500)
DOTS_Z = LENS.z - 2000
SPACING = 40
EXTENT = 900
# Light reaches the plate within about 7 degrees of straight on; the
# reference beam comes in at about 23.6 degrees, steeply enough that the
# light the hologram sends straight on, or doubles back, misses the eye.
CONE = math.tan(math.radians(7))
REFERENCE = math.tan(math.asin(0.4))
# The eye: its pupil's numerical aperture, and where it looks from.
PUPIL = 0.04
DIRECTIONS = {"straight on": (0.0, 0.0), "from the right": (0.06, 0.0)}

OUT = Path("out/magnifier")


def dots():
    """The grid of dots, with two missing to show which way is up: the
    one just right of the middle, and the one two above it."""
    n = int(EXTENT // SPACING)
    points = [
        (i * SPACING, j * SPACING, DOTS_Z)
        for j in range(-n, n + 1)
        for i in range(-n, n + 1)
        if (i, j) not in {(1, 0), (0, 2)}
    ]
    return np.array(points, float)


# Each worker's lens, made once: LENS itself, or traced by Mitsuba as a
# mesh of the given number of rings (see mesh.py).
_lenses = {}


def _lens(rings, alone=False):
    """The lens; `alone` if this process is the only one tracing, so
    Mitsuba may use every core."""
    if rings is None:
        return LENS
    if rings not in _lenses:
        from .mesh import MeshLens, _mitsuba

        # Mitsuba first: Dr.Jit loaded alone hangs.
        _mitsuba()
        if not alone:
            import drjit as dr

            # The workers already use every core between them.
            dr.set_thread_count(1)
        _lenses[rings] = MeshLens(LENS, rings)
    return _lenses[rings]


def _light_from(args):
    """The plate's light from some of the dots, worker `n`'s share; it
    reports how long each part took."""
    n, points, phases, rings = args
    lens = _lens(rings)
    plate = Plate(**PLATE)
    start = time.time()
    for point, phase in zip(points, phases):
        plate.add_direct(point, phase, CONE, LENS)
    direct = time.time() - start
    near = np.hypot(points[:, 0], points[:, 1]) < 500
    plate.add_through_lens(points[near], phases[near], CONE, lens)
    print(
        f"  worker {n}: {len(points)} dots' direct light in {direct:.0f} s, "
        f"{near.sum()} through the lens in {time.time() - start - direct:.0f} s",
        flush=True,
    )
    return plate.light


def sum_light(rings=None, gpu=False):
    """The plate with all the dots' light on it, through the lens as it
    is, or (with rings) as a mesh traced by Mitsuba: summed on the CPU's
    cores, or the GPU (see gpu.py). Prints how long it took."""
    points = dots()
    rng = np.random.default_rng(1)
    phases = rng.uniform(0, 2 * math.pi, len(points))
    lens = "the true lens" if rings is None else f"a mesh of {rings} rings"
    if gpu:
        return _sum_on_gpu(points, phases, rings, lens)
    workers = os.cpu_count() or 1
    chunks = [(i, points[i::workers], phases[i::workers], rings) for i in range(workers)]
    print(f"summing {len(points)} dots' light through {lens} on {workers} workers", flush=True)
    start = time.time()
    plate = Plate(**PLATE)
    with ProcessPoolExecutor(workers) as pool:
        for light in pool.map(_light_from, chunks):
            plate.light += light
    print(f"summed on {plate.size}^2 pixels in {time.time() - start:.0f} s", flush=True)
    return plate


# On the GPU, path lengths through the lens are blended by cubics, from a
# grid this many pixels apart.
GPU_COARSE = 32


def _sum_on_gpu(points, phases, rings, lens_name):
    import cupy as cp

    from .gpu import GpuPlate

    print(f"summing {len(points)} dots' light through {lens_name} on the GPU", flush=True)
    lens = _lens(rings, alone=True)
    plate = GpuPlate(**PLATE)
    start = time.time()
    for point, phase in zip(points, phases):
        plate.add_direct(point, phase, CONE, LENS)
    cp.cuda.Device().synchronize()
    direct = time.time() - start
    near = np.hypot(points[:, 0], points[:, 1]) < 500
    plate.add_through_lens(points[near], phases[near], CONE, lens, coarse=GPU_COARSE)
    plate.finish()
    print(
        f"summed on {plate.size}^2 pixels: {len(points)} dots' direct light in {direct:.1f} s, "
        f"{near.sum()} through the lens in {time.time() - start - direct:.1f} s",
        flush=True,
    )
    return plate


def replay(plate):
    """The hologram of the plate's light, lit by the reference beam again:
    it sends out the scene's light, with the rest heading off in other
    directions."""
    hologram, reference = plate.hologram(REFERENCE)
    return hologram * reference / np.abs(reference) ** 2


def predict():
    """Where the lens puts the dots' virtual image, and how magnified, by
    tracing two close rays from a dot and running them back to where they
    meet."""
    for height in (0.0, 100.0):
        point = np.array([height, 0.0, DOTS_Z])
        tangent = height / (LENS.z - DOTS_Z)
        rays = np.array([[tangent - 1e-4, 0.0], [tangent + 1e-4, 0.0]])
        end, _, out, _, _ = LENS.trace(point, rays)
        # Where the two rays, run back from the plate, cross.
        depth = -(end[1, 0] - end[0, 0]) / (out[1, 0] - out[0, 0])
        if height == 0:
            image_depth = -depth
        else:
            magnification = (end[0, 0] + depth * out[0, 0]) / height
    return image_depth, magnification


def spacing(view, pitch, half):
    """The spacing of the dots in the middle of a view, `half` either way
    of its middle: from the strongest repeat along x in it."""
    n = view.shape[0]
    m = int(half / pitch)
    patch = view[n // 2 - m : n // 2 + m, n // 2 - m : n // 2 + m]
    profile = patch.mean(axis=0)
    power = np.abs(np.fft.rfft(profile - profile.mean())) ** 2
    freqs = np.fft.rfftfreq(len(profile), pitch)
    peak = np.argmax(power[2:]) + 2
    return 1 / freqs[peak]


def sharpness(view, pitch, half):
    """How concentrated the light is in the middle of a view: highest
    when the dots there are in focus."""
    n = view.shape[0]
    m = int(half / pitch)
    patch = view[n // 2 - m : n // 2 + m, n // 2 - m : n // 2 + m]
    return float((patch**2).sum() / patch.sum() ** 2 * patch.size)


def picture(view, pitch, half=700, shrink=4):
    """The middle of a view, `half` either way, as a grey image, averaged
    over `shrink` pixels each way and brightened (square root)."""
    n = view.shape[0]
    m = int(half / pitch)
    patch = view[n // 2 - m : n // 2 + m, n // 2 - m : n // 2 + m]
    k = patch.shape[0] // shrink
    patch = patch[: k * shrink, : k * shrink].reshape(k, shrink, k, shrink).mean(axis=(1, 3))
    # Rows run up the image.
    patch = np.sqrt(patch[::-1] / np.percentile(patch, 99.9))
    return Image.fromarray((np.clip(patch, 0, 1) * 255).astype(np.uint8))


def main(gpu=False):
    OUT.mkdir(parents=True, exist_ok=True)
    image_depth, magnification = predict()
    print(f"lens: {LENS.thickness:.1f} um thick")
    print(
        f"predicted virtual image: {image_depth:.0f} um behind the plate, "
        f"{magnification:.3f} times magnified"
    )

    plate = sum_light(gpu=gpu)
    eye = Eye(replay(plate), plate.pitch, WAVELENGTH)
    exact = Eye(plate.light, plate.pitch, WAVELENGTH)

    # How sharp the middle, through the lens, is at each depth.
    depths = np.arange(4000, 8001, 250)
    sharp = [sharpness(eye.view((0, 0), PUPIL, d), plate.pitch, 250) for d in depths]
    best = depths[int(np.argmax(sharp))]
    print(f"sharpest through the lens at {best} um (scanned every 250 um)")

    focused = eye.view((0, 0), PUPIL, image_depth)
    measured = spacing(focused, plate.pitch, 250) / SPACING
    print(f"measured magnification: {measured:.2f}")
    direct = exact.view((0, 0), PUPIL, image_depth)
    match = np.corrcoef(focused.ravel(), direct.ravel())[0, 1]
    print(f"hologram's view against the light it recorded: correlation {match:.4f}")

    depths = {"on the dots": -DOTS_Z, "on the virtual image": image_depth}
    tiles = []
    for (where, direction) in DIRECTIONS.items():
        for (focus, depth) in depths.items():
            img = picture(eye.view(direction, PUPIL, depth), plate.pitch)
            name = f"{where} {focus}".replace(" ", "-")
            img.save(OUT / f"{name}.png")
            tiles.append((f"looking {where}, focused {focus}", img))
    sheet(tiles).save(OUT / "views.png")
    print(f"wrote {OUT}/views.png")


def sheet(tiles, columns=2):
    """The views in rows of `columns`, each labelled."""
    w, h = tiles[0][1].size
    label = 22
    rows = math.ceil(len(tiles) / columns)
    out = Image.new("L", (columns * (w + 10) - 10, rows * (h + label + 10) - 10), 40)
    draw = ImageDraw.Draw(out)
    for i, (text, img) in enumerate(tiles):
        x, y = (i % columns) * (w + 10), (i // columns) * (h + label + 10)
        draw.text((x + 4, y + 4), text, fill=230)
        out.paste(img, (x, y + label))
    return out
