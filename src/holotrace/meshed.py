"""Step 2: the magnifying glass, its lens traced by Mitsuba as a mesh.

First, how far off the light's path lengths through the mesh are, from
those through the true lens, for meshes of more and more rings: for a
few dots, aiming rays at a grid of points across where their light
lands. A path length off by L shifts the light's phase by 2 pi L /
wavelength; to look right it needs to be well under a tenth of a
wavelength. Then the whole hologram made through the finest mesh that
is good enough, against the one through the true lens, as the eye sees
them. The views go to out/mesh.
"""

import math
from pathlib import Path

import numpy as np

from .eye import Eye
from .magnifier import (
    DIRECTIONS,
    DOTS_Z,
    LENS,
    PUPIL,
    WAVELENGTH,
    picture,
    predict,
    replay,
    sheet,
    sum_light,
)
from .mesh import MeshLens

RINGS = [8, 16, 32, 64, 128]
SAMPLES = [(0.0, 0.0, DOTS_Z), (200.0, 0.0, DOTS_Z), (300.0, 300.0, DOTS_Z)]
# Good enough: no ray's phase off by more than this, in radians.
GOOD = 0.1

OUT = Path("out/mesh")


def targets(point, n=41):
    """A grid of n x n points across where the light of `point` lands
    through the lens, and a first guess at the ray to each (as in
    Plate.add_through_lens)."""
    px, py, pz = point
    focal = LENS.radius / (2 * (LENS.index - 1))
    before = LENS.z - pz
    after = before * focal / (focal - before)
    image = np.array([px, py]) * after / before
    image_z = LENS.z - after
    spread = -image_z / (LENS.z - image_z)
    center = image * (1 - spread)
    reach = 0.95 * LENS.aperture * spread
    g = np.linspace(-reach, reach, n)
    gx, gy = np.meshgrid(center[0] + g, center[1] + g)
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    through = image + (grid - image) * (LENS.z - image_z) / -image_z
    return grid, (through - [px, py]) / before


def errors(lens):
    """How far the path lengths through `lens` are from the true lens's,
    over the sample dots: the root mean square and the largest, in um,
    after taking out each dot's average (which only shifts all of its
    light's phase together, as its random starting phase does)."""
    off = []
    for point in SAMPLES:
        grid, guess = targets(point)
        true = LENS.aim(point, grid, guess)
        mesh = lens.aim(point, grid, guess)
        both = true.found & mesh.found & (true.rim <= LENS.aperture)
        d = mesh.path[both] - true.path[both]
        off.append(d - d.mean())
    off = np.concatenate(off)
    return float(np.sqrt(np.mean(off**2))), float(np.abs(off).max())


def main(gpu=False):
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"path length through the meshed lens, against the true lens (wavelength {WAVELENGTH} um):")
    print(" rings  triangles   rms (nm)   max (nm)   max phase (rad)")
    chosen = None
    for rings in RINGS:
        lens = MeshLens(LENS, rings)
        rms, worst = errors(lens)
        phase = 2 * math.pi * worst / WAVELENGTH
        print(f"{rings:6d} {lens.triangles:10d} {1000 * rms:10.2f} {1000 * worst:10.2f} {phase:12.3f}")
        if chosen is None and phase < GOOD:
            chosen = rings
    if chosen is None:
        print(f"no mesh tried is good enough (max phase under {GOOD} rad)")
        return
    print(f"using {chosen} rings")

    image_depth, _ = predict()
    true = Eye(replay(sum_light(gpu=gpu)), 0.5, WAVELENGTH)
    meshed = Eye(replay(sum_light(chosen, gpu=gpu)), 0.5, WAVELENGTH)
    tiles = []
    for where, direction in DIRECTIONS.items():
        a = true.view(direction, PUPIL, image_depth)
        b = meshed.view(direction, PUPIL, image_depth)
        match = np.corrcoef(a.ravel(), b.ravel())[0, 1]
        print(f"looking {where}, focused on the virtual image: views correlate {match:.4f}")
        tiles += [
            (f"true lens, looking {where}", picture(a, 0.5)),
            (f"mesh of {chosen} rings, looking {where}", picture(b, 0.5)),
        ]
    sheet(tiles).save(OUT / "views.png")
    print(f"wrote {OUT}/views.png")
