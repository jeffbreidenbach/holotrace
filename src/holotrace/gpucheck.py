"""Checks the GPU's sum of the light (gpu.py) against the CPU's (plate.py).

On some of the magnifier's dots, both straight to the plate and through
the lens: the CPU's sum on a fine grid, 8 pixels apart, is the
reference. Against it, the GPU blending the same way (bilinear) shows
whether the kernels and single precision are right; blending by cubics
on coarser and coarser grids shows how few rays are enough.

The error is the root mean square of the difference in the plate's
light, over that of the light: 0.01 is 1%, a phase error of about 0.01
radian.
"""

import math
import time

import cupy as cp
import numpy as np

from .gpu import GpuPlate
from .magnifier import CONE, LENS, PLATE, dots
from .plate import Plate


def _sum(plate, points, phases, near, coarse):
    """Puts the dots' light on `plate`; returns how long it took."""
    start = time.time()
    for point, phase in zip(points, phases):
        plate.add_direct(point, phase, CONE, LENS)
    plate.add_through_lens(points[near], phases[near], CONE, LENS, coarse=coarse)
    if isinstance(plate, GpuPlate):
        cp.cuda.Device().synchronize()
        plate.finish()
    return time.time() - start


def main():
    points = dots()
    rng = np.random.default_rng(1)
    phases = rng.uniform(0, 2 * math.pi, len(points))
    # Every 10th dot, and every 6th of those whose light goes through the lens.
    pick = np.arange(len(points)) % 10 == 0
    near = np.hypot(points[:, 0], points[:, 1]) < 500
    near_pick = near & (np.cumsum(near) % 6 == 0)
    chosen = pick | near_pick
    points, phases, near = points[chosen], phases[chosen], near_pick[chosen]
    print(f"{len(points)} dots, {near.sum()} of them through the lens")

    reference = Plate(**PLATE)
    seconds = _sum(reference, points, phases, near, coarse=8)
    print(f"CPU, bilinear, grid 8 pixels apart: {seconds:.1f} s (the reference)")
    scale = np.sqrt(np.mean(np.abs(reference.light) ** 2))

    for hermite, coarse in [(False, 8), (True, 16), (True, 32), (True, 64), (True, 128)]:
        plate = GpuPlate(**PLATE, hermite=hermite)
        seconds = _sum(plate, points, phases, near, coarse)
        error = np.sqrt(np.mean(np.abs(plate.light - reference.light) ** 2)) / scale
        how = "cubic" if hermite else "bilinear"
        print(f"GPU, {how}, grid {coarse} pixels apart: {seconds:.1f} s, error {error:.5f}")
