"""The plate, and the light the scene sends it.

The plate lies in the plane z = 0, square, its middle on the axis, and
records the light arriving there: a complex amplitude at each pixel, the
sum of the waves from every point of the scene. Each point's wave is
worked out exactly where it lands: e^(i k L) for L its optical path
length to the pixel, as strong as the light spreads out on its way
(see Plate.add_direct and Plate.add_through_lens).

Each point sends light only within `cone`, the largest tangent of the
angle it arrives at the plate with: the plate's pixels can only record
so steep a wave, and the reference beam must come in steeper still.
"""

import math
from dataclasses import dataclass, field

import numpy as np

from .lens import Lens


@dataclass
class Plate:
    """A plate `size` pixels square, `pitch` apart, for light of
    `wavelength`, recording `light`."""

    size: int
    pitch: float
    wavelength: float
    light: np.ndarray = field(init=False)

    def __post_init__(self):
        self.light = np.zeros((self.size, self.size), np.complex128)

    @property
    def k(self) -> float:
        return 2 * math.pi / self.wavelength

    def coordinate(self, index):
        """Where pixel `index` (along either side) is, from the middle."""
        return (np.asarray(index) - self.size / 2 + 0.5) * self.pitch

    def window(self, center, radius):
        """The pixels within the square of half-width `radius` around
        `center` (x, y): the slices that pick them out of `light` (rows run
        along y), and their x and y; or None if it misses the plate."""
        lo = [math.floor((c - radius) / self.pitch + self.size / 2) for c in center]
        hi = [math.ceil((c + radius) / self.pitch + self.size / 2) + 1 for c in center]
        lo = [max(0, v) for v in lo]
        hi = [min(self.size, v) for v in hi]
        if lo[0] >= hi[0] or lo[1] >= hi[1]:
            return None
        cols, rows = slice(lo[0], hi[0]), slice(lo[1], hi[1])
        x = self.coordinate(np.arange(lo[0], hi[0]))[None, :]
        y = self.coordinate(np.arange(lo[1], hi[1]))[:, None]
        return (rows, cols), x, y

    def add_direct(self, point, phase, cone, lens: Lens | None = None):
        """Adds the light of `point` (x, y, z behind the plate) that comes
        straight to the plate, missing the lens, with the starting
        `phase`."""
        px, py, pz = point
        depth = -pz
        found = self.window((px, py), cone * depth)
        if found is None:
            return
        where, x, y = found
        dx, dy = x - px, y - py
        tan2 = (dx * dx + dy * dy) / depth**2
        reach = tan2 < cone**2
        if lens is not None:
            # Where the line to the pixel crosses the lens's middle.
            f = (lens.z - pz) / depth
            reach &= (px + f * dx) ** 2 + (py + f * dy) ** 2 > lens.aperture**2
        distance = depth * np.sqrt(1 + tan2)
        # The light spreads over the plate as cos^3 / depth^2 per unit of
        # solid angle; the amplitude is its square root.
        strength = (1 + tan2) ** -0.75 / depth
        wave = strength * np.exp(1j * (self.k * distance + phase))
        self.light[where] += np.where(reach, wave, 0)

    def add_through_lens(self, point, phase, cone, lens: Lens, coarse=16):
        """Adds the light of `point` that reaches the plate through `lens`.

        The ray to each pixel is found exactly, by aiming through the lens
        (see Lens.aim), on a grid `coarse` pixels apart, and its path
        length and strength interpolated in between: they change so slowly
        that over 8 um the error is a nanometre or so."""
        px, py, pz = point
        # Where the light lands, roughly: as if from the lens's virtual
        # image of the point (thin lens), through its rim.
        focal = lens.radius / (2 * (lens.index - 1))
        before = lens.z - pz
        if before >= focal:
            raise ValueError("the point must be inside the focal length")
        after = before * focal / (focal - before)
        image = np.array([px, py]) * after / before
        image_z = lens.z - after
        spread = -image_z / (lens.z - image_z)
        center = image * (1 - spread)
        found = self.window(center, 1.05 * lens.aperture * spread)
        if found is None:
            return
        where, x, y = found

        # The grid the rays are found on, covering the window.
        step = coarse * self.pitch
        x0, y0 = x[0, 0], y[0, 0]
        nx = math.ceil((x[0, -1] - x0) / step) + 2
        ny = math.ceil((y[-1, 0] - y0) / step) + 2
        gx, gy = np.meshgrid(x0 + step * np.arange(nx), y0 + step * np.arange(ny))
        targets = np.column_stack([gx.ravel(), gy.ravel()])
        # Aim at each through the line from the virtual image.
        through = image + (targets - image) * (lens.z - image_z) / -image_z
        guess = (through - [px, py]) / before
        path, leave, arrive, spreading, ok = lens.aim(point, targets, guess)
        ok &= np.einsum("ij,ij->i", arrive, arrive) < cone**2
        # Light per unit of solid angle is cos^3 per unit of tangent.
        strength = np.sqrt((1 + np.einsum("ij,ij->i", leave, leave)) ** -1.5 / spreading)
        path = np.where(ok, path, 0).reshape(ny, nx)
        strength = np.where(ok, strength, 0).reshape(ny, nx)
        ok = ok.reshape(ny, nx)

        # Interpolate to every pixel in the window, where all four corners
        # around it were found.
        fx = (x - x0) / step
        fy = (y - y0) / step
        ix, iy = np.floor(fx).astype(int), np.floor(fy).astype(int)
        ax, ay = fx - ix, fy - iy

        def blend(a):
            return (
                a[iy, ix] * (1 - ax) * (1 - ay)
                + a[iy, ix + 1] * ax * (1 - ay)
                + a[iy + 1, ix] * (1 - ax) * ay
                + a[iy + 1, ix + 1] * ax * ay
            )

        reach = ok[iy, ix] & ok[iy, ix + 1] & ok[iy + 1, ix] & ok[iy + 1, ix + 1]
        wave = blend(strength) * np.exp(1j * (self.k * blend(path) + phase))
        self.light[where] += np.where(reach, wave, 0)

    def hologram(self, reference_tangent, beam_ratio=2.0):
        """The pattern the plate records: the brightness where the scene's
        light meets a reference beam, a plane wave tilted in y to
        `reference_tangent`, `beam_ratio` times the scene light's typical
        strength. Returns it, and the reference beam, to light it with."""
        y = self.coordinate(np.arange(self.size))[:, None]
        sine = reference_tangent / math.sqrt(1 + reference_tangent**2)
        typical = np.sqrt(np.mean(np.abs(self.light) ** 2))
        reference = beam_ratio * typical * np.exp(1j * self.k * sine * y)
        return np.abs(self.light + reference) ** 2, reference
