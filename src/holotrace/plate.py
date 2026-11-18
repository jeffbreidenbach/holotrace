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

from .lens import Aimed, Lens


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

    def add_through_lens(self, points, phases, cone, lens, coarse=16):
        """Adds the light of `points` (N x 3), with their starting `phases`,
        that reaches the plate through `lens` (a Lens, or one traced like
        it).

        The ray to each pixel is found exactly, by aiming through the lens
        (see Lens.aim), on a grid `coarse` pixels apart, and its path
        length and strength interpolated in between: they change so slowly
        that over 8 um the error is a nanometer or so. The rays for every
        point are aimed together, since tracing a few large batches is far
        faster than many small ones."""
        grids = [self._lens_grid(point, lens, coarse) for point in points]
        used = [i for i, grid in enumerate(grids) if grid is not None]
        if not used:
            return
        counts = [len(grids[i]["targets"]) for i in used]
        origins = np.concatenate(
            [np.broadcast_to(points[i], (n, 3)) for i, n in zip(used, counts)]
        )
        targets = np.concatenate([grids[i]["targets"] for i in used])
        guess = np.concatenate([grids[i]["guess"] for i in used])
        aimed = lens.aim(origins, targets, guess)
        bounds = np.cumsum([0] + counts)
        for k, i in enumerate(used):
            part = slice(bounds[k], bounds[k + 1])
            mine = Aimed(*(a[part] for a in aimed))
            self._add_lens_grid(grids[i], phases[i], cone, mine, lens.aperture)

    def _lens_grid(self, point, lens, coarse):
        """The grid of points across the plate the rays from `point`
        through `lens` are aimed at, `coarse` pixels apart, covering where
        its light lands, and a first guess at each ray; or None if its
        light misses the plate."""
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
        # Out to where the faces are traced, beyond the rim, so the grid
        # reaches past the rim's edge.
        found = self.window(center, lens.extent * spread)
        if found is None:
            return None
        where, x, y = found

        # The grid, covering the window.
        step = coarse * self.pitch
        x0, y0 = x[0, 0], y[0, 0]
        nx = math.ceil((x[0, -1] - x0) / step) + 2
        ny = math.ceil((y[-1, 0] - y0) / step) + 2
        gx, gy = np.meshgrid(x0 + step * np.arange(nx), y0 + step * np.arange(ny))
        targets = np.column_stack([gx.ravel(), gy.ravel()])
        # Aim at each through the line from the virtual image.
        through = image + (targets - image) * (lens.z - image_z) / -image_z
        guess = (through - [px, py]) / before
        return dict(
            where=where, x=x, y=y, coarse=coarse, step=step, shape=(ny, nx), targets=targets, guess=guess
        )

    def _add_lens_grid(self, grid, phase, cone, aimed, aperture):
        """Adds the light whose rays were aimed at `grid` (see _lens_grid):
        `aimed`, through a lens of `aperture`."""
        ny, nx = grid["shape"]
        ok = aimed.found
        values = self._lens_values(aimed, ok)
        path = np.where(ok, aimed.path, 0)

        # Interpolate to every pixel in the window, where all four corners
        # around it were found.
        x, y, step = grid["x"], grid["y"], grid["step"]
        fx = (x - x[0, 0]) / step
        fy = (y - y[0, 0]) / step
        ix, iy = np.floor(fx).astype(int), np.floor(fy).astype(int)
        ax, ay = fx - ix, fy - iy

        def blend(a):
            a = a.reshape(ny, nx)
            return (
                a[iy, ix] * (1 - ax) * (1 - ay)
                + a[iy, ix + 1] * ax * (1 - ay)
                + a[iy + 1, ix] * (1 - ax) * ay
                + a[iy + 1, ix + 1] * ax * ay
            )

        g = ok.reshape(ny, nx)
        reach = (
            g[iy, ix]
            & g[iy, ix + 1]
            & g[iy + 1, ix]
            & g[iy + 1, ix + 1]
            & (blend(values["rim"]) <= aperture)
            & (blend(values["gx"]) ** 2 + blend(values["gy"]) ** 2 < cone**2 / (1 + cone**2))
        )
        wave = blend(values["strength"]) * np.exp(1j * (self.k * blend(path) + phase))
        self.light[grid["where"]] += np.where(reach, wave, 0)

    @staticmethod
    def _lens_values(aimed, ok):
        """What is blended between the rays aimed through the lens, 0 where
        `ok` is not: how strong the light is where each lands (light per
        unit of solid angle is cos^3 per unit of tangent, spread over the
        area the lens spreads it to); the slopes of the path length across
        the plate, which are the sines of its direction (gx, gy); and how
        far from the axis it crossed the lens (rim)."""
        with np.errstate(all="ignore"):
            strength = np.sqrt((1 + np.einsum("ij,ij->i", aimed.leave, aimed.leave)) ** -1.5 / aimed.spreading)
        sines = aimed.arrive / np.sqrt(1 + np.einsum("ij,ij->i", aimed.arrive, aimed.arrive))[:, None]
        return {
            "strength": np.where(ok, strength, 0),
            "gx": np.where(ok, sines[:, 0], 0),
            "gy": np.where(ok, sines[:, 1], 0),
            "rim": np.where(ok, aimed.rim, 0),
        }

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
