"""A glass lens, and rays traced through it exactly.

Lengths are in micrometres. The lens's axis is the z axis, and light
travels toward +z: from the scene, through the lens, to the plate.

A direction is given by its tangent (dx/dz, dy/dz), which is convenient
because every ray here travels forward.

Rays are traced through the lens's faces a little beyond its rim, as if
the glass went on (to Lens.extent): the light just inside the rim is
then known smoothly up to it, from either side, and what is outside is
cut off by where each ray crosses, its `rim` radius (see Aimed).
"""

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np


@dataclass(frozen=True)
class Lens:
    """A biconvex lens: two spherical faces of `radius`, out to its rim
    `aperture` from the axis, where its edge is `edge` thick (blackened);
    its middle at height z along the axis."""

    z: float
    radius: float
    aperture: float
    index: float = 1.5
    edge: float = 20.0

    def sag(self, r):
        """How far each face curves in from its middle, r from the axis."""
        return self.radius - np.sqrt(self.radius**2 - np.asarray(r) ** 2)

    @property
    def thickness(self) -> float:
        """How thick the lens is at its middle."""
        return float(2 * self.sag(self.aperture) + self.edge)

    @property
    def extent(self) -> float:
        """How far from the axis the faces are traced: beyond the rim,
        nine tenths of the way to where, going on, they would meet."""
        meet = np.sqrt(self.radius**2 - (self.radius - self.thickness / 2) ** 2)
        return float(self.aperture + 0.9 * (meet - self.aperture))

    def trace(self, origin, tangents, z_end=0.0):
        """Traces rays from `origin` (x, y, z, or one per ray) in the
        directions `tangents` (M x 2) through both faces, refracting at
        each, to the plane at z_end. Returns where they land (M x 2), the
        optical path length of each (the distance, counted index times
        over in glass), their tangents after the lens (M x 2), which got
        through the faces as traced (out to `extent`), and how far from the
        axis each crossed them (beyond `aperture`, the edge stops it)."""
        d = np.column_stack([tangents, np.ones(len(tangents))])
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        o = np.broadcast_to(np.asarray(origin, float), d.shape)
        half = self.thickness / 2
        R = self.radius

        # Into the glass through the face toward the scene, whose sphere's
        # center is on the far side of the lens.
        c1 = np.array([0.0, 0.0, self.z - half + R])
        s1, hit1 = _sphere_hit(o, d, c1, R, near=True)
        p1 = o + s1[:, None] * d
        d1, ok1 = _refract(d, (p1 - c1) / R, 1 / self.index)

        # Out through the face toward the plate.
        c2 = np.array([0.0, 0.0, self.z + half - R])
        s2, hit2 = _sphere_hit(p1, d1, c2, R, near=False)
        p2 = p1 + s2[:, None] * d1
        d2, ok2 = _refract(d1, -(p2 - c2) / R, self.index)

        s3 = (z_end - p2[:, 2]) / d2[:, 2]
        end = p2 + s3[:, None] * d2
        path = s1 + self.index * s2 + s3
        rim = np.maximum(np.hypot(p1[:, 0], p1[:, 1]), np.hypot(p2[:, 0], p2[:, 1]))
        valid = hit1 & hit2 & ok1 & ok2 & (d2[:, 2] > 0) & (rim <= self.extent)
        return end[:, :2], path, d2[:, :2] / d2[:, 2:], valid, rim

    def aim(self, origin, targets, guess, z_end=0.0):
        """Finds the rays from `origin` through the lens that land on
        `targets` (see aim)."""
        return aim(self.trace, origin, targets, guess, z_end, step=1e-6)


class Aimed(NamedTuple):
    """Rays aimed at points on the plate (see aim)."""

    path: np.ndarray  # optical path length to the point
    leave: np.ndarray  # tangent leaving the source
    arrive: np.ndarray  # tangent arriving at the point
    spreading: np.ndarray  # area on the plate per unit of tangent
    found: np.ndarray  # whether a ray was found landing there
    rim: np.ndarray  # how far from the axis it crossed the lens


def aim(trace, origin, targets, guess, z_end=0.0, step=1e-6, iterations=10, tolerance=1e-3):
    """Finds, for each of `targets` (M x 2) on the plane at z_end, the ray
    from `origin` that `trace` (as Lens.trace) lands there, by Newton's
    method from the tangents `guess`, the landing point's change with the
    tangent found `step` apart. Returns them as Aimed: found means landing
    within `tolerance`, through the lens as traced, which goes beyond its
    rim; whether a ray really gets through is whether its `rim` is within
    the lens's aperture.

    The path length is carried the last little way to the target from
    where the ray landed along the plane, as the ray's tangent there says
    it grows (to first order: the path length is least, as Fermat's
    principle has it, so the error is second order)."""
    t = np.array(guess, float)
    dx, dy = np.array([step, 0.0]), np.array([0.0, step])
    landed = None
    with np.errstate(all="ignore"):
        for i in range(iterations + 1):
            end, path, out, valid, rim = trace(origin, t, z_end)
            ex = trace(origin, t + dx, z_end)[0]
            ey = trace(origin, t + dy, z_end)[0]
            # How the landing point moves with the tangent.
            j = np.stack([(ex - end) / step, (ey - end) / step], axis=2)
            det = j[:, 0, 0] * j[:, 1, 1] - j[:, 0, 1] * j[:, 1, 0]
            miss = targets - end
            # Stop once no more rays are landing: those that will have.
            # (Newton's method from a good guess lands most in 2 or 3.)
            close = valid & (np.hypot(miss[:, 0], miss[:, 1]) < tolerance / 10)
            if i == iterations or (i >= 3 and (close == landed).all()):
                break
            landed = close
            # Solve j @ change = miss, 2 x 2 at a time.
            change = np.stack(
                [
                    (j[:, 1, 1] * miss[:, 0] - j[:, 0, 1] * miss[:, 1]) / det,
                    (j[:, 0, 0] * miss[:, 1] - j[:, 1, 0] * miss[:, 0]) / det,
                ],
                axis=1,
            )
            # Rays that miss can send the step anywhere; keep it bounded.
            change = np.nan_to_num(change, nan=0.0, posinf=0.0, neginf=0.0)
            t = np.clip(t + change, -10.0, 10.0)
        # Along the plane, the path length grows by the sine of the ray's
        # angle to it per unit length.
        sines = out / np.sqrt(1 + np.einsum("ij,ij->i", out, out))[:, None]
        path = path + np.einsum("ij,ij->i", sines, miss)
    found = valid & (np.hypot(miss[:, 0], miss[:, 1]) < tolerance) & np.isfinite(det)
    return Aimed(path, t, out, np.abs(det), found, rim)


def _sphere_hit(o, d, center, radius, near):
    """How far along each ray (o + s d, d of unit length) it meets the
    sphere: where it enters (near) or leaves it; and whether it does."""
    oc = o - center
    b = np.einsum("ij,ij->i", oc, d)
    c = np.einsum("ij,ij->i", oc, oc) - radius**2
    disc = b * b - c
    root = np.sqrt(np.maximum(disc, 0.0))
    s = -b - root if near else -b + root
    return s, (disc >= 0) & (s > 0)


def _refract(d, normal, eta):
    """Bends unit directions d crossing a surface whose unit normal faces
    them, from index n1 into n2, eta = n1 / n2 (Snell's law). Returns the
    new directions, and False where the light is totally reflected."""
    cos_i = -np.einsum("ij,ij->i", normal, d)
    k = 1 - eta**2 * (1 - cos_i**2)
    out = eta * d + (eta * cos_i - np.sqrt(np.maximum(k, 0.0)))[:, None] * normal
    return out, k >= 0
