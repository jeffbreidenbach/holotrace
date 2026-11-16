"""A glass lens, and rays traced through it exactly.

Lengths are in micrometres. The lens's axis is the z axis, and light
travels toward +z: from the scene, through the lens, to the plate.

A direction is given by its tangent (dx/dz, dy/dz), which is convenient
because every ray here travels forward.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Lens:
    """A biconvex lens: two spherical faces of `radius`, meeting in a sharp
    rim `aperture` from the axis, its middle at height z along it."""

    z: float
    radius: float
    aperture: float
    index: float = 1.5

    @property
    def thickness(self) -> float:
        """How thick the lens is at its middle."""
        sag = self.radius - np.sqrt(self.radius**2 - self.aperture**2)
        return 2 * sag

    def trace(self, origin, tangents, z_end=0.0):
        """Traces rays from `origin` (x, y, z) in the directions `tangents`
        (M x 2) through both faces, refracting at each, to the plane at
        z_end. Returns where they land (M x 2), the optical path length of
        each (the distance, counted index times over in glass), their
        tangents after the lens (M x 2), and which got through the lens."""
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
        rim2 = self.aperture**2
        valid = (
            hit1
            & hit2
            & ok1
            & ok2
            & (p1[:, 0] ** 2 + p1[:, 1] ** 2 <= rim2)
            & (p2[:, 0] ** 2 + p2[:, 1] ** 2 <= rim2)
            & (d2[:, 2] > 0)
        )
        return end[:, :2], path, d2[:, :2] / d2[:, 2:], valid

    def aim(self, origin, targets, guess, z_end=0.0, iterations=10):
        """Finds, for each of `targets` (M x 2) on the plane at z_end, the
        ray from `origin` through the lens that lands there, by Newton's
        method from the tangents `guess`. Returns, for each, its optical
        path length, the tangent it leaves `origin` with and the one it
        arrives with, how much the lens spreads the light there (the area
        on the plane per unit of tangent), and whether it was found."""
        t = np.array(guess, float)
        step = 1e-6
        dx, dy = np.array([step, 0.0]), np.array([0.0, step])
        with np.errstate(all="ignore"):
            for _ in range(iterations + 1):
                end, path, out, valid = self.trace(origin, t, z_end)
                ex = self.trace(origin, t + dx, z_end)[0]
                ey = self.trace(origin, t + dy, z_end)[0]
                # How the landing point moves with the tangent.
                j = np.stack([(ex - end) / step, (ey - end) / step], axis=2)
                det = j[:, 0, 0] * j[:, 1, 1] - j[:, 0, 1] * j[:, 1, 0]
                miss = targets - end
                if _ == iterations:
                    break
                # Solve j @ change = miss, 2 x 2 at a time.
                change = np.stack(
                    [
                        (j[:, 1, 1] * miss[:, 0] - j[:, 0, 1] * miss[:, 1]) / det,
                        (j[:, 0, 0] * miss[:, 1] - j[:, 1, 0] * miss[:, 0]) / det,
                    ],
                    axis=1,
                )
                t += np.nan_to_num(change)
        found = valid & (np.hypot(miss[:, 0], miss[:, 1]) < 1e-4) & np.isfinite(det)
        return path, t, out, np.abs(det), found


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
