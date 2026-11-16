"""The lens as Mitsuba sees it: a mesh of flat triangles.

Mitsuba has no lens shape, so the lens is made of triangles, and Mitsuba
finds where rays meet them. Each triangle is flat, cutting a little
inside the sphere it stands in for, by up to (edge length)^2 / 8 radius;
light through it travels that much less glass, and its phase is off by
that much over the index less one. The normals, though, are the sphere's
own at the corners, blended smoothly across each triangle (as Mitsuba
does for shading), so rays bend almost as at the true surface. How fine
the mesh needs to be is what step 2 measures.

Mitsuba works in single precision. Its points are good to about 0.1 nm
over the few millimetres here, well within what phase needs.
"""

import math
from dataclasses import dataclass, field

import numpy as np

from .lens import Lens, aim

_scenes = {}


def _mitsuba():
    """Mitsuba, set to trace on the CPU (or GPU where it can)."""
    import mitsuba as mi

    if mi.variant() is None:
        from . import use_best_variant

        use_best_variant()
    return mi


def lens_mesh(lens: Lens, rings: int):
    """The lens's faces as triangles: on each, `rings` rings of vertices
    out to where the faces are traced (Lens.extent, a little beyond the
    rim), 6 x rings around, and one in the middle. Returns the vertex
    positions and normals (N x 3), and the triangles (F x 3)."""
    half = lens.thickness / 2
    r = lens.radius
    around = 6 * rings
    radii = lens.extent * np.arange(1, rings + 1) / rings
    angles = 2 * math.pi * np.arange(around) / around
    rr, aa = np.meshgrid(radii, angles, indexing="ij")
    x = np.concatenate([[0.0], (rr * np.cos(aa)).ravel()])
    y = np.concatenate([[0.0], (rr * np.sin(aa)).ravel()])
    sag = r - np.sqrt(r * r - x * x - y * y)

    # Triangles of one face: a fan around the middle, then two per
    # quadrilateral between rings.
    def ring(i, j):
        return 1 + i * around + (j % around)

    faces = [(0, ring(0, j), ring(0, j + 1)) for j in range(around)]
    for i in range(rings - 1):
        for j in range(around):
            a, b = ring(i, j), ring(i, j + 1)
            c, d = ring(i + 1, j), ring(i + 1, j + 1)
            faces += [(a, c, d), (a, d, b)]
    faces = np.array(faces)

    positions, normals, triangles = [], [], []
    # The face toward the scene bulges toward -z, the one toward the plate
    # toward +z; each sphere's center is on the far side.
    for sign in (-1, 1):
        z = lens.z + sign * (half - sag)
        center = lens.z + sign * (half - r)
        p = np.column_stack([x, y, z])
        n = (p - [0.0, 0.0, center]) / r
        offset = sum(len(q) for q in positions)
        positions.append(p)
        normals.append(n)
        triangles.append(faces + offset)
    return np.vstack(positions), np.vstack(normals), np.vstack(triangles)


@dataclass
class MeshLens:
    """`lens`, traced by Mitsuba as a mesh of `rings` rings a face."""

    lens: Lens
    rings: int
    _scene: object = field(init=False, repr=False)

    def __post_init__(self):
        mi = _mitsuba()
        positions, normals, faces = lens_mesh(self.lens, self.rings)
        mesh = mi.Mesh("lens", len(positions), len(faces), has_vertex_normals=True)
        params = mi.traverse(mesh)
        params["vertex_positions"] = mi.Float(positions.astype(np.float32).ravel())
        params["vertex_normals"] = mi.Float(normals.astype(np.float32).ravel())
        params["faces"] = mi.UInt32(faces.astype(np.uint32).ravel())
        params.update()
        self._scene = mi.load_dict({"type": "scene", "lens": mesh})

    # The lens's measurements, as Lens has them.
    def __getattr__(self, name):
        return getattr(self.lens, name)

    @property
    def triangles(self) -> int:
        return 2 * 6 * self.rings * (2 * self.rings - 1)

    def _hit(self, origins, directions):
        """Where rays first meet the mesh: the points and the (smoothly
        blended) unit normals there, and which did."""
        mi = _mitsuba()
        # Mitsuba can hang on rays that are not numbers.
        finite = np.isfinite(origins).all(axis=1) & np.isfinite(directions).all(axis=1)
        origins = np.where(finite[:, None], origins, 0.0)
        directions = np.where(finite[:, None], directions, [0.0, 0.0, 1.0])

        def vec(cls, a):
            a = a.astype(np.float32)
            return cls(mi.Float(a[:, 0]), mi.Float(a[:, 1]), mi.Float(a[:, 2]))

        ray = mi.Ray3f(o=vec(mi.Point3f, origins), d=vec(mi.Vector3f, directions))
        si = self._scene.ray_intersect(ray)
        p = np.column_stack([np.array(si.p.x), np.array(si.p.y), np.array(si.p.z)])
        n = si.sh_frame.n
        n = np.column_stack([np.array(n.x), np.array(n.y), np.array(n.z)])
        return p.astype(float), n.astype(float), np.array(si.is_valid()) & finite

    def trace(self, origin, tangents, z_end=0.0):
        """As Lens.trace, but finding the faces with Mitsuba."""
        from .lens import _refract

        d = np.column_stack([tangents, np.ones(len(tangents))])
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        o = np.broadcast_to(np.asarray(origin, float), d.shape)
        p1, n1, hit1 = self._hit(o, d)
        # Normals face the light coming in.
        n1 *= -np.sign(np.einsum("ij,ij->i", n1, d))[:, None]
        d1, ok1 = _refract(d, n1, 1 / self.index)
        # Start just inside, so as not to meet the same face again.
        p2, n2, hit2 = self._hit(p1 + 1e-3 * d1, d1)
        n2 *= -np.sign(np.einsum("ij,ij->i", n2, d1))[:, None]
        d2, ok2 = _refract(d1, n2, self.index)
        s3 = (z_end - p2[:, 2]) / d2[:, 2]
        end = p2 + s3[:, None] * d2
        path = (
            np.linalg.norm(p1 - o, axis=1)
            + self.index * np.linalg.norm(p2 - p1, axis=1)
            + s3
        )
        valid = hit1 & hit2 & ok1 & ok2 & (d2[:, 2] > 0) & (p2[:, 2] > p1[:, 2])
        rim = np.maximum(np.hypot(p1[:, 0], p1[:, 1]), np.hypot(p2[:, 0], p2[:, 1]))
        return end[:, :2], path, d2[:, :2] / d2[:, 2:], valid, rim

    def aim(self, origin, targets, guess, z_end=0.0):
        """As Lens.aim. Single precision needs a coarser step."""
        return aim(self.trace, origin, targets, guess, z_end, step=1e-5)
