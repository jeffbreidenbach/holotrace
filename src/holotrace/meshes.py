"""Meshes from files, and points sampled over their surfaces.

A scene's surfaces become points, each a source of light (see solid.py);
for a mesh, they are scattered over its triangles evenly by area, so
each stands for about the same patch of surface.
"""

import numpy as np


def read_stl(path):
    """The triangles (T x 3 x 3) of an STL file, binary or text."""
    data = open(path, "rb").read()
    count = int.from_bytes(data[80:84], "little")
    if len(data) == 84 + 50 * count:
        record = np.dtype([("normal", "<f4", 3), ("corners", "<f4", (3, 3)), ("attr", "<u2")])
        return np.frombuffer(data, record, count, 84)["corners"].astype(float)
    words = data.decode("ascii", "replace").split()
    # In text STL each "vertex" is followed by its three coordinates.
    values = []
    for i, w in enumerate(words):
        if w == "vertex":
            values += [float(v) for v in words[i + 1 : i + 4]]
    return np.array(values).reshape(-1, 3, 3)


def sample(triangles, spacing, rng):
    """Points scattered over the triangles about `spacing` apart, evenly
    by area: their positions, and the triangles' outward normals there
    (from the corners' winding, as STL files have it)."""
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    cross = np.cross(b - a, c - a)
    area = np.linalg.norm(cross, axis=1) / 2
    keep = area > 0
    a, b, c, cross, area = a[keep], b[keep], c[keep], cross[keep], area[keep]
    count = rng.poisson(area / spacing**2)
    which = np.repeat(np.arange(len(area)), count)
    # Uniform over each triangle: fold the unit square in half.
    u, v = rng.random(len(which)), rng.random(len(which))
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    points = a[which] + u[:, None] * (b - a)[which] + v[:, None] * (c - a)[which]
    normals = cross[which] / np.linalg.norm(cross[which], axis=1, keepdims=True)
    return points, normals
