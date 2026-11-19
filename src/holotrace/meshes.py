"""Meshes from files, and points sampled over their surfaces.

A scene's surfaces become points, each a source of light (see solid.py);
for a mesh, they are spread over its triangles evenly, so each stands
for about the same patch of surface.

Evenly, not at random: points scattered at random leave gaps and clumps
(the number in any patch varies by its square root), and with ten or so
to each spot the eye can resolve, the surface looks grainy, in any
light. So several times too many are scattered and the most crowded are
removed, down to the number wanted (sample elimination, after Yuksel):
what is left is as far apart as it can be.
"""

import math

import numpy as np

# How many times too many points are scattered before thinning, and how
# far apart, in spacings, two points still count as crowding each other:
# twice the distance apart in the densest packing.
OVER = 4
REACH = 2 * math.sqrt(1 / (2 * math.sqrt(3)))


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
    """Points spread evenly over the triangles about `spacing` apart, as
    many as their area has room for: their positions, and the triangles'
    outward normals there (from the corners' winding, as STL files have
    it)."""
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    cross = np.cross(b - a, c - a)
    area = np.linalg.norm(cross, axis=1) / 2
    keep = area > 0
    a, b, c, cross, area = a[keep], b[keep], c[keep], cross[keep], area[keep]
    # OVER times too many, scattered at random by area.
    count = rng.poisson(OVER * area / spacing**2)
    which = np.repeat(np.arange(len(area)), count)
    # Uniform over each triangle: fold the unit square in half.
    u, v = rng.random(len(which)), rng.random(len(which))
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    points = a[which] + u[:, None] * (b - a)[which] + v[:, None] * (c - a)[which]
    kept = thinned(points, round(area.sum() / spacing**2), spacing, rng)
    which = which[kept]
    return points[kept], cross[which] / np.linalg.norm(cross[which], axis=1, keepdims=True)


def thinned(points, want, spacing, rng):
    """Which `want` of the points to keep so that they are evenly spread,
    `spacing` apart: the rest, the most crowded, are removed.

    A point is crowded by each neighbor within REACH spacings, the more
    the closer (its weight: the sum of (1 - distance / reach)^8). Taking
    out the heaviest point, one at a time, is the method; here many go
    at once. A level comes down step by step, and at each step every
    point above it that is heavier than all its neighbors goes (no two
    such are neighbors, so none changes another's weight), and again,
    until none is above. So the most crowded places are thinned first,
    as one at a time would: taking out every point heavier than its
    neighbors, whatever its weight, thins sparse places as fast as
    crowded ones, and leaves the points less even than at random."""
    from scipy.spatial import cKDTree

    n = len(points)
    if want >= n:
        return np.arange(n)
    reach = REACH * spacing
    pairs = cKDTree(points).query_pairs(reach, output_type="ndarray")
    a, b = pairs[:, 0].astype(np.int32), pairs[:, 1].astype(np.int32)
    del pairs
    near = points.astype(np.float32)
    apart = np.sqrt(sum((near[a, k] - near[b, k]) ** 2 for k in range(3)))
    crowding = (1 - np.minimum(apart / reach, 1)) ** 8
    del near, apart
    # (A little at random, so that no two weights are equal.)
    weight = np.bincount(a, crowding, n) + np.bincount(b, crowding, n) + 1e-9 * rng.random(n)
    there = np.ones(n, bool)
    left, level = n, weight.max()
    while left > want:
        level *= 0.7
        if level < 1e-7:
            # Nothing crowds anything any more: any will do.
            there[rng.choice(np.flatnonzero(there), left - want, replace=False)] = False
            break
        # The pairs with a point above the level: no others matter yet.
        high = np.flatnonzero((weight[a] > level) | (weight[b] > level))
        ha, hb, hc = a[high], b[high], crowding[high]
        while left > want and len(ha):
            wa, wb = weight[ha], weight[hb]
            above, beaten = np.zeros(n, bool), np.zeros(n, bool)
            above[ha[wa > level]] = above[hb[wb > level]] = True
            beaten[ha[wa <= wb]] = beaten[hb[wb < wa]] = True
            go = np.flatnonzero(above & ~beaten)
            if len(go) == 0:
                break
            if len(go) > left - want:
                go = go[np.argsort(-weight[go])[: left - want]]
            there[go] = False
            left -= len(go)
            # Their neighbors are the less crowded for it.
            gone = ~(there[ha] & there[hb])
            weight -= np.bincount(ha[gone], hc[gone], n) + np.bincount(hb[gone], hc[gone], n)
            ha, hb, hc = ha[~gone], hb[~gone], hc[~gone]
        still = there[a] & there[b]
        a, b, crowding = a[still], b[still], crowding[still]
    return np.flatnonzero(there)
