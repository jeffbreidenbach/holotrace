"""Deep scenes: the plate's light carried plane to plane.

In sum_planes (solid.py) each tile of the plate gathers each slab's light
from a patch as wide as the tile and the margin that light spreads by on
its way to the plate: the deeper the scene, the bigger the margins, and
the more of each patch is worked out only to be thrown away, until the
patches no longer fit in memory at all.

Here the slabs are taken in groups a few millimetres deep instead. Each
group's slabs are carried, tile by tile as before but only a short way,
to a plane in the group's middle, so the margins stay small. That plane,
the whole plate's width of it, is then carried to the plate in one go:
transformed, each of its plane waves moved on by its own phase, and,
since what is kept of the plate is its spectrum, added to the spectrum
straight away, never transformed back. Transforms that big are done on
disk: the plane's rows on the GPU a band at a time as it is made, its
columns on the GPU a block at a time.

The spectrum's file is kept a block of columns at a time, BLOCK
columns wide (shape: blocks, rows, BLOCK), so that each block is read and
written in one piece; the viewer reads it either way (see viewer.Look).

The plane's transform is periodic: light leaving one side would come
back in the other. As long as all the scene's light lands on the plate,
nothing does. If some lands beyond, the planes are made wider than the
plate, by as much as it spreads, and the plate is cut out of the sum at
the end and transformed afresh.
"""

import math
import os
import time

import numpy as np

from . import solid
from .magnifier import WAVELENGTH

BLOCK = 512
# The biggest band of rows of a plane held in main memory, in bytes.
BAND = 6 * 2**30


def sum_by_groups(points, light, phases, size, path):
    """The spectrum of the plate's light, as solid.sum_planes and then
    solid.spectrum_in_place would make it (the plate `size` pixels square,
    kept every OPTICS.keep), made group by group as above, into a new file
    at `path`, which is returned (as a memmap, in blocks of columns)."""
    import cupy as cp

    o = solid.OPTICS
    n, pitch = size // o.keep, o.kept_pitch
    block = math.gcd(n, BLOCK)
    groups = _groups(points)

    # The planes' width: the plate, or more if the light spreads past it
    # (each point's light reaches as far as it spreads on the way to its
    # group's plane and from there to the plate).
    spread = 0.0
    for middle, slabs in groups:
        for plane, mine in slabs:
            p = points[mine]
            depth = -p[:, 2]
            reach = np.abs(p[:, :2]).max(axis=1) + o.cone * (np.abs(depth - middle) + abs(middle))
            spread = max(spread, reach.max())
    wide = 2 * (int(math.ceil(spread / pitch)) + 8)
    width = n
    while width < wide or solid._fast_len(width) != width:
        width += 2 * block
    edge = (width - n) // 2

    out = np.lib.format.open_memmap(path, "w+", np.complex64, (n // block, n, block))
    sums = out if width == n else _scratch(path, "sum", (width // block, width, block))
    plane = _scratch(path, "plane", (width // block, width, block))
    print(f"  kept every {pitch:g} um: {n}^2 pixels, {sum(len(s) for _, s in groups)} slabs in {len(groups)} "
          f"groups, carried plane to plane{f', on planes {width}^2' if width > n else ''}", flush=True)
    start = time.time()
    for g, (middle, slabs) in enumerate(groups):
        t = time.time()
        _plane(points, slabs, middle, light, phases, width, plane)
        made = time.time() - t
        _add_carried(plane, middle, sums, first=g == 0)
        print(f"  group {g + 1} of {len(groups)}, {len(slabs)} slabs, its plane {middle / 1000:+.2f} mm: made in "
              f"{made:.0f} s, carried in {time.time() - t - made:.0f} s; {time.time() - start:.0f} s", flush=True)
    _remove(plane)
    if width > n:
        _cut(sums, out, edge)
        _remove(sums)
    out.flush()
    cp.get_default_memory_pool().free_all_blocks()
    return out


def _groups(points):
    """The scene's slabs (as in sum_planes: each its recording plane's
    distance behind the plate and its points), in groups no deeper than
    lets a tile of the plate on the GPU carry each slab's light to the
    group's middle plane (its distance, behind the plate) with a margin
    of an eighth of the GPU's patch: [(middle, [(plane, points)])]."""
    o = solid.OPTICS
    depth = -points[:, 2]
    which = np.floor(depth / o.slab).astype(int)
    slabs = []
    for k in sorted(set(which)):
        plane = k * o.slab - o.clear if k >= 0 else (k + 1) * o.slab + o.clear
        slabs.append((plane, np.flatnonzero(which == k)))
    deepest = 2 * (solid.GPU_PATCH // 8 - 8) * o.kept_pitch / o.cone
    # (The planes of the slabs either side of the plate are beyond it, so
    # are not in order of depth: what counts is how far apart they are.)
    span = lambda s: (min(p for p, _ in s), max(p for p, _ in s))
    groups, current = [], []
    for slab in slabs:
        lo, hi = span(current + [slab])
        if current and hi - lo > deepest:
            groups.append(current)
            current = []
        current.append(slab)
    groups.append(current)
    return [(sum(span(s)) / 2, s) for s in groups]


def _plane(points, slabs, middle, light, phases, width, plane):
    """A group's light (its `slabs` of `points`) on its middle plane, `middle` behind the plate,
    `width` pixels square, into `plane` (in blocks of columns), already
    transformed along its rows: worked out a band of rows at a time, each
    tile of a band gathering each slab's light on its recording plane and
    carrying it to the middle (solid._carry), then the band transformed
    on the GPU."""
    import cupy as cp
    import cupyx.scipy.fft as cufft

    from .gpu import SeenPoints

    o = solid.OPTICS
    pitch = o.kept_pitch
    grid = dict(size=width, pitch=pitch, wavelength=WAVELENGTH)
    parts = []
    for at, mine in slabs:
        seen_from = points[mine] + [0.0, 0.0, at]
        margin = int(math.ceil(o.cone * abs(at - middle) / pitch)) + 8
        px, py, pz = seen_from.T
        r = o.cone * np.abs(pz)
        bounds = (np.floor((px - r) / pitch + width / 2), np.ceil((px + r) / pitch + width / 2) + 1,
                  np.floor((py - r) / pitch + width / 2), np.ceil((py + r) / pitch + width / 2) + 1)
        parts.append((at, mine, seen_from, margin, bounds))
    deepest = max(m for _, _, _, m, _ in parts)
    tile = min(width, (solid.GPU_PATCH - 2 * deepest) // 256 * 256, BAND // (8 * width) // 256 * 256)
    blocks, block = plane.shape[0], plane.shape[2]
    band = np.empty((tile, width), np.complex64)
    for y0 in range(0, width, tile):
        th = min(tile, width - y0)
        band[:th] = 0
        for x0 in range(0, width, tile):
            tw = min(tile, width - x0)
            out = None
            for at, mine, seen_from, m, (lo_x, hi_x, lo_y, hi_y) in parts:
                ox, oy, w, h = x0 - m, y0 - m, tw + 2 * m, th + 2 * m
                near = (hi_x > ox) & (lo_x < ox + w) & (hi_y > oy) & (lo_y < oy + h)
                if not near.any():
                    continue
                which = mine[near]
                seen = SeenPoints(**grid, points=seen_from[near], maps=cp.asarray(light[which], cp.float32),
                                  cone=o.cone, tile=(ox, oy, w, h))
                carried = solid._carry(seen.light(phases[which]), at - middle, m, pitch)
                out = carried if out is None else out + carried
                del seen, carried
                cp.get_default_memory_pool().free_all_blocks()
            if out is not None:
                band[:th, x0 : x0 + tw] = cp.asnumpy(out)
            del out
        for r in range(0, th, 256):
            rows = cufft.fft(cp.asarray(band[r : min(r + 256, th)]), axis=1, overwrite_x=True)
            band[r : r + len(rows)] = cp.asnumpy(rows)
            del rows
        plane[:, y0 : y0 + th, :] = band[:th].reshape(th, blocks, block).transpose(1, 0, 2)
    cp.get_default_memory_pool().free_all_blocks()
    cp.fft.config.get_plan_cache().clear()


def _add_carried(plane, middle, sums, first):
    """Transforms `plane` (already along its rows) along its columns, a
    block at a time, carries it `middle` to the plate (its plane waves
    within the cone and the band limit, each moved on by its own phase, as
    in solid._carry), and adds it to `sums` (or, if `first`, puts it
    there)."""
    import cupy as cp
    import cupyx.scipy.fft as cufft

    o, lam = solid.OPTICS, WAVELENGTH
    blocks, width, block = plane.shape
    pitch = o.kept_pitch
    fy = cp.fft.fftfreq(width, pitch).astype(cp.float32)[:, None]
    fx = np.fft.fftfreq(width, pitch).astype(np.float32)
    limit = 1 / (lam * math.sqrt((2 * middle / (width * pitch)) ** 2 + 1))
    for j in range(blocks):
        cols = cufft.fft(cp.asarray(plane[j]), axis=0, overwrite_x=True)
        x = cp.asarray(fx[j * block : (j + 1) * block])[None, :]
        s2 = (x * x + fy * fy) * lam * lam
        keep = (s2 < o.sine**2) & (cp.abs(x) < limit) & (cp.abs(fy) < limit)
        cols *= keep * cp.exp((2j * math.pi / lam * middle) * cp.sqrt(cp.maximum(0, 1 - s2))).astype(cp.complex64)
        del s2, keep
        if not first:
            cols += cp.asarray(sums[j])
        sums[j] = cp.asnumpy(cols)
        del cols
    cp.get_default_memory_pool().free_all_blocks()
    cp.fft.config.get_plan_cache().clear()


def _cut(sums, out, edge):
    """The plate, `edge` pixels in from each side of the planes' spectrum
    `sums`, as its own spectrum into `out`: `sums` transformed back (along
    its columns in place, then a band of rows at a time), the plate cut
    out, and transformed again."""
    import cupy as cp
    import cupyx.scipy.fft as cufft

    blocks, width, block = sums.shape
    n = out.shape[1]
    for j in range(blocks):
        sums[j] = cp.asnumpy(cufft.ifft(cp.asarray(sums[j]), axis=0, overwrite_x=True))
    rows = max(256, BAND // (8 * width) // 256 * 256)
    for r in range(edge, edge + n, rows):
        h = min(rows, edge + n - r)
        band = np.ascontiguousarray(sums[:, r : r + h, :].transpose(1, 0, 2)).reshape(h, width)
        back = cufft.ifft(cp.asarray(band), axis=1, overwrite_x=True)[:, edge : edge + n]
        again = cp.asnumpy(cufft.fft(back, axis=1))
        out[:, r - edge : r - edge + h, :] = again.reshape(h, n // block, block).transpose(1, 0, 2)
        del back
    for j in range(out.shape[0]):
        out[j] = cp.asnumpy(cufft.fft(cp.asarray(out[j]), axis=0, overwrite_x=True))
    cp.get_default_memory_pool().free_all_blocks()


def _scratch(path, what, shape):
    return np.lib.format.open_memmap(path.with_name(f"{what}.tmp.npy"), "w+", np.complex64, shape)


def _remove(a):
    """Deletes a scratch file (its memory map goes when the last use of it
    does)."""
    os.remove(a.filename)
