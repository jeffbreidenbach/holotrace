"""Deep scenes: the plate's light carried plane to plane.

In sum_planes (solid.py) each tile of the plate gathers each slab's light
from a patch as wide as the tile and the margin that light spreads by on
its way to the plate: the deeper the scene, the bigger the margins, and
the more of each patch is worked out only to be thrown away, until the
patches no longer fit in memory at all.

Here the slabs are taken in groups a few millimeters deep instead. Each
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
All the light is within the cone, a circle in the spectrum, so only that
part of each file is ever written or read: the rest stays as holes in
the file, which read as zeros.

The plane's transform is periodic: light leaving one side would come
back in the other. As long as all the scene's light lands on the plate,
nothing does. Groups some of whose light lands beyond are summed on
their own, on planes wider than the plate by enough that what comes back
falls short of it; the plate is cut out of their sum and transformed
afresh, and the other groups are added to that.
"""

import math
import os
import shutil
import time

import numpy as np

from . import solid
from .magnifier import WAVELENGTH

BLOCK = 512
# The biggest band of rows of a plane held in main memory, in bytes.
BAND = 6 * 2**30
# How many groups' planes to make before each pass over the spectrum: as
# many as the disk has room for, if None.
PER = None


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

    # Each group's planes' width: the plate, or more if its light spreads
    # past it. Light past a plane's side comes back in the other, so the
    # planes are wide enough that what comes back falls short of the
    # plate (each point's light reaches as far as it spreads on the way
    # to its group's plane and from there to the plate), and hold every
    # point.
    needs = []
    for middle, slabs in groups:
        most = 0.0
        for plane, mine in slabs:
            p = points[mine]
            depth = -p[:, 2]
            off = np.abs(p[:, :2]).max(axis=1)
            reach = off + o.cone * o.fade * (np.abs(depth - middle) + abs(middle))
            most = max(most, (reach + n * pitch / 2).max(), 2 * off.max())
        needs.append(int(math.ceil(most / pitch)) + 16)
    wide = [g for g, need in enumerate(needs) if need > n]
    narrow = [g for g, need in enumerate(needs) if need <= n]
    width = n
    while wide and (width < max(needs) or solid._fast_len(width) != width):
        width += 2 * block

    out = np.lib.format.open_memmap(path, "w+", np.complex64, (n // block, n, block))
    print(f"  kept every {pitch:g} um: {n}^2 pixels, {sum(len(s) for _, s in groups)} slabs in {len(groups)} "
          f"groups, carried plane to plane{f', {len(wide)} of them on planes {width}^2' if wide else ''}", flush=True)
    start = time.time()

    def add(some, width, sums, first):
        """Adds the groups `some`, on planes `width` wide, to `sums` (or,
        if `first`, puts them there). As many groups' planes at a time as
        the disk has room for: each pass over the spectrum then adds them
        all, and it is read and written less often."""
        blocks = width // block
        low, high = _cone(width, block)[:2]
        each = 8 * width * width * (low + blocks - high) / blocks
        per = PER or max(1, int((0.9 * shutil.disk_usage(path.parent).free - each) / each))
        for i in range(0, len(some), per):
            made = []
            for k, g in enumerate(some[i : i + per]):
                middle, slabs = groups[g]
                t = time.time()
                # A new file each time, so that what is not written is zero.
                plane = _scratch(path, f"plane{k}", (blocks, width, block))
                _plane(points, slabs, middle, light, phases, width, plane)
                made.append((plane, middle))
                print(f"  group {g + 1} of {len(groups)}, {len(slabs)} slabs, its plane {middle / 1000:+.2f} mm: "
                      f"made in {time.time() - t:.0f} s; {time.time() - start:.0f} s", flush=True)
            t = time.time()
            _add_carried(made, sums, first=first and i == 0)
            for plane, _ in made:
                _remove(plane)
            del made, plane
            print(f"  carried to the plate in {time.time() - t:.0f} s; {time.time() - start:.0f} s", flush=True)

    # The wide groups first, summed on their own and the plate cut out of
    # the sum; then the rest, at the plate's own width, added to that.
    if wide:
        sums = _scratch(path, "sum", (width // block, width, block))
        add(wide, width, sums, True)
        t = time.time()
        _cut(sums, out, (width - n) // 2, path)
        _remove(sums)
        del sums
        print(f"  the plate cut out in {time.time() - t:.0f} s; {time.time() - start:.0f} s", flush=True)
    add(narrow, n, out, not wide)
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
    """A group's light (its `slabs` of `points`) on its middle plane,
    `middle` behind the plate, `width` pixels square, into `plane` (in
    blocks of columns; new, so all zeros), already transformed along its
    rows: worked out a band of rows at a time, each tile of a band
    gathering each slab's light and carrying it to the middle (_tile),
    then the band transformed on the GPU. Bands with no light are not
    written at all, nor the columns outside the cone (see _cone)."""
    import cupy as cp
    import cupyx.scipy.fft as cufft

    o = solid.OPTICS
    pitch = o.kept_pitch
    parts = []
    for at, mine in slabs:
        seen_from = points[mine] + [0.0, 0.0, at]
        margin = int(math.ceil(o.cone * abs(at - middle) / pitch)) + 8
        px, py, pz = seen_from.T
        r = o.cone * o.fade * np.abs(pz)
        bounds = (np.floor((px - r) / pitch + width / 2), np.ceil((px + r) / pitch + width / 2) + 1,
                  np.floor((py - r) / pitch + width / 2), np.ceil((py + r) / pitch + width / 2) + 1)
        parts.append((at, mine, seen_from, margin, bounds))
    deepest = max(m for _, _, _, m, _ in parts)
    tile = min(width, (solid.GPU_PATCH - 2 * deepest) // 256 * 256, BAND // (8 * width) // 256 * 256)
    blocks, block = plane.shape[0], plane.shape[2]
    low, high = _cone(width, block)[:2]
    band = np.empty((tile, width), np.complex64)
    for y0 in range(0, width, tile):
        th = min(tile, width - y0)
        band[:th] = 0
        lit = False
        for x0 in range(0, width, tile):
            tw = min(tile, width - x0)
            out = _tile(parts, middle, light, phases, width, (x0, y0, tw, th), deepest)
            if out is not None:
                band[:th, x0 : x0 + tw] = cp.asnumpy(out)
                lit = True
            del out
        if not lit:
            continue
        for r in range(0, th, 256):
            rows = cufft.fft(cp.asarray(band[r : min(r + 256, th)]), axis=1, overwrite_x=True)
            band[r : r + len(rows)] = cp.asnumpy(rows)
            del rows
        blocked = band[:th].reshape(th, blocks, block).transpose(1, 0, 2)
        plane[:low, y0 : y0 + th, :] = blocked[:low]
        plane[high:, y0 : y0 + th, :] = blocked[high:]
    cp.get_default_memory_pool().free_all_blocks()
    cp.fft.config.get_plan_cache().clear()


def _tile(parts, middle, light, phases, width, box, deepest):
    """The light of a group's slabs (`parts`, as _plane lists them) on the
    tile `box` (x, y, width, height) of its middle plane, on the GPU, or
    None if none reaches it. Each slab's light is summed on its recording
    plane over the tile and the slab's own margin, into a patch with
    the deepest slab's margin all round, and transformed; its plane waves
    are moved on to the middle (solid._step) and added to the other
    slabs'; and the sum is transformed back, once for all of them, and
    the tile kept (overlap-save, as in solid._carry)."""
    import cupy as cp
    import cupyx.scipy.fft as cufft

    from .gpu import SeenPoints

    o = solid.OPTICS
    pitch = o.kept_pitch
    x0, y0, tw, th = box
    h, w = solid._fast_len(th + 2 * deepest), solid._fast_len(tw + 2 * deepest)
    total = None
    for at, mine, seen_from, m, (lo_x, hi_x, lo_y, hi_y) in parts:
        ox, oy, sw, sh = x0 - m, y0 - m, tw + 2 * m, th + 2 * m
        near = (hi_x > ox) & (lo_x < ox + sw) & (hi_y > oy) & (lo_y < oy + sh)
        if not near.any():
            continue
        which = mine[near]
        seen = SeenPoints(size=width, pitch=pitch, wavelength=WAVELENGTH, points=seen_from[near],
                          maps=cp.asarray(light[which], cp.float32), cone=o.cone, tile=(ox, oy, sw, sh),
                          edge=o.cone * o.fade)
        patch = cp.zeros((h, w), cp.complex64)
        seen.light(phases[which], into=(patch, x0 - deepest, y0 - deepest))
        del seen
        cp.get_default_memory_pool().free_all_blocks()
        patch = cufft.fft2(patch, overwrite_x=True)
        # (The transform's working memory is as big as the patch.)
        cp.fft.config.get_plan_cache().clear()
        solid._step(patch, at - middle, pitch)
        if total is None:
            total = patch
        else:
            total += patch
        del patch
        cp.get_default_memory_pool().free_all_blocks()
    if total is None:
        return None
    cp.get_default_memory_pool().free_all_blocks()
    total = cufft.ifft2(total, overwrite_x=True)
    out = total[deepest : deepest + th, deepest : deepest + tw].copy()
    del total
    cp.get_default_memory_pool().free_all_blocks()
    cp.fft.config.get_plan_cache().clear()
    return out


def _cone(width, block):
    """Where a spectrum `width` square, in blocks of `block` columns, can
    be other than zero: all the light is within the cone, a circle in the
    spectrum, so the blocks from `low` up to `high` are empty, and in
    block j only its first rows[j] rows and its last rows[j] hold
    anything (the spectrum's rows and columns run from 0 up, then from
    the most negative back up to 0). Returns low, high, rows."""
    o = solid.OPTICS
    edge = o.sine / WAVELENGTH
    f = np.abs(np.fft.fftfreq(width, o.kept_pitch)).reshape(-1, block).min(axis=1)
    inside = f < edge
    low = int(np.argmin(inside)) if not inside.all() else len(f)
    high = len(f) - int(np.argmin(inside[::-1])) if not inside.all() else len(f)
    reach = np.sqrt(np.maximum(0, edge**2 - f**2)) * width * o.kept_pitch
    rows = np.where(inside, np.minimum(width // 2, np.ceil(reach).astype(int) + 2), 0)
    return low, high, rows


def _add_carried(planes, sums, first):
    """Transforms each of `planes` (each a plane, already transformed
    along its rows, and its distance behind the plate) along its columns,
    a block at a time, carries it to the plate (its plane waves within
    the cone and the band limit, each moved on by its own phase, as in
    solid._carry), and adds them all to `sums` (or, if `first`, puts them
    there: `sums` is then new, all zeros). Only the blocks and rows
    within the cone are read or written (see _cone): the rest of the
    files stays unwritten, which reads as zeros and takes no room."""
    import cupy as cp
    import cupyx.scipy.fft as cufft

    o, lam = solid.OPTICS, WAVELENGTH
    blocks, width, block = sums.shape
    pitch = o.kept_pitch
    fy = cp.fft.fftfreq(width, pitch).astype(cp.float32)[:, None]
    fx = np.fft.fftfreq(width, pitch).astype(np.float32)
    rows = _cone(width, block)[2]
    for j in range(blocks):
        if rows[j] == 0:
            continue
        x = cp.asarray(fx[j * block : (j + 1) * block])[None, :]
        r = int(rows[j])
        parts = [slice(None)] if 2 * r >= width else [slice(0, r), slice(width - r, width)]
        total = [None] * len(parts)
        for plane, middle in planes:
            limit = 1 / (lam * math.sqrt((2 * middle / (width * pitch)) ** 2 + 1))
            cols = cufft.fft(cp.asarray(plane[j]), axis=0, overwrite_x=True)
            for i, part in enumerate(parts):
                there = solid.moved(cols[part], x, fy[part], middle, limit, limit)
                total[i] = there if total[i] is None else total[i] + there
            del cols, there
        for i, part in enumerate(parts):
            if not first:
                total[i] += cp.asarray(sums[j, part])
            sums[j, part] = cp.asnumpy(total[i])
        del total
    cp.get_default_memory_pool().free_all_blocks()
    cp.fft.config.get_plan_cache().clear()


def _cut(sums, out, edge, path):
    """The plate, `edge` pixels in from each side of the planes' spectrum
    `sums`, as its own spectrum into `out` (new, all zeros): `sums`
    transformed back along its columns in place, and then along its rows
    (_cut_rows). `path` is where scratch files go (see _scratch)."""
    import cupy as cp
    import cupyx.scipy.fft as cufft

    blocks, width, block = sums.shape
    for j in np.flatnonzero(_cone(width, block)[2]):
        sums[j] = cp.asnumpy(cufft.ifft(cp.asarray(sums[j]), axis=0, overwrite_x=True))
    cp.get_default_memory_pool().free_all_blocks()
    _cut_rows(sums, out, edge, path)


def _cut_rows(sums, out, edge, path):
    """The rest of _cut: `sums`, already transformed back along its
    columns, transformed back along its rows, a band at a time (on the
    GPU 256 rows at a time), the plate cut out, and transformed again:
    along its rows into a scratch file, then along its columns into
    `out`. Of the plate's spectrum only what is within the cone is kept
    (see _cone), as when the planes are no wider than the plate."""
    import cupy as cp
    import cupyx.scipy.fft as cufft

    blocks, width, block = sums.shape
    n = out.shape[1]
    low, high, reach = _cone(n, block)
    half = _scratch(path, "cut", out.shape)
    rows = max(256, BAND // (8 * width) // 256 * 256)
    for r in range(edge, edge + n, rows):
        h = min(rows, edge + n - r)
        band = np.ascontiguousarray(sums[:, r : r + h, :].transpose(1, 0, 2)).reshape(h, width)
        again = np.empty((h, n), np.complex64)
        for i in range(0, h, 256):
            back = cufft.ifft(cp.asarray(band[i : i + 256]), axis=1, overwrite_x=True)[:, edge : edge + n]
            again[i : i + 256] = cp.asnumpy(cufft.fft(back, axis=1))
            del back
        blocked = again.reshape(h, n // block, block).transpose(1, 0, 2)
        half[:low, r - edge : r - edge + h, :] = blocked[:low]
        half[high:, r - edge : r - edge + h, :] = blocked[high:]
        del band, again, blocked
    for j in np.flatnonzero(reach):
        cols = cufft.fft(cp.asarray(half[j]), axis=0, overwrite_x=True)
        r = int(reach[j])
        for part in [slice(None)] if 2 * r >= n else [slice(0, r), slice(n - r, n)]:
            out[j, part] = cp.asnumpy(cols[part])
        del cols
    _remove(half)
    cp.get_default_memory_pool().free_all_blocks()


def _scratch(path, what, shape):
    return np.lib.format.open_memmap(path.with_name(f"{what}.tmp.npy"), "w+", np.complex64, shape)


def _remove(a):
    """Deletes a scratch file (its memory map goes when the last use of it
    does)."""
    os.remove(a.filename)
