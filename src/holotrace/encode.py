"""The plate's pattern, as a writer would be given it.

    holotrace encode NAME [--pixel Q] [--levels | --gds] [--bold B]
    holotrace encode NAME [--pixel Q] --check [--spot W] [--bold B]

A scene's run keeps the plate's light as the eye can see it: sampled
only every few pixels, and without the reference beam (solid.render). A
writer needs the pattern itself, on its own pixels. This reads
out/NAME/spectrum.npy and writes out/NAME/pattern-<Q>um.pbm: the
plate's pixels, Q um apart (the plate's own, as it was made, or with
--pixel any whole fraction of what was kept), each clear or opaque: a
plate of chrome on glass.

- The spectrum is transformed back to the plate's light, into a scratch
  file as big as the spectrum whole (the plate's light has no holes): a
  block of columns and then a band of rows at a time.
- A tile at a time on the GPU, the light is made as fine as the pixels.
  It has no detail finer than what was kept, so its transform padded
  with zeros gives the samples between; only near a tile's edges is
  that off, so each tile is made with a margin, which is thrown away.
- Each pixel is set against the reference beam there: clear where
  cos(theta) > sqrt(1 - a^2), theta the light's phase against the
  beam's and a its strength thereabouts (1 where the plate is
  brightest). So the fringes are half clear where the light is
  brightest, thinner where it is dimmer, and gone in the dark; lit by
  the beam, the plate sends on a / pi of it as the scene's light.

The strength matters: the scene stands at the plate, so how bright each
part of it looks is how strong the light is there. With every fringe
half clear the boat comes out as a flat shape in a haze. It is the
light's strength thereabouts (its mean power over CELL x CELL of the
kept samples, smoothed), not each pixel's own as in speckle.py: that
follows the light better, speckle and all, but its lines are too thin
to write wherever the speckle is dim.

With --levels, out/NAME/pattern-<Q>um.pgm instead, for a writer that
shapes a relief in clear resist: a byte a pixel, how long the pixel
should hold the light back, in 256ths of a wave, 128 for not at all (as
that much more resist, or less, would). That is the light's phase
against the beam's, a sawtooth across each fringe, times m, less than 1
where the light (each pixel's own) is weak: a sawtooth m of a wave deep
sends on sinc(1 - m) of the light.

Both files are pictures of the plate as the viewer sees it, the top row
first: a PBM's opaque pixels are its black ones. Lit from behind by the
reference beam (climbing, at the angle spectrum.json gives as a sine),
the plate sends the scene's light on toward the viewer.

With --gds, out/NAME/pattern-<Q>um.gds instead of the PBM: the same
clear pixels as rectangles (each run of them along a row, runs alike
in rows one above another joined), in a GDSII file, which mask writers'
software takes: one structure, layer 1, the plate's middle at the
origin, x to the viewer's right and y up, to the nm. It is 64 bytes a
rectangle, so far bigger than the PBM.

With --bold B (-bold in the files' names), the fringes are all at their
strongest where the light's strength thereabouts is B or more, the
chrome's half clear and a relief's a whole wave deep, and there are
none where it is less: no thin lines and no shallow fringes, for a
writer to be tried on, but the scene comes out flat.

--check replays both patterns instead, lit by the reference beam, over
up to a hundred tiles spread across the plate: how much of the beam's
light comes out as the scene's, and how much other light comes out
within the cone. Each pixel is a square, on a grid no coarser than
0.2 um; and the same again as a spot W um wide at half its height would
write them (--spot, default 0.8): the chrome clear where it gets half
the full dose or more, the relief as deep as its dose.
"""

import json
import math
import struct
import sys
import time

import numpy as np

from . import planes
from .magnifier import WAVELENGTH
from .solid import results
from .viewer import _names, _option

# A tile's light, made as fine as the pixels, is at most FINE samples
# across (what the GPU holds beside its pattern); MARGIN of the kept
# samples either side of a tile are worked out and thrown away.
FINE = 12288
MARGIN = 64
# The light's strength thereabouts is its mean power over CELL x CELL
# kept samples (a few dozen of speckle's grains).
CELL = 8


def plate_light(path):
    """The plate's light, from the spectrum at `path`: transformed back
    into a scratch file beside it, in blocks of columns as planes.py's
    files are (a small plate's, kept whole, as one block). With it, its
    strength thereabouts (a value to each CELL x CELL of its samples:
    the root of their mean power, smoothed over the cells next to it, 1
    where it is brightest but for the top half percent), and how strong
    its own brightest samples are, but for the same."""
    import cupy as cp
    import cupyx.scipy.fft as cufft
    import scipy.fft

    spectrum = np.load(path, mmap_mode="r")
    n = spectrum.shape[1]
    cells = -(-n // CELL)
    power, sizes = np.zeros((cells, cells), np.float32), []

    def note(rows, r):
        """Takes note of the light's rows from r on (on the GPU): their
        power, cell by cell, and every sixteenth sample's size."""
        p = cp.pad(cp.abs(rows) ** 2, ((0, -len(rows) % CELL), (0, -n % CELL)))
        p = p.reshape(len(p) // CELL, CELL, cells, CELL).mean(axis=(1, 3))
        power[r // CELL : r // CELL + len(p)] = cp.asnumpy(p)
        sizes.append(cp.asnumpy(cp.abs(rows[::16, ::16])).ravel())

    if spectrum.ndim == 2:
        light = planes._scratch(path, "light", (1,) + spectrum.shape)
        light[0] = scipy.fft.ifft2(spectrum, workers=-1)
        for r in range(0, n, 1024):
            note(cp.asarray(light[0, r : r + 1024]), r)
    else:
        blocks, _, block = spectrum.shape
        light = planes._scratch(path, "light", spectrum.shape)
        for j in range(blocks):
            cols = cp.asarray(spectrum[j])
            # (Outside the cone the spectrum is empty.)
            if cols.any():
                light[j] = cp.asnumpy(cufft.ifft(cols, axis=0, overwrite_x=True))
        rows = max(256, planes.BAND // (8 * n) // 256 * 256)
        for r in range(0, n, rows):
            band = np.ascontiguousarray(light[:, r : r + rows, :].transpose(1, 0, 2)).reshape(-1, n)
            for i in range(0, len(band), 256):
                back = cufft.ifft(cp.asarray(band[i : i + 256]), axis=1, overwrite_x=True)
                note(back, r + i)
                band[i : i + 256] = cp.asnumpy(back)
            light[:, r : r + rows, :] = band.reshape(-1, blocks, block).transpose(1, 0, 2)
    cp.get_default_memory_pool().free_all_blocks()
    edged = np.pad(power, 1, mode="edge")
    near = (0.25, 0.5, 0.25)
    power = sum(near[i] * near[j] * edged[i : i + cells, j : j + cells] for i in range(3) for j in range(3))
    strength = np.sqrt(power)
    strength = np.minimum(strength / np.percentile(strength[strength > 0], 99.5), 1).astype(np.float32)
    sizes = np.concatenate(sizes)
    return light, strength, float(np.percentile(sizes[sizes > 0], 99.5))


def _patch(light, y0, x0, size):
    """The plate's light from row y0 and column x0 on, `size` square;
    nothing beyond the plate."""
    blocks, n, block = light.shape
    out = np.zeros((size, size), np.complex64)
    ya, yb = max(y0, 0), min(y0 + size, n)
    xa, xb = max(x0, 0), min(x0 + size, n)
    b0, b1 = xa // block, (xb - 1) // block + 1
    part = light[b0:b1, ya:yb, :].transpose(1, 0, 2).reshape(yb - ya, -1)
    out[ya - y0 : yb - y0, xa - x0 : xb - x0] = part[:, xa - b0 * block : xb - b0 * block]
    return out


def _finer(a, up):
    """`a` (on the GPU) with `up` times as many samples each way: its
    transform, padded with zeros, transformed back."""
    import cupy as cp
    import cupyx.scipy.fft as cufft

    p, h = a.shape[0], a.shape[0] // 2
    spectrum = cufft.fft2(a, overwrite_x=True)
    big = cp.zeros((p * up, p * up), cp.complex64)
    big[:h, :h] = spectrum[:h, :h]
    big[:h, -h:] = spectrum[:h, -h:]
    big[-h:, :h] = spectrum[-h:, :h]
    big[-h:, -h:] = spectrum[-h:, -h:]
    del spectrum
    big = cufft.ifft2(big, overwrite_x=True)
    big *= up * up
    return big


def _reference(y, sine):
    """The reference beam at heights `y` (um), as a column on the GPU: its
    phase worked out in double precision (it is thousands of turns)."""
    import cupy as cp

    turns = (sine / WAVELENGTH * y) % 1.0
    return cp.asarray(np.exp(2j * math.pi * turns).astype(np.complex64))[:, None]


class Patterns:
    """A plate's two patterns, a patch at a time: of its `light` (kept
    every `pitch` um, with its strength thereabouts and its brightest:
    see plate_light) against the reference beam (its sine), `up` pixels
    to each kept sample."""

    def __init__(self, light, strength, brightest, pitch, sine, up, bold=0.0):
        import cupy as cp

        self.light, self.pitch, self.sine, self.up, self.bold = light, pitch, sine, up, bold
        self.n = light.shape[1]
        # A tile of the plate, in kept samples, and it with its margins.
        self.size = 2 ** int(math.log2(FINE / up))
        self.tile = self.size - 2 * MARGIN
        self.cells = len(strength)
        self.strength = cp.asarray(strength.ravel())
        self.dimmest = cp.float32(1 / brightest)
        # How deep a sawtooth, as a share of a whole wave, sends on each
        # strength from 0 to 1: one m deep sends on sinc(1 - m).
        m = np.linspace(0, 1, 4097)
        self.depth = cp.asarray(np.interp(np.linspace(0, 1, 1025), np.sinc(1 - m), m).astype(np.float32))
        # The light's strength thereabouts, a, at a pixel of a patch
        # `wide` across: between the four cells about it.
        where = "raw float32 strength, int32 cells, int32 wide, int32 x0, int32 y0, float32 per, float32 bold"
        thereabouts = """
            float u = (x0 + (int)(i % wide) + 0.5f) * per - 0.5f, v = (y0 + (int)(i / wide) + 0.5f) * per - 0.5f;
            int u0 = (int)floorf(u), v0 = (int)floorf(v);
            float fu = u - u0, fv = v - v0;
            int ua = min(max(u0, 0), cells - 1), ub = min(max(u0 + 1, 0), cells - 1);
            int va = min(max(v0, 0), cells - 1), vb = min(max(v0 + 1, 0), cells - 1);
            float a = (1 - fv) * ((1 - fu) * strength[va * cells + ua] + fu * strength[va * cells + ub])
                + fv * ((1 - fu) * strength[vb * cells + ua] + fu * strength[vb * cells + ub]);
        """
        self.chrome = cp.ElementwiseKernel(
            "complex64 light, complex64 beam, " + where,
            "bool opaque",
            thereabouts + """
            if (bold > 0.0f) a = a >= bold ? 1.0f : 0.0f;
            complex<float> w = light * conj(beam);
            opaque = !(w.real() > sqrtf((w.real() * w.real() + w.imag() * w.imag()) * (1.0f - a * a)));
            """,
            "holotrace_chrome",
        )
        self.relief = cp.ElementwiseKernel(
            "complex64 light, complex64 beam, float32 dimmest, raw float32 depth, " + where,
            "uint8 level",
            """
            complex<float> w = light * conj(beam);
            float m;
            if (bold > 0.0f) {
            """ + thereabouts + """
                m = a >= bold ? 1.0f : 0.0f;
            } else {
                float t = fminf(abs(w) * dimmest, 1.0f) * 1024.0f;
                int k = min((int)t, 1023);
                m = depth[k] + (t - k) * (depth[k + 1] - depth[k]);
            }
            int l = (int)((atan2f(w.imag(), w.real()) * 0.15915494f * m + 0.5f) * 256.0f);
            level = l > 255 ? 255 : (l < 0 ? 0 : l);
            """,
            "holotrace_relief",
        )

    def patch(self, y0, x0, size, levels):
        """The pattern of the patch from kept row y0 and column x0 on,
        `size` kept samples square: which pixels are opaque, or with
        `levels` each pixel's level; its rows from the bottom up."""
        import cupy as cp

        up = self.up
        made = _finer(cp.asarray(_patch(self.light, y0, x0, size)), up)
        y = (y0 * up + np.arange(size * up) - self.n * up / 2 + 0.5) * self.pitch / up
        beam = _reference(y, self.sine)
        where = (self.strength, self.cells, size * up, x0 * up, y0 * up, cp.float32(1 / (CELL * up)), cp.float32(self.bold))
        if levels:
            return self.relief(made, beam, self.dimmest, self.depth, *where)
        return self.chrome(made, beam, *where)

    def core(self, y0, x0, levels):
        """The pattern of the tile whose first kept row and column are
        y0 and x0 (see patch), made with its margins and without them."""
        up, edge = self.up, MARGIN * self.up
        h, w = min(self.tile, self.n - y0) * up, min(self.tile, self.n - x0) * up
        return self.patch(y0 - MARGIN, x0 - MARGIN, self.size, levels)[edge : edge + h, edge : edge + w]


def encode(patterns, levels, path):
    """Writes a plate's pattern to the file at `path`: clear or opaque,
    or with `levels` 256 levels. Returns how many pixels it is across."""
    import cupy as cp

    n, up, tile = patterns.n, patterns.up, patterns.tile
    across = n * up
    with open(path, "wb") as f:
        f.write(f"P5\n{across} {across}\n255\n".encode() if levels else f"P4\n{across} {across}\n".encode())
        # From the top row of tiles down, each tile's rows turned top
        # first on the GPU.
        for y0 in reversed(range(0, n, tile)):
            h = min(tile, n - y0) * up
            band = np.empty((h, across if levels else -(-across // 8)), np.uint8)
            for x0 in range(0, n, tile):
                w = min(tile, n - x0) * up
                made = patterns.core(y0, x0, levels)[::-1]
                if levels:
                    band[:, x0 * up : x0 * up + w] = cp.asnumpy(made)
                else:
                    # Each tile's rows a whole number of bytes: all but
                    # the last are, and the last row's spare bits are
                    # the file's own.
                    bits = cp.packbits(cp.pad(made, ((0, 0), (0, -w % 8))).ravel())
                    band[:, x0 * up // 8 : (x0 * up + w + 7) // 8] = cp.asnumpy(bits.reshape(h, -1))
                del made
            f.write(band)
    cp.get_default_memory_pool().free_all_blocks()
    return across


def runs(clear):
    """The clear pixels of `clear` (rows by columns, in main memory or
    on the GPU) as rectangles: each run of them along a row, runs alike
    in rows one above another joined. Their first columns, last columns
    and one, first rows, last rows and one."""
    import cupy as cp

    xp = cp.get_array_module(clear)
    edges = xp.diff(xp.pad(clear, ((0, 0), (1, 1))).astype(xp.int8), axis=1)
    r, a = xp.nonzero(edges == 1)
    b = xp.nonzero(edges == -1)[1]
    order = xp.lexsort(xp.stack([r, b, a]))
    r, a, b = r[order], a[order], b[order]
    first = xp.ones(len(r), bool)
    first[1:] = (a[1:] != a[:-1]) | (b[1:] != b[:-1]) | (r[1:] != r[:-1] + 1)
    first = xp.flatnonzero(first)
    last = xp.append(first[1:], len(r))[: len(first)] - 1
    return a[first], b[first], r[first], r[last] + 1


def _record(kind, data=b""):
    return struct.pack(">HH", 4 + len(data), kind) + data


def _real(value):
    """A GDSII file's kind of real number: a sign, a power of 16 and 56
    bits."""
    power = 64
    while value >= 1:
        value, power = value / 16, power + 1
    while value < 1 / 16:
        value, power = value * 16, power - 1
    return bytes([power]) + int(value * 2**56).to_bytes(7, "big")


RECTANGLE = np.dtype([("boundary", ">u2", 2), ("layer", ">u2", 3), ("datatype", ">u2", 3), ("xy", ">u2", 2),
                      ("corners", ">i4", 10), ("end", ">u2", 2)])


class Gds:
    """A GDSII file of rectangles, in nm: one structure, layer 1."""

    def __init__(self, path):
        self.file, self.count = open(path, "wb"), 0
        now = list(time.localtime()[:6]) * 2
        self.file.write(
            _record(0x0002, struct.pack(">h", 600)) + _record(0x0102, struct.pack(">12h", *now))
            + _record(0x0206, b"HOLOTRACE\0") + _record(0x0305, _real(1e-3) + _real(1e-9))
            + _record(0x0502, struct.pack(">12h", *now)) + _record(0x0606, b"PLATE\0")
        )

    def add(self, x0, x1, y0, y1):
        """Rectangles from (x0, y0) to (x1, y1), in nm."""
        made = np.zeros(len(x0), RECTANGLE)
        made["boundary"], made["layer"], made["datatype"] = (4, 0x0800), (6, 0x0D02, 1), (6, 0x0E02, 0)
        made["xy"], made["end"] = (44, 0x1003), (4, 0x1100)
        made["corners"] = np.stack([x0, y0, x1, y0, x1, y1, x0, y1, x0, y0], axis=1)
        self.file.write(made.tobytes())
        self.count += len(made)

    def close(self):
        self.file.write(_record(0x0700) + _record(0x0400))
        self.file.close()


def encode_gds(patterns, path):
    """Writes a plate's clear pixels to the GDSII file at `path`, as
    rectangles (see runs). Returns how many."""
    import cupy as cp

    n, up, tile = patterns.n, patterns.up, patterns.tile
    nm = round(1000 * patterns.pitch / up)
    gds = Gds(path)
    for y0 in range(0, n, tile):
        for x0 in range(0, n, tile):
            a, b, r0, r1 = (cp.asnumpy(v).astype(np.int64) for v in runs(~patterns.core(y0, x0, False)))
            x, y = (x0 * up) * nm - n * up * nm // 2, (y0 * up) * nm - n * up * nm // 2
            gds.add(x + a * nm, x + b * nm, y + r0 * nm, y + r1 * nm)
    gds.close()
    cp.get_default_memory_pool().free_all_blocks()
    return gds.count


def check(patterns, spot):
    """Replays both patterns (see the top), and says what comes of it.
    Over tiles spread across the plate, each with its margin: the
    pattern's pixels as squares on a finer grid; lit by the reference
    beam; and of what comes out, the light within the cone, at the kept
    samples, set against the scene's there. The scene's share is as much
    of it as is the scene's light times one number, the same all over
    the plate; the rest is other light."""
    import cupy as cp
    import cupyx.scipy.fft as cufft

    n, up, pitch = patterns.n, patterns.up, patterns.pitch
    sub = math.ceil(pitch / up / 0.2 - 1e-9)
    size = 2 ** int(math.log2(FINE / (up * sub)))
    tile, fine = size - 2 * MARGIN, size * up * sub
    # The spot's blur, one way: a Gaussian, so the same along rows and columns.
    blur = cp.exp(-2 * math.pi**2 * (spot / 2.355) ** 2 * cp.fft.fftfreq(fine, pitch / (up * sub)).astype(cp.float32) ** 2)
    k = cp.fft.fftfreq(size, pitch).astype(cp.float32)
    cone = (k[None, :] ** 2 + k[:, None] ** 2) < (0.5 / pitch) ** 2
    free = cp.get_default_memory_pool().free_all_blocks

    def written(pattern):
        back = cufft.fft2(pattern.astype(cp.complex64), overwrite_x=True)
        back *= blur[None, :]
        back *= blur[:, None]
        return cufft.ifft2(back, overwrite_x=True).real

    def replay(through, beam):
        """What the patch sends on within the cone, at its kept samples."""
        out = cufft.fft2((through * beam).astype(cp.complex64, copy=False), overwrite_x=True)
        del through
        h = size // 2
        low = cp.empty((size, size), cp.complex64)
        low[:h, :h], low[:h, -h:], low[-h:, :h], low[-h:, -h:] = out[:h, :h], out[:h, -h:], out[-h:, :h], out[-h:, -h:]
        del out
        low = cufft.ifft2(low * cone) / (up * sub) ** 2
        return low[MARGIN : MARGIN + tile, MARGIN : MARGIN + tile]

    per_side = -(-n // tile)
    step = -(-per_side // 10)
    some = range(step // 2, per_side, step)
    kinds = [kind + ", bold" * bool(patterns.bold) + with_spot for kind in ("chrome", "relief")
             for with_spot in ("", f", written with a {spot:g} um spot")]
    out2, both, scene2, lit = np.zeros(4), np.zeros(4, complex), 0.0, 0
    for ty in some:
        for tx in some:
            y0, x0 = ty * tile - MARGIN, tx * tile - MARGIN
            scene = cp.asarray(_patch(patterns.light, y0, x0, size))[MARGIN : MARGIN + tile, MARGIN : MARGIN + tile]
            scene2 += float(cp.sum(cp.abs(scene) ** 2))
            # (The beam's light counts only where there is plate.)
            lit += min(tile, n - ty * tile) * min(tile, n - tx * tile)
            y = (y0 * up * sub + np.arange(fine) - n * up * sub / 2 + 0.5) * pitch / (up * sub)
            beam = _reference(y, patterns.sine)
            square = lambda p: cp.repeat(cp.repeat(p, sub, axis=0), sub, axis=1) if sub > 1 else p

            def note(i, through):
                out = replay(through, beam)
                out2[i] += float(cp.sum(cp.abs(out) ** 2))
                both[i] += complex(cp.vdot(scene, out))
                del out, through
                free()

            clear = square(~patterns.patch(y0, x0, size, False)).astype(cp.float32)
            note(0, clear)
            note(1, (written(clear) > 0.5).astype(cp.float32))
            del clear
            free()
            turns = square(patterns.patch(y0, x0, size, True)).astype(cp.float32) / 256
            note(2, cp.exp(2j * math.pi * turns).astype(cp.complex64))
            note(3, cp.exp(2j * math.pi * written(turns)).astype(cp.complex64))
            del turns, scene, beam
            free()
    print(f"{len(some) ** 2} tiles of {tile * pitch:.0f} um spread across the plate, pixels {pitch / up:g} um apart, "
          f"lit by the reference beam:", flush=True)
    for i, kind in enumerate(kinds):
        share = abs(both[i]) ** 2 / scene2
        print(f"  {kind}: {100 * share / lit:.2f}% of the beam becomes the scene's light; "
              f"other light within the cone, {100 * (out2[i] - share) / share:.0f}% of that", flush=True)


def main():
    names = _names()
    out = results() / (names[0] if names else "benchy")
    path = out / "spectrum.npy"
    info = json.loads(path.with_suffix(".json").read_text())
    if "reference" not in info:
        sys.exit(f"{path.with_suffix('.json')} does not say what reference beam the plate was made for")
    pitch, sine = info["pitch"], info["reference"]
    cone = info["cone"] / math.sqrt(1 + info["cone"] ** 2)
    pixel = _option("--pixel", info.get("pixel", pitch))
    up = round(pitch / pixel)
    if up < 1 or abs(up * pixel - pitch) > 1e-6 * pitch:
        sys.exit(f"the plate's light is kept every {pitch:g} um: --pixel must be a whole fraction of that")
    if sine + cone > WAVELENGTH / (2 * pixel):
        sys.exit(f"pixels {pixel:g} um apart cannot carry fringes {WAVELENGTH / (sine + cone):.2f} um apart")
    start = time.time()
    light, strength, brightest = plate_light(path)
    try:
        print(f"the plate's light, from its spectrum: {time.time() - start:.1f} s", flush=True)
        bold = _option("--bold", 0.0)
        patterns = Patterns(light, strength, brightest, pitch, sine, up, bold)
        start = time.time()
        if "--check" in sys.argv:
            check(patterns, _option("--spot", 0.8))
            print(f"replayed: {time.time() - start:.1f} s")
            return
        levels = "--levels" in sys.argv
        to = out / f"pattern-{pixel:g}um{'-bold' if bold else ''}.{'pgm' if levels else 'pbm'}"
        if "--gds" in sys.argv:
            if abs(round(1000 * pixel) - 1000 * pixel) > 1e-6:
                sys.exit("a GDSII file is to the nm: the pixels must be a whole number of them")
            to = to.with_suffix(".gds")
            count = encode_gds(patterns, to)
            print(f"{count} rectangles: {time.time() - start:.1f} s; wrote {to}, {to.stat().st_size / 2**30:.2f} GB")
            return
        across = encode(patterns, levels, to)
        print(f"{across}^2 pixels {pixel:g} um apart, {'256 levels' if levels else 'clear or opaque'}: "
              f"{time.time() - start:.1f} s; wrote {to}, {to.stat().st_size / 2**30:.2f} GB")
    finally:
        planes._remove(light)
