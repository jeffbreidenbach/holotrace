"""The wave sum on an NVIDIA GPU, with CuPy.

GpuPlate does what Plate does, adding up each point's waves across the
plate, but one GPU thread per pixel. Rays are still found on the CPU
(or by Mitsuba); only the sum, which is nearly all the work, moves.

GPUs are fast in single precision, which holds about 7 digits: too few
for a phase of tens of thousands of radians. So each kernel works only
with the small part of a point's path length that varies across the
plate (tens of micrometres at most), and the rest, the same for every
pixel, is folded into its starting phase, on the CPU in double
precision.

Through the lens, path lengths are known on a coarse grid. Between grid
points they are blended either as Plate does (bilinear), or, much more
closely, by cubic Hermite blending: a ray's arrival direction is the
slope of the path length across the plate (its sines), so the grid knows
the slopes too, and a cubic through the values and slopes is good to
the fourth order. That lets the grid be far coarser: fewer rays.
"""

import math

import cupy as cp
import numpy as np

from .plate import Plate

_SOURCE = r"""
#include <cupy/complex.cuh>

__device__ float coordinate(int index, int size, float pitch) {
    return (index - size / 2 + 0.5f) * pitch;
}

// The light straight from a point `depth` behind the plate, over the
// window of nx x ny pixels from (x0, y0). `phase` includes k times the
// depth; the rest of the distance, small, is worked out here.
extern "C" __global__ void direct(
    complex<float>* light, int size, float pitch, float k,
    int x0, int y0, int nx, int ny,
    float px, float py, float depth, float phase, float cone2,
    int lens, float lens_f, float lens_r2)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j >= ny) return;
    int col = x0 + i, row = y0 + j;
    float dx = coordinate(col, size, pitch) - px;
    float dy = coordinate(row, size, pitch) - py;
    float t2 = (dx * dx + dy * dy) / (depth * depth);
    if (t2 >= cone2) return;
    if (lens) {
        // Blocked where the line to the pixel crosses the lens.
        float cx = px + lens_f * dx, cy = py + lens_f * dy;
        if (cx * cx + cy * cy <= lens_r2) return;
    }
    float extra = depth * t2 / (sqrtf(1.0f + t2) + 1.0f);
    float strength = powf(1.0f + t2, -0.75f) / depth;
    float s, c;
    sincosf(k * extra + phase, &s, &c);
    light[row * size + col] += complex<float>(strength * c, strength * s);
}

// The light straight from a point `depth` behind the plate (in front, if
// negative: its light converges through the plate onto it, so its path
// to the plate is negative), as bright in each direction as `map` says:
// its amplitude over an n x n grid of tangents from -cone to cone. The
// window and phase are as for `direct`.
extern "C" __global__ void seen(
    complex<float>* light, int size, float pitch, float k,
    int x0, int y0, int nx, int ny,
    float px, float py, float depth, float phase, float cone,
    const float* map, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j >= ny) return;
    int col = x0 + i, row = y0 + j;
    // The light's direction, as tangents, the same either side.
    float tx = (coordinate(col, size, pitch) - px) / depth;
    float ty = (coordinate(row, size, pitch) - py) / depth;
    float t2 = tx * tx + ty * ty;
    if (t2 >= cone * cone) return;
    float u = (tx + cone) / (2 * cone) * (n - 1), v = (ty + cone) / (2 * cone) * (n - 1);
    int iu = min((int)u, n - 2), iv = min((int)v, n - 2);
    float au = u - iu, av = v - iv;
    const float* m = map + iv * n + iu;
    float bright = m[0] * (1 - au) * (1 - av) + m[1] * au * (1 - av)
                 + m[n] * (1 - au) * av + m[n + 1] * au * av;
    if (bright <= 0) return;
    float extra = depth * t2 / (sqrtf(1.0f + t2) + 1.0f);
    float strength = bright * powf(1.0f + t2, -0.75f) / fabsf(depth);
    float s, c;
    sincosf(k * extra + phase, &s, &c);
    light[row * size + col] += complex<float>(strength * c, strength * s);
}

// The light of a point through the lens, its path lengths (less the
// part in `phase`), their slopes and cross slopes, strengths, where the
// rays crossed the lens and which were found given on a gnx x gny grid
// `coarse` pixels apart, `step` long, from the window's corner. A pixel
// gets light if its ray, blended, crossed within `aperture` and arrives
// with its sine squared under `cone2`.
extern "C" __global__ void through_lens(
    complex<float>* light, int size, float k,
    int x0, int y0, int nx, int ny,
    int coarse, float step, int gnx, int gny,
    const float* path, const float* gx, const float* gy, const float* gxy,
    const float* strength, const float* rim, const unsigned char* ok,
    float phase, int hermite, float aperture, float cone2)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j >= ny) return;
    int ix = i / coarse, iy = j / coarse;
    if (ix >= gnx - 1 || iy >= gny - 1) return;
    float ax = (float)(i - ix * coarse) / coarse;
    float ay = (float)(j - iy * coarse) / coarse;
    int a = iy * gnx + ix, b = a + 1, c = a + gnx, d = c + 1;
    if (!(ok[a] && ok[b] && ok[c] && ok[d])) return;

    // Linear weights, and with hermite, cubic ones for values (v) and
    // slopes (s), at either end.
    float lx0 = 1 - ax, lx1 = ax, ly0 = 1 - ay, ly1 = ay;
    float w00 = lx0 * ly0, w10 = lx1 * ly0, w01 = lx0 * ly1, w11 = lx1 * ly1;
    if (rim[a] * w00 + rim[b] * w10 + rim[c] * w01 + rim[d] * w11 > aperture) return;
    float sx = gx[a] * w00 + gx[b] * w10 + gx[c] * w01 + gx[d] * w11;
    float sy = gy[a] * w00 + gy[b] * w10 + gy[c] * w01 + gy[d] * w11;
    if (sx * sx + sy * sy >= cone2) return;
    float p;
    if (hermite) {
        float vx0 = (1 + 2 * ax) * lx0 * lx0, vx1 = ax * ax * (3 - 2 * ax);
        float sx0 = ax * lx0 * lx0 * step, sx1 = ax * ax * (ax - 1) * step;
        float vy0 = (1 + 2 * ay) * ly0 * ly0, vy1 = ay * ay * (3 - 2 * ay);
        float sy0 = ay * ly0 * ly0 * step, sy1 = ay * ay * (ay - 1) * step;
        p = path[a] * vx0 * vy0 + gx[a] * sx0 * vy0 + gy[a] * vx0 * sy0 + gxy[a] * sx0 * sy0
          + path[b] * vx1 * vy0 + gx[b] * sx1 * vy0 + gy[b] * vx1 * sy0 + gxy[b] * sx1 * sy0
          + path[c] * vx0 * vy1 + gx[c] * sx0 * vy1 + gy[c] * vx0 * sy1 + gxy[c] * sx0 * sy1
          + path[d] * vx1 * vy1 + gx[d] * sx1 * vy1 + gy[d] * vx1 * sy1 + gxy[d] * sx1 * sy1;
    } else {
        p = path[a] * w00 + path[b] * w10 + path[c] * w01 + path[d] * w11;
    }
    float w = strength[a] * w00 + strength[b] * w10 + strength[c] * w01 + strength[d] * w11;
    float s, cs;
    sincosf(k * p + phase, &s, &cs);
    int at = (y0 + j) * size + x0 + i;
    light[at] += complex<float>(w * cs, w * s);
}
"""

_module = None


def _kernel(name):
    global _module
    if _module is None:
        _module = cp.RawModule(code=_SOURCE)
    return _module.get_function(name)


_BLOCK = (32, 8)


def _blocks(nx, ny):
    return (math.ceil(nx / _BLOCK[0]), math.ceil(ny / _BLOCK[1]))


class GpuPlate(Plate):
    """A Plate whose light is summed on the GPU. `hermite` chooses how
    path lengths through the lens are blended between grid points. Call
    finish() when all the light is on it, to bring it back as Plate's
    numpy array."""

    def __init__(self, size, pitch, wavelength, hermite=True):
        self.size, self.pitch, self.wavelength = size, pitch, wavelength
        self.hermite = hermite
        self.light = cp.zeros((size, size), cp.complex64)

    def finish(self):
        self.light = cp.asnumpy(self.light).astype(np.complex128)
        return self

    def add_direct(self, point, phase, cone, lens=None):
        px, py, pz = point
        depth = -pz
        found = self.window((px, py), cone * depth)
        if found is None:
            return
        (rows, cols), _, _ = found
        nx, ny = cols.stop - cols.start, rows.stop - rows.start
        start = (self.k * depth + phase) % (2 * math.pi)
        f32 = np.float32
        _kernel("direct")(
            _blocks(nx, ny),
            _BLOCK,
            (
                self.light, np.int32(self.size), f32(self.pitch), f32(self.k),
                np.int32(cols.start), np.int32(rows.start), np.int32(nx), np.int32(ny),
                f32(px), f32(py), f32(depth), f32(start), f32(cone**2),
                np.int32(lens is not None),
                f32(0 if lens is None else (lens.z - pz) / depth),
                f32(0 if lens is None else lens.aperture**2),
            ),
        )

    def add_seen(self, point, phase, cone, maps, index):
        """Adds the light of `point`, behind the plate or in front of it,
        that a viewer would see: as bright in each direction as its map,
        `maps[index]` (an n x n grid of amplitudes over tangents from
        -cone to cone, on the GPU)."""
        px, py, pz = point
        depth = -pz
        found = self.window((px, py), cone * abs(depth))
        if found is None:
            return
        (rows, cols), _, _ = found
        nx, ny = cols.stop - cols.start, rows.stop - rows.start
        start = (self.k * depth + phase) % (2 * math.pi)
        f32 = np.float32
        _kernel("seen")(
            _blocks(nx, ny),
            _BLOCK,
            (
                self.light, np.int32(self.size), f32(self.pitch), f32(self.k),
                np.int32(cols.start), np.int32(rows.start), np.int32(nx), np.int32(ny),
                f32(px), f32(py), f32(depth), f32(start), f32(cone),
                maps[index], np.int32(maps.shape[1]),
            ),
        )

    def _add_lens_grid(self, grid, phase, cone, aimed, aperture):
        ok = aimed.found
        if not ok.any():
            return
        ny, nx = grid["shape"]
        values = self._lens_values(aimed, ok)
        # The part of the path the same everywhere goes into the phase.
        base = float(aimed.path[ok].mean())
        rel = np.where(ok, aimed.path - base, 0).reshape(ny, nx)
        gx = np.where(ok, values["gx"], np.nan).reshape(ny, nx)
        gy = np.where(ok, values["gy"], np.nan).reshape(ny, nx)
        step = grid["step"]
        # Cross slopes, from how the slopes change across the grid.
        with np.errstate(all="ignore"):
            gxy = 0.5 * (np.gradient(gx, step, axis=0) + np.gradient(gy, step, axis=1))
        gx, gy, gxy = (np.nan_to_num(a) for a in (gx, gy, gxy))

        (rows, cols) = grid["where"]
        wx, wy = cols.stop - cols.start, rows.stop - rows.start
        start = (self.k * base + phase) % (2 * math.pi)
        f32 = np.float32

        def up(a, dtype=np.float32):
            return cp.asarray(np.ascontiguousarray(np.reshape(a, (ny, nx)), dtype))

        _kernel("through_lens")(
            _blocks(wx, wy),
            _BLOCK,
            (
                self.light, np.int32(self.size), f32(self.k),
                np.int32(cols.start), np.int32(rows.start), np.int32(wx), np.int32(wy),
                np.int32(grid["coarse"]), f32(step), np.int32(nx), np.int32(ny),
                up(rel), up(gx), up(gy), up(gxy), up(values["strength"]), up(values["rim"]),
                up(ok, np.uint8),
                f32(start), np.int32(self.hermite), f32(aperture), f32(cone**2 / (1 + cone**2)),
            ),
        )
