"""The wave sum on an NVIDIA GPU, with CuPy.

GpuPlate does what Plate does, adding up each point's waves across the
plate, but one GPU thread per pixel. Rays are still found on the CPU
(or by Mitsuba); only the sum, which is nearly all the work, moves.

GPUs are fast in single precision, which holds about 7 digits: too few
for a phase of tens of thousands of radians. So each kernel works only
with the small part of a point's path length that varies across the
plate (tens of micrometers at most), and the rest, the same for every
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

// The reverse of `seen`: how much the plate's error, `grad`, would change
// as the point's light changes, summed over its window (the light it
// sends each pixel, conjugated, times the error there). Each block adds
// its part into out[2 * index], out[2 * index + 1] (real, imaginary).
// `phase` here leaves out the point's own phase: that is what changes.
extern "C" __global__ void seen_reverse(
    const complex<float>* grad, int size, float pitch, float k,
    int x0, int y0, int nx, int ny,
    float px, float py, float depth, float phase, float cone,
    const float* map, int n, float* out, int index)
{
    __shared__ float re[256], im[256];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int t = threadIdx.y * blockDim.x + threadIdx.x;
    float r = 0, m = 0;
    if (i < nx && j < ny) {
        int col = x0 + i, row = y0 + j;
        float tx = (coordinate(col, size, pitch) - px) / depth;
        float ty = (coordinate(row, size, pitch) - py) / depth;
        float t2 = tx * tx + ty * ty;
        if (t2 < cone * cone) {
            float u = (tx + cone) / (2 * cone) * (n - 1), v = (ty + cone) / (2 * cone) * (n - 1);
            int iu = min((int)u, n - 2), iv = min((int)v, n - 2);
            float au = u - iu, av = v - iv;
            const float* mp = map + iv * n + iu;
            float bright = mp[0] * (1 - au) * (1 - av) + mp[1] * au * (1 - av)
                         + mp[n] * (1 - au) * av + mp[n + 1] * au * av;
            if (bright > 0) {
                float extra = depth * t2 / (sqrtf(1.0f + t2) + 1.0f);
                float strength = bright * powf(1.0f + t2, -0.75f) / fabsf(depth);
                float s, c;
                sincosf(k * extra + phase, &s, &c);
                complex<float> g = grad[row * size + col];
                // conj(strength e^(i a)) * g
                r = strength * (c * g.real() + s * g.imag());
                m = strength * (c * g.imag() - s * g.real());
            }
        }
    }
    re[t] = r;
    im[t] = m;
    __syncthreads();
    for (int h = 128; h > 0; h >>= 1) {
        if (t < h) { re[t] += re[t + h]; im[t] += im[t + h]; }
        __syncthreads();
    }
    if (t == 0) {
        atomicAdd(out + 2 * index, re[0]);
        atomicAdd(out + 2 * index + 1, im[0]);
    }
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
// ---- Many points at once ----
// Forward (seen_gather), each thread has a pixel of the plate and adds
// up the points that reach it. In reverse (seen_reverse_many), each
// point's window of pixels is cut into tiles of 32 x 8, one per block,
// numbered point after point: point j's from start[j] up to
// start[j + 1]. A block finds its point by binary search.

__device__ int owner(const int* start, int count, int b) {
    int lo = 0, hi = count;
    while (hi - lo > 1) {
        int mid = (lo + hi) / 2;
        if (start[mid] <= b) lo = mid; else hi = mid;
    }
    return lo;
}

// The light of a point at pixel (col, row), as `seen` works it out; false
// where it sends none.
__device__ bool seen_at(
    int col, int row, int size, float pitch, float k,
    float px, float py, float depth, float phase, float cone, float edge,
    const float* map, int n, complex<float>* value)
{
    float tx = (coordinate(col, size, pitch) - px) / depth;
    float ty = (coordinate(row, size, pitch) - py) / depth;
    float t2 = tx * tx + ty * ty;
    if (t2 >= edge * edge) return false;
    // Past the cone the light fades out smoothly up to `edge`, as bright
    // as the map's nearest direction: a sharp rim would ripple.
    float fade = 1.0f;
    if (t2 > cone * cone) {
        float c = cosf(1.5707963f * (sqrtf(t2) - cone) / (edge - cone));
        fade = c * c;
    }
    float u = (tx + cone) / (2 * cone) * (n - 1), v = (ty + cone) / (2 * cone) * (n - 1);
    u = fminf(fmaxf(u, 0.0f), (float)(n - 1));
    v = fminf(fmaxf(v, 0.0f), (float)(n - 1));
    int iu = min((int)u, n - 2), iv = min((int)v, n - 2);
    float au = u - iu, av = v - iv;
    const float* m = map + iv * n + iu;
    float bright = m[0] * (1 - au) * (1 - av) + m[1] * au * (1 - av)
                 + m[n] * (1 - au) * av + m[n + 1] * au * av;
    if (bright <= 0) return false;
    // (1 + t2)^(-3/4), by square roots: far cheaper than powf.
    float root = sqrtf(1.0f + t2);
    float extra = depth * t2 / (root + 1.0f);
    float strength = fade * bright / (root * sqrtf(root) * fabsf(depth));
    float s, c;
    sincosf(k * extra + phase, &s, &c);
    *value = complex<float>(strength * c, strength * s);
    return true;
}

// Which pixel of its point's window this thread has, and whether it is
// in the window.
__device__ bool tile_pixel(
    const int* start, int count, const int* nx, const int* ny,
    int* j, int* i, int* jj)
{
    int b = blockIdx.x;
    *j = owner(start, count, b);
    int local = b - start[*j];
    int across = (nx[*j] + 31) / 32;
    *i = (local % across) * 32 + threadIdx.x;
    *jj = (local / across) * 8 + threadIdx.y;
    return *i < nx[*j] && *jj < ny[*j];
}

// `seen` for every point, onto the tile of the plate tw x th pixels from
// (tx, ty). The tile is cut into square cells, `cell` pixels across,
// cells_x of them in a row, and each cell has a list of the points whose
// light reaches it: for cell c, list[cell_start[c]] up to
// list[cell_start[c + 1]]. Each thread has one pixel and adds up its
// cell's points there. Nothing else writes to its pixel, so the sum
// stays in the thread until it is whole. (With a thread to each wave,
// each added to the plate atomically as windows overlap, the sum took
// several times as long: the memory was what was slow, not the
// arithmetic.) `light` (interleaved real, imaginary) is the part of the
// plate from pixel (ox, oy), `width` pixels across; the tile lies
// within it.
extern "C" __global__ void seen_gather(
    float* light, int size, int ox, int oy, int width,
    int tx, int ty, int tw, int th, float pitch, float k, float cone, float edge,
    int cell, int cells_x, const int* cell_start, const int* list,
    const float* px, const float* py, const float* depth, const float* phase,
    const float* maps, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int jj = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= tw || jj >= th) return;
    int c = (jj / cell) * cells_x + i / cell;
    int at = cell_start[c], end = cell_start[c + 1];
    if (at == end) return;
    int col = tx + i, row = ty + jj;
    float re = 0, im = 0;
    for (; at < end; at++) {
        int j = list[at];
        complex<float> v;
        if (seen_at(col, row, size, pitch, k, px[j], py[j], depth[j], phase[j], cone, edge,
                    maps + (size_t)j * n * n, n, &v)) {
            re += v.real();
            im += v.imag();
        }
    }
    float* out = light + 2 * ((size_t)(row - oy) * width + (col - ox));
    out[0] += re;
    out[1] += im;
}

// `seen_reverse` for every point, into out (2 per point).
extern "C" __global__ void seen_reverse_many(
    const complex<float>* grad, int size, int ox, int oy, int width, float pitch, float k, float cone, float edge,
    int count, const int* start, const int* x0, const int* y0, const int* nx, const int* ny,
    const float* px, const float* py, const float* depth, const float* phase,
    const float* maps, int n, float* out)
{
    __shared__ float re[256], im[256];
    int t = threadIdx.y * blockDim.x + threadIdx.x;
    int j, i, jj;
    float r = 0, m = 0;
    if (tile_pixel(start, count, nx, ny, &j, &i, &jj)) {
        int col = x0[j] + i, row = y0[j] + jj;
        complex<float> v;
        if (seen_at(col, row, size, pitch, k, px[j], py[j], depth[j], phase[j], cone, edge,
                    maps + (size_t)j * n * n, n, &v)) {
            complex<float> g = grad[(row - oy) * width + (col - ox)];
            r = v.real() * g.real() + v.imag() * g.imag();
            m = v.real() * g.imag() - v.imag() * g.real();
        }
    }
    re[t] = r;
    im[t] = m;
    __syncthreads();
    for (int h = 128; h > 0; h >>= 1) {
        if (t < h) { re[t] += re[t + h]; im[t] += im[t + h]; }
        __syncthreads();
    }
    if (t == 0) {
        atomicAdd(out + 2 * j, re[0]);
        atomicAdd(out + 2 * j + 1, im[0]);
    }
}
"""

_modules = {}


def _kernel(name, fast=False):
    """A kernel of the source above, compiled; with `fast`, with CUDA's
    fast math (see SeenPoints.light)."""
    if fast not in _modules:
        _modules[fast] = cp.RawModule(code=_SOURCE, options=("--use_fast_math",) if fast else ())
    return _modules[fast].get_function(name)


_BLOCK = (32, 8)
# The cells the plate is cut into for SeenPoints.light, in pixels across
# (whole blocks), and how many (point, cell) pairs are listed for one
# launch: more would not fit on the GPU beside a big patch of the plate.
_CELL = 64
_PAIRS = 6_000_000


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

    def seen_reverse(self, grad, point, cone, maps, index, out):
        """The reverse of add_seen, for optimizing: adds into out[index]
        (out a float32 array of 2 per point, on the GPU) how the error
        `grad` across the plate (on the GPU) would change with the light
        of `point`: the sum, over the pixels it reaches, of its light
        there, without its own phase, conjugated, times `grad`."""
        px, py, pz = point
        depth = -pz
        found = self.window((px, py), cone * abs(depth))
        if found is None:
            return
        (rows, cols), _, _ = found
        nx, ny = cols.stop - cols.start, rows.stop - rows.start
        start = (self.k * depth) % (2 * math.pi)
        f32 = np.float32
        _kernel("seen_reverse")(
            _blocks(nx, ny),
            _BLOCK,
            (
                grad, np.int32(self.size), f32(self.pitch), f32(self.k),
                np.int32(cols.start), np.int32(rows.start), np.int32(nx), np.int32(ny),
                f32(px), f32(py), f32(depth), f32(start), f32(cone),
                maps[index], np.int32(maps.shape[1]), out, np.int32(index),
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


class SeenPoints:
    """Many points' light, as GpuPlate.add_seen sums it, but all at once:
    launching one sum per point costs far more than the sum. The points
    and their maps (on the GPU: one n x n map each) are fixed; their
    phases are given each time. With `tile` (x, y, width, height, in
    pixels), only that part of the plate is worked out."""

    def __init__(self, size, pitch, wavelength, points, maps, cone, tile=None, edge=None):
        self.size, self.pitch, self.cone = size, pitch, cone
        # Where each point's light has faded out (a tangent): at the cone, if not given.
        self.edge = edge or cone
        self.tile = tile or (0, 0, size, size)
        ox, oy, tw, th = self.tile
        self.k = 2 * math.pi / wavelength
        self.maps = maps
        px, py, pz = (np.asarray(a, float) for a in np.asarray(points).T)
        self.depth = -pz
        # Each point's window: the pixels of the tile its light reaches,
        # as Plate.window has it, but no farther than the directions it
        # sends any light in (_lit).
        (x_from, x_to), (y_from, y_to) = self._lit()

        def window(p, low, high, first, count):
            a, b = p + self.depth * low, p + self.depth * high
            lo = np.clip(np.floor(np.minimum(a, b) / pitch + size / 2), first, first + count).astype(np.int64)
            hi = np.clip(np.ceil(np.maximum(a, b) / pitch + size / 2) + 1, first, first + count).astype(np.int64)
            return lo, hi - lo

        lo_x, nx = window(px, x_from, x_to, ox, tw)
        lo_y, ny = window(py, y_from, y_to, oy, th)
        none = (nx == 0) | (ny == 0)
        nx[none] = ny[none] = 0
        self._window = lo_x, lo_y, nx, ny
        # The cells of the tile each window touches: its first, and how
        # many across and down; and the points in runs of at most _PAIRS
        # cells in all (first, last + 1), each run a launch (see light).
        first_x, first_y = (lo_x - ox) // _CELL, (lo_y - oy) // _CELL
        across = np.where(none, 0, (lo_x + nx - 1 - ox) // _CELL - first_x + 1)
        down = np.where(none, 0, (lo_y + ny - 1 - oy) // _CELL - first_y + 1)
        before = np.concatenate([[0], np.cumsum(across * down)])
        self._runs, first = [], 0
        while before[first] < before[-1]:
            last = max(first + 1, int(np.searchsorted(before, before[first] + _PAIRS, "right")) - 1)
            self._runs.append((first, last))
            first = last
        up = lambda a, t: cp.asarray(np.ascontiguousarray(a, t))
        self._cells = tuple(up(a, np.int64) for a in (first_x, first_y, across, down))
        self._where = up(px, np.float32), up(py, np.float32), up(self.depth, np.float32)

    def _lit(self):
        """The directions each point sends any light in, as tangents: from
        and to, along x and along y. Many points send none over much of
        the cone: they face away, or something is in front. Each map says
        where: a direction more either side, since the map is blended
        between directions, and out to the edge where the map's rim is
        lit, as it is the rim's brightness that fades out past the cone."""
        n = self.maps.shape[1]
        spans = []
        for axis in (1, 2):
            lit = cp.asnumpy(self.maps.max(axis=axis) > 0)
            first = lit.argmax(axis=1) - 1
            last = n - lit[:, ::-1].argmax(axis=1)
            at = lambda i: -self.cone + 2 * self.cone * i / (n - 1)
            spans.append((np.where(first <= 0, -self.edge, at(first)), np.where(last >= n - 1, self.edge, at(last))))
        return spans

    def _lists(self, first, last):
        """For the points from `first` up to `last`, the lists seen_gather
        wants (on the GPU, and made there): how many cells the tile is
        across, where each cell's list starts, and the lists, one after
        another."""
        ox, oy, tw, th = self.tile
        cells_x, cells_y = -(-tw // _CELL), -(-th // _CELL)
        first_x, first_y, across, down = (a[first:last] for a in self._cells)
        px, py, depth = (a[first:last] for a in self._where)
        # Every cell of every window: which point's, and which cell.
        cells = across * down
        ends = cp.cumsum(cells)
        index = cp.arange(int(ends[-1]), dtype=cp.int64)
        who = cp.searchsorted(ends, index, side="right")
        index -= (ends - cells)[who]
        x = first_x[who] + index % across[who]
        y = first_y[who] + index // across[who]
        del index
        # Not those wholly outside the point's disc of light, at the
        # window's corners.
        middle_x = (px / self.pitch + self.size / 2 - 0.5 - ox)[who]
        middle_y = (py / self.pitch + self.size / 2 - 0.5 - oy)[who]
        off_x = cp.maximum(0, cp.maximum(x * _CELL - middle_x, middle_x - (x * _CELL + _CELL - 1)))
        off_y = cp.maximum(0, cp.maximum(y * _CELL - middle_y, middle_y - (y * _CELL + _CELL - 1)))
        reach = (self.edge * cp.abs(depth) / self.pitch + 1)[who]
        keep = off_x * off_x + off_y * off_y <= reach * reach
        del middle_x, middle_y, off_x, off_y, reach
        # Sorted by cell, then by point.
        count = len(self.depth)
        key = cp.sort(((y * cells_x + x) * count + (who + first))[keep])
        del x, y, who, keep
        start = cp.searchsorted(key // count, cp.arange(cells_x * cells_y + 1, dtype=cp.int64))
        return cells_x, start.astype(cp.int32), (key % count).astype(cp.int32)

    def _phase(self, phases):
        return cp.asarray(((self.k * self.depth + phases) % (2 * math.pi)).astype(np.float32))

    def light(self, phases, into=None):
        """Their light on the plate (on the GPU), each starting with its
        phase. With `into` (an array and the pixel its corner is at, x
        then y), the light is added to that array instead, which must
        hold the whole tile.

        Each pixel adds up the points that reach it (seen_gather). The
        sum is compiled with CUDA's fast math, whose sines, square roots
        and divisions are a little less exact: nearly twice as fast, for
        a difference in the plate's light of 5 parts in 100,000."""
        tx, ty, tw, th = self.tile
        if into is None:
            light, ox, oy = cp.zeros((th, tw), cp.complex64), tx, ty
        else:
            light, ox, oy = into
        f32, i32 = np.float32, np.int32
        phase = self._phase(phases)
        for first, last in self._runs:
            cells_x, start, points = self._lists(first, last)
            _kernel("seen_gather", fast=True)(
                _blocks(tw, th), _BLOCK,
                (light, i32(self.size), i32(ox), i32(oy), i32(light.shape[1]), i32(tx), i32(ty), i32(tw), i32(th),
                 f32(self.pitch), f32(self.k), f32(self.cone), f32(self.edge),
                 i32(_CELL), i32(cells_x), start, points, *self._where, phase, self.maps, i32(self.maps.shape[1])),
            )
            del start, points
        return light

    def reverse(self, grad):
        """For each point, its light on the plate without its own phase,
        conjugated, times `grad` (on the GPU), summed."""
        count = len(self.depth)
        out = cp.zeros(2 * count, cp.float32)
        ox, oy, tw, th = self.tile
        # Each window in tiles of a block, one after another (see
        # tile_pixel).
        lo_x, lo_y, nx, ny = self._window
        start = np.concatenate([[0], np.cumsum(-(-nx // _BLOCK[0]) * -(-ny // _BLOCK[1]))])
        if start[-1] >= 2**31:
            raise ValueError("too many tiles for one launch")
        if start[-1] == 0:
            return np.zeros(count, complex)
        f32, i32 = np.float32, np.int32
        up = lambda a: cp.asarray(np.ascontiguousarray(a, i32))
        _kernel("seen_reverse_many")(
            (int(start[-1]),), _BLOCK,
            (grad, i32(self.size), i32(ox), i32(oy), i32(tw), f32(self.pitch), f32(self.k), f32(self.cone), f32(self.edge),
             i32(count), up(start), up(lo_x), up(lo_y), up(nx), up(ny), *self._where,
             self._phase(np.zeros(count)), self.maps, i32(self.maps.shape[1]), out),
        )
        out = cp.asnumpy(out)
        return out[0::2] + 1j * out[1::2]


class GpuEye:
    """Eye (eye.py), on the GPU: looks at `light` leaving a plate of
    pixels `pitch` apart, of `wavelength`."""

    def __init__(self, light, pitch, wavelength):
        self.wavelength = wavelength
        self.spectrum = cp.fft.fft2(cp.asarray(light, cp.complex64))
        n = self.spectrum.shape[0]
        freq = cp.fft.fftfreq(n, pitch).astype(cp.float32) * wavelength
        self.sx, self.sy = freq[None, :], freq[:, None]

    def view(self, direction, aperture, depth):
        """As Eye.view: the brightness across the plane `depth` behind the
        plate, seen along `direction` through a pupil of `aperture`. As a
        numpy array."""
        sx, sy = direction
        pupil = (self.sx - sx) ** 2 + (self.sy - sy) ** 2 < aperture**2
        cos = cp.sqrt(cp.maximum(0, 1 - self.sx**2 - self.sy**2))
        back = cp.exp((-2j * math.pi / self.wavelength) * cos * depth).astype(cp.complex64)
        return cp.asnumpy(cp.abs(cp.fft.ifft2(self.spectrum * pupil * back)) ** 2)
