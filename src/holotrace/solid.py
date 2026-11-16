"""Step 3: a solid scene at the image plane.

A ball, striped by longitude, sits across the plate: the plate cuts
through it, so its front floats in front of the plate and the rest is
behind. Behind it is a checkered wall, which it hides in part and on
which it casts a shadow. Light comes from the upper left; the ball is a
little glossy.

The surfaces are sampled into points, each a source of light, as in
the magnifier. What is new is that each point's light differs with
direction: Mitsuba works out, for each point and each of a grid of
directions toward the viewer, whether the point faces that way and
nothing stands in front of it (hidden surfaces), and whether the light
reaches it (shadows); the brightness, matte with a highlight, follows.
On the GPU each point's light then goes to the plate as bright in each
direction as that says (see the `seen` kernel in gpu.py).

A point in front of the plate sends the plate light converging onto it
(the light passes the plate first), as a lens's real image would; its
path to the plate counts as negative.

If the hologram is right, the eye should see the ball hide the wall,
and, moving, see more of the wall around it (parallax); focused on the
ball's front, in front of the plate, that is sharp and the wall blurred,
and the other way round focused on the wall. The views go to out/solid.
This step needs an NVIDIA GPU (the `gpu` extra).
"""

import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import planes
from .magnifier import PLATE, PUPIL, WAVELENGTH, sheet

# Lengths in micrometres.
BALL_Z, BALL_RADIUS = -150.0, 350.0
WALL_Z, WALL_HALF, CHECK = -600.0, 1100.0, 150.0
SPACING = 8.0
LIGHT = np.array([-0.5, 0.6, 0.8]) / np.linalg.norm([-0.5, 0.6, 0.8])
AMBIENT, GLOSS, SHINE = 0.15, 0.6, 40
# The light is a small disc this wide across (radians), not a point, so
# that shadows have soft edges: SHADOW_RAYS rays to points spread over it
# say how much of it each point sees.
LIGHT_SIZE = 0.05
SHADOW_RAYS = 16

VIEWS = {"straight on": (0.0, 0.0), "from the right": (0.06, 0.0)}
FOCUS = {"on the ball's front": -150.0, "on the wall": -WALL_Z}



def ball():
    """Points evenly over the ball (a Fibonacci spiral), those that can
    face the viewer: positions, normals, how much light each reflects,
    and how glossy it is."""
    n = round(4 * math.pi * BALL_RADIUS**2 / SPACING**2)
    i = np.arange(n) + 0.5
    z = 1 - 2 * i / n
    r = np.sqrt(1 - z * z)
    a = i * math.pi * (3 - math.sqrt(5))
    normals = np.column_stack([r * np.cos(a), r * np.sin(a), z])
    normals = normals[normals[:, 2] > -0.2]
    points = normals * BALL_RADIUS + [0.0, 0.0, BALL_Z]
    # Eight stripes around.
    longitude = np.arctan2(normals[:, 1], normals[:, 0])
    stripe = np.floor(longitude / (2 * math.pi) * 8).astype(int) % 2
    albedo = np.where(stripe == 0, 1.0, 0.1)
    return points, normals, albedo, np.full(len(points), GLOSS)


def wall(z=WALL_Z, half=WALL_HALF, light=1.0, dark=0.25, spacing=SPACING, check=CHECK):
    """Points about `spacing` apart over a wall at z, `half` either way,
    checkered `light` and `dark` in squares `check` across."""
    g = np.arange(-half, half + spacing / 2, spacing)
    x, y = (a.ravel() for a in np.meshgrid(g, g))
    # Moved a little at random off the grid, so that an eye sharp enough
    # to make them out sees no lattice.
    rng = np.random.default_rng(4)
    x = x + rng.uniform(-spacing / 2, spacing / 2, len(x))
    y = y + rng.uniform(-spacing / 2, spacing / 2, len(y))
    points = np.column_stack([x, y, np.full(len(x), z)])
    normals = np.tile([0.0, 0.0, 1.0], (len(x), 1))
    square = (np.floor(x / check) + np.floor(y / check)).astype(int) % 2
    albedo = np.where(square == 0, light, dark)
    return points, normals, albedo, np.zeros(len(x))


def scene():
    """The scene as Mitsuba sees it, for working out what hides what."""
    import mitsuba as mi

    from .mesh import _mitsuba

    _mitsuba()
    wall_at = mi.ScalarTransform4f().translate([0, 0, WALL_Z]).scale([WALL_HALF, WALL_HALF, 1])
    return mi.load_dict(
        {
            "type": "scene",
            "ball": {"type": "sphere", "center": [0, 0, BALL_Z], "radius": BALL_RADIUS},
            "wall": {"type": "rectangle", "to_world": wall_at},
        }
    )


def _blocked(world, origins, directions):
    """Whether anything is in the way of rays from `origins` along
    `directions` (both N x 3)."""
    import mitsuba as mi

    # Dr.Jit fails on empty arrays, rather than doing nothing.
    if len(origins) == 0:
        return np.zeros(0, bool)

    def vec(cls, a):
        a = np.ascontiguousarray(a, np.float32)
        return cls(mi.Float(a[:, 0]), mi.Float(a[:, 1]), mi.Float(a[:, 2]))

    ray = mi.Ray3f(o=vec(mi.Point3f, origins), d=vec(mi.Vector3f, directions))
    return np.array(world.ray_test(ray))


def _light_rays():
    """Directions to SHADOW_RAYS points spread evenly over the light's disc
    (a sunflower pattern), LIGHT_SIZE across about LIGHT."""
    i = np.arange(SHADOW_RAYS) + 0.5
    r = LIGHT_SIZE / 2 * np.sqrt(i / SHADOW_RAYS)
    a = i * math.pi * (3 - math.sqrt(5))
    u = np.cross(LIGHT, [0.0, 0.0, 1.0])
    u /= np.linalg.norm(u)
    v = np.cross(LIGHT, u)
    rays = LIGHT + (r * np.cos(a))[:, None] * u + (r * np.sin(a))[:, None] * v
    return rays / np.linalg.norm(rays, axis=1, keepdims=True)


def maps(world, points, normals, albedo, gloss, chunk=4000, shadows=True):
    """Each point's light, as bright in each of the optics' directions x
    directions directions toward the viewer (tangents across its cone;
    rows along y): the
    amplitude, the square root of the power it sends that way. That is
    its brightness (matte, from as much of the light as it sees, the
    rest in shadow, or with shadows off all of it wherever it faces the
    light; a little ambient light everywhere; and a highlight where
    glossy) times how much of it faces that way, or nothing where it
    faces away or something is in front of it."""
    o = OPTICS
    m = o.directions
    t = np.linspace(-o.cone, o.cone, m)
    tx, ty = np.meshgrid(t, t)
    d = np.column_stack([tx.ravel(), ty.ravel(), np.ones(m * m)])
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    half = LIGHT + d
    half /= np.linalg.norm(half, axis=1, keepdims=True)
    out = np.zeros((len(points), m * m), np.float32)
    lift = 0.05  # off the surface, so as not to meet it
    for s in range(0, len(points), chunk):
        p, n = points[s : s + chunk], normals[s : s + chunk]
        a, g = albedo[s : s + chunk], gloss[s : s + chunk]
        # How much of the light it sees: the share of rays to points
        # across the light that nothing stops.
        facing_light = n @ LIGHT > 0
        lit = facing_light.astype(float)
        if shadows:
            rays = _light_rays()
            idx = np.flatnonzero(facing_light)
            origins = np.repeat(p[idx] + lift * n[idx], len(rays), axis=0)
            clear = ~_blocked(world, origins, np.tile(rays, (len(idx), 1)))
            lit[idx] = clear.reshape(len(idx), len(rays)).mean(axis=1)
        diffuse = a * (AMBIENT + (1 - AMBIENT) * np.maximum(n @ LIGHT, 0) * lit)
        highlight = g[:, None] * lit[:, None] * np.maximum(n @ half.T, 0) ** SHINE
        # Seen, or facing away or hidden.
        facing = n @ d.T
        ok = facing > 0
        rows, cols = np.nonzero(ok)
        seen = ok.copy()
        seen[rows, cols] = ~_blocked(world, p[rows] + lift * n[rows], d[cols])
        power = (diffuse[:, None] + highlight) * np.maximum(facing, 0)
        out[s : s + chunk] = np.where(seen, np.sqrt(power), 0)
    return out.reshape(len(points), m, m)


@dataclass(frozen=True)
class Optics:
    """How the hologram is made: the plate's pixels, `pixel` um apart; the
    widest angle the scene's light reaches the plate at, `cone` (its
    tangent); the reference beam's angle (its sine); and what follows.

    - keep: what the eye can see of the plate's light is the light within
      the cone, which needs a sample only every so many of the plate's
      pixels, so that is all that is worked out and kept.
    - slab, clear: points are taken in slabs `slab` um deep, each slab's
      light gathered on a recording plane `clear` um beyond it (see
      sum_planes). A point's light is described by rays, its brightness
      in each direction, which is only right far enough from it; the
      wider the cone, the closer that is.
    - directions: each point's brightness is worked out for directions x
      directions directions across the cone (see maps).

    The default is the plate so far: 0.5 um pixels, the scene's light
    within 7 degrees, the reference at 23.6 degrees, kept every 2 um."""

    pixel: float = 0.5
    cone: float = math.tan(math.radians(7))
    reference: float = 0.4
    keep: int = 4
    slab: float = 1000.0
    clear: float = 400.0
    directions: int = 24

    @property
    def sine(self):
        """The sine of the widest angle."""
        return self.cone / math.sqrt(1 + self.cone**2)

    @property
    def kept_pitch(self):
        return self.pixel * self.keep

    @classmethod
    def wide(cls, pixel, degrees):
        """For pixels `pixel` um apart, the scene's light within `degrees`:
        kept as coarsely as that allows; the reference just outside the
        cone (the plate's pattern is to be encoded without the halo about
        the reference beam, so it need not be clear of that); slabs and
        their planes' distance shrunk as the cone grows, and directions
        grown with it."""
        cone = math.tan(math.radians(degrees))
        sine = cone / math.sqrt(1 + cone**2)
        keep = max(1, int(WAVELENGTH / (2 * sine) / pixel))
        reference = sine + 0.08
        if reference + sine > WAVELENGTH / (2 * pixel):
            raise ValueError("the pixels cannot carry fringes that fine")
        base = cls()
        ratio = base.sine / sine
        # At 24 degrees, planes 200 um from their points differ from
        # summing straight onto the plate by about 1.5%, 400 um by about
        # 0.6% (and at 7 degrees, 400 um by a few percent).
        return cls(pixel, cone, reference, keep, slab=max(200.0, base.slab * ratio),
                   clear=max(200.0, base.clear * ratio), directions=32)


# The optics in use (see Optics; benchy.py sets them).
OPTICS = Optics()
# The kept light is worked out a tile at a time, as big as the GPU's
# memory allows for its deepest recording plane: its patch is the tile
# and as far as light spreads either side, and is transformed whole.
GPU_PATCH = 15360


def sum_planes(points, light, phases, size, kept=None):
    """The plate's light, `size` pixels square, from `points` with their
    brightness maps `light` and starting `phases`, as the eye can see it:
    every OPTICS.keep pixels. It is worked out on the GPU a tile at a
    time, each tile finished before the next: nothing the size of the
    plate is made but what is kept, which goes into `kept` (made in
    memory if not given; a file on disk, say, for plates too big for
    memory).

    The points are taken in slabs OPTICS.slab deep, and each slab's light
    is summed onto a recording plane OPTICS.clear from the slab's nearer
    face to the plate (beyond the plate, for the slabs at it): so every
    point's light is described where its rays describe it well, and lands
    on many of the kept pixels. Each tile gets the patch of each plane
    its light can come from (the tile and a margin as wide as light
    spreads on the way), and that patch's light is carried on to the tile
    by its plane waves (see _carry): exact, as light crossing open space
    is the sum of its plane waves. Each pixel's light is its own sum, so
    working tile by tile is exact too. Tiles are as big as the GPU allows
    for the deepest plane's patch (GPU_PATCH)."""
    import cupy as cp

    from .gpu import SeenPoints

    o = OPTICS
    n, pitch = size // o.keep, o.kept_pitch
    grid = dict(size=n, pitch=pitch, wavelength=WAVELENGTH)
    depth = -points[:, 2]
    which_slab = np.floor(depth / o.slab).astype(int)
    # Each group: its plane's distance behind the plate (in front, if
    # negative), its points' indices, their positions as seen from the
    # plane, the margin light spreads by on the way to the plate (in kept
    # pixels), and their windows on the plane.
    groups = []
    for k in sorted(set(which_slab)):
        mine = np.flatnonzero(which_slab == k)
        # The plane, `clear` beyond the slab's face nearer the plate.
        plane = k * o.slab - o.clear if k >= 0 else (k + 1) * o.slab + o.clear
        seen_from = points[mine] + [0.0, 0.0, plane]
        margin = int(math.ceil(o.cone * abs(plane) / pitch)) + 8
        px, py, pz = seen_from.T
        r = o.cone * np.abs(pz)
        bounds = (
            np.floor((px - r) / pitch + n / 2), np.ceil((px + r) / pitch + n / 2) + 1,
            np.floor((py - r) / pitch + n / 2), np.ceil((py + r) / pitch + n / 2) + 1,
        )
        groups.append((plane, mine, seen_from, margin, bounds))
    deepest = max(m for _, _, _, m, _ in groups)
    tile = min(n, (GPU_PATCH - 2 * deepest) // 256 * 256)
    if tile < 1024:
        raise ValueError("too deep a scene to carry tile by tile: see planes.py")

    if kept is None:
        kept = np.zeros((n, n), np.complex64)
    tiles, done, start = (-(-n // tile)) ** 2, 0, time.time()
    print(f"  kept every {pitch:g} um: {n}^2 pixels, in {tiles} tiles of {tile}^2, {len(groups)} slabs", flush=True)
    for y0 in range(0, n, tile):
        for x0 in range(0, n, tile):
            th, tw = min(tile, n - y0), min(tile, n - x0)
            out = cp.zeros((th, tw), cp.complex64)
            for plane, mine, seen_from, m, (lo_x, hi_x, lo_y, hi_y) in groups:
                ox, oy, w, h = x0 - m, y0 - m, tw + 2 * m, th + 2 * m
                near = (hi_x > ox) & (lo_x < ox + w) & (hi_y > oy) & (lo_y < oy + h)
                if not near.any():
                    continue
                which = mine[near]
                seen = SeenPoints(**grid, points=seen_from[near], maps=cp.asarray(light[which], cp.float32),
                                  cone=o.cone, tile=(ox, oy, w, h))
                out += _carry(seen.light(phases[which]), plane, m, pitch)
                del seen
                # Each slab's patch is a different size; the GPU memory
                # kept for reuse would otherwise pile up.
                cp.get_default_memory_pool().free_all_blocks()
            kept[y0 : y0 + th, x0 : x0 + tw] = cp.asnumpy(out)
            del out
            done += 1
            if done < tiles and (tiles <= 20 or done % max(1, tiles // 20) == 0):
                print(f"  {done} of {tiles} tiles, {time.time() - start:.0f} s", flush=True)
    return kept


def _carry(patch, distance, margin, pitch):
    """The light a patch of a recording plane (on the GPU, its pixels
    `pitch` apart), `distance` behind the plate (in front, if negative),
    sends the tile `margin` pixels in from its edges: each of its plane waves moved on by its own
    phase (the angular spectrum method), keeping only those within the cone,
    as all the scene's light is, and within what the sampling can carry
    without aliasing (band-limited, after Matsushima). The margin is as
    far as light spreads on the way, so the tile's light is whole, and
    what wraps round the patch's edges in the transform stays outside it
    (overlap-save). Deep planes make big patches, so the transform is
    done in place and the step applied a block of rows at a time."""
    import cupy as cp
    import cupyx.scipy.fft as cufft

    # Padded to sizes the FFT handles well (small prime factors): others
    # take it far more memory.
    h, w = patch.shape
    ph, pw = cufft.next_fast_len(h), cufft.next_fast_len(w)
    if (ph, pw) != (h, w):
        padded = cp.zeros((ph, pw), cp.complex64)
        padded[:h, :w] = patch
        patch = padded
    lam = WAVELENGTH
    fx = cp.fft.fftfreq(pw, pitch).astype(cp.float32)[None, :]
    fy = cp.fft.fftfreq(ph, pitch).astype(cp.float32)
    limit_x = 1 / (lam * math.sqrt((2 * distance / (pw * pitch)) ** 2 + 1))
    limit_y = 1 / (lam * math.sqrt((2 * distance / (ph * pitch)) ** 2 + 1))
    spectrum = cufft.fft2(patch, overwrite_x=True)
    del patch
    for r in range(0, ph, 1024):
        y = fy[r : r + 1024, None]
        s2 = (fx * fx + y * y) * lam * lam
        cone = OPTICS.cone
        keep = (s2 < cone**2 / (1 + cone**2)) & (cp.abs(fx) < limit_x) & (cp.abs(y) < limit_y)
        phase = (2 * math.pi / lam) * distance * cp.sqrt(cp.maximum(0, 1 - s2))
        spectrum[r : r + 1024] *= keep * cp.exp(1j * phase).astype(cp.complex64)
    moved = cufft.ifft2(spectrum, overwrite_x=True)
    out = moved[margin : h - margin, margin : w - margin].copy()
    del moved, spectrum
    cp.fft.config.get_plan_cache().clear()
    return out


def _fast_len(n):
    import cupyx.scipy.fft as cufft

    return cufft.next_fast_len(n)


def spectrum_in_place(a, block=512):
    """Replaces `a` (square, complex) with its 2D Fourier transform, in
    place: the rows a block at a time, then the columns, so that only a
    block more is needed than the array itself."""
    import scipy.fft

    n = a.shape[0]
    for i in range(0, n, block):
        a[i : i + block] = scipy.fft.fft(a[i : i + block], axis=1, workers=-1)
    for j in range(0, n, block):
        a[:, j : j + block] = scipy.fft.fft(a[:, j : j + block], axis=0, workers=-1)
    return a


def render(name, parts, world, focus, plate_size=PLATE["size"], shadows=True):
    """Makes the hologram of a scene and the eye's views of it, into
    out/<name>: `parts` maps a name to each part's points, normals,
    albedo and gloss (see ball and wall), `world` is the scene as Mitsuba
    sees it, and `focus` maps a name to each depth the eye focuses at.
    The plate is `plate_size` pixels square, as OPTICS has it, summed a
    tile at a time, through recording planes (sum_planes), and kept as
    the eye can see it; too big for memory, it is summed plane to plane
    instead, into the spectrum's file on disk (see planes.py). With
    shadows off, nothing casts shadows (see
    maps).

    Lit by the reference beam again, the hologram sends out the scene's
    light, and the rest of what it sends goes in other directions: the
    beam itself straight on, at the reference's angle, and the light about
    it and the twin image farther off still. Within the cone, where the
    eye can see,
    it is the scene's light alone, so that is what the views are made
    from. (A binary plate, printed, would add noise of its own.)

    Saves the spectrum of the plate's light, as kept (spectrum.npy, for
    the viewer), and a sheet of views (views.png)."""
    from .viewer import Look

    out = Path("out") / name
    out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    points, normals, albedo, gloss = (np.concatenate(a) for a in zip(*parts.values()))
    counts = ", ".join(f"{len(p[0])} on the {k}" for k, p in parts.items())
    print(f"{len(points)} points: {counts}", flush=True)
    light = maps(world, points, normals, albedo, gloss, shadows=shadows)
    alive = light.max(axis=(1, 2)) > 0
    print(
        f"what each sees and lights, {OPTICS.directions}x{OPTICS.directions} directions each, by Mitsuba: "
        f"{time.time() - start:.1f} s; {alive.sum()} points seen at all",
        flush=True,
    )

    start = time.time()
    rng = np.random.default_rng(1)
    phases = rng.uniform(0, 2 * math.pi, len(points))
    o = OPTICS
    n = plate_size // o.keep
    big = n * n * 8 > 6 * 2**30
    # Only the points seen at all, their maps no longer kept twice, and
    # in half precision (plenty for brightness), to leave memory for
    # carrying the plate's light.
    light = light[alive].astype(np.float16)
    if big:
        plate = planes.sum_by_groups(points[alive], light, phases[alive], plate_size, out / "spectrum.npy")
    else:
        plate = sum_planes(points[alive], light, phases[alive], plate_size)
    del light
    print(
        f"summed for {plate_size}^2 pixels {o.pixel:g} um apart on the GPU, light within "
        f"{math.degrees(math.atan(o.cone)):.0f} degrees: {time.time() - start:.1f} s",
        flush=True,
    )

    start = time.time()
    if big:
        Look.save(out / "spectrum.npy", None, o)
    else:
        spectrum_in_place(plate)
        Look.save(out / "spectrum.npy", plate, o)
    look = Look(plate, o.kept_pitch, cone=o.cone, reference=o.reference)
    print(f"its spectrum saved: {time.time() - start:.1f} s", flush=True)

    start = time.time()
    tiles = []
    for where, direction in VIEWS.items():
        for label, depth in focus.items():
            view = look.view(direction, PUPIL, depth)
            tiles.append((f"looking {where}, focused {label}", view))
    scale = look.scale(tiles[0][1])
    sheet([(t, look.image(v, scale, "")) for t, v in tiles]).save(out / "views.png")
    print(f"eye's views: {time.time() - start:.1f} s; wrote {out}/views.png and spectrum.npy")


def main():
    render("solid", {"ball": ball(), "wall": wall()}, scene(), FOCUS)
