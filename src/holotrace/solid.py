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
from pathlib import Path

import numpy as np

from .magnifier import CONE, PLATE, PUPIL, REFERENCE, WAVELENGTH, sheet

# Lengths in micrometres.
BALL_Z, BALL_RADIUS = -150.0, 350.0
WALL_Z, WALL_HALF, CHECK = -600.0, 1100.0, 150.0
SPACING = 8.0
LIGHT = np.array([-0.5, 0.6, 0.8]) / np.linalg.norm([-0.5, 0.6, 0.8])
AMBIENT, GLOSS, SHINE = 0.15, 0.6, 40
# Each point's brightness is worked out for MAP x MAP directions.
MAP = 24
# Points closer to the plate than this are left out: their light would
# land on a pixel or two.
NEAREST = 2.0

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
    """Points on a grid `spacing` apart over a wall at z, `half` either
    way, checkered `light` and `dark` in squares `check` across."""
    g = np.arange(-half, half + spacing / 2, spacing)
    x, y = (a.ravel() for a in np.meshgrid(g, g))
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


def maps(world, points, normals, albedo, gloss, chunk=4000):
    """Each point's light, as bright in each of MAP x MAP directions
    toward the viewer (tangents from -CONE to CONE; rows along y): the
    amplitude, the square root of the power it sends that way. That is
    its brightness (matte, from the light where it is not in shadow, a
    little ambient light everywhere, and a highlight where glossy) times
    how much of it faces that way, or nothing where it faces away or
    something is in front of it."""
    t = np.linspace(-CONE, CONE, MAP)
    tx, ty = np.meshgrid(t, t)
    d = np.column_stack([tx.ravel(), ty.ravel(), np.ones(MAP * MAP)])
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    half = LIGHT + d
    half /= np.linalg.norm(half, axis=1, keepdims=True)
    out = np.zeros((len(points), MAP * MAP), np.float32)
    lift = 0.05  # off the surface, so as not to meet it
    for s in range(0, len(points), chunk):
        p, n = points[s : s + chunk], normals[s : s + chunk]
        a, g = albedo[s : s + chunk], gloss[s : s + chunk]
        # Lit, or in shadow.
        facing_light = n @ LIGHT > 0
        lit = facing_light.copy()
        lit[facing_light] = ~_blocked(
            world, p[facing_light] + lift * n[facing_light], np.tile(LIGHT, (facing_light.sum(), 1))
        )
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
    return out.reshape(len(points), MAP, MAP)


# The plate is worked out a tile this many pixels square at a time.
TILE = 4096
# Points at least this far from the plate (um) have their light gathered
# on recording planes first (see sum_planes); 0 sums all of it directly.
SLAB = 1000.0
# Points are kept at least this far (um) from their recording plane. A
# point's light is described by rays (its brightness in each direction),
# which is only right far enough from it, compared with how quickly its
# brightness changes with direction: with the cone about 0.1 rad across,
# some hundreds of um. Closer, the sums differ by some percent from the
# light summed straight onto the plate; 400 um keeps that to a few.
CLEAR = 400.0


def sum_planes(points, light, phases, size, slab=SLAB):
    """The plate's light (in main memory), `size` pixels square, from
    `points` with their brightness maps `light` and starting `phases`,
    worked out on the GPU a TILE square at a time, each tile finished
    before the next: nothing else the size of the plate is made, so
    plates can be as big as main memory holds them.

    Points within `slab` of the plate send their light to the tile
    directly. The rest are taken in slabs of that depth: each slab's
    light is summed onto a recording plane just on the plate's side of
    it, where it covers little and so costs little, over the patch of the
    plane the tile's light can come from (the tile and a margin as wide
    as light spreads on the way), and that patch's light carried on to
    the tile by its plane waves (see _carry). That step is exact: light
    crossing open space is the sum of its plane waves. But each point's
    light is described by rays, which is right only some way from the
    point (see CLEAR), so the result differs from summing straight onto
    the plate by a few percent. Each pixel's light is its own sum, so
    working tile by tile is exact."""
    import cupy as cp

    from .gpu import SeenPoints

    pitch = PLATE["pitch"]
    depth = -points[:, 2]
    if slab:
        slice_ = np.where(depth >= 0, np.floor(depth / slab), -np.floor(-depth / slab)).astype(int)
    else:
        slice_ = np.zeros(len(points), int)
    # Each group: its plane's distance behind the plate (0: the plate
    # itself), its points' indices, their positions as seen from the
    # plane, the margin light spreads by, and their windows (in pixels).
    groups = []
    for k in sorted(set(slice_)):
        mine = np.flatnonzero(slice_ == k)
        plane = 0.0 if k == 0 else np.sign(k) * (abs(k) * slab - CLEAR)
        seen_from = points[mine] + [0.0, 0.0, plane]
        margin = 0 if k == 0 else int(math.ceil(CONE * abs(plane) / pitch)) + 32
        px, py, pz = seen_from.T
        r = CONE * np.abs(pz)
        bounds = (
            np.floor((px - r) / pitch + size / 2), np.ceil((px + r) / pitch + size / 2) + 1,
            np.floor((py - r) / pitch + size / 2), np.ceil((py + r) / pitch + size / 2) + 1,
        )
        groups.append((plane, mine, seen_from, margin, bounds))

    plate = np.zeros((size, size), np.complex64)
    for y0 in range(0, size, TILE):
        for x0 in range(0, size, TILE):
            th, tw = min(TILE, size - y0), min(TILE, size - x0)
            tile = cp.zeros((th, tw), cp.complex64)
            for plane, mine, seen_from, m, (lo_x, hi_x, lo_y, hi_y) in groups:
                ox, oy, w, h = x0 - m, y0 - m, tw + 2 * m, th + 2 * m
                near = (hi_x > ox) & (lo_x < ox + w) & (hi_y > oy) & (lo_y < oy + h)
                if not near.any():
                    continue
                which = mine[near]
                seen = SeenPoints(**dict(PLATE, size=size), points=seen_from[near], maps=cp.asarray(light[which]),
                                  cone=CONE, tile=(ox, oy, w, h))
                patch = seen.light(phases[which])
                tile += patch if plane == 0 else _carry(patch, plane, m)
                del seen, patch
            plate[y0 : y0 + th, x0 : x0 + tw] = cp.asnumpy(tile)
            del tile
            cp.get_default_memory_pool().free_all_blocks()
    return plate


def _carry(patch, distance, margin):
    """The light a patch of a recording plane (on the GPU), `distance`
    behind the plate (in front, if negative), sends the tile `margin`
    pixels in from its edges: each of its plane waves moved on by its own
    phase (the angular spectrum method), keeping only those within CONE,
    as all the scene's light is, and within what the sampling can carry
    without aliasing (band-limited, after Matsushima). The margin is as
    far as light spreads on the way, so the tile's light is whole, and
    what wraps round the patch's edges in the transform stays outside it
    (overlap-save)."""
    import cupy as cp

    h, w = patch.shape
    pitch, lam = PLATE["pitch"], WAVELENGTH
    fft = 1 << int(math.ceil(math.log2(max(h, w))))
    f = cp.fft.fftfreq(fft, pitch).astype(cp.float32)
    fx, fy = f[None, :], f[:, None]
    s2 = (fx * fx + fy * fy) * lam * lam
    limit = 1 / (lam * math.sqrt((2 * distance / (fft * pitch)) ** 2 + 1))
    keep = (s2 < CONE**2 / (1 + CONE**2)) & (cp.abs(fx) < limit) & (cp.abs(fy) < limit)
    step = keep * cp.exp((2j * math.pi / lam) * distance * cp.sqrt(cp.maximum(0, 1 - s2))).astype(cp.complex64)
    region = cp.zeros((fft, fft), cp.complex64)
    region[:h, :w] = patch
    moved = cp.fft.ifft2(cp.fft.fft2(region) * step)
    return moved[margin : h - margin, margin : w - margin]


def replay_tiled(plate, ratio=2.0, strip=1024):
    """As replay() in magnifier.py, a strip of rows at a time, in place:
    the hologram of the plate's light against the reference beam (see
    Plate.hologram), lit by that beam again."""
    pitch, n = PLATE["pitch"], plate.shape[0]
    typical = math.sqrt(sum(float(np.sum(np.abs(plate[i : i + strip]) ** 2)) for i in range(0, n, strip)) / n**2)
    sine = REFERENCE / math.sqrt(1 + REFERENCE**2)
    k = 2 * math.pi / WAVELENGTH
    for i in range(0, n, strip):
        y = ((np.arange(i, min(i + strip, n)) - n / 2 + 0.5) * pitch)[:, None]
        beam = ratio * typical * np.exp(1j * k * sine * y)
        hologram = np.abs(plate[i : i + strip] + beam) ** 2
        plate[i : i + strip] = hologram * beam / np.abs(beam) ** 2
    return plate


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


def render(name, parts, world, focus, plate_size=PLATE["size"], slab=SLAB):
    """Makes the hologram of a scene and the eye's views of it, into
    out/<name>: `parts` maps a name to each part's points, normals,
    albedo and gloss (see ball and wall), `world` is the scene as Mitsuba
    sees it, and `focus` maps a name to each depth the eye focuses at.
    The plate is `plate_size` pixels square, PLATE's pitch apart, summed a
    tile at a time, through recording planes `slab` apart (sum_planes; 0
    for none), then replayed and transformed in place: the plate is the
    only big array. Saves the spectrum of the plate's light,
    lit by the reference beam again (spectrum.npy, for the viewer), and a
    sheet of views (views.png)."""
    from .viewer import Look

    out = Path("out") / name
    out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    points, normals, albedo, gloss = (np.concatenate(a) for a in zip(*parts.values()))
    counts = ", ".join(f"{len(p[0])} on the {k}" for k, p in parts.items())
    print(f"{len(points)} points: {counts}", flush=True)
    light = maps(world, points, normals, albedo, gloss)
    alive = (light.max(axis=(1, 2)) > 0) & (np.abs(points[:, 2]) >= NEAREST)
    print(
        f"what each sees and lights, {MAP}x{MAP} directions each, by Mitsuba: "
        f"{time.time() - start:.1f} s; {alive.sum()} points seen at all",
        flush=True,
    )

    start = time.time()
    rng = np.random.default_rng(1)
    phases = rng.uniform(0, 2 * math.pi, len(points))
    plate = sum_planes(points[alive], light[alive], phases[alive], plate_size, slab=slab)
    del light
    tiles = (-(-plate_size // TILE)) ** 2
    how = f"recording planes every {slab:g} um" if slab else "directly"
    print(f"summed on {plate_size}^2 pixels on the GPU, {tiles} tiles, {how}: {time.time() - start:.1f} s", flush=True)

    start = time.time()
    replay_tiled(plate)
    spectrum_in_place(plate)
    np.save(out / "spectrum.npy", plate)
    look = Look(plate)
    print(f"replayed, its spectrum saved: {time.time() - start:.1f} s", flush=True)

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
