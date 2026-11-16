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

from .magnifier import CONE, PLATE, PUPIL, WAVELENGTH, picture, replay, sheet

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


def render(name, parts, world, focus, plate_size=PLATE["size"]):
    """Makes the hologram of a scene and the eye's views of it, into
    out/<name>: `parts` maps a name to each part's points, normals,
    albedo and gloss (see ball and wall), `world` is the scene as Mitsuba
    sees it, and `focus` maps a name to each depth the eye focuses at.
    The plate is `plate_size` pixels square, PLATE's pitch apart.
    Saves the plate's light, lit by the reference beam again (replay.npy,
    for the viewer), and a sheet of views (views.png)."""
    import cupy as cp

    from .gpu import GpuEye, GpuPlate, SeenPoints

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
    size = dict(PLATE, size=plate_size)
    plate = GpuPlate(**size)
    seen = SeenPoints(**size, points=points[alive], maps=cp.asarray(light[alive]), cone=CONE)
    plate.light = seen.light(phases[alive])
    cp.cuda.Device().synchronize()
    plate.finish()
    print(f"summed on {plate.size}^2 pixels on the GPU: {time.time() - start:.1f} s", flush=True)

    start = time.time()
    lit = replay(plate).astype(np.complex64)
    np.save(out / "replay.npy", lit)
    eye = GpuEye(lit, plate.pitch, WAVELENGTH)
    tiles = []
    for where, direction in VIEWS.items():
        for label, depth in focus.items():
            view = eye.view(direction, PUPIL, depth)
            half = 0.44 * plate.size * plate.pitch
            tiles.append((f"looking {where}, focused {label}", picture(view, plate.pitch, half=half, shrink=4 * plate.size // PLATE["size"])))
    sheet(tiles).save(out / "views.png")
    print(f"eye's views: {time.time() - start:.1f} s; wrote {out}/views.png and replay.npy")


def main():
    render("solid", {"ball": ball(), "wall": wall()}, scene(), FOCUS)
