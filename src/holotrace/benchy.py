"""3DBenchy, the 3D printing test boat, as a hologram.

    holotrace benchy [--scale S] [--spacing D] [--turn A]
                     [--pixel P --cone C [--slab T]] [--plate W] [--wall]

makes it S times as big (default 1), on a plate S times as wide, with
its surface sampled into points D um apart (default solid.SPACING),
turned A degrees (default TURN). With --pixel and --cone, the plate's
pixels are P um apart and the scene's light spreads C degrees either
way (see solid.Optics.wide); without, the plate is as at first: 0.5 um
pixels, 7 degrees; --slab sets the slabs' depth, T um (see
solid.Optics). With --plate, the plate is W mm wide, whatever the boat's
size. The results go to out/benchy-<width>mm-<D>um, with -<C>deg-<P>um
added for wide plates, -boat<S> for a plate of its own width, and -wall
with --wall: a dim checkered wall just behind the stern, which the boat
hides in part and casts its shadow on. It reaches past the plate as far
as the light spreads on the way there, so that no edge of it is seen from
anywhere: it looks endless. (The planes are then wider than the plate:
see planes.py.)

The boat (scenes/3dbenchy, public domain since 2025), scaled to about
1.7 mm long, stands upright, turned 30 degrees to show its bow, its
middle on the plate, so that the plate cuts through it: the bow is in
front of the plate, the stern behind, against darkness. As in step 3
(solid.py), its surface is sampled into points, and Mitsuba works out
what each sees and is lit by: the boat hides and shadows parts of
itself, through the cabin's windows and doorways too. The views and the
plate's light go to out/benchy-... (see the viewer, viewer.py).
"""

import dataclasses
import math
import sys
from pathlib import Path

import numpy as np

from . import solid
from .meshes import read_stl, sample

STL = Path("scenes/3dbenchy/3DBenchy.stl")
LENGTH = 1700.0  # um
TURN = math.radians(30)
MIDDLE_Z = 0.0
ALBEDO, GLOSS = 0.85, 0.25
# The wall: how far behind the boat's hindmost point, and its squares'
# size, for a boat of LENGTH.
WALL_GAP, WALL_CHECK = 60.0, 150.0
FOCUS = {"on the bow": -500.0, "on the stern": 500.0}


def boat(scale=1.0):
    """The boat's triangles, in the scene, `scale` times as big: upright
    (the file's z is up, the scene's y), its length along x turned by TURN
    about the vertical, its middle at MIDDLE_Z (times scale)."""
    t = read_stl(STL)
    v = t.reshape(-1, 3)
    size = scale * LENGTH / (v[:, 0].max() - v[:, 0].min())
    middle = (v.min(axis=0) + v.max(axis=0)) / 2
    x, y, z = ((v - middle) * size).T
    # File (x along, y across, z up) to scene (x, y up, z toward the
    # viewer): a rotation, not a reflection, which would turn the faces
    # inside out. Then turned about the vertical.
    across = -y
    c, s = math.cos(TURN), math.sin(TURN)
    scene = np.column_stack([c * x + s * across, z, -s * x + c * across + scale * MIDDLE_Z])
    return scene.reshape(-1, 3, 3)


def world(triangles, wall=None):
    """The boat as Mitsuba sees it, and the wall, if any: how far behind
    the plate it is, and half its width."""
    import mitsuba as mi

    from .mesh import _mitsuba

    _mitsuba()
    mesh = mi.Mesh("boat", 3 * len(triangles), len(triangles))
    params = mi.traverse(mesh)
    params["vertex_positions"] = mi.Float(triangles.astype(np.float32).ravel())
    params["faces"] = mi.UInt32(np.arange(3 * len(triangles), dtype=np.uint32))
    params.update()
    scene = {"type": "scene", "boat": mesh}
    if wall:
        depth, half = wall
        at = mi.ScalarTransform4f().translate([0, 0, -depth]).scale([half, half, 1])
        scene["wall"] = {"type": "rectangle", "to_world": at}
    return mi.load_dict(scene)


def _option(name, default):
    if name in sys.argv:
        return float(sys.argv[sys.argv.index(name) + 1])
    return default


def main():
    global TURN
    scale = _option("--scale", 1.0)
    spacing = _option("--spacing", solid.SPACING)
    TURN = math.radians(_option("--turn", math.degrees(TURN)))
    if "--cone" in sys.argv:
        pixel, degrees = _option("--pixel", 0.25), _option("--cone", 24.0)
        solid.OPTICS = solid.Optics.wide(pixel, degrees)
        if "--slab" in sys.argv:
            solid.OPTICS = dataclasses.replace(solid.OPTICS, slab=_option("--slab", 0.0))
        wide = f"-{degrees:g}deg-{pixel:g}um"
    else:
        wide = ""
    o = solid.OPTICS
    triangles = boat(scale)
    points, normals = sample(triangles, spacing, np.random.default_rng(3))
    hull = (points, normals, np.full(len(points), ALBEDO), np.full(len(points), GLOSS))
    focus = {k: scale * v for k, v in FOCUS.items()}
    # The plate, S times 2048 um across, in whole kept pixels.
    width = _option("--plate", 2.048 * scale) * 1000
    size = o.keep * math.ceil(width / o.pixel / o.keep)
    name = f"benchy-{width / 1000:g}mm-{spacing:g}um{wide}"
    if "--plate" in sys.argv:
        name += f"-boat{scale:g}"
    if TURN != math.radians(30):
        name += f"-turn{math.degrees(TURN):g}"
    parts, wall = {"boat": hull}, None
    if "--wall" in sys.argv:
        depth = -triangles[..., 2].min() + scale * WALL_GAP
        half = width / 2 + o.cone * o.fade * depth + 2 * spacing
        wall = (depth, half)
        # Dim, so that the boat stands out against it.
        parts["wall"] = solid.wall(z=-depth, half=half, light=0.3, dark=0.1, spacing=spacing, check=scale * WALL_CHECK)
        focus["on the wall"] = depth
        name += "-wall"
        print(f"a wall {2 * half / 1000:.1f} mm wide, {depth / 1000:.1f} mm behind the plate", flush=True)
    solid.render(name, parts, world(triangles, wall), focus, plate_size=size)
