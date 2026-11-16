"""Computed holograms from ray-traced scenes.

Mitsuba traces the scene; the waves are summed separately.

    holotrace check       checks that Mitsuba works and can measure how
                          far rays go, which the waves' phases come from
    holotrace magnifier   step 1: a hologram of a magnifying glass, with
                          the lens traced exactly, no Mitsuba yet
    holotrace mesh        step 2: the same, the lens a mesh traced by
                          Mitsuba, against the exact one
    holotrace gpucheck    checks the GPU's sum against the CPU's

Add --gpu to magnifier or mesh to sum the light on an NVIDIA GPU.
"""

import sys

# Mitsuba's variants, best first: on the GPU (which needs an NVIDIA card
# of compute capability 7.5 or newer), on the CPU, then plain Python. All
# are monochrome: a hologram is recorded at one wavelength.
VARIANTS = ["cuda_ad_mono", "llvm_ad_mono", "scalar_mono"]


def use_best_variant() -> str:
    """Selects the fastest Mitsuba variant that works here, and returns it."""
    import mitsuba as mi

    for variant in VARIANTS:
        if variant not in mi.variants():
            continue
        try:
            mi.set_variant(variant)
            # The GPU variant only fails once it is used.
            mi.load_dict({"type": "scene"})
            return variant
        except Exception:
            continue
    raise RuntimeError("no usable Mitsuba variant")


def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else "check"
    gpu = "--gpu" in sys.argv[2:]
    if command == "magnifier":
        from .magnifier import main as magnifier

        magnifier(gpu)
    elif command == "mesh":
        from .meshed import main as meshed

        meshed(gpu)
    elif command == "gpucheck":
        from .gpucheck import main as gpucheck

        gpucheck()
    elif command == "check":
        check()
    else:
        sys.exit(__doc__)


def check() -> None:
    import mitsuba as mi

    variant = use_best_variant()
    print(f"Mitsuba {mi.__version__}, variant {variant}")

    # A ray straight at a unit sphere 5 units away should hit it 4 units out.
    scene = mi.load_dict({
        "type": "scene",
        "sphere": {"type": "sphere", "center": [0, 0, 5], "radius": 1},
    })
    ray = mi.Ray3f(o=mi.Point3f(0, 0, 0), d=mi.Vector3f(0, 0, 1))
    hit = scene.ray_intersect(ray)
    print(f"ray hit the sphere at distance {hit.t} (expected 4)")
