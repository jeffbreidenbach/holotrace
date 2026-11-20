"""Computed holograms from ray-traced scenes.

Mitsuba traces the scene; the waves are summed separately.

    holotrace check       checks that Mitsuba works and can measure how
                          far rays go, which the waves' phases come from
    holotrace magnifier   step 1: a hologram of a magnifying glass, with
                          the lens traced exactly, no Mitsuba yet
    holotrace mesh        step 2: the same, the lens a mesh traced by
                          Mitsuba, against the exact one
    holotrace solid       step 3: a solid scene across the plate, hidden
                          surfaces and shadows from Mitsuba (GPU only)
    holotrace benchy      3DBenchy, the 3D printing test boat, as in step 3
                          (--scale S, --spacing D: see benchy.py)
    holotrace view NAME   a movie of the eye moving and refocusing, from
                          out/NAME/spectrum.npy (from solid or benchy;
                          --light laser or led: see viewer.py)
    holotrace compare NAME...
                          the scenes side by side, by laser and by LED
    holotrace encode NAME the plate's pattern for a writer, on its own
                          pixels: clear or opaque (a picture, or a GDSII
                          file), or 256 levels (see encode.py)
    holotrace testplate   two small plates to try a writer on: gratings,
                          ladders of thin lines, a staircase of levels,
                          small holograms (see testplate.py)
    holotrace speckle     step 4: the solid scene with each point's phase
                          optimized against speckle (GPU only; --check
                          checks its gradient)
    holotrace gpucheck    checks the GPU's sum against the CPU's

Add --gpu to magnifier or mesh to sum the light on an NVIDIA GPU.
"""

import os
import sys

# Mitsuba's variants, best first: on the GPU (which needs an NVIDIA card
# of compute capability 7.5 or newer), on the CPU, then plain Python. All
# are monochrome: a hologram is recorded at one wavelength.
VARIANTS = ["cuda_ad_mono", "llvm_ad_mono", "scalar_mono"]


def _gpu_can_trace() -> bool:
    """Whether there is an NVIDIA GPU new enough for Mitsuba (compute
    capability 7.5). Asked of nvidia-smi, not Mitsuba: on an older card,
    Mitsuba can hang starting up (see use_best_variant)."""
    import subprocess

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        ).stdout
        return any(float(line) >= 7.5 for line in out.split())
    except (OSError, ValueError, subprocess.SubprocessError):
        return False


_variant = None


def use_best_variant() -> str:
    """Selects the fastest Mitsuba variant that works here, and returns
    it. Mitsuba must not be imported before this is called.

    Dr.Jit, which Mitsuba's arrays are made of, looks for a GPU as it is
    imported, on a thread of its own, and on finding one too old says so.
    Saying so needs Python's lock; if the main thread, holding that lock,
    has meanwhile asked Dr.Jit for anything, it is waiting for that
    thread, and each waits on the other for ever: one core busy, nothing
    printed. Whether it happens is a matter of timing (about one start
    in five, here). So with no GPU it can use, Dr.Jit is told there is
    no CUDA to look for, which it notes in a line of its own, "Unable to
    load", and all is well."""
    global _variant
    if _variant:
        return _variant
    gpu = _gpu_can_trace()
    if not gpu:
        os.environ.setdefault("DRJIT_LIBCUDA_PATH", "no CUDA for Mitsuba (this GPU is too old for it)")
    import mitsuba as mi

    for variant in VARIANTS:
        if variant not in mi.variants() or (variant.startswith("cuda") and not gpu):
            continue
        try:
            mi.set_variant(variant)
            # The GPU variant only fails once it is used.
            mi.load_dict({"type": "scene"})
            _variant = variant
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
    elif command == "solid":
        from .solid import main as solid

        solid()
    elif command == "benchy":
        from .benchy import main as benchy

        benchy()
    elif command == "view":
        from .viewer import main as view

        view()
    elif command == "compare":
        from .viewer import compare

        compare()
    elif command == "encode":
        from .encode import main as encode

        encode()
    elif command == "testplate":
        from .testplate import main as testplate

        testplate()
    elif command == "speckle":
        from .speckle import main as speckle

        speckle()
    elif command == "gpucheck":
        from .gpucheck import main as gpucheck

        gpucheck()
    elif command == "check":
        check()
    else:
        sys.exit(__doc__)


def check() -> None:
    variant = use_best_variant()
    import mitsuba as mi

    print(f"Mitsuba {mi.__version__}, variant {variant}")

    # A ray straight at a unit sphere 5 units away should hit it 4 units out.
    scene = mi.load_dict({
        "type": "scene",
        "sphere": {"type": "sphere", "center": [0, 0, 5], "radius": 1},
    })
    ray = mi.Ray3f(o=mi.Point3f(0, 0, 0), d=mi.Vector3f(0, 0, 1))
    hit = scene.ray_intersect(ray)
    print(f"ray hit the sphere at distance {hit.t} (expected 4)")
