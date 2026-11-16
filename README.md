# holotrace

Computed off-axis (Leith–Upatnieks) holograms of ray-traced scenes, with
the subject on the image plane: on and around the plate.

The aim is the best hologram a computer can make, the test being that
optics in the scene still work: look through a recorded magnifying glass,
and what is behind it is magnified, sharp, and changes as you move.

## How it works

The hologram is an exact coherent sum, not a stereogram or hogels, which
blur fine angular detail. Each point of the scene sends a wave to every
point of the plate it can reach; the plate records their sum against a
reference beam.

1. **Ray tracing ([Mitsuba 3](https://www.mitsuba-renderer.org/)).** For
   each point of the scene: whether it can be seen from each direction,
   how bright it is that way, and, where its light passes through lenses
   or off mirrors, how long the light's path is (which sets its phase).
2. **Coherent sum.** For each tile of the plate, add up every point's
   wave: brightness × visibility × e^(ik·path length) / distance.
3. **Output.** Add the reference beam, encode the fringes (binary or
   grayscale), and write tiles for the printer.

A simulated eye, with a real pupil and focus, looks at the result to
check it before anything is printed.

## Plan

1. A tiny analytic test, no renderer: a fine pattern behind a lens, a
   plate a few millimetres across, and the simulated eye. It passes if the
   pattern looks magnified and sharp, and the magnification changes as the
   eye moves.
2. The same test with the scene traced by Mitsuba.
3. Scale up: tiling, real meshes and materials from Blender, then a real
   scene.
4. Choose the output medium, which fixes the pixel pitch and encoding.

## Running

```
uv run holotrace check        # Mitsuba works and can measure how far rays go
uv run holotrace magnifier    # step 1: a hologram of a magnifying glass
uv run holotrace mesh         # step 2: the lens as a mesh traced by Mitsuba
```

Results go to `out/`. Add `--gpu` to `magnifier` or `mesh` to sum the
light on an NVIDIA GPU with CuPy, many times faster:

```
uv run --extra gpu holotrace mesh --gpu
uv run --extra gpu holotrace solid      # step 3: a solid scene across the plate
uv run --extra gpu holotrace benchy     # 3DBenchy, the 3D printing test boat
uv run --extra gpu holotrace benchy --scale 5 --spacing 16   # a 1 cm plate, tiled, with recording planes
uv run --extra gpu holotrace benchy --scale 10 --spacing 32  # a 2 cm plate: about 1 min on an 8 GB GPU from 2016
uv run --extra gpu holotrace benchy --scale 20 --spacing 40  # a 4 cm plate: about 2.5 min, 10 GB of memory
# Wide plates: 0.25 um pixels, the scene's light within 35 degrees either way.
uv run --extra gpu holotrace benchy --plate 20.48 --scale 6 --spacing 12 --turn 15 --pixel 0.25 --cone 35 --slab 600
#   a 2 cm plate, the boat 1 cm long: about 20 min, the spectrum 13 GB on disk
# Plates too big for memory are summed plane to plane (see planes.py): groups of
# slabs carried a short way, then each group's plane to the plate in one go.
uv run --extra gpu holotrace benchy --plate 40.96 --scale 16 --spacing 32 --pixel 0.25 --cone 35 --slab 600
#   a 4 cm plate, the boat 2.7 cm long: the spectrum 54 GB on disk, and as much again while it is made
uv run --extra gpu holotrace view benchy --focus 0   # a movie: eye moving side to side, up and down
uv run --extra gpu holotrace view NAME --eye 300     # ... seen by a real eye 30 cm away, 4 mm pupil
uv run --extra gpu holotrace view NAME --eye 300 --source 0.02   # ... lit by a lamp 2 cm across, 1 m off: far less speckle
uv run --extra gpu holotrace speckle    # step 4 (parked): phases optimized against speckle
uv run --extra gpu holotrace gpucheck   # the GPU's sum against the CPU's
```

Mitsuba itself runs on the GPU only on NVIDIA cards of compute
capability 7.5 or newer (Turing onward); on older cards and other
machines it runs on the CPU, more slowly. The GPU sum (CuPy) works on
older cards too.
