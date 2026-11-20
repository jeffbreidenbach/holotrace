# holotrace: design and decisions

This document explains how holotrace works, why it is built the way it
is, and what each optimization bought. It is written for people and for
language models picking up the code: it states the conventions and
invariants that are easy to break, the numbers each decision was
checked against, and what is known not to work.

## 1. What it makes

holotrace computes **off-axis (Leith–Upatnieks) holograms of ray-traced
scenes recorded at the image plane**: the subject straddles the plate,
part in front and part behind, as in an image-plane hologram. The test
scene is 3DBenchy (`scenes/3dbenchy`, public domain), a 3D-printing test
boat with a cabin, windows, a doorway and a hollow interior, which
exercises hidden surfaces, self-shadowing and parallax.

The goal is the best hologram a computer can make, printable in the end.
The guiding test from the start was that optics inside a scene should
still work: look through a recorded magnifying glass, and what is behind
it should be magnified, sharp at the right depth, and change as you
move. A stereogram or hogel hologram, which records a set of 2D views,
cannot do that, so holotrace computes the **exact coherent sum**: every
point of the scene sends a spherical wave to every pixel of the plate it
can reach, with its true path length, and the plate records their sum.

The latest result is a 4.1 cm plate (163,840² pixels 0.25 µm apart)
holding a 27 mm boat whose light spreads ±35°, viewed by a simulated eye
30 cm away. With its light by rule it takes 11 min on one 8 GB graphics
card from 2016 with an NVMe disk; path traced, as it is unless asked
otherwise (§3.1), about a minute more, going by the 2 cm boat, which
has as many points (not timed as a whole). With a
checkered wall behind the boat that looks endless from every angle
(§6.5) it takes 46 min, path traced.

A 9.8 cm plate whose fringes are coarse enough for a laser writer (2 µm
apart at their finest, the light within ±7°) takes 5.5 min, and its
pattern on the writer's pixels 2 min more (§6.6).

## 2. Conventions

Anyone changing the code needs these.

| Thing | Convention |
|---|---|
| Units | Micrometers throughout. Wavelength `WAVELENGTH = 0.633` (HeNe red) in `magnifier.py`. |
| Geometry | The plate is the plane z = 0, centered on the axis. The viewer is at +z. A point's **depth** is −z: positive behind the plate, negative in front. |
| Pixels | Pixel index i sits at x = (i − n/2 + 0.5) · pitch (`coordinate` in `gpu.py`); grids are centered. |
| Cone | `cone` is the **tangent** of the widest angle the scene's light reaches the plate at; `Optics.sine` is its sine. Light outside the cone is never computed or kept. |
| Reference beam | Given as a sine (`Optics.reference`), tilted in y. |
| Phases | Each point gets a random starting phase from `np.random.default_rng(1)`, so runs are repeatable. |
| Kept light | What is stored is not the full-resolution plate but the scene's light at the plate, sampled every `Optics.keep` pixels (`kept_pitch`). See §5.2. |
| Spectrum file | `out/NAME/spectrum.npy` is the **unshifted 2D FFT** of the kept light. It is stored either whole, shape (n, n), or in blocks of columns, shape (n/B, n, B) with B = 512 (§6.4). `spectrum.json` beside it holds `pitch`, `cone`, `reference` and the plate's `pixel` (which `encode.py` writes its pattern on, §6.6). `viewer.Look` reads both layouts. |
| Global settings | `solid.OPTICS` is a module-level `Optics`; `benchy.py` replaces it from the command line. Code that needs it reads `solid.OPTICS` at call time (`planes.py` does too). |

## 3. The pipeline

```
scene (STL) ──sample──▶ points + normals
                          │
                Mitsuba 3 │  for each point and 32×32 directions:
                          │  visible? lit? how bright?   (solid.maps)
                          ▼
              brightness maps (float16) + random phases
                          │
            GPU point sum │  each point's wave onto a recording plane
                          │  near it                     (gpu.SeenPoints)
                          ▼
         carry to the plate by plane waves               (solid._carry,
                          │                               planes.py)
                          ▼
     spectrum of the kept light, on disk                 (spectrum.npy)
                          │
             viewer.Look  │  eye with pupil, focus, position, light source
                          ▼
                 views.png, movie-*.mp4
```

### 3.1 Points and brightness maps (`meshes.py`, `solid.maps`)

Surfaces are sampled into points spread evenly, about `spacing` µm
apart (`meshes.sample`). Evenly, not at random: random points leave gaps
and clumps that the eye sees as grain (§8.4). So four times too many are
scattered at random by area, and the most crowded are removed until the
wanted number is left (sample elimination, after Yuksel). A point is
crowded by each neighbor within 1.07 spacings, the more the closer.

- **Many at a time.** The method removes the most crowded point, one at
  a time. `meshes.thinned` removes many at once: a level comes down in
  steps, and at each step every point above it that is more crowded than
  all its neighbors goes. Removing every such point whatever its level
  left the points less even than random ones, because sparse places
  were thinned as fast as crowded ones.
- **Cost:** 36 s for the 1.9 million points of the 4 cm boat, from 7.6
  million scattered.
- **Result:** the nearest neighbor is 0.68 spacings away at the 1st
  percentile, where at random it is 0.06.

Each point is a light source whose brightness depends on direction. For each point and each of
`directions × directions` (32 × 32) directions across the cone, Mitsuba
3 answers three questions by ray casting:

- does the surface face that way;
- does anything stand between the point and the viewer that way (hidden
  surfaces, including through the cabin's windows);
- how much of the light source reaches the point. The light is a small
  disc, not a point, sampled by 16 shadow rays, so shadows have soft
  edges.

The brightness is matte shading plus a little ambient light plus a
glossy highlight, times how much the surface faces that direction. The
map stores the **amplitude**, the square root of the power. On the GPU,
each point's map is read bilinearly in the tangent of the direction to
each pixel (`seen_at` in `gpu.py`).

The shading is worked out where the rays are traced, on Mitsuba's
arrays (Dr.Jit, on every core), a chunk of 4000 points and their 4
million directions at a time. In NumPy it took five times as long as
the rays themselves: the 4 mm boat's 170,000 points took 12.8 s, and
now take 1.9 s; the 2 cm boat's 1.9 million took 150–174 s, and now
take 32 s. The maps agree to 6 × 10⁻⁶ (one direction in 52 million is
lit in one and not the other).

**The matte light, path traced.** The matte part of the boat's
brightness is the light falling on each point as Mitsuba's path tracer
finds it (`solid._falling`): the lamp's with shadows as soft as its
disc is wide, the sky's as far as the point can see the sky, and what
other surfaces send on, after as many as four bounces. At first it was
a rule, and with `benchy --rule` it still is: the lamp's light by how
squarely it falls, times how much of the lamp the point sees, plus the
same ambient light everywhere. The rule is quicker (see the cost,
below); step 3's scene (`holotrace solid`) still uses it.

- **The scene** (`solid.lit`): the lamp is a disc 0.05 rad across far
  off toward `LIGHT`, and the sky is the same all round. They are as
  bright as make the two ways agree in the open: a surface squarely
  facing the lamp gets 1 − `AMBIENT` from it and `AMBIENT` from the
  sky. The boat is matte with its own reflectance, 0.85; the wall is
  matte with the mean of its squares', 0.2.
- **How.** Each point is looked at from 1 µm above, straight down, 256
  times over. What a matte surface sends back is the light falling on
  it times its reflectance / π, so dividing gives the light. Looking at
  the point leaves it to the path tracer to aim at the small lamp.
- **Checked** on a flat sheet alone, where the answer is known (the
  lamp's light by its tilt, and the whole sky's): tilted 0°, 30° and
  60°, the light came to 0.7585, 0.4488 and 0.1501, and should be
  0.7582, 0.4486 and 0.1500.
- **Noise.** One point's light differs between runs by 5% where the
  lamp reaches it, and 1% where only the sky does. It adds no grain
  that can be measured: on the 2 cm boat with points 32 µm apart, the
  contrast under a 0.02 rad lamp was 0.270 by rule and 0.258 traced
  (§8.4's measure); on the 4 cm boat and its wall (§6.4), 0.297 and
  0.296.
- **Cost:** about 47 µs for each point traced, in the maps step. The
  sum is not touched. With every point traced, as at first, the 2 cm
  boat's 1.9 million points took 120 s instead of 30 s by rule; the
  4 mm boat's 170,000 took 11 s instead of 3 s; the 4 cm boat and
  wall's 5.0 million took 240 s instead of 78 s, of a run of 46
  minutes.
- **Only the points seen from somewhere are traced.** The rest send
  the plate nothing, whatever falls on them. Within ±35° two thirds of
  the boat's points are seen, and 87% of the boat and wall's; within
  ±7°, 42% of the boat's. The 2 cm plate at ±7° (739,000 points) takes
  38 s in all: it took 66 s with every point traced, and 29 s by rule.
  The 4 cm plate at ±7° (1.9 million) takes 113 s: 196 s, and 81 s by
  rule. The 2 cm boat at ±35° takes 3.9 min, its maps 85 s of that; by
  rule, 3.0 min. The maps are the same but for the path tracer's
  noise: on a 4 mm boat the same points and directions are seen, and
  the seen points' light is the same on average to 0.04%.
- **What it changes** in this scene is modest: corners and the shadowed
  side get the light they would, not a fixed share, and the views are 4
  to 6% brighter from light bounced between surfaces. On the 4 cm boat
  and wall, seen straight on under a 0.02 rad lamp and smoothed over
  300 µm, the view as a whole is 3% brighter. Its brightest tenth, the
  lit boat, is 10% brighter, and a twentieth of that by more than 27%:
  the deck around the cabin and the chimney's shaded side, which the
  lit surfaces beside them light. Its darkest tenth, the boat's shadow
  on the wall, is 8% darker, most where the wall is nearest the hull:
  the boat hides part of the sky there, and by rule the ambient light
  is the same everywhere.
- **What is still by rule:** the highlight, and so anything whose
  brightness changes with the direction it is seen from. Tracing that
  would take paths for each of a point's 1,024 directions, not one set
  for the point: one direction's light at 8 paths was 29% off, and the
  boat traces at 5.2 million paths a second, which makes hours for the
  4 cm boat. Not run.

Mitsuba runs on the CPU (LLVM variant): its GPU variants need compute
capability 7.5 or newer, so they are only tried when `nvidia-smi`
reports a new enough card.

**The start-up hang, and its cause.** On an older card Mitsuba used to
hang at start now and then, with no output and one core busy.

- **Cause.** Dr.Jit, which Mitsuba's arrays are made of, looks for a GPU
  on a thread of its own as it is imported. On finding one too old it
  prints a warning, which needs Python's lock. If the main thread,
  holding that lock, has meanwhile called into Dr.Jit, it is waiting
  for that thread, and each waits on the other for ever.
- **Reproduced:** a Dr.Jit call straight after `import drjit` hung in
  7 of 8 starts. In holotrace the call came a little later, so it hung
  only sometimes.
- **Cure** (`use_best_variant`): when the card cannot trace, Dr.Jit is
  told there is no CUDA library to look for (`DRJIT_LIBCUDA_PATH`),
  before Mitsuba is imported. It prints one line, "Unable to load", and
  goes on. 0 of 40 starts hung, with a Dr.Jit call straight after.
- **Dr.Jit's own cure** is in its main branch since 20 August 2026
  (its issue 524, the same two locks): what it has to say is kept
  until its lock is let go. No release has it yet: 1.5.0, of 7 August,
  is the newest and the one used here. Not tried here.
- **Mitsuba must be had from `mesh._mitsuba()`,** not imported first, or
  the setting comes too late.
- A rerun of a scene whose maps are kept does not load Mitsuba at all.

### 3.2 The point sum (`gpu.py`)

The heart of it is one CUDA kernel, `seen_gather`. Each GPU thread has
one pixel of the plate, and adds up the waves of the points that reach
it:

```
t      = (pixel − point) / depth                    (tangent, 2D)
extra  = depth · |t|² / (sqrt(1 + |t|²) + 1)        (path beyond depth, computed stably)
value  = map(t) · (1 + |t|²)^(−3/4) / |depth| · e^{i (k·extra + phase)}
```

**Precision.** GPUs are fast in single precision, about 7 digits, too
few for a phase of tens of thousands of radians. So the large part of
each path, k · depth, is folded into the point's starting phase on the
CPU in double precision, and the kernel only handles `extra`, which is
at most tens of micrometers.

**Which points reach a pixel.** The tile being worked out is cut into
cells 64 pixels square. For each cell, the points whose light reaches
it are listed; the lists are made on the GPU (`SeenPoints._lists`), at
most 6 million (point, cell) pairs at a time so that they fit beside a
big patch. A block of 32 × 8 threads lies within one cell, and each
thread loops over that cell's list.

**Why a thread per pixel.** The first batched kernel, `seen_many`, gave
a thread to each wave and added it to the plate with `atomicAdd`. It
was limited by memory traffic, not arithmetic: two atomic adds per wave.
A thread per pixel keeps its sum in registers and writes once.

**Lit windows.** A point's window on the plate was the whole disc its
cone reaches. But only 46% of a seen point's directions carry any
light: the rest face away or are hidden. So each window is shrunk to
the box of directions its map lights, one direction more either side
because the map is blended (`SeenPoints._lit`).

**Fast math.** The kernel is compiled with CUDA's fast math, whose
sines, square roots and divisions are a little less exact.

| On 150,000 random points, a 7168² tile | Time | Against `seen_many` |
|---|---|---|
| `seen_many` (a thread per wave) | 30.5 s | |
| `seen_gather`, lit windows | 7.2 s | 8 × 10⁻⁷ |
| + fast math (in use) | 4.0 s | 5 × 10⁻⁵ |

| Whole sum, same points and maps | Before | Now | Spectrum differs |
|---|---|---|---|
| 4 mm plate | 28.6 s | 4.4 s | 4.7 × 10⁻⁵ |
| 2 cm plate, on NVMe | 540 s | 138 s | 4.6 × 10⁻⁵ |

With a sharp cone (no fade), fast math puts a few pixels on the cone's
very edge on the other side of it: 63 of 4 million in one test. Without
fast math that test is bit for bit the same as `seen_many`.

**Batching.** One launch per point cost more than the sum itself;
launching all points at once made it about 3× faster, before the
thread-per-pixel kernel. There is a reverse kernel
(`seen_reverse_many`) for gradients, used by the parked speckle
optimizer (§8.1). It still gives a block of threads to each tile of
each point's window, since it sums over a point's pixels.

**Points in front of the plate** send light converging through it, as a
real image would. Their depth is negative, and the same formula gives
the right converging wave.

## 4. Correctness checks built into the project

Every structural change was checked against the simpler method it
replaced, on a case small enough for both. In order:

| Change | Checked against | Agreement |
|---|---|---|
| GPU sum | CPU sum (`holotrace gpucheck`) | under 1% |
| Lens traced as a Mitsuba mesh (64 rings, 98k triangles) | exact analytic lens | phase within 0.03 rad; views correlate 0.9997–1.0000 |
| Tiled plate | untiled plate, 4 mm | 2 × 10⁻⁷ |
| Viewing from the spectrum patch | full-size replay | point for point |
| Recording planes (§5.1) | direct sum, 1 cm | view correlation ≈ 0.98 |
| Kept light every 2 µm | full replay, 4 mm | correlation 0.992–0.995 |
| Deep slabs carried at the kept pitch | all-0.5 µm sum, 2 cm | correlation > 0.999, brightness within 0.2% |
| Per-slab patch sizes | common patch size | 0.8% (band-limit differences) |
| Half-precision maps | float32 maps | 0.013% |
| Plane-to-plane sum (§6) | tile-by-tile sum, 2 cm spectrum | 0.76% overall; views 0.04–0.06% |
| Plane-to-plane sum, 4 cm | the in-memory carry's views | 0.07–0.09% |
| Wide-plane carry with a cut-out plate (`planes._cut`) | direct NumPy FFTs, synthetic field | 0.09% (float32 at the cone's edge) |
| Slabs summed before transforming back; zeros never written | the earlier plane-to-plane result, 2 cm and 4 cm | views 0.04–0.09%, unchanged |
| Wide planes for only the groups that need them (§6.4, step 12) | every group on wide planes, 4 cm with the endless wall, spectrum | 0.65% overall; 0.03–0.13% under sine 0.55; 3.3% in the cone's last degree |
| Real-eye view (`Look.eye_view`) | brute force (whole plane × sphere's phase, one filter), 4 mm plate | one cell: 0.4–1.2%, correlation 0.9999; 2–3 cells: 2–7%, correlation 0.997–0.999 |
| A thread per pixel, lit windows, fast math (§3.2) | a thread per wave (`seen_many`), same points and maps, spectrum | 4 mm: 4.7 × 10⁻⁵; 2 cm: 4.6 × 10⁻⁵ |
| All points at once (`SeenPoints`) | point by point (`holotrace speckle --check`) | light 9 × 10⁻⁷, reverse 2 × 10⁻⁷ |
| Maps shaded on Mitsuba's arrays (§3.1) | shaded in NumPy, 4 mm boat | 6 × 10⁻⁶ |
| Scratch planes in half precision, only lit rows written (§6.4, steps 16 and 17) | the run before them, 4 cm spectrum | 2 × 10⁻⁴ inside sine 0.55; 0.5% in the cone's last degree |

The recording planes themselves differ from a direct sum by 1.2–1.6% in
the brightness of views at ±35° with planes 200 µm from their points,
and the later changes were held to stay below that.

**The field itself is less exact than that figure suggests.** Comparing
the plate's light (amplitude, by region of the cone) against planes
1200 µm from their points, on a 4 mm plate:

| Slab / clearance (µm) | Sum time | Inner half of the cone | 50–80% | 80–95% |
|---|---|---|---|---|
| 200 / 800 | 100 s | 2.9% | 3.3% | 8.4% |
| 600 / 400 | 60 s | 3.3% | 3.9% | 8.8% |
| 600 / 200 (in use) | 32 s | 4.1% | 4.8% | 10.4% |
| 300 / 200 | 18 s | 4.6% | 5.3% | 11.6% |
| 200 / 200 | 15 s | 4.9% | 5.7% | 12.4% |

- **The differences shrink only slowly with distance,** so the reference
  is not converged either.
- **Cause (confirmed):** each point's light was cut off sharply at the
  cone's edge on its recording plane, and a sharp edge ripples.
- **Cure (in use for wide plates):** each point's light now fades out
  between the cone and 1.13 times it (`Optics.fade`; as far as the kept
  pixels can carry), and the carry's cone mask does the cutting.

**Against a brute-force truth.** 3000 of the points, more than 100 µm
from the plate, summed straight onto the plate at 0.25 µm with a fade
far past the cone:

| Slab / clearance (µm) | Fade | Inner half of the cone | 50–80% | 80–95% | 95–100% |
|---|---|---|---|---|---|
| 600 / 200 | none | 3.6% | 4.2% | 8.7% | 30% |
| 200 / 1200 | none | 3.2% | 3.5% | 5.9% | 24% |
| 600 / 200 | 1.13 | 2.0% | 2.2% | 2.4% | 4.2% |
| 300 / 200 | 1.14 | 2.1% | 2.3% | 2.5% | 4.6% |

- **Both sums agree with it equally:** tile by tile and plane to plane
  give the same figures to 0.03%.
- **What is left** is mostly from points near the plate: with only
  points more than 400 µm away it is 1.3–1.6% (0.9% with planes 800 µm
  from their points). The truth is itself less sure for near points.
- **Cost:** the faded light covers 28% more of each plane, so the GPU
  sum takes that much longer at the same slab depth.
- **Thinner slabs no longer cost accuracy.** On the 4 mm plate, all the
  points: 600 µm slabs without the fade 32–36 s; with it 41–44 s; 300 µm
  slabs with it 23–25 s. Not yet timed on a big plate.

## 5. The core ideas that make it tractable

A 4 cm plate at 0.25 µm has 2.7 × 10¹⁰ pixels and the scene about 1.3
million visible points. Each point's ±35° cone reaches a disc of a few
hundred million pixels, so a direct sum would be nearly 10¹⁵ wave
evaluations. Four ideas brought that down to a couple of hours, and the
steps of §6.4 since to minutes.

### 5.1 Recording planes: sum points close, carry the rest by plane waves

Points are taken in slabs `slab` µm deep (300 µm at ±35°). Each slab's
points are summed onto a **recording plane** `clear` µm (200 µm) beyond
the slab's face nearer the plate, where each point's cone covers only a
small disc. That plane's light is then **carried to the plate by its
plane waves** (the angular spectrum method): FFT, multiply each plane
wave by e^{i k z cos θ}, inverse FFT. This is exact for light crossing
empty space.

Why the planes are not closer: a point's light is described by rays, its
brightness map by direction. That description is only right some
distance from the point, and the wider the cone, the closer that
distance is. At 24°, 200 µm differs from the direct sum by about 1.5%
and 400 µm by about 0.6%; at 7°, 400 µm still differed by a few
percent. `Optics.wide` shrinks slabs and clearance as the cone grows,
with floors of 200 µm.

Details that matter:

- **Band limit (Matsushima).** The transfer function is band-limited to
  what the patch's sampling can carry without aliasing,
  1 / (λ · sqrt((2z / (N · pitch))² + 1)), and masked to the cone.
- **Overlap-save margins.** A tile gathers each slab's plane over the
  tile plus a margin as wide as light spreads on the way
  (cone · |distance| / pitch + 8 pixels). After the carry it keeps only
  the middle, so what wraps around the FFT stays in the margin.
- **FFT sizes.** Patches are padded to sizes with small prime factors
  (`cufft.next_fast_len`). Sizes with large prime factors made cuFFT take
  far more memory. Sizes with no factors but 2 and 3 are faster again: a
  13,824² patch transforms in 65 ms, a 13,608² one in 92 ms. The big
  plates' tiles are padded to one of those when it is within 4%
  (`planes._patch_len`). The tile-by-tile sum is not: a different patch
  size moves the cone's sharp cut (§6.4), and its results would change
  by 2% in the cone's last degree for no gain at that size.

### 5.2 Keep only what the eye can see

Lit by its reference beam again, an ideal hologram sends out three
things: the reference beam itself straight on, the scene's light, and
the twin image. The reference angle puts the other two outside the
scene's cone. **Within the cone, what leaves the plate is just the
scene's light**, so that is all that needs to be kept, and it needs a
sample only every λ / (2 · sine) µm, the Nyquist spacing for the cone.

At ±35° that is 0.55 µm, so the kept light is sampled every 0.5 µm, every
2nd pixel of the 0.25 µm plate. That is 4× less data than the plate.
The full-resolution plate with its fringes is never formed.

**The pixel size bounds the cone.** The finest fringe is
(reference sine + cone sine) / λ = (0.654 + 0.574) / 0.633 ≈ 1.94 per µm.
That just fits under the 2 per µm a 0.25 µm pixel can carry; `Optics.wide`
raises an error when it cannot.

### 5.3 Tiles: never hold the plate

The plate is worked out a tile at a time on the GPU. Each tile takes
only the points whose light reaches it and, from each slab, only the
patch of that slab's recording plane its light can come from. Each
pixel's light is its own sum, so tiling is exact (2 × 10⁻⁷ against an
untiled plate).

### 5.4 View from the spectrum, not the plate

An eye far away takes in only plane waves within its pupil's small cone
around the direction it looks from: a small patch of the light's
spectrum. So the spectrum of the kept light is computed once, and each
view is built from just the patch its pupil selects, moved back to the
focus depth by phase. That costs a few hundred pixels square for a 1 cm
plate, and the cost doesn't grow with the plate (`viewer.Look.view`).

## 6. Scaling up: the optimization history

Each step below solved the bottleneck the previous one exposed. Times
are on an 8 GB graphics card from 2016, about 32 GB of RAM, and a SATA
SSD, except where NVMe is named.

### 6.1 From 2 mm to 4 cm at ±7°

| Plate | What changed | Sum time |
|---|---|---|
| 4 mm, 8192² | batched GPU kernel | 75 s |
| 1 cm, 20480² | tiles; viewer from the spectrum | 6.5 min |
| 1 cm | recording planes in 1 mm slabs | 38 s (from 320 s) |
| 2 cm, 40960² | tile pipeline: only each tile's patch of each plane | 3.25 min, 19 GB |
| 4 cm, 81920² | keep only the cone's light, every 2 µm | 19 min, 16 GB |
| 4 cm | deep slabs summed and carried at the kept 2 µm; 8192 tiles; boat without its wall | 68 s (from 987 s); 2.6 min total |
| 4 cm | a thread per pixel in the point sum, and the maps shaded on Mitsuba's arrays (§3) | 21 s; 81 s total, 37 s of it spreading the points evenly |
| 4 cm | the light path traced, for the points that are seen (§3.1) | 23 s; 113 s total, 44 s of it the maps |

### 6.2 Wide viewing angle: ±35° at 0.25 µm

±7° was too narrow to look around the boat, so `Optics.wide(pixel,
degrees)` sets everything from two numbers:

- **reference** = cone sine + 0.08: just outside the cone;
- **keep** = the largest whole number of pixels within the Nyquist spacing;
- **slab and clearance** scaled down with the cone, with floors of 200 µm;
- **32 × 32 directions** per map, about 2.2° apart at ±35°.

Every point now goes through a recording plane; none is summed straight
onto the plate.

At ±35° the deep slabs' margins become large, around 15,000 pixels for a
boat ±10 mm deep. Patches outgrew the GPU (`GPU_PATCH` = 15,360), and
then main memory.

- **In-memory carry (since removed).** Patches up to 44,000² were built
  in RAM a strip of rows at a time, FFT'd by rows on the GPU, then by
  columns a block at a time. It worked, but the waste grew as
  ((tile + 2 · margin) / tile)², up to about 10× for the deepest slabs.
  The 27 mm boat needed 26 GB of RAM plus 3 GB of swap.
- **Per-slab patch sizes.** Each slab got its own margin, so shallow slabs
  went back to the GPU. That made the 27 mm boat fit at all. 0.8%
  difference.
- **Half-precision maps.** 2.5 GB instead of 5 GB; 0.013% difference.
- **Spectrum on disk.** 81,920² complex64 is 54 GB, held as a NumPy
  memmap. `spectrum_in_place` transformed it rows then columns, which took
  26 min, mostly column reads that weren't contiguous.

| Run | Sum | Spectrum | Total |
|---|---|---|---|
| 2 cm, 1 cm boat | 878 s | 75 s | ~16 min |
| 4 cm, 2 cm boat (scale 12, turned 15°) | 76 min | 26 min | ~1 h 45 min |
| 4 cm, 27 mm boat (scale 16, at 30°) | 92 min | 26 min | ~2 h |

### 6.3 Choosing the boat's size: no clipping at the edges

The boat should be as large as possible without any of it falling
outside the plate's window when viewed from the edge of the cone. The
criterion: for every direction at the cone's edge, each point's ray to
the plate must land within the plate,
|x + z · tan θ · dₓ| and |y + z · tan θ · d_y| ≤ half the plate's width.

At a 30° turn on a 4.1 cm plate, the largest scale is 19.3; we chose 16
(27 mm long) for margin. A real eye 30 cm away sweeping ±17–19 cm across
and up and down then never clips.

### 6.4 Plane to plane (`planes.py`), the current method

**Problem.** In the tile-by-tile method, the margin for a slab is its
whole distance to the plate. The deeper the scene, the more of each
patch is computed only to be thrown away, and past some depth the
patches don't fit in memory at all.

**Design.** Plates too big for memory (more than 6 GB of kept light)
go through `planes.sum_by_groups`:

1. **Group the slabs.** Each group is no deeper than lets its slabs reach
   a plane in the group's middle with a margin of at most 1/8 of the GPU
   patch: about 2.7 mm at ±35° and 0.5 µm. The planes of the slabs either
   side of z = 0 lie beyond the plate, so they are not in depth order;
   grouping uses each group's span (max − min), not its first plane.
2. **Make each group's middle plane, tile by tile on the GPU**, carrying
   each slab only from its recording plane to the middle: short
   distances, small margins, no host-memory patches.
3. **Transform the plane by rows as it is made.** A band of rows, at most
   6 GB in RAM, is FFT'd along x on the GPU and written to a scratch file
   in column-block layout (n/B, n, B), B = 512. Every later access is one
   contiguous block. (Since step 16 the file is in half precision.)
4. **Carry the whole plane to the plate in one step.** For each column
   block: FFT along y on the GPU, multiply by the transfer function for
   the group's distance (cone mask, band limit, phase), and **add it into
   the spectrum file**.

   The output wanted is the plate's spectrum, and the carry is a
   multiplication in that domain. So the plane is never transformed back,
   and the separate 26 min spectrum pass is gone:
   **spectrum = Σ over groups of H(distance) · FFT(group plane)**.
5. **Wrap-around.** The plate-wide transform is periodic. If all the
   scene's light lands on the plate, which the no-clipping size rule
   (§6.3) guarantees, nothing wraps. Otherwise planes are made wider
   than the plate, by enough that what comes back round falls short of
   it. Their spectrum is summed on that wider domain, transformed back,
   the plate cut out, and transformed again (`_cut`; see step 12).

6. **Sum a tile's slabs before transforming back** (`planes._tile`). Each
   slab's light is summed straight into a patch with the group's deepest
   margin, transformed, and moved on to the middle plane; the slabs'
   spectra are added, and one inverse transform serves them all. S slabs
   cost S + 1 transforms, not 2S. The patch, the running sum and the
   transform's workspace must fit the GPU together: three patch-sized
   arrays, which is why the transform's plan cache is cleared after each
   slab.
7. **Never write zeros.** All the light is within the cone, a circle in
   the spectrum covering 65% of it. Blocks of columns outside the circle
   are skipped, and within a block only the rows inside it are read or
   written (`planes._cone`). Bands of a plane with no light are not
   written at all. The files are sparse: what is not written reads as
   zeros and takes no room. Each group gets a new scratch file so this
   holds. The 4 cm spectrum takes 33 GB on disk, not 51.
8. **The carry step in one pass** (`solid.moved`). Moving a patch's
   plane waves on was eight or so array operations, and took longer than
   the transform itself (0.19 s against 0.15 s for a 15360² patch). It
   is now one GPU kernel. Its phase is the distance's own, kept to a
   whole turn in double precision, plus what the wave's slant takes off,
   so single precision holds it to a thousandth of a radian.
9. **Several groups a pass.** The disk's sustained writing (0.5–1 GB/s
   here) limits the carry. As many groups' planes are made as the disk
   has room for, then one pass over the spectrum adds them all: the
   4 cm spectrum is rewritten twice, not seven times.
10. **Faded light and 300 µm slabs** (§4). The fade costs 28% more point
    work at a given slab depth; thinner slabs win it back. Each slab
    costs a transform a tile, about 16 s a group at 4 cm, so the best
    depth there is 300–400 µm, not thinner.
11. **Brightness maps kept on disk** (`solid.seen_maps`): a rerun of the
    same scene skips Mitsuba (then 5 minutes, and its start-up hang;
    now half a minute by rule, a minute and a half path traced, and
    no hang: §3.1).
12. **Wide planes only where needed.** A wide plane costs its area: at
    4 cm with the endless wall (§6.5) the planes are 114,688² against
    the plate's 81,920², twice the work and about 90 GB of scratch each.
    So each group's need is worked out on its own: how far its slabs'
    light lands from the axis (their points' offset, plus the cone's
    spread to the middle plane and from there to the plate), plus half
    the plate. Groups that need more than the plate's width are summed
    first, on planes of the widest need, into a scratch spectrum; the
    plate is cut out of that into the result; the rest are then added at
    the plate's own width. With the wall, one group of seven is wide.
13. **The cut, a few rows at a time** (`planes._cut_rows`). Cutting the
    plate out transforms the wide sum back along columns, block by
    block, then along rows on the GPU 256 rows at a time (a whole band
    of rows at that width does not fit an 8 GB card), keeps the plate's
    part, and transforms it again at the plate's width.
14. **A thread per pixel in the point sum** (§3.2). The point sum was a
    third of making the planes, about 5.5 minutes at 4 cm. It now takes
    67 s.
15. **Faster transform sizes** for the tiles' patches (§5.1).
16. **Scratch planes in half precision.** With the point sum fast, the
    disk was over half the run, and most of its traffic was the groups'
    planes: 49 GB each, written once and read once. They are now kept
    as two float16 a sample. Each run of 256 rows is divided by a scale
    of its own, so that its largest part is 30,000 (half precision
    holds up to 65,504); the scales, one a row, stay in memory and are
    put back as the plane is read (`planes._plane`, `_add_carried`).
    The spectrum itself stays in single precision.
17. **Only the rows light reaches.** Within a band, rows beyond every
    point's window and the margin its slab's light spreads by are
    neither transformed nor written (`planes._tile` reports the rows).
    Before, every row of a band with any light was.

    With steps 16 and 17 the 4 cm boat's seven planes take 66 GB on
    disk, not 341 GB. All seven then fit on the disk at once, so one
    pass adds them all and the spectrum is written once.

**Results.**

| 4 cm, 27 mm boat | Sum | Total with maps | Peak RAM |
|---|---|---|---|
| In-memory carry (removed) | 92 min + 26 min spectrum | ~2 h | 26 GB + 3 GB swap |
| Plane to plane, SATA SSD | 105 min | ~1 h 50 min | ~4 GB |
| + steps 6 and 7, on NVMe | 35 min (24 min making planes, 11 min carrying) | ~40 min | ~4 GB (RSS is higher: page cache) |
| + steps 8 to 11 (faded, 300 µm slabs) | 24 min (16.5 min making planes, 7.5 min carrying) | ~29 min, 24 min on a rerun | ~4 GB |
| + even points (§3.1), steps 14 and 15 | 20.9 min (9.4 min making planes, 11.4 min carrying) | 22 min | |
| + steps 16 and 17 | **9.9 min** (6.8 min making planes, 3.0 min carrying) | **11.1 min** | |
| With the endless wall, every group on wide planes | 99 min + 8 min cut | ~2 h | |
| With the endless wall, steps 12 and 13 | 87 min (38 min the wall's plane, 4.5 min its carry, 16 min the cut, 28 min the boat's groups) | ~1.5 h on a rerun | |
| With the endless wall, steps 14 to 17 | **43 min** (9.5 min the wall's plane, 4 min its carry, 15 min the cut, 14 min the boat's groups) | **45 min** | |
| The same, its light path traced (§3.1) | 41 min (9.4 min the wall's plane, 4 min its carry, 15 min the cut, 13 min the boat's groups) | **46 min** | |

- **All but the last row were timed with the light by rule** (now
  `--rule`). Path traced, only the maps take longer (§3.1): in the last
  row 4.0 min, not 1.3, with every one of its points traced.

- **The 20.9 min run carried in three passes,** not two: 229 GB of disk
  was free, room for three 49 GB planes at a time. Making the planes
  is the like-for-like part: 16.5 min before, 9.4 min after.
- **2 cm:** 878 s by GPU tiles; 1066 s plane to plane; 892 s with steps 6
  and 7 on SATA. On NVMe: 540 s with steps 8 to 13, 138 s with step 14,
  113 s with steps 16 and 17 (131 s on SATA). The whole run from
  scratch takes 3.0 min on NVMe and 3.3 min on SATA, by rule; path
  traced, 3.9 min on NVMe.
- **Accuracy:** views agree with the old method to 0.04–0.09% throughout.
- **Steps 16 and 17 against the run before them** (4 cm spectrum, same
  points): 2 × 10⁻⁴ inside sine 0.55, which is half precision's (a
  small plate with every row lit differs by 1.7 × 10⁻⁴ throughout), and
  0.5% in the cone's last degree. That last is step 17: the cone's
  sharp cut leaves a faint ripple beyond where light reaches, and the
  rows that held only that ripple are no longer kept. The same region
  already differs by a few percent between grids (next item).
- **A sharp cone on different grids.** Sums made on planes of different
  widths place the cone's sharp cut slightly differently, and differ by
  a few percent in its last degree (sine 0.565–0.574) and well under
  0.5% inside sine 0.55. It is not wrap-around: the difference is not
  gathered at the plate's edges.
- **With the wall, the cut is now the largest part:** 15 of 43 minutes,
  and steps 14 to 17 do not touch it. The wide sum it works on is still
  in single precision, 68 GB of it. §12 has a way to do without the
  wide planes and the cut altogether.
- **Where the time goes now** (4 cm without the wall, each part timed,
  in a run of 670 s made just before the last small saving):

  | Part | Time |
  |---|---|
  | Point sum | 67 s |
  | Tiles' transforms and steps | 95 s |
  | Other tile work (maps to the GPU, clearing and adding patches) | 79 s |
  | Bands: tiles off the GPU, row transforms, writing the planes | 211 s |
  | The carry: reading the planes, column transforms, writing the spectrum | 215 s |

  Nearly two thirds is moving data: between GPU and memory, and to and
  from the disk, which reads at 1.3–1.5 GB/s through the file system.
  A run writes about 100 GB (it was about 400 GB).
- **Tried and left out:** recording planes closer than 200 µm (error up
  from 2.1% to 2.6% at 100 µm); slabs thinner than 300 µm (slower at
  4 cm).
- **Results on another disk:** `HOLOTRACE_OUT=/path` puts results (and
  the scratch plane) there; the viewer reads the same variable.
- **Scaling:** cost now grows with plate area × number of groups, not
  depth squared. Depth is no longer limited by memory.

**Disk needed:** the spectrum plus one scratch plane in half precision,
1.5 × (n² × 8 bytes), at the least; with more, more groups go in each
pass. Wide planes need more: the 4 cm plate with the endless wall took
174 GB at most while it was made.

The plate that fits, worked out from that (not run):

| Viewing cone | Kept pitch | Plate for 91 GB free | For ~160 GB | For ~670 GB |
|---|---|---|---|---|
| ±35° | 0.5 µm | 4.3 cm | 5.7 cm | 11.5 cm |
| ±24° | 0.75 µm | 6.4 cm | 8.5 cm | 17 cm |
| ±15° | 1.0 µm | 8.4 cm | 11.4 cm | 23 cm |

The disk matters: moving from the SATA SSD to NVMe cut the 4 cm run by
well over half, before steps 16 and 17 made the planes small. (At 2 cm
the SATA disk now costs 18 s of a 3 minute run; 4 cm on SATA has not
been rerun.) 10 cm at ±35° was estimated at 7–9 h on that GPU with
enough NVMe, before steps 14 to 17 more than halved the 4 cm run. Not
run.

### 6.5 The endless wall (`benchy --wall`)

A checkered wall stands behind the boat, which hides part of it and
casts a shadow on it. A wall the size of the plate shows its edges as
soon as the eye moves off the axis, so the wall reaches past the plate
as far as the light spreads on the way there:

half its width = half the plate + cone · fade · depth + 2 · spacing.

At 4 cm that is a wall 56.6 mm wide, 9.8 mm behind a 40.96 mm plate:
3.1 million points on the wall beside the boat's 1.9 million, 4.4
million of them seen at all.

- **Checked by eye:** from 32° to the side the wall fills the view. From
  33° toward a corner one corner is dark; that is the plate's own far
  corner beyond the 35° cone, not the wall's edge.
- **Maps in half precision from the start** (`solid.maps`): five million
  32 × 32 maps are 10 GB as float16; as float32, with the copies made on
  the way, they did not fit in memory.
- **The wall's points are not thinned** like the boat's (§3.1): they
  are a grid, each point moved at random within its square
  (`solid.wall`), which is already nearly as even. On a flat sheet such
  a grid's contrast under a 0.04 rad lamp, averaged over 12 sets of
  phases, was 0.18 (0.044 after the retina's blur), against 0.41
  (0.195) at random.
- **Its light lands beyond the plate,** which is what steps 12 and 13 of
  §6.4 are for.

### 6.6 A plate for a laser writer, and its pattern (`encode.py`)

A laser writer's finest lines are about 1 µm, so its plate has a narrow
cone again. `benchy --scale 48 --spacing 64 --pixel 0.4 --cone 7` makes
a 98.3 mm plate, 245,760² pixels 0.4 µm apart, the boat 8 cm long: the
scene's light within ±7°, the reference beam at 11.6° (`Optics.wide`),
so fringes from 1.96 to 7.9 µm apart. Its light is kept every 2.4 µm,
40,960² samples: a quarter of what the 4 cm plate at ±35° keeps. On
NVMe:

| Step | Time |
|---|---|
| The boat's surface as 4.26 million points, 64 µm apart | 84 s |
| Path-traced maps; 1.78 million points seen | 124 s (with a short test running beside it) |
| The sum, plane to plane: 59 slabs, one group | 113 s |
| The preview | 4 s |
| The whole run | 5.5 min; 20.6 GB of memory at most; the spectrum 8.6 GB on disk |

Two things failed at this width, both in the viewer, both for want of
GPU memory:

- **The run's preview.** `Look.view` covers the whole plate, four
  samples across to each of the spectrum's that the pupil reaches:
  25,000² samples for the preview's pupil of 0.04. The preview's pupil
  is now the widest the GPU holds a view for (`Look.widest_pupil`: 0.020
  for this plate), and `holotrace view` and `compare` narrow theirs the
  same way.
- **The real eye's view.** From 60 cm, `eye_view`'s 5 × 5 cells were
  19,200² samples each. It now takes as many cells as it needs for the
  GPU to hold one (8 × 8 here for an eye on the axis, 12,288² each):
  17–24 s a view. Plates up
  to 4 cm seen from 30 cm are done as before (their cells are 7,200²
  samples, 8,640² by lamp light).

**The pattern.** A run keeps the plate's light as the eye can see it,
without the reference beam (§5.2); a writer needs the pattern on its
own pixels. `holotrace encode NAME` transforms the spectrum back to the
plate's light (a scratch file on disk, as big as the spectrum whole),
makes it as fine as the pixels a tile at a time on the GPU (the tile's
transform padded with zeros), and sets each pixel against the reference
beam. The files are plain pictures of the plate, PBM and PGM:

- **Chrome on glass** (clear or opaque): clear where
  cos θ > sqrt(1 − a²), θ the light's phase against the beam's and a
  its strength thereabouts, 1 where the plate is brightest. The fringes
  are half clear where the light is brightest, thinner where it is
  dimmer, gone in the dark.
- **A relief in resist** (`--levels`, 256 of them): how long each pixel
  holds the light back. That is the light's phase against the beam's, a
  sawtooth across each fringe, made shallower where the light is weak:
  a sawtooth m of a wave deep sends on sinc(1 − m) of the light.

For the 98.3 mm plate, on 0.4 µm pixels:

| | Time | File |
|---|---|---|
| The plate's light from its spectrum (12.5 GB of scratch) | 53–57 s | |
| Clear or opaque | 71 s | 7.0 GB |
| 256 levels | 104 s | 56 GB |

- **Tiles' margins.** A tile's light made finer is off near its edges,
  so 64 kept samples either side are worked out and thrown away. A tile
  made with that margin differed from the same tile made with a margin
  seven times as wide by 0.29% of the light (rms), 0.45% within 8 kept
  samples of its edge.
- **Read back.** A 4.9 mm plate's two files, read back, lit by the
  reference beam and viewed, show the boat as the scene's own light
  does (a prototype, not in the code): the files are laid out as
  `encode.py` says.

**The light's strength has to be in the pattern.** The scene stands at
the plate, so how bright each part of it looks is how strong the light
is there. The first version made every fringe half clear, whatever the
light's strength, and on one tile at the plate's middle that looked
good: 7.4% of the beam became the scene's light. Over a whole plate it
is useless. Viewed, a 4.9 mm plate with the same optics showed the boat
as a flat shape in a haze, and so did a relief of the light's phase
alone. Replayed over that whole plate (a prototype, each pixel a point;
other light within the cone as a share of the scene's, as drawn and as
a 0.8 µm spot would write it):

| Pattern | Other light | With the spot |
|---|---|---|
| Chrome, every fringe half clear | 276% | 279% |
| The same, opaque where the light thereabouts is under a tenth of the brightest | 30% (still a flat shape) | 31% |
| Chrome, fringes as wide as each pixel's own light is strong (`speckle.py`) | 26% | 72% |
| Chrome, fringes as wide as the light thereabouts is strong (the code's) | 26% | 41% |
| Relief, the light's phase alone | 267% | 272% |
| Relief, shallower where each pixel's own light is weak (the code's) | 7% | 7% |
| Relief, shallower where the light thereabouts is weak | 14% | 16% |

The strength thereabouts is the light's mean power over 8 × 8 kept
samples, smoothed over the cells next to it. Each pixel's own strength
follows the light better, speckle and all, but as line widths it asks
for lines too thin to write wherever the speckle is dim; as depths it
asks nothing a spot cannot give.

**Replayed** (`encode NAME --check`): both patterns over up to a hundred
tiles spread across the plate, each pixel a square on a grid no coarser
than 0.2 µm, lit by the reference beam. How much of the beam becomes
the scene's light (most of this plate is dark), and how much other
light comes out within the cone, as a share of the scene's:

| Pixels | Chrome | Written with a 0.8 µm spot | Relief | Written with a 0.8 µm spot |
|---|---|---|---|---|
| 0.8 µm | 0.84%, 55% | 0.75%, 59% | 4.2%, 16% | 2.0%, 19% |
| 0.4 µm | 1.07%, 24% | 0.72%, 48% | 5.3%, 6% | 2.3%, 7% |
| 0.2 µm | 1.13%, 21% | 0.81%, 40% | 5.6%, 5% | 2.3%, 5% |

The spot is a guess at a writer's, not a measurement of one: a Gaussian
0.8 µm wide at half its height, the chrome clear where it gets half the
full dose or more, the relief as deep as its dose. The check takes
1.5 min after the plate's light.

- **A relief is five times as bright as chrome and far cleaner** (three
  times as bright, written with the spot), and the spot costs it
  brightness, not cleanness. Chrome's thin fringes, where the scene is
  dim, are what the spot spoils.
- **Pixels half as wide as the finest line are about fine enough.** The
  fringes' places carry the scene, so the pixels must be finer than the
  lines: 0.8 µm pixels more than double the other light. 0.2 µm pixels
  help chrome a little (48% to 40% with the spot) for four times the
  file.
- **Much of the other light is the pattern's second order.** A pattern
  that is not the light itself also sends out light with twice the
  scene's phase, spread twice as wide, about the mirror image of the
  reference beam's direction. With the beam only just outside the cone
  (`Optics.wide`) that overlaps the cone, and shows as a faint second
  outline in the views. On the 4.9 mm plate, 0.2 µm pixels, moving the
  beam out to three times the cone's sine took the relief's other light
  from 5% to 0% and chrome's from 19% to 12%; but the fringes are then
  1.3 µm apart at their finest, and written with the 0.8 µm spot chrome
  was at 80%.
- **A narrower cone is cleaner.** A 4.1 mm plate at ±4.5° (fringes
  2.7 µm apart at their finest): chrome 15%, 20% with the spot; relief
  2% either way.

**Wide plates go through the same code,** as an electron-beam writer
would need. After the viewer's changes the 2 cm plate at ±35° was made
afresh in 236 s (234 s before). Its pattern on its own 0.25 µm pixels
took 14 s (0.78 GB), after about a minute for the plate's light. The
4 cm plate's pattern (163,840² pixels, 3.1 GB) took 9.6 min, 7.3 of
them the plate's light: its 51 GB spectrum read from the SATA disk into
54 GB of scratch on NVMe. Replayed, the 2 cm plate's 0.25 µm pixels are
too coarse for fringes 0.52 µm apart: chrome's other light is 81% of
the scene's, the relief's 25%. With `--pixel 0.125` they are 32% and
11%.

Nothing has been printed. No writer has been given these files, so
their format is a first guess, and the views are still of the scene's
light, not of the pattern's (§9).

### 6.7 The boat at its printed size, and plates to try a writer on (`testplate.py`)

The printed boat is 60 mm long (its file is 60 × 31 × 48 mm). For a
writer whose finest lines are 0.9 µm wide, fringes may be 1.8 µm apart
at their finest, which by `Optics.wide` lets the scene's light spread
±7.8°, the reference beam at 12.5°:

`benchy --plate 68.8128 --scale 35.2947 --spacing 48 --pixel 0.3 --cone 7.8`

The pixels are a third of the finest line, the light is kept every
2.1 µm (32,768² samples), and the plate is 68.8 mm wide, as much wider
than the boat as the first plate was. (The boat's light reaches 31.7 mm
to the side of the plate's middle and 26.7 mm up and down, so 63.5 mm
would hold it.) On NVMe:

| | Time | |
|---|---|---|
| The hologram: 4.09 million points 48 µm apart, 1.74 million seen | 4.5 min | 14.8 GB of memory at most |
| The plate's light from its spectrum (8.6 GB of scratch) | 21–27 s | |
| Chrome, 229,376² pixels, as a picture (PBM) | 56 s | 6.1 GB |
| The same as rectangles (GDSII, `encode --gds`) | 166 s | 576 million of them, 34 GB |
| Relief, 256 levels (PGM) | 85 s | 49 GB |

Replayed (`--check --spot 0.9`, 100 tiles): chrome turns 1.18% of the
beam into the scene's light, with other light 23% of that; written with
a 0.9 µm spot, 0.67% and 59%. The relief: 5.8% and 6%; with the spot,
1.9% and 8%.

- **GDSII** is what mask writers' software takes: `encode --gds` writes
  the clear pixels as rectangles, each run of them along a row, runs
  alike in rows one above another joined. A 4.9 mm plate's file, read
  back by a separate reader and painted, gave the PBM's pixels exactly
  (1.96 million rectangles, 151 million pixels). At 64 bytes a
  rectangle a whole plate's file is 34 GB; whether any writer's
  software takes one that big has not been tried.
- **A plate's width, rounded.** A width that was a whole number of kept
  pixels could come out one more: 68.8128 mm gave 229,383 pixels, not
  229,376. `benchy` now rounds up from a hair less.

**Test plates.** What the replays above cannot say is what a writer
does with lines thinner than its finest, which is how chrome shows the
dim parts of a scene, or how deep each of a relief's levels comes out.
`holotrace testplate` writes two small plates to find out, each thing
on them named beside it (`layout.txt` says what and where):

- **Chrome** (28.9 × 16.2 mm; PBM 0.6 GB; GDSII 22.9 million
  rectangles, 1.4 GB):
  - twelve gratings 1.9 mm square, lines from 0.6 to 2.4 µm wide and as
    far apart, lying and standing. A grating whose lines came out as
    wide as its gaps sends a tenth of a laser's light (1/π²) into each
    first order and none into the second.
  - twenty ladders 1 mm square: lines from 0.3 to 1.5 µm wide, 3.6 and
    7.2 µm from one to the next, clear on chrome and chrome on clear.
    Lines w wide, P apart send sin²(πw/P)/π² into a first order, which
    says how wide each came out.
  - three holograms of the boat, 8.6 mm wide: as `encode` makes it; with
    every fringe half clear (`--bold`), which has no thin lines and
    shows the boat flat; and with the light within ±5°, its fringes
    coarser.
- **Relief** (27.9 × 16.2 mm; PGM 4.7 GB): a staircase of sixteen
  levels from 0 to 255 in squares 1 mm wide; twelve sawtooth gratings,
  teeth from 1.8 to 9.6 µm wide, a whole wave deep, lying and standing
  (one with sharp teeth of the right depth sends all its light into one
  first order); and three holograms, as the chrome's: as `encode
  --levels` makes it; with every fringe a whole wave deep (`--bold`),
  which has no shallow fringes (a writer may not make those, any more
  than thin lines) and shows the boat flat; and with the light within
  ±5°.

Both take 57 s, after a minute for the holograms the first time.
Nothing has been printed.

`chrome.png` and `relief.png` show the plates, each dot the mean of 48
pixels each way. For chrome that is how much is clear, and the boats
show, since where the scene is brighter more is clear. A relief's
fringes are as much above level 128 as below it however deep they are,
so their mean is flat gray (the first `relief.png` showed no boats at
all). `relief.png` therefore shows a thing with fringes by how deep
they are, white for teeth of all 256 levels, and a flat thing by its
level.

The plates' files were read back. `chrome.gds`, its 22,910,712
rectangles drawn again by a separate reader, is `chrome.pbm` pixel for
pixel. And the holograms, cut out of `chrome.pbm` and `relief.pgm`
where `layout.txt` says they are and lit by the reference beam, show
the boat, the right way up (as drawn; no writer's spot):

| Hologram | Of the beam, becomes the scene's light | Other light in the kept band, against that |
|---|---|---|
| H1, chrome, ±7.8° | 1.12% | 28% |
| H2, chrome, ±7.8°, bold | 2.20% | 27% |
| H3, chrome, ±5° | 1.10% | 18% |
| R1, relief, ±7.8° | 5.48% | 7% |
| R2, relief, ±7.8°, bold | 21.68% | 25% |
| R3, relief, ±5° | 5.52% | 3% |

The bold ones are brighter, and their other light is mostly their own
flatness: they are set against the scene's light, which is not flat.
What a spot 0.9 µm wide would make of the ±7.8° ones (`encode --check
--spot 0.9`, with `--bold 0.25` for the bold; 25 tiles; the spot is a
guess at the writer's, as in §6.6), the same two numbers:

| Pattern | As drawn | Written with the spot |
|---|---|---|
| Chrome | 1.14%, 25% | 0.70%, 54% |
| Chrome, bold | 2.32%, 26% | 2.34%, 29% |
| Relief | 5.57%, 6% | 1.84%, 8% |
| Relief, bold | 22.91%, 24% | 8.65%, 32% |

Bold chrome is all but untouched by the spot, as it was made to be. A
bold relief is not: it keeps 38% of its light, the relief proper 33%.
So it stands in for the shallow fringes, not for what a spot does to a
sawtooth's steep sides; the sawtooth gratings are there for that.

**A writer's own pixels.** The manuals give this writer's pixels as
200 or 400 nm, by its mode, and a relief's picture ends up on them
(`docs/heidelberg-laser-writer.md`). `testplate --pixel Q` makes the plates on
pixels of Q µm, into `testplate-<Q>um`, the holograms' finest fringes
still two lines wide; every line and tooth is a whole number of pixels,
so their widths change with Q. Made for 0.2 and 0.4 µm as well, the
±7.8° boat replayed on each (`encode --check --spot 0.9`, 25 tiles; the
same two numbers, as drawn and written with the spot):

| Pixels | The plates, in all | `chrome.gds` | `relief.pgm` | Chrome | Relief |
|---|---|---|---|---|---|
| 0.2 µm | 2.7 min after the holograms, 18.6 GB of memory at most | 35.7 million rectangles, 2.13 GB | 10.5 GB | 1.12%, 20%; 0.72%, 46% | 5.49%, 5%; 1.79%, 7% |
| 0.3 µm | 57 s after the holograms | 22.9 million, 1.37 GB | 4.7 GB | 1.14%, 25%; 0.70%, 54% | 5.57%, 6%; 1.84%, 8% |
| 0.4 µm | 32 s after the holograms | 17.0 million, 1.01 GB | 2.6 GB | 1.05%, 27%; 0.66%, 61% | 5.18%, 8%; 1.72%, 10% |

Each `chrome.gds`, drawn again by the separate reader, is its
`chrome.pbm` pixel for pixel.

**A count of shapes.** One site's rules for this writer allow a
million corners in a whole design (Heidelberg's own guide: a million to
a polygon), and `chrome.gds` has 92 million, four to a rectangle.
Nearly all are the holograms'. So `testplate` also writes
`chrome_lines.gds`, the plate without its holograms: 30,455 rectangles
at 0.3 µm pixels (1.9 MB), 61,262 at 0.2 µm, 18,957 at 0.4 µm. The
separate reader finds each to be exactly `chrome.gds`'s rectangles but
for those within the holograms.

**A boat to try the way in with**: a plate 2.1 mm wide (`benchy --plate
2.1 --scale 1.025 --spacing 16 --pixel 0.3 --cone 7.8`, 2.4 s), its
GDSII file 611,048 rectangles and 37 MB, its PGM 47 MB. So small a
plate, of fewer than 1024 kept pixels, was refused as "too deep a scene
to carry tile by tile": a slip in `solid.sum_planes`, mended.

`docs/heidelberg-laser-writer.md` is what to ask, take, write and measure on a
first visit to the writer.

## 7. The viewer (`viewer.py`)

`Look` holds the spectrum and makes views.

- **`view(direction, pupil, depth)`**: an eye infinitely far away,
  looking along `direction` (sines), pupil of numerical aperture `pupil`,
  focused `depth` behind the plate. It takes the spectrum patch in the
  pupil, applies the propagation phase back to the focus plane, and
  inverse-FFTs the patch, padded 2× so the view is smooth. One view covers
  the whole plate, so a wide plate needs a narrower pupil for the GPU to
  hold it (`widest_pupil`, §6.6).
- **`eye_view(eye, pupil_radius, depth)`**: a real eye at a finite
  distance, which sees each point along its own line to the eye.
  - **Method.** Light headed for the pupil's middle from all over the
    plane is a spherical wave closing in on it. Multiplying the plane's
    light by the opposite phase, e^{+ikr} with r each sample's distance
    from the eye, brings the waves the pupil takes in to about straight
    on everywhere, so one filter (the pupil) serves every point. This is
    what the eye's lens does.
  - **Cells.** The plane is done in up to 5 × 5 cells (more for a plate
    too wide for the GPU to hold a cell, §6.6), each made from
    the spectrum patch about the direction from its middle to the eye,
    wide enough for the whole cell, its border and every lamp direction.
    Cells overlap by an eighth and are cross-faded.
  - **Details.** The phase is computed in double precision (millions of
    radians, nearly canceling). The pupil is a disc facing the cell, so
    seen obliquely it is an ellipse in sine space. Its edge is shaded
    over one spectrum sample. A lamp tilt moves the pupil's place in the
    cell's spectrum, so all of a wavelength's tilts share one transform.
  - **Accuracy.** Against brute force: 0.4–1.2% with one cell; 2–7% with
    2–3 cells on a 4 mm plate (correlation 0.997–0.999), from the far
    tails of the pupil's blur being cut at each cell's edge. Larger
    plates have larger cells and smaller error.
  - **The earlier version was wrong about speckle.** It blended far-away
    views toward a 9 × 9 lattice of points. Neighboring views were up
    to 1° apart, wider than the pupil (0.76°), so each pixel averaged up
    to four independent speckle patterns. It reported a laser contrast
    of 0.72 where a single view, and the new method, give 1.04.
- **Light sources.** Each source is a list of replays
  (wavelength offset, tilt), whose brightness is added:
  - `LASER`: one replay.
  - `LED`: 3 wavelengths (±8 nm) and a tiny tilt; nearly coherent.
  - `lamp(across)`: a disc `across` radians wide at 3 wavelengths, 111
    replays.

  A wavelength change shifts the spectrum in y, because the fringes were
  made for the reference wavelength. A tilt shifts it by tilt / λ: in y
  scaled by the cosine of the reference angle, in x as is. All replays'
  views are made the same size so they can be added; a bug where they
  differed was fixed.
- **Movies.** `holotrace view NAME --eye 300 --focus 0 [--source 0.02]`
  sweeps the eye across and back, then up and back, as far as it can
  while still seeing the whole plate: 96 frames.
  - Laser: about 3 s a frame at 4 cm.
  - Lamp (111 replays): about 10 s a frame at 4 cm.

  Two changes made these two to three times faster. Spectrum patches
  are copied from the blocked file a run of rows and columns at a time,
  not picked out sample by sample (`Look._patch`, `_runs`), and the
  sphere's phase for each cell is worked out on the GPU, still in double
  precision.

  The movies in `out/` made before this change used the blended view,
  which understates laser speckle.

**Known viewer artifact.** With the sharp far-away test eye (numerical
aperture 0.03), the views focused on the bow show a dark diagonal band
on the hull. That is the test eye's shallow depth of field, not the
hologram; the realistic eye doesn't show it.

## 8. Speckle

A coherent hologram of a diffuse surface shows speckle. Each eye
resolution cell sums the waves of many points with random phases. For a
4 mm pupil 30 cm away, the cell is about 115 µm across on the object,
and points 12–32 µm apart put 10–80 points in each cell.

Contrast below is std / local mean on smooth lit surfaces in real-eye
views (30 cm, 4 mm pupil). The second number includes the retina's blur:
1 arcmin, about 87 µm at 30 cm. Fully developed speckle has contrast 1.

**The numbers in §8.2 were measured with the earlier, blended real-eye
view, which understated contrast (0.72 where the truth is 1.04). Its
conclusion stands, since the trend was measured under one viewer. §8.3
is re-measured with the corrected view.**

### 8.1 Optimizing the phases: parked, and why

`speckle.py` optimizes each point's phase by gradient descent (Adam),
through a simulated binary plate, an LED replay and a realistic eye.

| Views | Error, random phases | Error, optimized phases |
|---|---|---|
| Eye positions trained on | 0.40 | 0.10–0.23 |
| Eye positions between them | 0.40 | 0.40–0.44 |

The reason is counting: each point has one phase, but every eye
position sees its own independent speckle pattern. Across a ±35° viewing
zone there are many more independent patterns than phases, so no choice
of phases fixes them all.

### 8.2 Sparse points: tried, and they make it worse

If the points were farther apart than the eye's resolution cell, their
light would not overlap and there would be nothing to interfere. 2 cm
boat, laser:

| Point spacing | Contrast | After the retina's blur |
|---|---|---|
| 12 µm | 0.72 | 0.30 |
| 40 µm | 0.83 | 0.35 |
| 80 µm | 1.11 | 0.50 |
| 120 µm | 1.45 | 0.68 |

The eye's blur spot (about 115 µm) is about the size the retina resolves
(about 90 µm). So dots far enough apart not to interfere are far enough
apart to be seen as dots, with dark gaps between them. There is no
spacing that avoids both.

### 8.3 An extended light source: works

Each part of a lamp of angular width θ replays the plate from its own
direction. That shifts which plane waves enter the pupil, so each part
gives an independent speckle pattern, and their intensities add. Once θ
exceeds the pupil's angular width seen from the scene (2 × NA ≈ 0.013
rad), speckle averages away. The cost is blur of each point's distance
from the plate × θ, small for an image-plane hologram.

| Source width (rad) | 2 cm boat | 4 cm boat | 4 cm, after the retina's blur |
|---|---|---|---|
| 0 (laser) | 1.04 | 1.05 | 0.39 |
| 0.01 | 0.50 | 0.50 | 0.25 |
| 0.02 | 0.40 | 0.41 | 0.22 |
| 0.04 | 0.31 | 0.30 | 0.18 |

0.02 rad (a 2 cm diffuser 1 m away) suits the 4 cm boat: speckle cut to
about 40% of a laser's, and the deepest parts still sharp. Nothing in the hologram
changes. The practical conclusion for a print is to **light it with a
red LED behind a diffuser 1–2 cm across, about a meter away**, not a
laser.

The retina-blurred contrast levels off near 0.18. Part of that is the
boat's real shading, which the metric also counts, and part was grain
from the points' random scatter (§8.4). These numbers are for points
scattered at random.

### 8.4 Grain from random points: cured by spreading them evenly

Points scattered at random over a surface leave gaps and clumps: a
patch that holds n points on average holds n ± √n. The eye's resolution
cell holds about 10 points 32 µm apart, so its brightness varies by
about 30% from the scatter alone. It looks like speckle, but it is in
the scene, not the light, and a lamp does not average it away.

- **It is not speckle.** A flat matte sheet of random points 32 µm
  apart, lit by a 0.04 rad lamp, showed contrast 0.46 (0.22 after the
  retina's blur). Averaged over 12 sets of random phases, which would
  wash speckle out, it was still 0.41 (0.195).
- **Without any speckle** (points of equal brightness added as
  intensities, through the eye's blur), a random sheet gives 0.41
  (0.19) at 32 µm and 0.20 (0.095) at 16 µm. Spread evenly
  (`meshes.sample`, §3.1) it gives 0.14 (0.06) and 0.065 (0.03).

On the boat, 1 cm long on a 2 cm plate, with the points at random and
spread evenly. Contrast is std over the local mean (a Gaussian of
σ = 300 µm) on the smooth, lit parts of the hull, the same parts in
both, from two eye positions 30 cm away:

| Light | 12 µm, random | 12 µm, even | 32 µm, random | 32 µm, even |
|---|---|---|---|---|
| Laser | 0.985 (0.370) | 0.971 (0.352) | 1.042 (0.405) | 0.940 (0.352) |
| Lamp 0.01 rad | 0.457 (0.231) | 0.444 (0.224) | 0.557 (0.281) | 0.374 (0.194) |
| Lamp 0.02 rad | 0.370 (0.204) | 0.342 (0.191) | 0.475 (0.261) | 0.277 (0.167) |
| Lamp 0.04 rad | 0.283 (0.176) | 0.248 (0.154) | 0.383 (0.238) | 0.215 (0.154) |

- **At 32 µm,** the spacing of the 4 cm boat, even points cut the
  contrast under a 0.02 rad lamp from 0.475 to 0.277.
- **Even points 32 µm apart look smoother under a lamp than random
  points 12 µm apart,** with a seventh as many points.
- **At 12 µm** the gain is small: some 70 points to a cell were already
  nearly smooth.
- **By laser** it changes little: speckle dominates.

## 9. What the results do not include (compromises)

- **An ideal plate.** Views assume the plate records the light exactly.
  A printed plate has finite pixels and few levels, often binary.
  Encoding adds scattered light and costs brightness. `holotrace encode`
  writes the pattern and replays it (§6.6): beside the scene's light a
  chrome plate sends other light a quarter as strong into the cone,
  half as strong if written with a 0.8 µm spot; a relief in resist a
  sixteenth. Views of the pattern itself are made only in a prototype.
- **Speckle** in laser light (§8). The lamp reduces it in viewing only.
- **Simple materials:** a single white plastic: matte, a little ambient,
  one glossy highlight, one disc light. The boat's matte light is path
  traced (§3.1), but the materials are no richer for it.
- **32 × 32 directions per point** (about 2.2° apart at ±35°): highlights
  and occlusion edges are blended between those directions.
- **Occlusion edges are soft, and finer maps alone would not sharpen
  them.** Where a point passes from seen to hidden, the edge is about
  sqrt(λ / z) wide in angle, z being the recording plane's distance: it
  is set by diffraction between the point and its plane. In a 1D model
  with planes 200–500 µm from the point, the edge was 2.0–2.7° wide
  whether the map had 32 directions, 128, or was perfectly sharp (the
  pupil takes in 0.76°). A plane 1 mm away gave 1.2°, and 3.6 mm gave
  0.74°. On a wall 10 mm behind what hides it, 2.0–2.7° is a soft edge
  0.3–0.5 mm wide. Sharper edges need both far planes for such points
  and finer maps. Not tried in the pipeline.
- **Recording-plane error:** about 2% in the field's amplitude within
  the cone, 4% at its very edge (§4).
- **Viewer approximations:** the real eye's view is cut into cells (§7),
  and the LED and lamp use a handful of discrete wavelengths and
  directions.
- **Monochrome**, at 633 nm.

## 10. File map

| File | Role |
|---|---|
| `__init__.py` | Command-line entry: `holotrace check / magnifier / mesh / solid / benchy / view / compare / encode / testplate / speckle / gpucheck`. |
| `magnifier.py` | Step 1: dots behind an analytic lens; shared constants (`WAVELENGTH`, `PLATE`, `CONE`, `REFERENCE`, `PUPIL`) and image helpers. |
| `lens.py`, `plate.py`, `eye.py` | The analytic lens with exact ray aiming; the CPU plate (reference implementation); the far-away eye of the early steps. |
| `mesh.py`, `meshed.py` | Step 2: the lens as a Mitsuba mesh; Mitsuba setup (`_mitsuba`, which picks a variant the card supports). |
| `gpu.py` | CUDA kernels via CuPy: `direct`, `through_lens` (Hermite-blended path lengths), `seen` / `seen_gather` and their reverse kernels; `GpuPlate`, `SeenPoints`. |
| `gpucheck.py` | GPU against CPU. |
| `meshes.py` | STL reading and sampling surfaces into evenly spread points. |
| `solid.py` | Step 3 scene (ball and wall); `maps` (Mitsuba visibility and shading; `lit` and `_falling` for path-traced light); `Optics` / `Optics.wide`; `sum_planes` (tile by tile, for plates that fit in memory); `_carry`; `render` (the whole pipeline). |
| `planes.py` | Plane-to-plane sum for plates too big for memory (§6.4). |
| `benchy.py` | The boat: `--scale`, `--spacing`, `--turn`, `--plate`, `--pixel`, `--cone`, `--slab`, `--wall`, `--rule`. |
| `viewer.py` | `Look` (views from the spectrum), light sources (`LASER`, `LED`, `lamp`), movies, `compare`. |
| `encode.py` | The plate's pattern on a writer's pixels, clear or opaque (a picture, or a GDSII file) or in 256 levels, and a replay of it (§6.6, §6.7). |
| `testplate.py` | Two small plates to try a writer on: gratings, ladders of thin lines, a staircase of levels, small holograms (§6.7). |
| `speckle.py` | Parked phase optimization against speckle (§8.1). |
| `docs/heidelberg-laser-writer.md` | Not code: what to ask, take, write and measure on a first visit to a laser writer, with the test plates (§6.7). |

Output names encode the settings, for example
`out/benchy-40.96mm-32um-35deg-0.25um-boat16`: plate width, point
spacing, cone, pixel, and boat scale (when the plate width is given
separately). A non-default turn is added as `-turn15`, the wall as
`-wall`, and light by rule as `-rule`.

## 11. Reproducing the main results

```
uv run --extra gpu holotrace benchy --plate 20.48 --scale 6 --spacing 12 --turn 15 --pixel 0.25 --cone 35 --slab 300
#   2 cm plate, 1 cm boat: 4 min (3 by rule), 13 GB spectrum (8 GB on disk)
uv run --extra gpu holotrace benchy --plate 40.96 --scale 16 --spacing 32 --pixel 0.25 --cone 35 --slab 300
#   4 cm plate, 27 mm boat: 11 min by rule on NVMe (set HOLOTRACE_OUT to a folder on it);
#   the spectrum 33 GB on disk, and up to 66 GB of scratch while running
#   add --wall for the endless wall: 46 min, up to 174 GB of disk while running
#   add --rule for the light by rule, not path traced (§3.1): quicker, and how the
#   11 min above was measured
uv run --extra gpu holotrace benchy --scale 48 --spacing 64 --pixel 0.4 --cone 7
#   a 9.8 cm plate for a laser writer (§6.6): 5.5 min, 21 GB of memory
uv run --extra gpu holotrace encode benchy-98.304mm-64um-7deg-0.4um
#   its pattern, clear or opaque: 2 min, 7 GB (--levels for 256 levels: 2.7 min, 56 GB;
#   --check to replay both: 2.5 min)
uv run --extra gpu holotrace benchy --plate 68.8128 --scale 35.2947 --spacing 48 --pixel 0.3 --cone 7.8
#   the boat at its printed size, 60 mm, for a writer whose finest lines are 0.9 um (§6.7): 4.5 min
uv run --extra gpu holotrace encode benchy-68.8128mm-48um-7.8deg-0.3um-boat35.2947
#   its pattern in chrome: 1.3 min, 6 GB (--levels: 1.8 min, 49 GB; --gds: 3.2 min, 34 GB)
uv run --extra gpu holotrace testplate
#   two small plates to try the writer on: 2 min
uv run --extra gpu holotrace view benchy-40.96mm-32um-35deg-0.25um-boat16 --eye 300 --focus 0
#   real-eye movie, laser: ~5 min
uv run --extra gpu holotrace view benchy-40.96mm-32um-35deg-0.25um-boat16 --eye 300 --focus 0 --source 0.02
#   the same lit by a lamp 0.02 rad across: ~17 min
```

A run's first line comes after the points are spread evenly, 37 s for
the 4 cm boat. On a card too old for Mitsuba to trace on, Dr.Jit then
prints one line, "Unable to load": that is the cure for the start-up
hang (§3.1), not a fault.

## 12. Where to go next

In the order recommended:

1. **View the printed plate.** `encode` writes the pattern and replays
   it as numbers (§6.6); a prototype views a 5 mm plate's. Still to do:
   view any plate's pattern with the same eye and lamp, to see the
   encoding's other light against the remaining speckle; put a real
   writer into the replay (its spot, its stripes), once one has been
   measured; and choose the cone and the reference beam's angle by what
   the pattern's second order does (§6.6).
2. **Choose a real medium** (e-beam or laser lithography on chrome on
   glass, a service for computed holograms, a photoplotter). Its pixel
   size and levels fix the optics: §6.6 is a first guess at a laser
   writer's, ±35° at 0.25 µm at an electron beam's.
3. **Go bigger** (§6.4). It needs only disk: 10 cm at ±35° needs about
   500 GB. Not run, and not timed.
4. **Richer scenes.** Mitsuba's path tracer works out the light
   falling on each point (§3.1); the scene it lights is still one
   matte boat, a matte wall, one lamp and an even sky. Mitsuba will
   take textures, several materials, and lamps and skies of any shape,
   and a scene could be built in Blender and exported to it. Each
   point's reflectance could then come from Mitsuba too
   (`eval_diffuse_reflectance`, already used in `solid._falling`). Shiny
   materials are another matter: see §3.1 and §9.
5. **The endless wall without wide planes.** Leave out, in the point
   sum, each ray that would land beyond the plate, and make the planes
   periodic at the plate's own width: no wide planes, no cut. Tried on
   a 4 mm plate as a prototype (not in the code): the plate's light
   400 µm in from its edges matched the wide planes' to 0.00% for sines
   under 0.5, and differed by 2.9% in the cone's last degree. The cost
   is a faded border, 1.6% off at 50–100 µm from the edge with a 25 µm
   fade. Not run at 4 cm.
6. **Smaller items:**
   - richer materials;
   - a glass scene, to test the original magnifying-glass idea at scale;
   - automated tests for the checks in §4;
   - resuming a plane-to-plane run after the last finished group, for
     multi-day runs.
