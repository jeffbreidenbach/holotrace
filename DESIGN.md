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
30 cm away. It takes about 40 min on one 8 GB GPU from 2016 with an NVMe
disk.

## 2. Conventions

Anyone changing the code needs these.

| Thing | Convention |
|---|---|
| Units | Micrometres throughout. Wavelength `WAVELENGTH = 0.633` (HeNe red) in `magnifier.py`. |
| Geometry | The plate is the plane z = 0, centred on the axis. The viewer is at +z. A point's **depth** is −z: positive behind the plate, negative in front. |
| Pixels | Pixel index i sits at x = (i − n/2 + 0.5) · pitch (`coordinate` in `gpu.py`); grids are centred. |
| Cone | `cone` is the **tangent** of the widest angle the scene's light reaches the plate at; `Optics.sine` is its sine. Light outside the cone is never computed or kept. |
| Reference beam | Given as a sine (`Optics.reference`), tilted in y. |
| Phases | Each point gets a random starting phase from `np.random.default_rng(1)`, so runs are repeatable. |
| Kept light | What is stored is not the full-resolution plate but the scene's light at the plate, sampled every `Optics.keep` pixels (`kept_pitch`). See §5.2. |
| Spectrum file | `out/NAME/spectrum.npy` is the **unshifted 2D FFT** of the kept light. It is stored either whole, shape (n, n), or in blocks of columns, shape (n/B, n, B) with B = 512 (§6.4). `spectrum.json` beside it holds `pitch`, `cone` and `reference`. `viewer.Look` reads both layouts. |
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

Surfaces are sampled into points spread evenly by area, about `spacing`
µm apart (`meshes.sample`). Each point is a light source whose
brightness depends on direction. For each point and each of
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

Mitsuba runs on the CPU (LLVM variant): its GPU variants need compute
capability 7.5 or newer and **can hang on start** on older cards, so they
are only tried when `nvidia-smi` reports a new enough card. Even on the
CPU, Mitsuba/Dr.Jit sometimes hangs at start with no output and one core
busy. It is random. The cure is to kill the process (by PID, never with
`pkill` patterns that match the calling shell) and start again.

### 3.2 The point sum (`gpu.py`)

The heart of it is one CUDA kernel, `seen_many`, launched once for many
points at a time: the thread blocks are divided among the points' windows
(the pixels each point's cone can reach), and each thread adds one
point's wave at one pixel with `atomicAdd`:

```
t      = (pixel − point) / depth                    (tangent, 2D)
extra  = depth · |t|² / (sqrt(1 + |t|²) + 1)        (path beyond depth, computed stably)
value  = map(t) · (1 + |t|²)^(−3/4) / |depth| · e^{i (k·extra + phase)}
```

**Precision.** GPUs are fast in single precision, about 7 digits, too
few for a phase of tens of thousands of radians. So the large part of
each path, k · depth, is folded into the point's starting phase on the
CPU in double precision, and the kernel only handles `extra`, which is
at most tens of micrometres.

**Batching.** One launch per point cost more than the sum itself;
launching all points at once made it about 3× faster. There is a matching
reverse kernel (`seen_reverse_many`) for gradients, used by the parked
speckle optimizer (§8.1).

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
| Real-eye view (`Look.eye_view`) | brute force (whole plane × sphere's phase, one filter), 4 mm plate | one cell: 0.4–1.2%, correlation 0.9999; 2–3 cells: 2–7%, correlation 0.997–0.999 |

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
evaluations. Four ideas bring that down to a couple of hours.

### 5.1 Recording planes: sum points close, carry the rest by plane waves

Points are taken in slabs `slab` µm deep (600 µm at ±35°). Each slab's
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
  far more memory.

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
are on an 8 GB GPU from 2016, about 32 GB of RAM, and a SATA SSD.

### 6.1 From 2 mm to 4 cm at ±7°

| Plate | What changed | Sum time |
|---|---|---|
| 4 mm, 8192² | batched GPU kernel | 75 s |
| 1 cm, 20480² | tiles; viewer from the spectrum | 6.5 min |
| 1 cm | recording planes in 1 mm slabs | 38 s (from 320 s) |
| 2 cm, 40960² | tile pipeline: only each tile's patch of each plane | 3.25 min, 19 GB |
| 4 cm, 81920² | keep only the cone's light, every 2 µm | 19 min, 16 GB |
| 4 cm | deep slabs summed and carried at the kept 2 µm; 8192 tiles; boat without its wall | 68 s (from 987 s); 2.6 min total |

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
   contiguous block.
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
   (§6.3) guarantees, nothing wraps. Otherwise the planes are made wider
   than the plate by as far as the light spreads. The spectrum is then
   summed on that wider domain, transformed back, the plate cut out, and
   transformed again (`_cut`).

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

**Results.**

| 4 cm, 27 mm boat | Sum | Total with maps | Peak RAM |
|---|---|---|---|
| In-memory carry (removed) | 92 min + 26 min spectrum | ~2 h | 26 GB + 3 GB swap |
| Plane to plane, SATA SSD | 105 min | ~1 h 50 min | ~4 GB |
| + steps 6 and 7, on NVMe | **35 min** (24 min making planes, 11 min carrying) | ~40 min | ~4 GB (RSS is higher: page cache) |

- **2 cm:** 878 s by GPU tiles; 1066 s plane to plane; 892 s with steps 6
  and 7 on SATA.
- **Accuracy:** views agree with the old method to 0.04–0.09% throughout.
- **Where the time goes now:** about two thirds in the GPU's point sum,
  one third carrying planes. On SATA the disk dominated: most of the
  "making" time there was writing the scratch plane.
- **Results on another disk:** `HOLOTRACE_OUT=/path` puts results (and
  the scratch plane) there; the viewer reads the same variable.
- **Scaling:** cost now grows with plate area × number of groups, not
  depth squared. Depth is no longer limited by memory.

**Disk needed:** the spectrum plus one scratch plane, 2 × (n² × 8 bytes).

| Viewing cone | Kept pitch | Plate for 91 GB free | For ~160 GB | For ~670 GB |
|---|---|---|---|---|
| ±35° | 0.5 µm | 3.7 cm | 4.9 cm | 10 cm |
| ±24° | 0.75 µm | 5.5 cm | 7.4 cm | 15 cm |
| ±15° | 1.0 µm | 7.3 cm | 9.9 cm | 20 cm |

The disk matters: moving from the SATA SSD to NVMe cut the 4 cm run by
well over half. 10 cm at ±35° is estimated at 10–12 h on that GPU with
enough NVMe, or 1–2 h on a rented 4090-class card (not run).

## 7. The viewer (`viewer.py`)

`Look` holds the spectrum and makes views.

- **`view(direction, pupil, depth)`**: an eye infinitely far away,
  looking along `direction` (sines), pupil of numerical aperture `pupil`,
  focused `depth` behind the plate. It takes the spectrum patch in the
  pupil, applies the propagation phase back to the focus plane, and
  inverse-FFTs the patch, padded 2× so the view is smooth. One view covers
  the whole plate.
- **`eye_view(eye, pupil_radius, depth)`**: a real eye at a finite
  distance, which sees each point along its own line to the eye.
  - **Method.** Light headed for the pupil's middle from all over the
    plane is a spherical wave closing in on it. Multiplying the plane's
    light by the opposite phase, e^{+ikr} with r each sample's distance
    from the eye, brings the waves the pupil takes in to about straight
    on everywhere, so one filter (the pupil) serves every point. This is
    what the eye's lens does.
  - **Cells.** The plane is done in up to 5 × 5 cells, each made from
    the spectrum patch about the direction from its middle to the eye,
    wide enough for the whole cell, its border and every lamp direction.
    Cells overlap by an eighth and are cross-faded.
  - **Details.** The phase is computed in double precision (millions of
    radians, nearly cancelling). The pupil is a disc facing the cell, so
    seen obliquely it is an ellipse in sine space. Its edge is shaded
    over one spectrum sample. A lamp tilt moves the pupil's place in the
    cell's spectrum, so all of a wavelength's tilts share one transform.
  - **Accuracy.** Against brute force: 0.4–1.2% with one cell; 2–7% with
    2–3 cells on a 4 mm plate (correlation 0.997–0.999), from the far
    tails of the pupil's blur being cut at each cell's edge. Larger
    plates have larger cells and smaller error.
  - **The earlier version was wrong about speckle.** It blended far-away
    views toward a 9 × 9 lattice of points. Neighbouring views were up
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
  - Laser: about 6 s a frame at 4 cm.
  - Lamp (111 replays): about 26 s a frame at 4 cm, 5 s at 2 cm.

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
red LED behind a diffuser 1–2 cm across, about a metre away**, not a
laser.

The retina-blurred contrast levels off near 0.18. Part of that is the
boat's real shading, which the metric also counts. A speckle-free
reference, averaged over many random phase sets, was not made, so the
split is not measured.

## 9. What the results do not include (compromises)

- **An ideal plate.** Views assume the plate records the light exactly.
  A printed plate has finite pixels and few levels, often binary.
  Encoding adds scattered noise and costs brightness; that is not yet
  simulated for the large plates. `speckle.py` contains a binary encoding
  (clear where cos θ > sqrt(1 − a²)) that can be reused.
- **Speckle** in laser light (§8). The lamp reduces it in viewing only.
- **Simple materials:** a single white plastic: matte, a little ambient,
  one glossy highlight, one disc light.
- **32 × 32 directions per point** (about 2.2° apart at ±35°): highlights
  and occlusion edges are blended between those directions.
- **Recording-plane error:** about 2% in the field's amplitude within
  the cone, 4% at its very edge (§4).
- **Viewer approximations:** the real eye's view is cut into cells (§7),
  and the LED and lamp use a handful of discrete wavelengths and
  directions.
- **Monochrome**, at 633 nm.

## 10. File map

| File | Role |
|---|---|
| `__init__.py` | Command-line entry: `holotrace check / magnifier / mesh / solid / benchy / view / compare / speckle / gpucheck`. |
| `magnifier.py` | Step 1: dots behind an analytic lens; shared constants (`WAVELENGTH`, `PLATE`, `CONE`, `REFERENCE`, `PUPIL`) and image helpers. |
| `lens.py`, `plate.py`, `eye.py` | The analytic lens with exact ray aiming; the CPU plate (reference implementation); the far-away eye of the early steps. |
| `mesh.py`, `meshed.py` | Step 2: the lens as a Mitsuba mesh; Mitsuba setup (`_mitsuba`, which picks a variant the card supports). |
| `gpu.py` | CUDA kernels via CuPy: `direct`, `through_lens` (Hermite-blended path lengths), `seen` / `seen_many` and their reverse kernels; `GpuPlate`, `SeenPoints`. |
| `gpucheck.py` | GPU against CPU. |
| `meshes.py` | STL reading and sampling surfaces into points. |
| `solid.py` | Step 3 scene (ball and wall); `maps` (Mitsuba visibility and shading); `Optics` / `Optics.wide`; `sum_planes` (tile by tile, for plates that fit in memory); `_carry`; `render` (the whole pipeline). |
| `planes.py` | Plane-to-plane sum for plates too big for memory (§6.4). |
| `benchy.py` | The boat: `--scale`, `--spacing`, `--turn`, `--plate`, `--pixel`, `--cone`, `--slab`. |
| `viewer.py` | `Look` (views from the spectrum), light sources (`LASER`, `LED`, `lamp`), movies, `compare`. |
| `speckle.py` | Parked phase optimization against speckle (§8.1). |

Output names encode the settings, for example
`out/benchy-40.96mm-32um-35deg-0.25um-boat16`: plate width, point
spacing, cone, pixel, and boat scale (when the plate width is given
separately). A non-default turn is added as `-turn15`.

## 11. Reproducing the main results

```
uv run --extra gpu holotrace benchy --plate 20.48 --scale 6 --spacing 12 --turn 15 --pixel 0.25 --cone 35 --slab 600
#   2 cm plate, 1 cm boat: ~17 min, 13 GB spectrum (8 GB on disk)
uv run --extra gpu holotrace benchy --plate 40.96 --scale 16 --spacing 32 --pixel 0.25 --cone 35 --slab 600
#   4 cm plate, 27 mm boat: ~40 min on NVMe (set HOLOTRACE_OUT to a folder on it);
#   the spectrum 33 GB on disk, and up to 54 GB of scratch while running
uv run --extra gpu holotrace view benchy-40.96mm-32um-35deg-0.25um-boat16 --eye 300 --focus 0
#   real-eye movie, laser: ~10 min
uv run --extra gpu holotrace view benchy-40.96mm-32um-35deg-0.25um-boat16 --eye 300 --focus 0 --source 0.02
#   the same lit by a lamp 0.02 rad across: ~40 min
```

If a run prints nothing for more than a minute after starting, Mitsuba
has hung at start (§3.1): kill it and run it again.

## 12. Where to go next

In the order recommended:

1. **Simulate the printed plate.** Encode the 4 cm hologram tile by tile
   to a printer's pixel size and levels, add the reference beam, and view
   it with the same eye and lamp. This measures the encoding noise
   against the remaining speckle. It also checks whether ±35° at 0.25 µm
   is realistic: a coarser printer pixel forces a narrower cone, and that
   decides the settings for every larger run.
2. **Choose a real medium** (e-beam or laser lithography on chrome on
   glass, a service for computed holograms, a photoplotter). Its pixel
   size and levels fix the optics.
3. **Go bigger** (§6.4). It needs only disk: 10 cm at ±35° needs about
   670 GB and roughly 2–3 days on a SATA SSD, much less on NVMe and a
   newer GPU.
4. **Thinner slabs on a big plate** (§4): with the fade, 300 µm slabs
   are as accurate as 600 µm and were 1.8× faster on a 4 mm plate. Time
   them at 4 cm.
5. **Smaller items:**
   - retry Mitsuba's start automatically when it hangs;
   - add two groups per pass over the spectrum (a few minutes a run);
   - richer materials;
   - a glass scene, to test the original magnifying-glass idea at scale;
   - automated tests for the checks in §4;
   - resuming a plane-to-plane run after the last finished group, for
     multi-day runs.
