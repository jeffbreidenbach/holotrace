"""Test plates: what a writer makes of fine lines and of levels, found
out before a whole plate is given to it.

    holotrace testplate [--line W]

writes into out/testplate the patterns of two small plates, for a
writer whose finest lines are thought to be W um wide (default 0.9), on
pixels a third of that:

chrome.pbm, chrome.gds: clear or opaque (see encode.py for the files).

- Gratings, 2 mm square: lines 2, 3, 4, 5, 6 and 8 pixels wide and as
  far apart, lying (as a hologram's fringes do) and standing. Lit by a
  laser, a grating whose lines came out as wide as its gaps sends a
  tenth of the light (1 / pi^2) into each first order and none into
  the second: so what the writer makes of each width, each way.
- Ladders, 1 mm square: clear lines 1 to 5 pixels wide, 12 pixels apart
  and 24, lying; and the same of chrome on clear. Lines w wide, P apart
  send sin^2(pi w / P) / pi^2 of the light into a first order: so how
  wide each came out. A chrome hologram shows the dim parts of its
  scene by lines thinner than W (encode.py).
- Three holograms of the boat, 8.6 mm wide: as encode.py makes it;
  bold, all its fringes half clear, which has no thin lines and shows
  the boat flat; and one whose light spreads only 5 degrees, with
  coarser fringes.

relief.pgm: 256 levels (see encode.py), 128 where there is nothing.

- A staircase: sixteen squares 1 mm wide, levels 0 to 255: how deep
  each level comes out.
- Sawtooth gratings, 2 mm square, levels 0 to 255 across each tooth
  (a whole wave, if the depths are right), teeth 6, 8, 12, 16, 24 and
  32 pixels wide, lying and standing: one with sharp teeth sends all
  its light into one first order.
- Two holograms: at the widest angle W allows, and at 5 degrees.

Each has its name beside it and the plate's own along the top, upright
when the plate is: layout.txt says what each is and where, chrome.png
and relief.png show the plates. The holograms are made first, as
benchy makes them, into out/testplate too.
"""

import math
import os
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import encode, planes, solid
from .magnifier import WAVELENGTH
from .solid import results
from .viewer import _option

LINES = (2, 3, 4, 5, 6, 8)
LADDER, APART = (1, 2, 3, 4, 5), (12, 24)
TEETH = (6, 8, 12, 16, 24, 32)
STEPS = 16
# Sizes in um: the holograms, about; the gratings, ladders and stairs;
# the gaps between things; a dot of a name's letters, and of the plate's.
HOLOGRAM, GRATING, LADDERS, GAP, DOT, TITLE = 8600, 2000, 1000, 300, 40, 100


def letters(text, dot):
    """`text` as pixels, rows from the bottom up, each dot of its
    letters `dot` pixels square."""
    font = ImageFont.load_default()
    box = font.getbbox(text)
    img = Image.new("L", (box[2] + 2, box[3] + 2), 0)
    ImageDraw.Draw(img).text((1, 1), text, fill=255, font=font)
    return np.repeat(np.repeat(np.array(img)[::-1] > 127, dot, axis=0), dot, axis=1)


def hologram(out, pixel, degrees):
    """Makes the boat's hologram about HOLOGRAM wide, its light within
    `degrees`, on pixels `pixel` apart (unless it is there already), and
    returns where its spectrum is and how many pixels wide it is."""
    import cupyx.scipy.fft as cufft

    o = solid.Optics.wide(pixel, degrees)
    n = cufft.next_fast_len(round(HOLOGRAM / o.kept_pitch))
    width = float(f"{n * o.kept_pitch / 1000:.6g}")
    scale = float(f"{width / 2.048:.4g}")
    path = out / f"benchy-{width:g}mm-16um-{degrees:g}deg-{pixel:g}um-boat{scale:g}" / "spectrum.npy"
    if not path.exists():
        run = "import sys, holotrace; sys.argv = ['holotrace'] + sys.argv[1:]; holotrace.main()"
        options = ["--plate", repr(width), "--scale", repr(scale), "--spacing", "16", "--pixel", repr(pixel),
                   "--cone", repr(degrees)]
        subprocess.run([sys.executable, "-c", run, "benchy", *options], check=True,
                       env={**os.environ, "HOLOTRACE_OUT": str(out)}, stdout=subprocess.DEVNULL)
    return path, n * o.keep


def lay_out(rows, title, gap, dot):
    """Where everything goes: `rows` of things (name, pixels wide and
    high), each row left to right, `gap` apart, the rows from the top
    down under the plate's name `title`, with room under each thing for
    its name in letters of `dot`. Returns the plate's size, each thing's
    place (pixels from the left and from the bottom), and the title's."""
    name_high = 12 * dot + gap // 2
    wide = max([title.shape[1]] + [sum(w for _, w, _ in row) + gap * (len(row) - 1) for row in rows]) + 2 * gap
    high = gap + title.shape[0] + gap + sum(max(h for _, _, h in row) + name_high + gap for row in rows)
    places, top = {}, high - gap - title.shape[0] - gap
    for row in rows:
        x = gap
        for name, w, h in row:
            places[name] = (x, top - h)
            x += w + gap
        top -= max(h for _, _, h in row) + name_high + gap
    return wide, high, places, (gap, high - gap - title.shape[0])


def main():
    import cupy as cp

    line = _option("--line", 0.9)
    pixel = line / 3
    nm = round(1000 * pixel)
    if abs(nm - 1000 * pixel) > 1e-6:
        sys.exit("the pixels, a third of --line, must be a whole number of nm")
    out = results() / "testplate"
    out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    # The widest angle whose finest fringes (see Optics.wide) are two
    # lines wide, to a tenth of a degree.
    widest = math.floor(10 * math.degrees(math.asin((WAVELENGTH / (2 * line) - 0.08) / 2))) / 10
    px = lambda um: round(um / pixel)
    gap, dot, title_dot = px(GAP), px(DOT), px(TITLE)
    # Whole numbers of every grating's lines, and of every tooth.
    grating = px(GRATING) // 240 * 240
    ladder = px(LADDERS) // 24 * 24
    wide_at, wide_px = hologram(out, pixel, widest)
    narrow_at, narrow_px = hologram(out, pixel, 5.0)
    print(f"the holograms, {widest:g} and 5 degrees: {time.time() - start:.1f} s", flush=True)

    def stripes(size, apart, clear, lying, sawtooth=False):
        """Lines `clear` of every `apart` pixels (or teeth `apart` wide,
        as levels), lying or standing."""
        along = np.arange(size) % apart
        across = (along * 256 // apart).astype(np.uint8) if sawtooth else along < clear
        return np.broadcast_to(across[:, None] if lying else across[None, :], (size, size))

    def of_boat(at, bold=0.0, levels=False):
        def paint(view):
            import json

            info = json.loads(at.with_suffix(".json").read_text())
            light, strength, brightest = encode.plate_light(at)
            try:
                up = round(info["pitch"] / pixel)
                patterns = encode.Patterns(light, strength, brightest, info["pitch"], info["reference"], up, bold)
                for y0 in range(0, patterns.n, patterns.tile):
                    for x0 in range(0, patterns.n, patterns.tile):
                        made = patterns.core(y0, x0, levels)
                        made = cp.asnumpy(made if levels else ~made)
                        view[y0 * up : y0 * up + made.shape[0], x0 * up : x0 * up + made.shape[1]] = made
            finally:
                planes._remove(light)
        return paint

    um = lambda pixels: f"{pixels * pixel:g} um"
    chrome = [
        [("H1", f"the boat, light within {widest:g} degrees", wide_px, wide_px, of_boat(wide_at)),
         ("H2", f"the same, bold: every fringe half clear where the light thereabouts is a quarter of the brightest "
                f"or more", wide_px, wide_px, of_boat(wide_at, bold=0.25)),
         ("H3", "the boat, light within 5 degrees", narrow_px, narrow_px, of_boat(narrow_at))],
        [(f"G{1 + i + 6 * (not lying)}", f"lines {um(k)} wide, {um(k)} apart, {'lying' if lying else 'standing'}",
          grating, grating, stripes(grating, 2 * k, k, lying)) for lying in (True, False) for i, k in enumerate(LINES)],
        [(f"L{1 + i + 5 * j + 10 * dark}",
          f"{'chrome' if dark else 'clear'} lines {um(k)} wide, {um(apart)} from one to the next, lying",
          ladder, ladder, stripes(ladder, apart, k, True) ^ bool(dark))
         for dark in (0, 1) for j, apart in enumerate(APART) for i, k in enumerate(LADDER)],
    ]
    relief = [
        [("R1", f"the boat, light within {widest:g} degrees", wide_px, wide_px, of_boat(wide_at, levels=True)),
         ("R2", "the boat, light within 5 degrees", narrow_px, narrow_px, of_boat(narrow_at, levels=True))],
        [(f"S{1 + i}", f"level {i * 255 // (STEPS - 1)}", ladder, ladder,
          np.broadcast_to(np.uint8(i * 255 // (STEPS - 1)), (ladder, ladder))) for i in range(STEPS)],
        [(f"T{1 + i + 6 * (not lying)}", f"teeth {um(k)} wide, levels 0 to 255 across each, {'lying' if lying else 'standing'}",
          grating, grating, stripes(grating, k, 0, lying, sawtooth=True)) for lying in (True, False) for i, k in enumerate(TEETH)],
    ]
    legend = []
    for plate, rows, levels in (("chrome", chrome, False), ("relief", relief, True)):
        start = time.time()
        title = letters(f"HOLOTRACE  TEST   {plate.upper()}   LINES  {line:g}  UM   THIS  WAY  UP", title_dot)
        wide, high, places, title_at = lay_out([[(t[0], t[2], t[3]) for t in row] for row in rows], title, gap, dot)
        # The plate, rows from the bottom up: clear or not, or levels.
        made = np.lib.format.open_memmap(out / f"{plate}.tmp.npy", "w+", np.uint8 if levels else bool, (high, wide))
        if levels:
            made[:] = 128
        ink = 255 if levels else True
        made[title_at[1] : title_at[1] + title.shape[0], title_at[0] : title_at[0] + title.shape[1]][title] = ink
        legend.append(f"{plate}: {wide * pixel / 1000:.3f} mm wide, {high * pixel / 1000:.3f} mm high, pixels {pixel:g} um; "
                      f"places are of each thing's lower left corner, in mm from the plate's, as the viewer sees it")
        for row in rows:
            for name, what, w, h, paint in row:
                x, y = places[name]
                if callable(paint):
                    paint(made[y : y + h, x : x + w])
                else:
                    made[y : y + h, x : x + w] = paint
                word = letters(name, dot)
                made[y - gap // 4 - word.shape[0] : y - gap // 4, x : x + word.shape[1]][word] = ink
                legend.append(f"  {name:4s} {x * pixel / 1000:7.3f} {y * pixel / 1000:7.3f}  {w * pixel / 1000:.3f} mm square  {what}")
        # The files: rows from the top down; a PBM's black is opaque.
        band = 2048
        small = []
        with open(out / f"{plate}.{'pgm' if levels else 'pbm'}", "wb") as f:
            f.write(f"P5\n{wide} {high}\n255\n".encode() if levels else f"P4\n{wide} {high}\n".encode())
            for top in range(high, 0, -band):
                rows_ = np.asarray(made[max(top - band, 0) : top][::-1])
                f.write(rows_.tobytes() if levels else np.packbits(~rows_, axis=1).tobytes())
                step = 48
                cut = rows_[: len(rows_) // step * step, : wide // step * step]
                small.append(cut.reshape(len(cut) // step, step, wide // step, step).mean(axis=(1, 3)))
        picture = np.concatenate(small)
        Image.fromarray((picture if levels else 255 * picture).astype(np.uint8)).save(out / f"{plate}.png")
        said = f"{plate}: {wide} x {high} pixels {pixel:g} um apart, {wide * pixel / 1000:.1f} x {high * pixel / 1000:.1f} mm"
        if not levels:
            gds = encode.Gds(out / "chrome.gds")
            for bottom in range(0, high, band):
                a, b, r0, r1 = (cp.asnumpy(v).astype(np.int64) for v in encode.runs(cp.asarray(made[bottom : bottom + band])))
                gds.add(a * nm - wide * nm // 2, b * nm - wide * nm // 2,
                        (bottom + r0) * nm - high * nm // 2, (bottom + r1) * nm - high * nm // 2)
            gds.close()
            said += f"; chrome.gds, {gds.count} rectangles, {(out / 'chrome.gds').stat().st_size / 2**30:.2f} GB"
        planes._remove(made)
        del made
        print(f"{said}: {time.time() - start:.1f} s", flush=True)
    (out / "layout.txt").write_text("\n".join(legend) + "\n")
    print(f"wrote {out}/chrome.pbm, chrome.gds, relief.pgm, their pictures and layout.txt")
