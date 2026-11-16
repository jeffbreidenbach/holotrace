"""An eye looking at the plate from far away.

Far from the plate, an eye takes in the light leaving it in a narrow
range of directions: those within its pupil's angle, `aperture` (the
numerical aperture: the sine of the half-angle), around the direction
it looks from. Focused at a depth behind the plate, it sees the light
as it is in that plane, so it is found by running the plate's light
back to there. Both are done with plane waves (the angular spectrum
method): the light is split into plane waves by a Fourier transform,
those outside the pupil are dropped, and each is moved back by its own
phase.

The views are of the plane in focus, in its own lengths: far away, the
eye sees no perspective, so moving it only shifts things at other depths
across what is in focus (parallax).
"""

import numpy as np


class Eye:
    """Looks at `light` leaving a plate of pixels `pitch` apart, of
    `wavelength`."""

    def __init__(self, light, pitch, wavelength):
        self.wavelength = wavelength
        self.spectrum = np.fft.fft2(light)
        n = light.shape[0]
        # Each plane wave's direction, as sines along x (columns) and y.
        self.sx = (np.fft.fftfreq(n, pitch) * wavelength)[None, :]
        self.sy = (np.fft.fftfreq(n, pitch) * wavelength)[:, None]

    def view(self, direction, aperture, depth):
        """What the eye sees looking along `direction` (sines along x and
        y), through a pupil of numerical aperture `aperture`, focused
        `depth` behind the plate: the brightness across that plane."""
        sx, sy = direction
        pupil = (self.sx - sx) ** 2 + (self.sy - sy) ** 2 < aperture**2
        # Each plane wave's phase moving back by depth: k cos(angle) depth.
        cos = np.sqrt(np.maximum(0.0, 1 - self.sx**2 - self.sy**2))
        back = np.exp(-2j * np.pi / self.wavelength * cos * depth)
        return np.abs(np.fft.ifft2(self.spectrum * pupil * back)) ** 2
