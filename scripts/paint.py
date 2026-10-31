# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy", "pillow"]
# ///
"""Paint the site's watercolour plates.

Every image is rendered from a fixed seed, so a run reproduces the committed
files exactly. Pigment is modelled as optical density on paper: each wash adds
absorbance, and the colour seen is paper * exp(-absorbance), which mixes like
glazed transparent watercolour. Washes carry edge darkening (pigment drying at
the rim), granulation (pigment settling in the paper's tooth) and low-frequency
density variation; mist is painted by lifting density back out.

    uv run scripts/paint.py            # all plates
    uv run scripts/paint.py landscape  # one plate: landscape | paper | stroke | cover

Outputs:
    public/paint/landscape.webp   porch-morning ridges behind every page
    public/paint/paper.webp       tileable watercolour-paper grain for the page sheet
    public/paint/stroke.webp      a creek-water brush mark (alpha) for rules and the active nav item
    public/paint/almanac-cover.webp the roundel on the Almanac's title page (RGBA)
    public/paint/almanac-glaze.webp the golden glaze of the title page
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, zoom

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "public" / "paint"

PAPER = np.array([243, 238, 228], dtype=np.float32) / 255.0  # cold-press cotton, warm


def hex_rgb(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return np.array([int(h[i : i + 2], 16) for i in (0, 2, 4)], dtype=np.float32) / 255.0


def pigment(hex_colour: str) -> np.ndarray:
    """Absorbance per channel that turns paper into this colour at density 1."""
    c = np.clip(hex_rgb(hex_colour), 1e-3, None)
    return -np.log(np.minimum(c / PAPER, 0.999))


# ---------------------------------------------------------------- noise

def smooth_noise(rng, h, w, scale, periodic=False):
    """Unit-variance Gaussian-smoothed noise with feature size ~scale px."""
    f = max(1, int(scale // 6))
    hh, ww = max(4, h // f + 4), max(4, w // f + 4)
    n = rng.standard_normal((hh, ww)).astype(np.float32)
    n = gaussian_filter(n, scale / f, mode="wrap" if periodic else "reflect")
    if f > 1:
        n = zoom(n, f, order=3, mode="grid-wrap" if periodic else "reflect")
    n = n[:h, :w]
    return (n - n.mean()) / (n.std() + 1e-6)


def fbm(rng, h, w, base, octaves=4, gain=0.5, periodic=False):
    out = np.zeros((h, w), np.float32)
    amp, total, s = 1.0, 0.0, base
    for _ in range(octaves):
        out += amp * smooth_noise(rng, h, w, s, periodic)
        total += amp
        amp *= gain
        s = max(1.0, s / 2)
    return out / total


def noise1d(rng, n, scale):
    f = max(1, int(scale // 6))
    m = rng.standard_normal(n // f + 8).astype(np.float32)
    m = gaussian_filter(m, scale / f, mode="reflect")
    if f > 1:
        m = zoom(m, f, order=3)
    m = m[:n]
    return (m - m.mean()) / (m.std() + 1e-6)


def ridge_profile(rng, n, octaves):
    """A ridgeline: sum of 1D noises, (scale px, amplitude px)."""
    y = np.zeros(n, np.float32)
    for scale, amp in octaves:
        y += amp * noise1d(rng, n, scale)
    return y


# ---------------------------------------------------------------- paper

class Sheet:
    """A sheet of watercolour paper accumulating absorbance."""

    def __init__(self, h, w, seed, periodic=False):
        self.h, self.w = h, w
        self.rng = np.random.default_rng(seed)
        self.periodic = periodic
        self.A = np.zeros((h, w, 3), np.float32)
        tooth = 0.6 * smooth_noise(self.rng, h, w, 1.6, periodic) + 0.4 * smooth_noise(self.rng, h, w, 4.0, periodic)
        self.tooth = tooth  # paper relief: positive = hollow, where pigment settles
        self.yy, self.xx = np.mgrid[0:h, 0:w].astype(np.float32)

    def noise(self, scale, octaves=3):
        return fbm(self.rng, self.h, self.w, scale, octaves, periodic=self.periodic)

    def wash(self, mask, colour, density=1.0, edge=0.6, edge_width=3.0, gran=0.25,
             vary=0.18, vary_scale=120.0, cover=0.0):
        """Lay a wash over `mask` (0..1).

        edge     pigment pooled at the drying rim
        gran     granulation into the paper tooth
        vary     low-frequency unevenness of the wash
        cover    fraction of the existing glaze this wash hides (opaque-ish
                 overpainting, used where a near ridge sits over far mist)
        """
        m = np.clip(mask, 0, 1).astype(np.float32)
        d = density * m
        if vary:
            d *= 1.0 + vary * self.noise(vary_scale, 3)
        if edge:
            rim = np.clip(m - gaussian_filter(m, edge_width), 0, 1)
            d += edge * density * rim / (rim.max() + 1e-6) * 0.9
        if gran:
            fine = smooth_noise(self.rng, self.h, self.w, 2.2, self.periodic)
            d *= 1.0 + gran * (0.6 * self.tooth + 0.4 * fine)
        d = np.clip(d, 0, None)
        if np.any(cover):
            self.A *= (1.0 - cover * m)[..., None]
        self.A += d[..., None] * pigment(colour)[None, None, :]

    def lift(self, mask, amount):
        """Lift pigment back out with a damp brush (mist, light)."""
        self.A *= (1.0 - amount * np.clip(mask, 0, 1))[..., None]

    def stroke(self, pts, colour, width, density=1.0, soft=0.8):
        """An ink/paint line along polyline pts (N,2) in px."""
        m = np.zeros((self.h, self.w), np.float32)
        pts = np.asarray(pts, np.float32)
        for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
            n = int(max(abs(x1 - x0), abs(y1 - y0)) * 2) + 1
            xs = np.linspace(x0, x1, n).round().astype(int)
            ys = np.linspace(y0, y1, n).round().astype(int)
            ok = (xs >= 0) & (xs < self.w) & (ys >= 0) & (ys < self.h)
            m[ys[ok], xs[ok]] = 1.0
        m = gaussian_filter(m, width / 2.0)
        m = np.clip(m / (m.max() + 1e-6) * 2.2, 0, 1)
        if soft:
            m = gaussian_filter(m, soft)
        self.wash(m, colour, density, edge=0.3, edge_width=1.2, gran=0.15, vary=0.15, vary_scale=40)

    def render(self, light=0.018):
        img = PAPER[None, None, :] * np.exp(-self.A)
        # raking light across the paper relief
        relief = gaussian_filter(self.tooth, 0.8, mode="wrap" if self.periodic else "reflect")
        img *= (1.0 - light * relief)[..., None]
        return np.clip(img, 0, 1)


def save(img, path, quality=82, size=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.fromarray((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8))
    if size:
        im = im.resize(size, Image.LANCZOS)
    im.save(path, "WEBP", quality=quality, method=6)
    print(f"wrote {path.relative_to(ROOT)}  {im.size[0]}x{im.size[1]}  {path.stat().st_size // 1024} KB")


def soft_step(x, width):
    """0 below 0, 1 above, smooth over `width`."""
    return np.clip(0.5 + x / (2 * width), 0, 1) ** 1.0


# ---------------------------------------------------------------- landscape

def landscape():
    H, W = 1400, 2400
    s = Sheet(H, W, seed=1987)
    yy, xx = s.yy, s.xx
    rng = s.rng

    # Sky: wet-in-wet, violet-grey overhead warming to apricot over the ridges.
    t = yy / H
    overhead = np.clip(1.0 - t / 0.42, 0, 1) ** 1.3
    s.wash(overhead * (0.55 + 0.25 * s.noise(260, 2)), "#CDCAD9", density=0.6, edge=0, gran=0.0, vary=0.25, vary_scale=300)
    glow = np.exp(-((t - 0.40) / 0.11) ** 2) * (0.75 + 0.25 * np.cos((xx / W - 0.28) * 2.2))
    s.wash(glow, "#EFD3BA", density=0.6, edge=0, gran=0.0, vary=0.3, vary_scale=220)
    # long low cloud streaks, laid wet-in-wet
    streak = fbm(rng, H // 2 + 2, W // 10 + 2, 12, 3)
    streak = zoom(streak, (2, 10), order=3)[:H, :W]
    cloud = np.clip(streak * 0.9 - 0.1, 0, 1) * np.exp(-((t - 0.24) / 0.1) ** 2)
    s.wash(gaussian_filter(cloud, 4), "#BFB8CE", density=0.4, edge=0.25, edge_width=12, gran=0.0, vary=0.2)
    # morning light low on the left, lifted out of the sky
    sun = np.exp(-(((xx - 0.22 * W) / (0.22 * W)) ** 2 + ((yy - 0.37 * H) / (0.13 * H)) ** 2))
    s.lift(sun, 0.45)

    # Ridges, far to near: (crest y, colour, density, octaves, canopy, edge sharpness, granulation, fall-off depth)
    ridges = [
        (0.455, "#B3A6C9", 0.72, [(520, 30), (190, 11), (60, 3)], 0.0, 3.5, 0.05, 0.13),
        (0.535, "#A2A3BC", 0.75, [(430, 34), (150, 12), (45, 3)], 0.0, 2.2, 0.08, 0.13),
        (0.615, "#8A9DA6", 0.78, [(380, 36), (120, 13), (30, 4)], 1.6, 1.6, 0.12, 0.14),
        (0.705, "#6F8A84", 0.82, [(340, 40), (110, 15), (24, 5)], 3.2, 1.3, 0.16, 0.16),
        (0.800, "#587466", 0.86, [(300, 34), (90, 12), (14, 6)], 5.0, 1.1, 0.20, 0.22),
    ]
    for i, (cy, col, dens, octs, canopy, sharp, gran, fall) in enumerate(ridges):
        crest = cy * H + ridge_profile(rng, W, octs)
        if canopy:
            # tree canopy: fine bumps along the near ridgelines
            bumps = np.abs(noise1d(rng, W, 3.0)) * canopy + np.abs(noise1d(rng, W, 7.0)) * canopy * 0.8
            crest = crest - bumps
        # wet edge: the boundary itself wanders a little in 2D
        wander = 3.0 * s.noise(30, 2)
        depth = yy - crest[None, :] + wander
        m = soft_step(depth, sharp)
        # density falls away below the crest into mist
        fade = np.exp(-np.clip(depth, 0, None) / (fall * H))
        body = m * (0.30 + 0.70 * fade)
        s.wash(body, col, density=dens, edge=0.75, edge_width=2.5 + i * 0.4, gran=gran,
               vary=0.22, vary_scale=160, cover=0.55 * m)
        # mist pooling in the hollow beneath this ridge
        mist_y = crest + (0.06 + 0.008 * i) * H
        band = np.exp(-((yy - mist_y[None, :]) / (0.034 * H)) ** 2)
        wisps = np.clip(0.6 + 0.7 * s.noise(160, 3), 0, 1)
        s.lift(band * wisps, 0.85 - 0.05 * i)

    # Foreground: the gravel drive darkened by rain.
    ground = 0.925 * H + ridge_profile(rng, W, [(600, 12), (140, 5)])
    gm = soft_step(yy - ground[None, :] + 3.0 * s.noise(24, 2), 1.5)
    s.lift(gm, 0.7)
    s.wash(gm, "#C9B89C", density=0.75, edge=0.8, edge_width=2.5, gran=0.55, vary=0.25, vary_scale=90)
    # the rain darkens the gravel in long puddled patches: the same stone, richer
    wet = fbm(rng, H // 2 + 2, W // 6 + 2, 10, 3)
    wet = zoom(wet, (2, 6), order=3)[:H, :W]
    wet = np.clip(wet * 1.4 - 0.15, 0, 1) * gm
    s.wash(gaussian_filter(wet, 1.5), "#9C8C7B", density=0.6, edge=0.9, edge_width=3.0, gran=0.5, vary=0.2, vary_scale=40)
    stones = (rng.random((H, W)) > 0.9965).astype(np.float32)
    stones = np.clip(gaussian_filter(stones, 1.1) * 12, 0, 1) * gm
    s.wash(stones, "#8C7C6A", density=0.45, edge=0, gran=0.1, vary=0)

    img = s.render()
    save(img, OUT / "landscape.webp", quality=80, size=(1920, 1120))


# ---------------------------------------------------------------- paper tile

def paper():
    N = 512
    s = Sheet(N, N, seed=7, periodic=True)
    # under text the sheet must stay quiet: a fine tooth, no blotches.
    # The sizing is even to within a breath, and the tooth is mostly the
    # finest grain, with only a trace of the coarser pits that read as mottling.
    s.tooth = 0.85 * smooth_noise(s.rng, N, N, 1.3, True) + 0.15 * smooth_noise(s.rng, N, N, 4.0, True)
    s.wash(np.ones((N, N), np.float32) * 0.5 * (1 + 0.04 * s.noise(110, 3)), "#E9E1D2",
           density=0.10, edge=0, gran=0.08, vary=0.0)
    img = s.render(light=0.010)
    save(img, OUT / "paper.webp", quality=90)


# ---------------------------------------------------------------- brush marks

def strokes():
    """A single-pigment creek-water brush mark with alpha.

    The mark is a dry-ish horizontal stroke: dense where the loaded brush
    touched down on the left, breaking up and thinning toward the lift-off.
    """
    H, W = 72, 640
    s = Sheet(H, W, seed=11)
    yy, xx = s.yy, s.xx
    u = xx / W
    u1 = np.arange(W, dtype=np.float32) / W
    centre = H * 0.52 + 3 * noise1d(s.rng, W, 160) - 4.0 * u1
    # round touch-down, a long even body, a feathered lift-off
    profile = np.sqrt(np.clip(u1 / 0.035, 0, 1)) * (1 - np.clip((u1 - 0.55) / 0.45, 0, 1) ** 1.6)
    half = H * 0.25 * profile * (1 + 0.08 * noise1d(s.rng, W, 50))
    d = np.abs(yy - centre[None, :]) - half[None, :]
    m = soft_step(-d + 0.8 * s.noise(4, 2), 1.2)
    # the brush runs dry toward the end: hair-streaks of bare paper along the stroke
    hairs = noise1d(s.rng, H * 8, 14)[::8][:H, None] * np.ones((1, W), np.float32)
    hairs += 0.5 * s.noise(18, 2)
    dry = soft_step(hairs + 2.0 - 3.2 * u1[None, :], 1.2)
    s.wash(m * (0.3 + 0.7 * dry), "#5A7B7B", density=1.0, edge=0.8, edge_width=1.6, gran=0.35, vary=0.25, vary_scale=80)
    A = s.A.mean(axis=2)
    alpha = np.clip(1 - np.exp(-A), 0, 1)
    alpha = alpha / alpha.max()
    rgba = np.zeros((H, W, 4), np.float32)
    rgba[..., :3] = hex_rgb("#4F7473")
    rgba[..., 3] = alpha
    path = OUT / "stroke.webp"
    Image.fromarray((rgba * 255 + 0.5).astype(np.uint8), "RGBA").save(path, "WEBP", quality=85, method=6)
    print(f"wrote {path.relative_to(ROOT)}  {W}x{H}  {path.stat().st_size // 1024} KB")


# ---------------------------------------------------------------- almanac cover

def cover():
    """The roundel on the Almanac's title page: the porch view at first light.

    The morning sun clears the far ridge on the left while the waning moon is
    still up on the right; mist lies in the hollows, and the porch rail runs
    across the foreground. Painted inside a circle on bare paper, which renders
    white so the plate prints onto the sheet with multiply blending; the month
    ring around it is lettered in SVG on the page.
    """
    N = 1000
    s = Sheet(N, N, seed=1802)
    yy, xx = s.yy, s.xx
    rng = s.rng
    R = 0.43 * N
    rr = np.hypot(xx - N / 2, yy - N / 2)
    disc = np.clip((R - rr) / 1.5 + 0.5, 0, 1)
    t = (yy - (N / 2 - R)) / (2 * R)   # 0 at the top of the roundel, 1 at the bottom

    # Sky: violet overhead, warming to apricot over the ridges
    s.wash(disc * np.clip(1 - t / 0.55, 0, 1) ** 1.2, "#C9C3DB", density=0.75, edge=0, gran=0.0, vary=0.25, vary_scale=200)
    glow = np.exp(-((t - 0.47) / 0.13) ** 2) * (0.7 + 0.3 * np.cos((xx / N - 0.3) * 2.6))
    s.wash(disc * glow, "#EFD0B0", density=0.75, edge=0, gran=0.0, vary=0.3, vary_scale=160)

    # The sun, just clear of the far ridge: a lifted disc with a halo of warm glaze
    sx, sy = 0.34 * N, 0.475 * N
    sd = np.hypot(xx - sx, yy - sy)
    s.lift(disc * np.exp(-(sd / 120) ** 2), 0.8)
    s.wash(disc * np.exp(-(sd / 90) ** 2), "#F0CF98", density=0.6, edge=0, gran=0.0, vary=0.1)
    sun = np.clip((34 - sd + 0.5 * s.noise(10, 2)) / 1.5 + 0.5, 0, 1)
    s.lift(sun, 1.0)
    s.wash(sun, "#EDC774", density=0.75, edge=0.9, edge_width=2.5, gran=0.08, vary=0.15, vary_scale=20)

    # The waning moon, still up in the west: a pale crescent lifted from the violet
    mx, my = 0.69 * N, 0.255 * N
    md = np.hypot(xx - mx, yy - my)
    lit = np.clip((26 - md) / 1.5 + 0.5, 0, 1) * np.clip((np.hypot(xx - mx - 11, yy - my + 4) - 23) / 1.5 + 0.5, 0, 1)
    s.lift(np.clip((26 - md) / 1.5 + 0.5, 0, 1), 0.35)
    s.lift(gaussian_filter(lit, 0.8), 0.9)
    s.wash(gaussian_filter(lit, 0.8), "#EDE6D2", density=0.3, edge=0.5, edge_width=1.2, gran=0.0, vary=0)

    # Ridges, far to near, with mist pooled beneath each
    ridges = [
        (0.515, "#B1A4C8", 0.72, [(260, 16), (90, 6), (30, 2)], 0.0, 2.4, 0.10),
        (0.585, "#9FA0BA", 0.75, [(220, 18), (75, 7), (24, 2)], 0.0, 1.8, 0.11),
        (0.655, "#879AA4", 0.80, [(190, 18), (60, 7), (16, 2)], 1.2, 1.4, 0.12),
        (0.730, "#6C8882", 0.84, [(170, 18), (55, 7), (12, 3)], 2.2, 1.2, 0.13),
    ]
    for i, (cy, col, dens, octs, canopy, sharp, fall) in enumerate(ridges):
        crest = cy * N + ridge_profile(rng, N, octs)
        if canopy:
            crest -= np.abs(noise1d(rng, N, 2.5)) * canopy + np.abs(noise1d(rng, N, 5.0)) * canopy * 0.8
        depth = yy - crest[None, :] + 1.5 * s.noise(18, 2)
        m = soft_step(depth, sharp) * disc
        s.wash(m * (0.3 + 0.7 * np.exp(-np.clip(depth, 0, None) / (fall * N))), col, density=dens,
               edge=0.75, edge_width=2.0 + 0.3 * i, gran=0.12 + 0.03 * i, vary=0.2, vary_scale=120, cover=0.55 * m)
        band = np.exp(-((yy - crest[None, :] - 0.045 * N) / (0.02 * N)) ** 2)
        s.lift(disc * band * np.clip(0.6 + 0.7 * s.noise(110, 3), 0, 1), 0.8 - 0.05 * i)

    # The near ridge in hemlock, dark against the mist
    crest = 0.80 * N + ridge_profile(rng, N, [(150, 12), (45, 5)])
    crest -= np.abs(noise1d(rng, N, 2.2)) * 4.0 + np.abs(noise1d(rng, N, 5.0)) * 3.0
    m = soft_step(yy - crest[None, :] + 1.2 * s.noise(10, 2), 1.0) * disc
    s.wash(m, "#587466", density=0.9, edge=0.8, edge_width=2.0, gran=0.22, vary=0.22, vary_scale=60, cover=0.6 * m)

    # The porch rail across the foreground: weathered top rail and balusters
    wood = "#7D6B5B"
    rail = np.zeros((N, N), np.float32)
    top, bot = 0.835 * N, 0.905 * N
    rail += (np.abs(yy - top) < 6) * 1.0
    for bx in np.arange(0.5 * N - 7 * 62, 0.5 * N + 7 * 62 + 1, 62):
        bx += rng.normal(0, 1.0)
        rail += ((np.abs(xx - bx) < 6) & (yy > top) & (yy < bot)) * 1.0
    rail += (yy > bot) * 1.0   # the porch floor
    rail = gaussian_filter(np.clip(rail + 0.25 * s.noise(3, 2) * (rail > 0), 0, 1), 0.8) * disc
    s.wash(rail, wood, density=1.0, edge=0.8, edge_width=1.5, gran=0.35, vary=0.3, vary_scale=40, cover=0.8 * rail)
    # first light along the top of the rail
    s.lift(gaussian_filter((np.abs(yy - (top - 4)) < 2).astype(np.float32), 0.8) * rail, 0.45)

    # A pencil ring around the roundel
    a = np.linspace(0, 2 * np.pi, 720)
    for rad, wdt, dn in ((R + 3, 1.6, 0.55), (R + 11, 0.9, 0.4)):
        ring = np.stack([N / 2 + rad * np.cos(a), N / 2 + rad * np.sin(a)], 1)
        s.stroke(ring, "#7E7062", width=wdt, density=dn, soft=0.5)

    # The roundel is a vignette set into the yellow cover: inside the disc it
    # keeps its own paper, so it is saved opaque there; outside, only the
    # pencil rings remain, as ink with alpha.
    img = s.render(light=0.0)
    ink = hex_rgb("#7E7062")
    dark = np.clip(1 - img / PAPER[None, None, :], 0, 1).max(axis=2) / (1 - ink / PAPER).max()
    alpha = np.maximum(disc, np.clip(dark, 0, 1) * (rr > R))
    rgb = np.where((rr > R)[..., None], ink[None, None, :], img)
    rgba = np.concatenate([rgb, alpha[..., None]], axis=2)
    save(rgba, OUT / "almanac-cover.webp", quality=88, size=(800, 800))

    glaze()


def glaze():
    """The golden glaze of the Almanac's title page.

    Farmers' almanacs come in a yellow cover; this one is laid in gamboge over
    raw sienna, a little uneven, darker where the wash dried at its rim, with
    the paper left bare in a ragged margin inside the ruled border. Rendered
    white-normalised for multiply blending and stretched to the title page, so
    the variation is kept broad and the granulation fine.
    """
    Hh, Ww = 1100, 900
    s = Sheet(Hh, Ww, seed=1792)
    yy, xx = s.yy, s.xx
    inset = 16
    edge = np.minimum.reduce([xx - inset, Ww - inset - xx, yy - inset, Hh - inset - yy])
    m = soft_step(edge + 2.2 * s.noise(40, 3) + 0.8 * s.noise(6, 2), 1.6)
    s.wash(m, "#E3BD62", density=0.72, edge=0.9, edge_width=10.0, gran=0.22, vary=0.16, vary_scale=320)
    # a second, warmer glaze pooled toward the foot of the page, as the board tilted
    foot = np.clip((yy / Hh - 0.25) / 0.75, 0, 1) ** 1.5
    s.wash(m * foot * (0.7 + 0.3 * s.noise(200, 2)), "#D9A65A", density=0.22, edge=0, gran=0.1, vary=0.2, vary_scale=260)
    # light lifted where the sun sits behind the lettering
    s.lift(np.exp(-(((xx - Ww / 2) / (0.45 * Ww)) ** 2 + ((yy - 0.18 * Hh) / (0.2 * Hh)) ** 2)), 0.18)
    img = np.clip(s.render(light=0.0) / PAPER[None, None, :], 0, 1)
    save(img, OUT / "almanac-glaze.webp", quality=84)


PLATES = {"landscape": landscape, "paper": paper, "stroke": strokes, "cover": cover}

if __name__ == "__main__":
    for name in sys.argv[1:] or PLATES:
        PLATES[name]()
