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
    uv run scripts/paint.py landscape  # one plate: landscape | paper | stroke | cover | map

Outputs:
    public/paint/landscape.webp   porch-morning ridges behind every page
    public/paint/paper.webp       tileable watercolour-paper grain for the page sheet
    public/paint/stroke.webp      a creek-water brush mark (alpha) for rules and the active nav item
    public/paint/almanac-cover.webp the roundel on the Almanac's title page
    public/paint/almanac-map.webp the Almanac's plate of the holdings
    src/data/almanac-map.json     region outlines and mark positions for the map overlay
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, zoom

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "public" / "paint"
DATA = ROOT / "src" / "data"

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
    im = Image.fromarray((img * 255 + 0.5).astype(np.uint8))
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

    img = np.clip(s.render(light=0.0) / PAPER[None, None, :], 0, 1)
    save(img, OUT / "almanac-cover.webp", quality=84, size=(800, 800))


# ---------------------------------------------------------------- map

def blob(cx, cy, rx, ry, rng, wob=0.12, n=96, rot=0.0):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = 1 + wob * (0.6 * noise_loop(rng, n, 3) + 0.4 * noise_loop(rng, n, 7))
    x, y = rx * r * np.cos(a), ry * r * np.sin(a)
    c, si = np.cos(rot), np.sin(rot)
    return np.stack([cx + c * x - si * y, cy + si * x + c * y], 1)


def noise_loop(rng, n, k):
    """Periodic smooth noise around a loop with ~k undulations."""
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    out = np.zeros(n)
    for j in range(1, k + 1):
        out += rng.standard_normal() * np.cos(j * a) / j + rng.standard_normal() * np.sin(j * a) / j
    return out / (np.abs(out).max() + 1e-6)


def poly_mask(pts, h, w):
    from PIL import ImageDraw
    im = Image.new("L", (w, h), 0)
    ImageDraw.Draw(im).polygon([tuple(p) for p in pts], fill=255)
    return np.asarray(im, np.float32) / 255.0


def path_d(pts, scale=1.0):
    p = np.asarray(pts) * scale
    return "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in p) + " Z"


def clip_halfplane(poly, p, n):
    """Keep the part of polygon `poly` where (x - p) . n <= 0 (Sutherland-Hodgman)."""
    out = []
    k = len(poly)
    for i in range(k):
        a, b = poly[i], poly[(i + 1) % k]
        da, db = np.dot(a - p, n), np.dot(b - p, n)
        if da <= 0:
            out.append(a)
        if da * db < 0:
            out.append(a + (b - a) * (da / (da - db)))
    return np.array(out)


def voronoi_cells(region, seeds):
    """Split a polygon into the Voronoi cells of `seeds`."""
    cells = []
    for i, si in enumerate(seeds):
        poly = region.copy()
        for j, sj in enumerate(seeds):
            if i != j and len(poly):
                poly = clip_halfplane(poly, (si + sj) / 2, sj - si)
        cells.append(poly)
    return cells


def dab_mask(s, pts, radius, jitter=0.25):
    """Round brush dabs (trees, shrubs) at pts."""
    from PIL import ImageDraw
    im = Image.new("L", (s.w, s.h), 0)
    dr = ImageDraw.Draw(im)
    for x, y in pts:
        r = radius * max(0.4, 1 + jitter * s.rng.standard_normal())
        # a clump is a few overlapping touches of the brush, not a disc
        for _ in range(3):
            ox, oy = s.rng.normal(0, r * 0.35, 2)
            rr = r * (0.6 + 0.25 * s.rng.random())
            dr.ellipse([x + ox - rr, y + oy - rr * 0.85, x + ox + rr, y + oy + rr * 0.85], fill=255)
    m = np.asarray(im, np.float32) / 255.0
    m = gaussian_filter(m, 0.9)
    return np.clip(m + 0.3 * s.noise(2.5, 2) * (m > 0.05), 0, 1)


def mapplate():
    """Plate I of the Almanac: the holdings, in plan, with the hills drawn in elevation.

    The creek runs down from the woods under the ridge. West of it lie the tool
    fields in two blocks either side of the lane: agentic tools above, science
    infrastructure below, one field per tool in the order of the `agentic` and
    `science` arrays in `almanac/index.astro`. East of the creek is the orchard
    of writing, its first rows planted, the rest left in meadow for what comes.
    The farmhouse sits by the ford; the far corner is left to mist.
    """
    from scipy.ndimage import distance_transform_edt

    H, W = 1000, 1600
    s = Sheet(H, W, seed=2026)
    rng = s.rng
    yy, xx = s.yy, s.xx
    P = lambda fx, fy: np.array([fx * W, fy * H])

    # ---- the hills along the top, drawn in elevation as old estate maps do
    for cy, col, dens, octs, canopy, fall in [
        (0.075, "#B5ABCB", 0.55, [(420, 18), (140, 7)], 0.0, 0.05),
        (0.125, "#9D90BB", 0.68, [(330, 22), (110, 8), (35, 3)], 0.0, 0.06),
        (0.180, "#8A9DA6", 0.70, [(300, 20), (90, 7), (25, 3)], 1.2, 0.06),
        (0.235, "#6F8A84", 0.72, [(260, 16), (70, 6), (18, 3)], 2.2, 0.06),
    ]:
        crest = cy * H + ridge_profile(rng, W, octs)
        if canopy:
            crest -= np.abs(noise1d(rng, W, 3.0)) * canopy + np.abs(noise1d(rng, W, 6.0)) * canopy
        depth = yy - crest[None, :] + 2.0 * s.noise(24, 2)
        m = soft_step(depth, 1.6)
        s.wash(m * np.exp(-np.clip(depth, 0, None) / (fall * H)), col, density=dens, edge=0.8,
               edge_width=2.2, gran=0.15, vary=0.2, cover=0.5 * m * np.exp(-np.clip(depth, 0, None) / (0.03 * H)))
        band = np.exp(-((yy - crest[None, :] - 0.045 * H) / (0.018 * H)) ** 2)
        s.lift(band * np.clip(0.6 + 0.7 * s.noise(100, 2), 0, 1), 0.75)

    # ---- the valley floor: a pale wash of meadow, wet-in-wet
    floor = soft_step(yy - (0.28 * H + 25 * s.noise(200, 2)), 40)
    s.wash(floor * (0.6 + 0.4 * s.noise(220, 2)), "#DCDABF", density=0.42, edge=0, gran=0.15, vary=0.35, vary_scale=180)

    # ---- the creek, from the woods under the ridge down past the ford
    t = np.linspace(0, 1, 500)
    cx = W * (0.600 - 0.10 * t + 0.040 * np.sin(t * 7.0 + 0.4) + 0.012 * np.sin(t * 23 + 1.0))
    cy = H * (0.27 + 0.76 * t)
    creek = np.stack([cx, cy], 1)
    dist = np.full((H, W), 1e4, np.float32)
    rad = np.zeros((H, W), np.float32)
    for (x, y), tt in zip(creek, t):
        xi, yi = int(x), int(y)
        y0, y1, x0, x1 = max(0, yi - 60), min(H, yi + 60), max(0, xi - 60), min(W, xi + 60)
        if y0 >= y1 or x0 >= x1:
            continue
        dd = np.hypot(xx[y0:y1, x0:x1] - x, yy[y0:y1, x0:x1] - y)
        closer = dd < dist[y0:y1, x0:x1]
        dist[y0:y1, x0:x1] = np.where(closer, dd, dist[y0:y1, x0:x1])
        rad[y0:y1, x0:x1] = np.where(closer, 2.0 + 5.5 * tt, rad[y0:y1, x0:x1])
    water = np.clip((rad + 1.2 * s.noise(8, 2) - dist) / 1.0 + 0.5, 0, 1) * soft_step(yy - 0.275 * H, 4)
    banks = np.exp(-np.clip(dist - rad, 0, None) / 18.0) * (dist < 200)
    s.wash(banks * (0.7 + 0.3 * s.noise(30, 2)), "#A5B79D", density=0.5, edge=0, gran=0.3, vary=0.3)

    def densify(pts, wobble, closed=True):
        """A hand-drawn line through the given corners, densified and wavered."""
        pts = np.asarray(pts, np.float64)
        seq = list(zip(pts, np.roll(pts, -1, axis=0))) if closed else list(zip(pts[:-1], pts[1:]))
        out = []
        for p0, p1 in seq:
            n = max(2, int(np.hypot(*(p1 - p0)) // 16))
            for k in range(n):
                out.append(p0 + (p1 - p0) * k / n)
        if not closed:
            out.append(pts[-1])
        out = np.array(out)
        return out + wobble * np.stack([noise_loop(rng, len(out), 9), noise_loop(rng, len(out), 9)], 1)

    def block(corners, cuts):
        """A block of fields: a quadrilateral (TL, TR, BR, BL) cut into strips across its width."""
        tl, tr, br, bl = (P(*c) for c in corners)
        fr = [0.0, *cuts, 1.0]
        top = [tl + (tr - tl) * f for f in fr]
        bot = [bl + (br - bl) * f for f in fr]
        cells = [densify([top[i], top[i + 1], bot[i + 1], bot[i]], 3.0) for i in range(len(fr) - 1)]
        return densify([tl, tr, br, bl], 4.0), cells

    agentic, agentic_cells = block([(0.060, 0.375), (0.470, 0.360), (0.455, 0.565), (0.045, 0.585)], [0.34, 0.68])
    science, science_cells = block([(0.050, 0.665), (0.455, 0.650), (0.430, 0.865), (0.080, 0.885)], [0.33, 0.66])

    # Each field its own crop: (colour, texture, furrow angle)
    paint = [
        ("#CDB97E", "furrow", 0.25), ("#A9B784", "plain", 0.0), ("#C9A774", "furrow", 1.35),
        ("#B7BE8E", "stipple", 0.0), ("#D4C391", "furrow", -0.15), ("#9EB29A", "furrow", 1.5),
    ]
    hedge_pts = []
    for cell, (col, tex, ang) in zip(agentic_cells + science_cells, paint):
        cm = poly_mask(cell, H, W)
        inner = distance_transform_edt(cm)
        fm = soft_step(inner - 6 + 2.0 * s.noise(10, 2), 2.0) * cm
        s.wash(fm, col, density=0.75, edge=0.9, edge_width=3.0, gran=0.45, vary=0.3, vary_scale=70)
        if tex == "furrow":
            ca, sa = np.cos(ang), np.sin(ang)
            stripe = 0.5 + 0.5 * np.sin((xx * ca + yy * sa) / 4.0 + 0.6 * s.noise(60, 2))
            s.wash(fm * stripe ** 4 * soft_step(inner - 13, 3), col, density=0.5, edge=0, gran=0.4, vary=0.2)
        elif tex == "stipple":
            dots = (rng.random((H, W)) > 0.986).astype(np.float32)
            dots = np.clip(gaussian_filter(dots, 1.2) * 10, 0, 1)
            s.wash(dots * soft_step(inner - 12, 2), "#7C8F62", density=0.55, edge=0, gran=0.2, vary=0)
        # hedgerows: a close line of small hemlock dabs along every field edge
        for a, b in zip(cell, np.roll(cell, -1, axis=0)):
            n = int(np.hypot(*(b - a)) // 7)
            for k in range(n):
                if rng.random() < 0.7:
                    hedge_pts.append(a + (b - a) * (k + 0.5) / max(n, 1) + rng.normal(0, 1.2, 2))
    s.wash(dab_mask(s, hedge_pts, 3.4, jitter=0.3), "#62806A", density=0.6, edge=0.8, edge_width=1.0, gran=0.3, vary=0.25, vary_scale=30)

    # Writing: the orchard across the creek, fenced; the first rows are planted
    oc, ow, oh, rot = P(0.765, 0.485), 0.205 * W, 0.30 * H, -0.06
    c_, s_ = np.cos(rot), np.sin(rot)
    corner = lambda u, v: oc + np.array([c_ * u * ow / 2 - s_ * v * oh / 2, s_ * u * ow / 2 + c_ * v * oh / 2])
    writing = densify([corner(-1, -1), corner(1, -1), corner(1, 1), corner(-1, 1)], 3.0)
    wm = gaussian_filter(poly_mask(writing, H, W), 2.0)
    s.wash(wm, "#C3CB9F", density=0.45, edge=0.8, edge_width=4.0, gran=0.3, vary=0.3, vary_scale=90)
    plot = densify([corner(-1, -1), corner(1, -1), corner(1, 0.05), corner(-1, 0.05)], 2.0)
    pm = poly_mask(plot, H, W)
    pin = distance_transform_edt(pm)
    s.wash(soft_step(pin - 6, 2) * pm * (0.8 + 0.2 * s.noise(40, 2)), "#AFC08E", density=0.5, edge=0.6, gran=0.3)
    trees = []
    for i in range(-6, 7):
        for j in range(-4, 1):
            q = corner(i / 6.6, (j + 0.5) / 4.4 - 0.05) + rng.normal(0, 1.2, 2)
            if pin[int(q[1]), int(q[0])] > 10:
                trees.append(q)
    s.wash(dab_mask(s, trees, 6.0, jitter=0.1), "#6C8A5E", density=0.8, edge=1.0, edge_width=1.5, gran=0.35, vary=0.2, vary_scale=40)
    # the post-and-rail fence
    fence = np.vstack([writing, writing[:1]])
    s.stroke(fence, "#8E7B66", width=1.6, density=0.55, soft=0.4)

    # Uncharted: left as paper, only the mist that hides it
    uncharted = blob(*P(0.83, 0.845), 0.15 * W, 0.13 * H, rng, wob=0.12, rot=-0.1)
    um = gaussian_filter(poly_mask(uncharted, H, W), 16.0)
    s.lift(um, 0.85)
    s.wash(um * np.clip(0.5 + 0.6 * s.noise(60, 2), 0, 1), "#CFC8DD", density=0.55, edge=0, gran=0.05, vary=0.3)

    # Woods: the forest margin under the ridge, a woodlot, and trees along the creek.
    # Each is a soft wash for the canopy with crowns dabbed into it.
    def woodland(mask, n, r):
        ys, xs = np.nonzero(mask > 0.5)
        if not len(xs):
            return
        pick = rng.choice(len(xs), size=min(n, len(xs)), replace=False)
        crowns = dab_mask(s, np.stack([xs[pick], ys[pick]], 1).astype(float), r, jitter=0.25)
        body = gaussian_filter(mask, 3.0) * (0.6 + 0.4 * s.noise(30, 2))
        s.wash(np.clip(body * 0.5 + crowns, 0, 1), "#6F8C72", density=0.6, edge=0.8, edge_width=2.0, gran=0.35, vary=0.3, vary_scale=30)
        shade = np.clip(crowns - np.roll(np.roll(crowns, -4, axis=0), -3, axis=1), 0, 1)
        s.wash(gaussian_filter(shade, 1.0), "#3F5C4E", density=0.6, edge=0.3, gran=0.3, vary=0.2)

    edge_y = 0.30 * H + 14 * s.noise(90, 2) + 10 * s.noise(25, 2)
    margin = soft_step(edge_y - yy, 2) * soft_step(yy - 0.245 * H, 6)
    margin *= 1 - np.clip(water * 3, 0, 1)
    woodland(margin, 520, 6.5)
    lot = gaussian_filter(poly_mask(blob(*P(0.03, 0.96), 0.09 * W, 0.09 * H, rng, wob=0.2), H, W), 2)
    woodland(lot, 120, 7.0)
    lot2 = gaussian_filter(poly_mask(blob(*P(0.97, 0.36), 0.05 * W, 0.08 * H, rng, wob=0.2), H, W), 2)
    woodland(lot2, 70, 6.5)
    river_trees = []
    for (x, y), tt in zip(creek[::4], t[::4]):
        if 0.30 * H < y < 0.97 * H and rng.random() < 0.85:
            side = rng.choice([-1, 1])
            off = (14 + 6 * tt + rng.normal(0, 3)) * side
            q = np.array([x + off, y + rng.normal(0, 3)])
            if 0 <= q[1] < H and 0 <= q[0] < W:
                river_trees.append(q)
    s.wash(dab_mask(s, river_trees, 6.5, jitter=0.25), "#6F8C72", density=0.65, edge=0.9, edge_width=1.5, gran=0.35, vary=0.3, vary_scale=30)

    # The water itself, last, so it reads through everything
    s.wash(water, "#5A7B7B", density=1.0, edge=1.0, edge_width=1.5, gran=0.3, vary=0.2, vary_scale=50, cover=0.6 * water)

    # The farmhouse by the ford, and the gravel lane: west over the ford between
    # the tool blocks, north-east to the orchard gate
    house = P(0.585, 0.745)
    lane_w = np.array([house + [-14, -6], P(0.52, 0.64), P(0.45, 0.615), P(0.25, 0.618), P(0.05, 0.625)])
    lane_e = np.array([house + [10, -12], P(0.62, 0.69), P(0.665, 0.655), corner(-0.55, 1)])
    for ln in (lane_w, lane_e):
        sm = np.array([np.interp(np.linspace(0, len(ln) - 1, 80), np.arange(len(ln)), ln[:, k]) for k in (0, 1)]).T
        s.stroke(sm, "#A89577", width=5, density=0.55)
    hx, hy = house
    roof = np.array([[hx - 20, hy - 2], [hx, hy - 18], [hx + 20, hy - 2]])
    body = np.array([[hx - 15, hy - 3], [hx + 15, hy - 3], [hx + 15, hy + 14], [hx - 15, hy + 14]])
    s.wash(gaussian_filter(poly_mask(body, H, W), 0.7), "#C9B79C", density=0.9, edge=0.8, edge_width=1.0, gran=0.2)
    s.wash(gaussian_filter(poly_mask(roof, H, W), 0.7), "#8E6B5E", density=0.9, edge=0.8, edge_width=1.0, gran=0.2)
    porch = np.array([[hx - 15, hy + 10], [hx + 15, hy + 10], [hx + 15, hy + 14], [hx - 15, hy + 14]])
    s.wash(gaussian_filter(poly_mask(porch, H, W), 0.6), "#6E5A4C", density=0.8, edge=0.3, gran=0.1)

    # Neat line: a double pencil rule around the plate.
    for mm, wdt, dn in ((12, 2.0, 0.6), (21, 1.0, 0.45)):
        box = np.array([[mm, mm], [W - mm, mm], [W - mm, H - mm], [mm, H - mm], [mm, mm]], np.float32)
        s.stroke(box, "#7E7062", width=wdt, density=dn, soft=0.5)

    # Printed onto the page with multiply blending: bare paper must come out
    # white so the page's own sheet shows through.
    img = np.clip(s.render(light=0.0) / PAPER[None, None, :], 0, 1)
    save(img, OUT / "almanac-map.webp", quality=84)

    def centroid(pts):
        p = np.asarray(pts)
        return [round(float(p[:, 0].mean()), 1), round(float(p[:, 1].mean()), 1)]

    def at(fx, fy):
        return [round(fx * W, 1), round(fy * H, 1)]

    data = {
        "viewBox": [W, H],
        "regions": {
            "agentic": {"d": path_d(agentic), "label": at(0.255, 0.335)},
            "science": {"d": path_d(science), "label": at(0.245, 0.925)},
            "writing": {"d": path_d(writing), "label": at(0.765, 0.675)},
            "uncharted": {"d": path_d(uncharted), "label": centroid(uncharted)},
        },
        "fields": {
            "agentic": [{"d": path_d(c), "label": centroid(c)} for c in agentic_cells],
            "science": [{"d": path_d(c), "label": centroid(c)} for c in science_cells],
            "writing": [{"d": path_d(plot), "label": centroid(plot)}],
        },
        "house": [round(hx, 1), round(hy, 1)],
    }
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "almanac-map.json").write_text(json.dumps(data, indent=1) + "\n")
    print(f"wrote {(DATA / 'almanac-map.json').relative_to(ROOT)}")


PLATES = {"landscape": landscape, "paper": paper, "stroke": strokes, "cover": cover, "map": mapplate}

if __name__ == "__main__":
    for name in sys.argv[1:] or PLATES:
        PLATES[name]()
