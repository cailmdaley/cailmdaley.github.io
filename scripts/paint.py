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
    uv run scripts/paint.py landscape  # one plate: landscape | paper | stroke | map

Outputs:
    public/paint/landscape.webp   porch-morning ridges behind every page
    public/paint/paper.webp       tileable watercolour-paper grain for the page sheet
    public/paint/stroke.webp      a creek-water brush mark (alpha) for rules and the active nav item
    public/paint/almanac-map.webp the Almanac estate map
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
    from scipy.ndimage import distance_transform_edt

    H, W = 1150, 1600
    s = Sheet(H, W, seed=2025)
    rng = s.rng
    yy, xx = s.yy, s.xx
    P = lambda fx, fy: np.array([fx * W, fy * H])

    # ---- the hills along the top, drawn in elevation as old estate maps do
    for cy, col, dens, octs, canopy, fall in [
        (0.085, "#B5ABCB", 0.55, [(420, 22), (140, 8)], 0.0, 0.05),
        (0.135, "#9D90BB", 0.70, [(330, 26), (110, 9), (35, 3)], 0.0, 0.06),
        (0.195, "#8A9DA6", 0.70, [(300, 24), (90, 8), (25, 3)], 1.2, 0.06),
        (0.255, "#6F8A84", 0.72, [(260, 20), (70, 7), (18, 3)], 2.2, 0.07),
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
    floor = soft_step(yy - (0.30 * H + 30 * s.noise(200, 2)), 40)
    s.wash(floor * (0.6 + 0.4 * s.noise(220, 2)), "#D9D8BC", density=0.45, edge=0, gran=0.15, vary=0.35, vary_scale=180)

    # ---- the creek, from a spring under the ridge down through the valley
    t = np.linspace(0, 1, 500)
    cx = W * (0.600 - 0.13 * t + 0.055 * np.sin(t * 8.5 + 0.4) + 0.018 * np.sin(t * 27 + 1.0))
    cy = H * (0.30 + 0.72 * t)
    creek = np.stack([cx, cy], 1)
    dist = np.full((H, W), 1e4, np.float32)
    rad = np.zeros((H, W), np.float32)
    for (x, y), tt in zip(creek, t):
        xi, yi = int(x), int(y)
        y0, y1, x0, x1 = max(0, yi - 50), min(H, yi + 50), max(0, xi - 50), min(W, xi + 50)
        if y0 >= y1 or x0 >= x1:
            continue
        dd = np.hypot(xx[y0:y1, x0:x1] - x, yy[y0:y1, x0:x1] - y)
        closer = dd < dist[y0:y1, x0:x1]
        dist[y0:y1, x0:x1] = np.where(closer, dd, dist[y0:y1, x0:x1])
        rad[y0:y1, x0:x1] = np.where(closer, 2.0 + 5.0 * tt, rad[y0:y1, x0:x1])
    water = np.clip((rad + 1.2 * s.noise(8, 2) - dist) / 1.0 + 0.5, 0, 1)
    banks = np.exp(-np.clip(dist - rad, 0, None) / 16.0) * (dist < 200)
    s.wash(banks * (0.7 + 0.3 * s.noise(30, 2)), "#A5B79D", density=0.5, edge=0, gran=0.3, vary=0.3)

    # ---- regions
    def outline(fracs, wobble=5.0):
        """A hand-drawn boundary through the given corners, densified and wavered."""
        pts = np.array([P(*f) for f in fracs])
        out = []
        for p0, p1 in zip(pts, np.roll(pts, -1, axis=0)):
            n = max(2, int(np.hypot(*(p1 - p0)) // 18))
            for k in range(n):
                out.append(p0 + (p1 - p0) * k / n)
        out = np.array(out)
        return out + wobble * np.stack([noise_loop(rng, len(out), 9), noise_loop(rng, len(out), 9)], 1)

    tools = outline([(0.07, 0.40), (0.13, 0.31), (0.24, 0.285), (0.33, 0.315), (0.405, 0.36), (0.435, 0.45),
                     (0.415, 0.555), (0.445, 0.64), (0.40, 0.745), (0.31, 0.80), (0.20, 0.815), (0.115, 0.765),
                     (0.06, 0.665), (0.085, 0.55), (0.05, 0.47)])
    writing = outline([(0.625, 0.40), (0.70, 0.355), (0.79, 0.365), (0.85, 0.425), (0.865, 0.52), (0.83, 0.615),
                       (0.76, 0.675), (0.68, 0.685), (0.625, 0.635), (0.605, 0.53)])
    uncharted = blob(*P(0.835, 0.865), 0.135 * W, 0.105 * H, rng, wob=0.12, rot=-0.1)

    # Tools & infrastructure: a patchwork, one field per tool
    seeds = np.array([P(*q) for q in [
        (0.140, 0.400), (0.265, 0.365), (0.365, 0.460), (0.110, 0.580),
        (0.250, 0.520), (0.345, 0.660), (0.205, 0.720)]])
    cells = voronoi_cells(tools, seeds)
    field_paint = [
        ("#CDB97E", "furrow", 0.35), ("#A9B784", "plain", 0.0), ("#C9A774", "furrow", 1.2),
        ("#B7BE8E", "stipple", 0.0), ("#D4C391", "furrow", -0.6), ("#9EB29A", "plain", 0.0),
        ("#C8AE86", "stipple", 0.0),
    ]
    for cell, (col, tex, ang) in zip(cells, field_paint):
        cm = poly_mask(cell, H, W)
        inner = distance_transform_edt(cm)
        fm = soft_step(inner - 6 + 2.0 * s.noise(10, 2), 2.0) * cm
        s.wash(fm, col, density=0.75, edge=0.9, edge_width=3.0, gran=0.45, vary=0.3, vary_scale=70)
        if tex == "furrow":
            ca, sa = np.cos(ang), np.sin(ang)
            stripe = 0.5 + 0.5 * np.sin((xx * ca + yy * sa) / 4.2 + 0.8 * s.noise(60, 2))
            s.wash(fm * stripe ** 4 * soft_step(inner - 14, 3), col, density=0.45, edge=0, gran=0.4, vary=0.2)
        elif tex == "stipple":
            dots = (rng.random((H, W)) > 0.985).astype(np.float32)
            dots = np.clip(gaussian_filter(dots, 1.3) * 10, 0, 1)
            s.wash(dots * soft_step(inner - 12, 2), "#7C8F62", density=0.55, edge=0, gran=0.2, vary=0)
    # hedgerows between the fields: dabs of hemlock along the cell boundaries
    hedge_pts = []
    for cell in cells:
        for a, b in zip(cell, np.roll(cell, -1, axis=0)):
            n = int(np.hypot(*(b - a)) // 11)
            for k in range(n):
                q = a + (b - a) * (k + 0.5) / max(n, 1)
                if rng.random() < 0.22:
                    hedge_pts.append(q + rng.normal(0, 1.5, 2))
    s.wash(dab_mask(s, hedge_pts, 4.0), "#62806A", density=0.6, edge=0.9, edge_width=1.2, gran=0.3, vary=0.2, vary_scale=30)

    # Writing: the orchard across the creek; the dated plot is planted, the rest is meadow
    wm = gaussian_filter(poly_mask(writing, H, W), 2.0)
    s.wash(wm, "#C3CB9F", density=0.45, edge=0.8, edge_width=4.0, gran=0.3, vary=0.3, vary_scale=90)
    plot = clip_halfplane(writing, P(0.735, 0.545), np.array([0.3, 1.0]))  # the northern part
    pm = poly_mask(plot, H, W)
    pin = distance_transform_edt(pm)
    trees = []
    for gy in range(-12, 12):
        for gx in range(-14, 14):
            q = P(0.725, 0.50) + np.array([gx * 27 + gy * 8, gy * 24]) + rng.normal(0, 1.8, 2)
            xi, yi = int(q[0]), int(q[1])
            if 0 <= yi < H and 0 <= xi < W and pin[yi, xi] > 14:
                trees.append(q)
    s.wash(soft_step(pin - 6, 2) * pm * (0.8 + 0.2 * s.noise(40, 2)), "#AFC08E", density=0.5, edge=0.6, gran=0.3)
    s.wash(dab_mask(s, trees, 5.2, jitter=0.12), "#6C8A5E", density=0.75, edge=1.0, edge_width=1.5, gran=0.35, vary=0.2, vary_scale=40)

    # Uncharted: left as paper, only the mist that hides it
    um = gaussian_filter(poly_mask(uncharted, H, W), 14.0)
    s.lift(um, 0.8)
    s.wash(um * np.clip(0.5 + 0.6 * s.noise(60, 2), 0, 1), "#D6D1DF", density=0.35, edge=0, gran=0.05, vary=0.3)

    # Woods: hemlock clumps in the hollows and along the creek
    wood_pts = []
    for (cxw, cyw, n, spread) in [(0.06, 0.34, 70, 38), (0.47, 0.35, 90, 45), (0.92, 0.40, 80, 42),
                                   (0.505, 0.60, 45, 24), (0.635, 0.82, 40, 24), (0.07, 0.92, 55, 32),
                                   (0.37, 0.31, 40, 26), (0.955, 0.62, 40, 24), (0.48, 0.92, 30, 22)]:
        c = P(cxw, cyw)
        for _ in range(n):
            wood_pts.append(c + rng.normal(0, spread, 2) * np.array([1.4, 0.8]))
    wood_pts = [q for q in wood_pts
                if 0 <= q[1] < H and 0 <= q[0] < W and water[int(q[1]), int(q[0])] < 0.1]
    woods = dab_mask(s, wood_pts, 9.0, jitter=0.3)
    s.wash(woods, "#6F8C72", density=0.65, edge=0.9, edge_width=2.0, gran=0.35, vary=0.35, vary_scale=30)
    # shade on the lower side of each clump, a second, darker touch
    shade = np.clip(woods - np.roll(np.roll(woods, -5, axis=0), -3, axis=1), 0, 1)
    s.wash(gaussian_filter(shade, 1.2), "#3F5C4E", density=0.7, edge=0.4, gran=0.3, vary=0.2)

    # The water itself, last, so it reads through everything
    s.wash(water, "#5A7B7B", density=1.0, edge=1.0, edge_width=1.5, gran=0.3, vary=0.2, vary_scale=50, cover=0.6 * water)

    # The house, and the gravel lane from it to the fields and over the ford to the orchard
    house = P(0.575, 0.875)
    lane = np.array([house + [-6, -10], P(0.48, 0.80), P(0.415, 0.775), P(0.37, 0.745)])
    lane2 = np.array([house + [8, -14], P(0.585, 0.76), P(0.625, 0.69), P(0.655, 0.655)])
    for ln in (lane, lane2):
        sm = np.array([np.interp(np.linspace(0, len(ln) - 1, 60), np.arange(len(ln)), ln[:, k]) for k in (0, 1)]).T
        s.stroke(sm, "#A89577", width=5, density=0.55)
    hx, hy = house
    roof = np.array([[hx - 18, hy - 2], [hx, hy - 16], [hx + 18, hy - 2]])
    body = np.array([[hx - 14, hy - 3], [hx + 14, hy - 3], [hx + 14, hy + 13], [hx - 14, hy + 13]])
    s.wash(gaussian_filter(poly_mask(body, H, W), 0.7), "#C9B79C", density=0.9, edge=0.8, edge_width=1.0, gran=0.2)
    s.wash(gaussian_filter(poly_mask(roof, H, W), 0.7), "#8E6B5E", density=0.9, edge=0.8, edge_width=1.0, gran=0.2)
    porch = np.array([[hx - 14, hy + 9], [hx + 14, hy + 9], [hx + 14, hy + 13], [hx - 14, hy + 13]])
    s.wash(gaussian_filter(poly_mask(porch, H, W), 0.6), "#6E5A4C", density=0.8, edge=0.3, gran=0.1)

    # Neat line: a double pencil rule around the plate.
    for mm, wdt, dn in ((14, 2.0, 0.6), (24, 1.0, 0.45)):
        box = np.array([[mm, mm], [W - mm, mm], [W - mm, H - mm], [mm, H - mm], [mm, mm]], np.float32)
        s.stroke(box, "#7E7062", width=wdt, density=dn, soft=0.5)

    # Printed onto the page with multiply blending: bare paper must come out
    # white so the page's own sheet shows through.
    img = np.clip(s.render() / PAPER[None, None, :], 0, 1)
    save(img, OUT / "almanac-map.webp", quality=84)

    def centroid(pts):
        p = np.asarray(pts)
        return [round(float(p[:, 0].mean()), 1), round(float(p[:, 1].mean()), 1)]

    data = {
        "viewBox": [W, H],
        "regions": {
            "tools": {"d": path_d(tools), "label": [round(0.25 * W, 1), round(0.86 * H, 1)]},
            "writing": {"d": path_d(writing), "label": [round(0.745 * W, 1), round(0.73 * H, 1)]},
            "uncharted": {"d": path_d(uncharted), "label": centroid(uncharted)},
        },
        "fields": {
            "tools": [{"d": path_d(c), "label": centroid(c)} for c in cells],
            "writing": [{"d": path_d(plot), "label": centroid(plot)}],
        },
        "house": [round(hx, 1), round(hy, 1)],
    }
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "almanac-map.json").write_text(json.dumps(data, indent=1) + "\n")
    print(f"wrote {(DATA / 'almanac-map.json').relative_to(ROOT)}")


PLATES = {"landscape": landscape, "paper": paper, "stroke": strokes, "map": mapplate}

if __name__ == "__main__":
    for name in sys.argv[1:] or PLATES:
        PLATES[name]()
