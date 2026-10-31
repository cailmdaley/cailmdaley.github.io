# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "healpy", "pillow"]
# ///
"""Bake the textures for the UNIONS mass-map globe on /research.

The globe is the sky seen as a celestial sphere: the Gaia EDR3 all-sky map
(galactic, rotated to equatorial) with the UNIONS Kaiser–Squires κ_E map laid
over it inside the survey footprint. Everything is baked into one
equirectangular texture in equatorial coordinates (u = RA / 360°,
v = (90° − Dec) / 180°), which `src/scripts/globe.ts` wraps onto a sphere.

The mass map is shown as in the First Principles demo (`massmap_display.py`):
the delivered 30′ map smoothed to 60′ in total, with the smoothing normalised
by the smoothed mask so holes and edges don't bleed in, then rescaled to unit
standard deviation inside the footprint. The amplitude stays blind: only
κ/σ is shown, and no colour bar.

Colour is the site's: voids in ridge violet, the mean in morning mist, mass
peaks in hemlock and creek green.

    uv run scripts/globe.py [--kappa FILE] [--gaia FILE]

Outputs:
    public/globe/sky.webp      equirectangular sky + κ texture
    public/globe/globe.webp    a still of the globe at its opening view (no-JS and pre-load fallback)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import healpy as hp
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "public" / "globe"
LOOM = Path.home() / "loom/.felt/work/positions/career-futures/prospects/first-principles-foundation"
KAPPA = LOOM / "shots/massmap/kappa_E_2026-10-30.fits.gz"
GAIA = Path.home() / "Downloads/unions-eiffel-composites/assets/gaia_edr3_colour_equirectangular.png"

TEX_W, TEX_H = 2048, 1024      # published texture
SS = 2                         # supersampling while baking
NSIDE = 512                    # working resolution of the mass map
SMOOTH_FWHM_ARCMIN = 30.0      # smoothing of the delivered KS map
DISPLAY_FWHM_ARCMIN = 60.0     # total smoothing of the map shown
LO, HI = -1.6, 3.5             # display range, in units of the shown map's σ
MAP_ALPHA = 0.94               # opacity of the map over the sky

# The opening view (data-elev, data-phi on /research): camera latitude and the RA it faces.
ELEV, PHI = 55.0, 195.0
STILL = 720                    # fallback still, px

# low κ → high κ: voids in ridge violet, the mean in morning mist, mass
# peaks in hemlock and deep creek green
STOPS = [
    (-1.6, "#8578AE"),
    (-0.7, "#B6ACCF"),
    (0.0, "#E8E5E6"),
    (0.7, "#B4C6B8"),
    (1.5, "#739381"),
    (2.4, "#476B6B"),
    (3.5, "#27413F"),
]

# J2000 equatorial → galactic rotation
EQ2GAL = np.array([[-0.0548755604, -0.8734370902, -0.4838350155],
                   [0.4941094279, -0.4448296300, 0.7469822445],
                   [-0.8676661490, -0.1980763734, 0.4559837762]])


def hex_rgb(h):
    return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], np.float64) / 255.0


def colorize(k):
    xs = np.array([s[0] for s in STOPS])
    cs = np.stack([hex_rgb(s[1]) for s in STOPS])
    kk = np.clip(k, LO, HI)
    return np.stack([np.interp(kk, xs, cs[:, c]) for c in range(3)], -1)


def display_map(path):
    """massmap_display.py's recipe: 60′ total smoothing, mask-normalised, unit σ."""
    m = hp.read_map(path, dtype=np.float64)
    m = hp.ud_grade(np.where(m == hp.UNSEEN, hp.UNSEEN, m), NSIDE)
    bad = m == hp.UNSEEN
    extra = np.radians(np.sqrt(DISPLAY_FWHM_ARCMIN ** 2 - SMOOTH_FWHM_ARCMIN ** 2) / 60)
    num = hp.smoothing(np.where(bad, 0.0, m), fwhm=extra)
    den = hp.smoothing((~bad).astype(float), fwhm=extra)
    out = np.where(~bad & (den > 0.5), num / np.maximum(den, 1e-3), np.nan)
    ok = np.isfinite(out)
    out[ok] /= out[ok].std()
    area = ok.sum() * hp.nside2pixarea(NSIDE, degrees=True)
    print(f"κ_E display map: {area:,.0f} deg² shown, nside {NSIDE}")
    return out


def unit(ra, dec):
    ra, dec = np.radians(ra), np.radians(dec)
    return np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], -1)


def sample_bilinear(tex, x, y):
    th, tw = tex.shape[:2]
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = (x - x0)[..., None], (y - y0)[..., None]
    x0 %= tw
    x1 = (x0 + 1) % tw
    y0 = np.clip(y0, 0, th - 1)
    y1 = np.minimum(y0 + 1, th - 1)
    return (tex[y0, x0] * (1 - fx) * (1 - fy) + tex[y0, x1] * fx * (1 - fy)
            + tex[y1, x0] * (1 - fx) * fy + tex[y1, x1] * fx * fy)


def gaia_sky(v, gaia_path):
    """Gaia all-sky colour at unit vectors v (equatorial). The source map is galactic
    plate carrée with l = 0 at the centre, increasing to the left."""
    Image.MAX_IMAGE_PIXELS = None
    tex = np.asarray(Image.open(gaia_path).convert("RGB"), np.float32) / 255.0
    g = v @ EQ2GAL.T
    l = np.degrees(np.arctan2(g[..., 1], g[..., 0]))
    b = np.degrees(np.arcsin(np.clip(g[..., 2], -1, 1)))
    th, tw = tex.shape[:2]
    s = sample_bilinear(tex, ((0.5 - l / 360) % 1) * tw - 0.5, (90 - b) / 180 * th - 0.5)
    # quieter and a little warmer, as in the video; the night lifted from black to slate ink
    grey = s.mean(-1, keepdims=True)
    s = (grey + 0.8 * (s - grey)) * np.array([0.92, 0.89, 0.84])
    return np.array([40, 44, 52]) / 255.0 + 1.35 * s


def bake(kappa_path, gaia_path):
    w, h = TEX_W * SS, TEX_H * SS
    ra = (np.arange(w) + 0.5) / w * 360.0
    dec = 90.0 - (np.arange(h) + 0.5) / h * 180.0
    RA, DEC = np.meshgrid(ra, dec)
    v = unit(RA, DEC)

    sky = gaia_sky(v, gaia_path)
    k = display_map(kappa_path)
    kk = k[hp.ang2pix(NSIDE, RA, DEC, lonlat=True)]
    inside = np.isfinite(kk)
    rgb = colorize(np.nan_to_num(kk))
    a = MAP_ALPHA * inside[..., None]
    out = sky * (1 - a) + rgb * a
    im = Image.fromarray((np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8)).resize((TEX_W, TEX_H), Image.LANCZOS)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "sky.webp"
    im.save(path, "WEBP", quality=80, method=6)
    print(f"wrote {path.relative_to(ROOT)}  {TEX_W}x{TEX_H}  {path.stat().st_size // 1024} KB")
    return np.asarray(im, np.float32) / 255.0


def still(tex):
    """The opening view, rendered as globe.ts renders it (limb-darkened, RA increasing to the left)."""
    n = STILL * 2
    p, e = np.radians(PHI), np.radians(ELEV)
    c = np.array([np.cos(e) * np.cos(p), np.cos(e) * np.sin(p), np.sin(e)])
    r = np.array([np.sin(p), -np.cos(p), 0.0])
    u = np.array([-np.sin(e) * np.cos(p), -np.sin(e) * np.sin(p), np.cos(e)])
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float64)
    a, b = (xx + 0.5) / n * 2 - 1, 1 - (yy + 0.5) / n * 2
    rr = a * a + b * b
    mu = np.sqrt(np.clip(1 - rr, 0, 1))
    v = a[..., None] * r + b[..., None] * u + mu[..., None] * c
    ra = np.degrees(np.arctan2(v[..., 1], v[..., 0])) % 360
    dec = np.degrees(np.arcsin(np.clip(v[..., 2], -1, 1)))
    th, tw = tex.shape[:2]
    s = sample_bilinear(tex, ra / 360 * tw - 0.5, (90 - dec) / 180 * th - 0.5)
    s *= (0.78 + 0.22 * mu)[..., None]
    alpha = (rr <= 1).astype(np.float64)[..., None]
    rgba = np.concatenate([s, alpha], -1)
    im = Image.fromarray((np.clip(rgba, 0, 1) * 255 + 0.5).astype(np.uint8), "RGBA").resize((STILL, STILL), Image.LANCZOS)
    path = OUT / "globe.webp"
    im.save(path, "WEBP", quality=80, method=6)
    print(f"wrote {path.relative_to(ROOT)}  {STILL}x{STILL}  {path.stat().st_size // 1024} KB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kappa", type=Path, default=KAPPA)
    ap.add_argument("--gaia", type=Path, default=GAIA)
    args = ap.parse_args()
    still(bake(args.kappa, args.gaia))
