# Cail Daley's Personal Website

Personal site built on the "Observational Patience" design philosophy.

## Tech Stack

- **Astro** with TypeScript (strict mode)
- **EB Garamond** (serif) + **JetBrains Mono** (code)
- No build-time CSS framework - custom CSS with design tokens

## Design Philosophy

See `design/design-philosophy.md` for the full manifesto. Key points:

**Porch Morning, in watercolour**: the view from a West Virginia porch on a misty morning — violet on the far ridge, mist pooled in the hollows, hemlock ridges, gravel darkened by rain — painted as transparent glazes on cold-press paper. Content sits on a sheet of that paper (#F2EDE3) laid over the painting; the accent is creek water after rain (#476B6B). Nothing is animated: the painting is a still image, and only link colours change on hover. Two things on Research move: the UNIONS mass-map globe, which turns slowly and answers the hand (see below), and the cross-correlation panorama, which pans across cosmic time as the page scrolls.

**Two Modes**:
- **Laboratory** - Scientific work, data, technical analysis. After Cajal. Sepia/graphite tones.
- **Illuminated** - Philosophy, AI alignment, creative writing. After manuscripts. Gold accents, drop caps.

**Typography**:
- No italics anywhere (`grep -rn italic src public` finds only comments in global.css); emphasis is weight, small capitals, or a brush mark
- Small caps for h1; h2 upright at weight 600 over a creek-water brush stroke; h3 in all-small-caps
- Drop caps use EBGaramondInitials F1/F2 fonts (layered effect)
- Oldstyle numerals throughout

## Project Structure

```
src/
├── layouts/
│   └── BaseLayout.astro     # Layout: fixed painted landscape behind the paper sheet
├── pages/
│   ├── index.astro          # Home
│   ├── research.astro       # Research: one glazed panel per project
│   ├── almanac/
│   │   ├── index.astro      # Almanacs: intro, this year's title page, tool and writing lists
│   │   └── science-ai.astro # Science & AI with claims demo
│   └── about.astro          # About page
└── styles/
    └── global.css           # Design tokens and base styles

scripts/
└── paint.py                 # Seeded watercolour renderer for everything in public/paint/

public/
├── paint/                   # landscape.webp, paper.webp, stroke.webp, almanac-cover.webp, almanac-glaze.webp
├── globe/                   # sky.webp, globe.webp (scripts/globe.py)
└── claims-demo/
    ├── index.html           # Standalone D3 visualization
    └── EBGaramond-*.otf     # Local fonts for drop caps
```

## Painted assets (`scripts/paint.py`)

`uv run scripts/paint.py [landscape|paper|stroke|cover]` re-renders the plates; every plate is seeded, so a run reproduces the committed files. Pigment is optical density on paper (colour = paper · e^(−absorbance)), so washes glaze like transparent watercolour; each wash has edge darkening, granulation into the paper tooth and low-frequency unevenness, and mist is painted by lifting density out.

- `landscape.webp` — behind every page; on phones it shows as a band above the sheet
- `paper.webp` — tileable grain for the sheet
- `stroke.webp` — creek-water brush mark (alpha) under h2, dividers and the active nav item
- `almanac-cover.webp` — the roundel on the almanac's title page: sunrise and the waning moon over the ridges, seen over the porch rail. It is RGBA, a vignette with its own paper set into the yellow cover (normal blending, not multiply). The month ring around it, and all the title-page lettering, are HTML/SVG in `almanac/index.astro`; the year and its place after Leap Year are reckoned at build time.
- `almanac-glaze.webp` — the yellow of the title page, a gamboge-over-sienna glaze in the manner of a farmer's almanac cover; rendered white-normalised and stretched to the title page with `mix-blend-mode: multiply`. Rendered by the `cover` plate.

The page is "Almanacs", one almanac a year: the plural names it in the nav (BaseLayout) and the page title, and the URL is `/almanac`. The title page's h1 is the singular "Almanac" of the current year's number. Home (`src/content/pages/index.md`) links to it as "the Almanacs".

## Mass-map globe (`/research`)

The Optical Cosmic Shear card carries the UNIONS κ_E mass map on an interactive celestial globe, over the Gaia EDR3 all-sky map. It is the site's one moving thing: it turns slowly (about 5°/s), pauses off screen, and does not turn at all under `prefers-reduced-motion`. Drag (or arrow keys) turns it; pinch, ctrl+wheel (trackpad pinch), `+`/`-`, or the wheel once the globe is focused zoom it, so the wheel never takes over page scrolling. On touch, a vertical swipe scrolls the page until the globe is zoomed in.

- `scripts/globe.py` (`uv run scripts/globe.py`) bakes `public/globe/sky.webp`, one 2048×1024 equirectangular texture in equatorial coordinates (Gaia sky with the mass map composited inside the footprint), and `public/globe/globe.webp`, a still of the opening view that shows until WebGL draws and stands in without JS or WebGL2. The map follows the demo video's recipe (60′ mask-normalised smoothing, unit σ, amplitude blinded) in the video's press palette, pale cyan for underdense sky to deep blue for the mass peaks. Inputs default to the files in the loom and `~/Downloads`; pass `--kappa` and `--gaia` to point elsewhere.
- `src/scripts/globe.ts` is a self-written WebGL2 renderer: one fragment shader ray-casts an orthographic sphere and samples the texture with seam-safe gradients. Textures load only when the globe nears the viewport. The opening view (`data-elev`, `data-phi`) matches `ELEV, PHI` in `globe.py`. RA increases to the left, as in the video.

## Claims Demo (`/claims-demo/`)

Interactive DAG visualization for epistemic claims. Embedded via iframe on science-ai page.

**Tech**: D3.js (loaded with `defer`), vanilla JS, CSS

**Design**: See `design/epistemic-dendrites.md` for full philosophy. Tree rings + radial Perlin noise aesthetic.

**Core concepts**:
- **Organic nodes**: 2D noise-based ellipses with concentric rings radiating outward
- **Ring-based connectors**: Each strand connects to a specific ring (confidence → strand count)
- **Unified opacity system**: Nodes and connectors share `ringOpacity(i) = 0.5 × (1 - i/4)`

**Key functions**:
- `renderDAG()` - Main visualization
- `noise2D(x, y, seed)` - Layered sine waves for smooth organic shapes
- `organicEllipse(rx, ry, seed, scale)` - Generate noise-displaced ellipse path
- `ringOpacity(index)` - Shared opacity function for rings and connectors
- `updateHighlighting()` - Selection with ring-hierarchy-aware opacity/width
- `selectClaim(id)` - Update selection and show claim card with dendrite art
- `drawClaimArt(claim)` - Canvas-based dendrite visualization per claim

**Ring system**:
- Scales: [1.0, 1.15, 1.3, 1.45] (core + 3 outer rings)
- Fill: organic shapes painted outside-in with fading opacity
- Strokes: core at 0.8px (opacity × 0.7), outer at 0.5px

**Highlighting**:
- Nodes: selected 1.0, connected 0.6, dimmed 0.3
- Connectors: ring-based differentiation preserved when highlighted
- Stroke width stored in `data-base-width`, opacity in `data-base-opacity`

**Animation**: 600ms duration, 200ms between tiers, cubicOut easing

## CSS Variables

From `global.css`:

```css
--paper: #F2EDE3;          /* Cold-press sheet (also --bg-card) */
--text-primary: #2E2A26;   /* Graphite */
--text-secondary: #4A4540;
--text-muted: #6E675F;
--border-light: #D9CFBF;   /* Pencil rule */
--accent: #476B6B;         /* Creek water after rain (--creek); --creek-wet on hover */
--ridge: #9D90BB;  --slate: #8A9DA6;  --hemlock: #587466;  --gravel: #B4A48C;

/* Correctness colors (1-5 scale) */
--correctness-1: #A87070;  /* Red - low */
--correctness-3: #C4A86A;  /* Yellow - mid */
--correctness-5: #6B8B6B;  /* Green - high */
```

## Commands

```bash
npm run dev      # Dev server at localhost:4321
npm run build    # Production build
npm run preview  # Preview production build
```

## Conventions

- Sheet (`.container`) max-width: 100ch; `<BaseLayout wide>` gives 118ch (Research, whose panels carry inset figures)
- `.wash` gives a glazed panel; set `--wash: r, g, b` for the pigment
- Figures blend into the paper with `mix-blend-mode: multiply`
- Nothing animates or reacts to scroll, except on Research: the mass-map globe and the scroll-driven cross-correlation panorama (`.scroll-reveal`)
- Drop caps only on major section starts (h1 level)
- Illuminated mode uses F1 (gold overlay) + F2 (dark letter) fonts layered
- Iframe embeds should have no padding, match container width exactly
- Prefer `defer` on external scripts to avoid blocking
