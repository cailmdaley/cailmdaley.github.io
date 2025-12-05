# Cail Daley's Personal Website

Personal site built on the "Observational Patience" design philosophy.

## Tech Stack

- **Astro** with TypeScript (strict mode)
- **EB Garamond** (serif) + **JetBrains Mono** (code)
- No build-time CSS framework - custom CSS with design tokens

## Design Philosophy

See `design/design-philosophy.md` for the full manifesto. Key points:

**Porch Morning Palette**: Weathered wood, drafting paper, creek water, mountain mist. Warm natural tones with cold creek accent (#5A7B7B).

**Two Modes**:
- **Laboratory** - Scientific work, data, technical analysis. After Cajal. Sepia/graphite tones.
- **Illuminated** - Philosophy, AI alignment, creative writing. After manuscripts. Gold accents, drop caps.

**Typography**:
- Drop caps use EBGaramondInitials F1/F2 fonts (layered effect)
- Small-caps for h1, italic for h2
- Oldstyle numerals throughout

## Project Structure

```
src/
├── layouts/
│   └── BaseLayout.astro     # Main layout with ridge parallax background
├── pages/
│   ├── index.astro          # Home
│   ├── research.astro       # Research page
│   ├── science-ai.astro     # Science & AI with claims demo
│   └── about.astro          # About page
├── components/
│   └── RidgeParallax.astro  # Parallax mountain ridges background
└── styles/
    └── global.css           # Design tokens and base styles

public/
└── claims-demo/
    ├── index.html           # Standalone D3 visualization
    └── EBGaramond-*.otf     # Local fonts for drop caps
```

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
--bg-card: #EDE8E0;        /* Drafting paper */
--bg-secondary: #BEB0A0;   /* Aged pine */
--text-primary: #2E2A26;   /* Graphite */
--text-secondary: #4A4540;
--text-muted: #7A7368;
--border: #A89888;         /* Pencil rule */
--accent: #5A7B7B;         /* Creek water */

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

- Container max-width: 85ch
- Drop caps only on major section starts (h1 level)
- Illuminated mode uses F1 (gold overlay) + F2 (dark letter) fonts layered
- Iframe embeds should have no padding, match container width exactly
- Prefer `defer` on external scripts to avoid blocking
