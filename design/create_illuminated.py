#!/usr/bin/env python3
"""
Observational Patience — Illuminated Mode

After manuscripts. For philosophy, AI alignment, creative writing.
Drop caps, celestial imagery, decorative borders. What things mean.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch, Wedge
import numpy as np
from pathlib import Path

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['EB Garamond', 'Garamond', 'Times New Roman']

# Illuminated palette - warmed toward gold
COLORS = {
    'bg': '#FAF8F2',
    'cream': '#FFFCF5',
    'ink': '#2A2520',
    'ink_light': '#5A5550',
    'ink_faint': '#8A8580',
    'gold': '#9A7B35',
    'gold_light': '#C4A86A',
    'border': '#B8A888',
    'accent': '#5A7B7B',
}


def draw_drop_cap(ax, letter, x, y, size=1.8):
    """Illuminated initial letter"""
    pad = size * 0.12

    # Decorative frame
    ax.add_patch(FancyBboxPatch((x - pad, y - pad), size + 2*pad, size + 2*pad,
                                boxstyle="round,pad=0,rounding_size=0.03",
                                facecolor=COLORS['cream'], edgecolor=COLORS['gold'],
                                linewidth=2))
    ax.add_patch(FancyBboxPatch((x + 0.05, y + 0.05), size - 0.1, size - 0.1,
                                boxstyle="round,pad=0,rounding_size=0.02",
                                facecolor='none', edgecolor=COLORS['gold_light'],
                                linewidth=0.8, linestyle='--'))

    # Corner dots
    for cx, cy in [(x, y), (x + size, y), (x, y + size), (x + size, y + size)]:
        ax.plot(cx, cy, 'o', color=COLORS['gold'], markersize=4)

    # The letter
    ax.text(x + size/2, y + size/2, letter, fontsize=size * 38,
            ha='center', va='center', color=COLORS['ink'],
            fontfamily='serif', style='italic')

    # Vine decoration
    np.random.seed(ord(letter))
    t = np.linspace(0, 2 * np.pi, 40)
    r = size * 0.38
    vx = x + size/2 + r * np.cos(t) * (1 + 0.08 * np.sin(4 * t))
    vy = y + size/2 + r * np.sin(t) * (1 + 0.08 * np.cos(3 * t))
    ax.plot(vx, vy, color=COLORS['gold_light'], linewidth=0.4, alpha=0.4)


def draw_celestial_border(ax, x, y, w, h):
    """Decorative border with stars"""
    # Double border
    ax.add_patch(Rectangle((x, y), w, h, facecolor='none',
                           edgecolor=COLORS['gold_light'], linewidth=1.5))
    ax.add_patch(Rectangle((x + 0.08, y + 0.08), w - 0.16, h - 0.16,
                           facecolor='none', edgecolor=COLORS['gold_light'],
                           linewidth=0.5, alpha=0.5))

    # Corner stars
    for cx, cy in [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]:
        for angle in np.linspace(0, 2*np.pi, 9)[:-1]:
            ax.plot([cx, cx + 0.12 * np.cos(angle)],
                   [cy, cy + 0.12 * np.sin(angle)],
                   color=COLORS['gold'], linewidth=0.8, alpha=0.7)

    # Stars along top
    np.random.seed(42)
    for sx in np.linspace(x + 0.4, x + w - 0.4, 15):
        sy = y + h - 0.12 + np.random.uniform(-0.02, 0.02)
        ax.plot(sx, sy, '*', color=COLORS['gold_light'],
                markersize=np.random.uniform(2, 5), alpha=0.7)


def draw_moon(ax, x, y, r=0.4):
    """Moon phase"""
    ax.add_patch(Circle((x, y), r, facecolor=COLORS['cream'],
                        edgecolor=COLORS['border'], linewidth=0.5))
    ax.add_patch(Wedge((x, y), r, -90, 90, facecolor=COLORS['cream']))

    # Shadow crescent
    t = np.linspace(-np.pi/2, np.pi/2, 30)
    term_x = x + r * 0.3 * np.cos(t)
    term_y = y + r * np.sin(t)
    ax.fill_betweenx(term_y, term_x, x + r * np.cos(t),
                     color='#E8E4DC', alpha=0.8)


def draw_constellation(ax, x, y, r=1.0):
    """Small constellation diagram"""
    ax.add_patch(Circle((x, y), r, facecolor=COLORS['cream'],
                        edgecolor=COLORS['gold_light'], linewidth=0.8, alpha=0.5))

    np.random.seed(888)
    stars = []
    for _ in range(15):
        angle = np.random.uniform(0, 2 * np.pi)
        dist = np.random.uniform(0, r * 0.85)
        sx = x + dist * np.cos(angle)
        sy = y + dist * np.sin(angle)
        size = np.random.uniform(2, 5)
        stars.append((sx, sy, size))
        ax.plot(sx, sy, '*', color=COLORS['gold'], markersize=size, alpha=0.8)

    # Connect some stars
    for i in range(0, min(8, len(stars) - 1), 2):
        ax.plot([stars[i][0], stars[i+1][0]], [stars[i][1], stars[i+1][1]],
               color=COLORS['gold_light'], linewidth=0.4, alpha=0.5)


def draw_sun(ax, x, y, r=0.25):
    """Sun symbol"""
    ax.add_patch(Circle((x, y), r, facecolor=COLORS['gold_light'],
                        edgecolor=COLORS['gold'], linewidth=1))
    ax.plot(x, y, '.', color=COLORS['gold'], markersize=4)
    for angle in np.linspace(0, 2*np.pi, 9)[:-1]:
        ax.plot([x + r * 1.1 * np.cos(angle), x + r * 1.4 * np.cos(angle)],
               [y + r * 1.1 * np.sin(angle), y + r * 1.4 * np.sin(angle)],
               color=COLORS['gold'], linewidth=0.8)


def create_canvas():
    fig = plt.figure(figsize=(14, 10), facecolor=COLORS['bg'])
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_facecolor(COLORS['bg'])

    # Celestial border
    draw_celestial_border(ax, 0.4, 0.4, 13.2, 9.2)

    # Drop cap
    draw_drop_cap(ax, 'I', 1, 6.2, size=2.0)

    # Title
    ax.text(3.4, 7.7, 'lluminated', fontsize=30, color=COLORS['ink'],
            fontfamily='serif', style='italic')
    ax.text(7.6, 7.7, 'Mode', fontsize=30, color=COLORS['ink_light'],
            fontfamily='serif', style='italic')

    ax.text(7, 7.1, '— what things mean —', fontsize=11,
            color=COLORS['ink_faint'], ha='center', style='italic')
    ax.plot([3.5, 10.5], [6.85, 6.85], color=COLORS['gold_light'],
            linewidth=0.5, alpha=0.5)

    # Main text block
    lines = [
        "In the beginning was the question: not what is,",
        "but what ought to be. The alignment problem lives here—",
        "in the space between measurement and meaning,",
        "between what systems do and what we want.",
        "",
        "Wonder is epistemically valid. Beauty is not opposed",
        "to rigor. Some ideas deserve ceremonial containers,",
        "illuminated beginnings, the gold leaf of careful attention.",
    ]

    for i, line in enumerate(lines):
        if line:
            ax.text(3.3, 6.3 - i * 0.42, line, fontsize=11,
                   color=COLORS['ink_light'], fontfamily='serif')

    # Constellation
    draw_constellation(ax, 11.5, 5.5, r=1.2)
    ax.text(11.5, 4.0, 'Celestial Map', fontsize=8, color=COLORS['ink_faint'],
            ha='center', style='italic')

    # Moon
    draw_moon(ax, 1.8, 2.8, r=0.5)
    ax.text(1.8, 2.0, 'First Quarter', fontsize=7, color=COLORS['ink_faint'],
            ha='center', style='italic')

    # Sun
    draw_sun(ax, 12.3, 2.0, r=0.22)

    # Vine decoration (left margin)
    vine_y = np.linspace(1.8, 8.0, 40)
    vine_x = 0.75 + 0.08 * np.sin(vine_y * 2.5)
    ax.plot(vine_x, vine_y, color=COLORS['gold_light'], linewidth=0.8, alpha=0.4)

    # Leaves
    for i in range(0, len(vine_y), 6):
        lx, ly = vine_x[i], vine_y[i]
        direction = 1 if i % 2 == 0 else -1
        ax.annotate('', xy=(lx + 0.12 * direction, ly + 0.08),
                   xytext=(lx, ly),
                   arrowprops=dict(arrowstyle='->', color=COLORS['gold_light'],
                                  lw=0.4, mutation_scale=6))

    # Quote box
    qx, qy, qw, qh = 4.2, 1.3, 5.5, 1.4
    ax.add_patch(FancyBboxPatch((qx, qy), qw, qh,
                                boxstyle="round,pad=0.02,rounding_size=0.04",
                                facecolor=COLORS['cream'], edgecolor=COLORS['gold_light'],
                                linewidth=0.8, alpha=0.8))
    ax.text(qx + 0.15, qy + qh - 0.25, '"', fontsize=20, color=COLORS['gold'], alpha=0.5)
    ax.text(qx + qw/2, qy + qh/2 + 0.12, 'Wonder is the beginning of wisdom.',
            fontsize=11, color=COLORS['ink_light'], ha='center', style='italic')
    ax.text(qx + qw/2, qy + qh/2 - 0.25, '— the practice of observational patience',
            fontsize=8, color=COLORS['ink_faint'], ha='center')

    # Ornament
    ax.text(7, 6.7, '✦', fontsize=8, ha='center', color=COLORS['gold'], alpha=0.6)

    # Plate notation
    ax.text(13.3, 0.6, 'Plate II', fontsize=8, color=COLORS['ink_light'],
            ha='right', style='italic')

    plt.savefig(Path(__file__).parent / 'illuminated-mode.png', format='png',
                bbox_inches='tight', facecolor=fig.get_facecolor(), dpi=300)
    plt.close()
    print("Created illuminated-mode.png")


if __name__ == '__main__':
    create_canvas()
