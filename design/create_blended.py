#!/usr/bin/env python3
"""
Observational Patience — Blended Mode

Where measurement meets meaning. Technical AI safety, pattern recognition,
epistemic claims with philosophical weight. Laboratory precision with
illuminated thresholds.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch
import numpy as np
from pathlib import Path
from scipy import interpolate

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['EB Garamond', 'Garamond', 'Times New Roman']

# Blended palette - between graphite and gold
COLORS = {
    'bg': '#F6F2EA',
    'cream': '#FDFAF5',
    'ink': '#2A2520',
    'ink_mid': '#4A4540',
    'ink_light': '#6A6560',
    'ink_faint': '#9A9590',
    'gold': '#9A7B35',
    'gold_light': '#C4A86A',
    'accent': '#5A7B7B',
    'wash': '#D4D0C8',
    'green': '#6B8B6B',
    'yellow': '#C4A86A',
    'red': '#A87070',
}


def draw_small_drop_cap(ax, letter, x, y, size=1.2):
    """Smaller illuminated initial - threshold marker"""
    pad = size * 0.1
    ax.add_patch(FancyBboxPatch((x - pad, y - pad), size + 2*pad, size + 2*pad,
                                boxstyle="round,pad=0,rounding_size=0.02",
                                facecolor=COLORS['cream'], edgecolor=COLORS['gold_light'],
                                linewidth=1.5))
    for cx, cy in [(x, y), (x + size, y), (x, y + size), (x + size, y + size)]:
        ax.plot(cx, cy, 'o', color=COLORS['gold_light'], markersize=2.5)
    ax.text(x + size/2, y + size/2, letter, fontsize=size * 32,
            ha='center', va='center', color=COLORS['ink'],
            fontfamily='serif', style='italic')


def draw_confidence_bar(ax, x, y, width, confidence, label):
    """Epistemic confidence bar"""
    bar_height = 0.22

    # Background
    ax.add_patch(Rectangle((x, y - bar_height/2), width, bar_height,
                           facecolor=COLORS['wash'], alpha=0.5))

    # Fill based on confidence
    if confidence > 0.7:
        fill_color = COLORS['green']
    elif confidence > 0.5:
        fill_color = COLORS['yellow']
    else:
        fill_color = COLORS['red']

    ax.add_patch(Rectangle((x, y - bar_height/2), width * confidence, bar_height,
                           facecolor=fill_color, alpha=0.6))

    # Value
    ax.text(x - 0.12, y, f'{confidence:.2f}', fontsize=7, va='center', ha='right',
            color=COLORS['ink_mid'], fontfamily='monospace')

    # Label
    ax.text(x + width + 0.15, y, label, fontsize=8, va='center',
            color=COLORS['ink_light'], fontfamily='monospace')


def draw_branching_structure(ax, x, y, scale=1.0):
    """Abstract branching - simplified, label-free"""
    np.random.seed(123)

    def branch(start, angle, length, depth):
        if depth > 3 or length < 0.1:
            return
        end = (start[0] + length * np.cos(angle), start[1] + length * np.sin(angle))

        # Bezier curve
        mid = ((start[0] + end[0])/2 + np.random.uniform(-0.1, 0.1),
               (start[1] + end[1])/2 + np.random.uniform(-0.1, 0.1))
        t = np.linspace(0, 1, 20)
        bx = (1-t)**2 * start[0] + 2*(1-t)*t * mid[0] + t**2 * end[0]
        by = (1-t)**2 * start[1] + 2*(1-t)*t * mid[1] + t**2 * end[1]

        lw = 1.0 * (0.7 ** depth)
        ax.plot(bx, by, color=COLORS['ink_mid'], linewidth=lw, alpha=0.7)

        for _ in range(np.random.randint(1, 3)):
            new_angle = angle + np.random.uniform(-0.6, 0.6)
            new_length = length * np.random.uniform(0.5, 0.7)
            branch(end, new_angle, new_length, depth + 1)

    branch((x, y), np.pi/2, 0.6 * scale, 0)
    branch((x, y), -np.pi/2, 0.4 * scale, 1)


def draw_dag_node(ax, x, y, label, size=0.18):
    """DAG node"""
    ax.add_patch(Circle((x, y), size, facecolor=COLORS['cream'],
                        edgecolor=COLORS['accent'], linewidth=1.2))
    ax.text(x, y, label, fontsize=9, ha='center', va='center',
            color=COLORS['ink_mid'], style='italic')


def draw_dag_edge(ax, start, end):
    """DAG edge with bezier curve"""
    mid_y = (start[1] + end[1]) / 2 + 0.2
    t = np.linspace(0, 1, 30)
    bx = (1-t)**2 * start[0] + 2*(1-t)*t * ((start[0]+end[0])/2) + t**2 * end[0]
    by = (1-t)**2 * start[1] + 2*(1-t)*t * mid_y + t**2 * end[1]
    ax.plot(bx, by, color=COLORS['wash'], linewidth=1, alpha=0.6)


def create_canvas():
    fig = plt.figure(figsize=(14, 10), facecolor=COLORS['bg'])
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_facecolor(COLORS['bg'])

    # Subtle border (between modes)
    ax.add_patch(FancyBboxPatch((0.35, 0.35), 13.3, 9.3, boxstyle="round,pad=0",
                                facecolor='none', edgecolor=COLORS['gold_light'],
                                linewidth=0.8, alpha=0.4))
    # Corner accents only
    for cx, cy in [(0.35, 0.35), (13.65, 0.35), (0.35, 9.65), (13.65, 9.65)]:
        ax.plot(cx, cy, '*', color=COLORS['gold_light'], markersize=6, alpha=0.5)

    # Title with small drop cap
    draw_small_drop_cap(ax, 'B', 1, 7.5, size=1.5)
    ax.text(2.8, 8.2, 'lended', fontsize=26, color=COLORS['ink'],
            fontfamily='serif', style='italic')
    ax.text(5.8, 8.2, 'Mode', fontsize=26, color=COLORS['ink_light'],
            fontfamily='serif', style='italic')

    ax.text(7, 7.6, '— where measurement meets meaning —', fontsize=10,
            color=COLORS['ink_faint'], ha='center', style='italic')

    # === EPISTEMIC CLAIMS (left-center) ===
    ax.text(1.5, 6.6, 'epistemic status', fontsize=9, color=COLORS['ink_faint'],
            fontfamily='monospace')

    claims = [
        (0.82, 'value specification'),
        (0.65, 'corrigibility'),
        (0.48, 'mesa-optimization'),
        (0.71, 'interpretability'),
        (0.55, 'distributional shift'),
    ]

    for i, (conf, label) in enumerate(claims):
        draw_confidence_bar(ax, 1.5, 6.1 - i * 0.6, 3.0, conf, label)

    # === REASONING DAG (center) ===
    nodes = [
        (7, 5.5, 'α'), (8.5, 6.2, 'β'), (8.5, 4.8, 'γ'),
        (10, 5.5, 'δ'), (10, 6.5, 'ε')
    ]
    edges = [(0, 1), (0, 2), (1, 3), (2, 3), (1, 4), (3, 4)]

    for i, j in edges:
        draw_dag_edge(ax, (nodes[i][0], nodes[i][1]), (nodes[j][0], nodes[j][1]))

    for x, y, label in nodes:
        draw_dag_node(ax, x, y, label)

    ax.text(8.5, 4.0, 'reasoning structure', fontsize=8, color=COLORS['ink_faint'],
            ha='center', fontfamily='monospace')

    # === BRANCHING STRUCTURE (right) ===
    draw_branching_structure(ax, 12, 5.5, scale=1.8)
    ax.text(12, 3.8, 'pattern', fontsize=8, color=COLORS['ink_faint'],
            ha='center', style='italic')

    # === QUOTE (philosophical weight) ===
    qx, qy = 1.2, 2.0
    ax.text(qx, qy + 0.3, '"', fontsize=18, color=COLORS['gold_light'], alpha=0.6)
    ax.text(qx + 0.25, qy + 0.1, 'The trace is more valuable than the result.',
            fontsize=10, color=COLORS['ink_light'], style='italic')
    ax.text(qx + 0.25, qy - 0.25, 'Confidence is metadata, not binary.',
            fontsize=10, color=COLORS['ink_light'], style='italic')

    # === SCALE / NOTATION ===
    ax.text(12.5, 1.5, 'confidence: 0.67', fontsize=7, color=COLORS['ink_faint'],
            fontfamily='monospace', ha='center')
    ax.text(12.5, 1.2, 'importance: high', fontsize=7, color=COLORS['ink_faint'],
            fontfamily='monospace', ha='center')

    ax.text(13.3, 0.6, 'Plate III', fontsize=8, color=COLORS['ink_light'],
            ha='right', style='italic')

    # Decorative line separator
    ax.plot([1, 6], [3.0, 3.0], color=COLORS['wash'], linewidth=0.5)

    plt.savefig(Path(__file__).parent / 'blended-mode.png', format='png',
                bbox_inches='tight', facecolor=fig.get_facecolor(), dpi=300)
    plt.close()
    print("Created blended-mode.png")


if __name__ == '__main__':
    create_canvas()
