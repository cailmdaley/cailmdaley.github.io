#!/usr/bin/env python3
"""
Observational Patience — Laboratory Mode

After Cajal. For observational cosmology, technical analysis, data pipelines.
Flowing lines, organic branching, stippled density. How things are measured.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch
import numpy as np
from pathlib import Path
from scipy import interpolate

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['EB Garamond', 'Garamond', 'Times New Roman']

# Laboratory palette - contracted toward graphite
COLORS = {
    'bg': '#F5F0E8',
    'ink': '#1A1815',
    'ink_mid': '#3A3530',
    'ink_light': '#5A5550',
    'ink_faint': '#8A8578',
    'wash': '#C8C4BC',
    'accent': '#4A5A5A',
}


def smooth_curve(points, n_smooth=80):
    """Smooth curve through points"""
    if len(points) < 3:
        return points[:, 0], points[:, 1]
    t = np.linspace(0, 1, len(points))
    t_smooth = np.linspace(0, 1, n_smooth)
    try:
        cs_x = interpolate.CubicSpline(t, points[:, 0], bc_type='natural')
        cs_y = interpolate.CubicSpline(t, points[:, 1], bc_type='natural')
        return cs_x(t_smooth), cs_y(t_smooth)
    except:
        return points[:, 0], points[:, 1]


def draw_dendrite(ax, start, angle, length, depth=0, max_depth=4, base_width=1.2):
    """Recursive branching structure"""
    if depth > max_depth or length < 0.08:
        return

    np.random.seed(int(start[0] * 1000 + start[1] * 100 + depth * 10))
    width = base_width * (0.7 ** depth)

    # Curved path
    curve = np.random.uniform(-0.3, 0.3)
    end = (start[0] + length * np.cos(angle), start[1] + length * np.sin(angle))
    mid = (start[0] + length * 0.5 * np.cos(angle + curve),
           start[1] + length * 0.5 * np.sin(angle + curve))

    points = np.array([start, mid, end])
    x, y = smooth_curve(points, 30)
    ax.plot(x, y, color=COLORS['ink_mid'], linewidth=width, alpha=0.85,
            solid_capstyle='round')

    # Dendritic spines
    if depth > 1 and np.random.random() > 0.4:
        for _ in range(np.random.randint(2, 4)):
            t = np.random.uniform(0.3, 0.7)
            idx = int(t * (len(x) - 1))
            spine_angle = angle + np.random.choice([-1, 1]) * np.pi/2
            spine_len = 0.04
            ax.plot([x[idx], x[idx] + spine_len * np.cos(spine_angle)],
                   [y[idx], y[idx] + spine_len * np.sin(spine_angle)],
                   color=COLORS['ink_mid'], linewidth=0.3, alpha=0.5)

    # Branch
    if depth < max_depth:
        for _ in range(np.random.randint(1, 3)):
            new_angle = angle + np.random.uniform(-0.7, 0.7)
            new_length = length * np.random.uniform(0.5, 0.75)
            draw_dendrite(ax, end, new_angle, new_length, depth + 1, max_depth, base_width)


def draw_cell_body(ax, x, y, size=0.2):
    """Neuron soma"""
    theta = np.linspace(0, 2 * np.pi, 50)
    r = size * (1 + 0.06 * np.sin(5 * theta))
    ax.fill(x + r * np.cos(theta), y + r * np.sin(theta),
            color=COLORS['bg'], edgecolor=COLORS['ink_mid'], linewidth=1)
    ax.add_patch(Circle((x, y), size * 0.35, facecolor=COLORS['wash'],
                        edgecolor=COLORS['ink_light'], linewidth=0.5, alpha=0.6))
    ax.plot(x, y, '.', color=COLORS['ink'], markersize=2)


def draw_filament(ax, start, end, thickness=0.8):
    """Cosmic filament with stippling"""
    np.random.seed(int(start[0] * 100 + end[1] * 50))

    n_ctrl = np.random.randint(3, 5)
    t_ctrl = np.sort(np.concatenate([[0], np.random.uniform(0.2, 0.8, n_ctrl - 2), [1]]))

    dx, dy = end[0] - start[0], end[1] - start[1]
    length = np.sqrt(dx**2 + dy**2)
    perp_x, perp_y = -dy / length, dx / length

    ctrl_pts = []
    for t in t_ctrl:
        disp = np.random.uniform(-0.2, 0.2) * length * 0.3
        ctrl_pts.append([start[0] + t * dx + disp * perp_x,
                        start[1] + t * dy + disp * perp_y])

    x, y = smooth_curve(np.array(ctrl_pts), 60)

    for offset in np.linspace(-0.01, 0.01, 3) * thickness:
        lw = thickness * (1 - abs(offset) * 30)
        ax.plot(x + offset * perp_x, y + offset * perp_y,
               color=COLORS['ink_mid'], linewidth=lw, alpha=0.5)

    # Stippling
    n_dots = int(length * 12 * thickness)
    for _ in range(n_dots):
        t = np.random.uniform(0, 1)
        idx = int(t * (len(x) - 1))
        offset = np.random.normal(0, 0.015 * thickness)
        ax.plot(x[idx] + offset * perp_x, y[idx] + offset * perp_y,
               '.', color=COLORS['ink_mid'], markersize=np.random.uniform(0.2, 0.5), alpha=0.4)


def draw_cluster(ax, x, y, size=0.12):
    """Galaxy cluster node"""
    np.random.seed(int(x * 1000 + y * 100))
    for _ in range(int(40 * size)):
        r = np.random.exponential(size * 0.4)
        theta = np.random.uniform(0, 2 * np.pi)
        dot_size = np.random.uniform(0.3, 1.2) * (1 - r / (size * 2))
        ax.plot(x + r * np.cos(theta), y + r * np.sin(theta),
               '.', color=COLORS['ink'], markersize=max(0.2, dot_size), alpha=0.6)


def create_canvas():
    fig = plt.figure(figsize=(14, 10), facecolor=COLORS['bg'])
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_facecolor(COLORS['bg'])

    # Title
    ax.text(7, 9.3, 'Laboratory Mode', fontsize=28, color=COLORS['ink'],
            ha='center', fontfamily='serif', style='italic')
    ax.text(7, 8.85, 'after Cajal — how things are measured',
            fontsize=11, color=COLORS['ink_faint'], ha='center', style='italic')
    ax.plot([4, 10], [8.65, 8.65], color=COLORS['wash'], linewidth=0.5)

    # === NEURAL STRUCTURE (left) ===
    draw_cell_body(ax, 2.5, 5.0, size=0.25)
    for angle in [np.pi/2 - 0.3, np.pi/2, np.pi/2 + 0.3]:
        draw_dendrite(ax, (2.5, 5.2), angle, 0.7, max_depth=5)
    # Axon
    axon_y = np.linspace(4.75, 3.2, 30)
    axon_x = 2.5 + 0.03 * np.sin(axon_y * 8)
    ax.plot(axon_x, axon_y, color=COLORS['ink_mid'], linewidth=0.8)

    ax.text(2.5, 2.8, 'Purkinje cell', fontsize=8, color=COLORS['ink_faint'],
            ha='center', style='italic')
    ax.text(2.5, 2.5, 'cerebellum', fontsize=7, color=COLORS['ink_faint'],
            ha='center', fontfamily='monospace')

    # === COSMIC WEB (center-right) ===
    np.random.seed(42)
    nodes = [(6, 6), (7.5, 5), (9, 6.5), (8, 4), (10, 5), (11, 6),
             (6.5, 4.5), (9.5, 4), (10.5, 4.5), (7, 7), (11.5, 5.5)]

    # Filaments
    edges = [(0, 1), (1, 2), (1, 3), (2, 4), (4, 5), (3, 4), (0, 6),
             (3, 7), (4, 8), (0, 9), (5, 10), (2, 5)]
    for i, j in edges:
        draw_filament(ax, nodes[i], nodes[j], thickness=0.6)

    # Clusters
    for x, y in nodes:
        draw_cluster(ax, x, y, size=np.random.uniform(0.08, 0.15))

    # Void indication
    ax.add_patch(Ellipse((8, 5.5), 1.2, 0.8, facecolor=COLORS['bg'],
                         edgecolor='none', alpha=0.3))

    ax.text(8.5, 3.2, 'cosmic web', fontsize=8, color=COLORS['ink_faint'],
            ha='center', style='italic')
    ax.text(8.5, 2.9, 'z ~ 0', fontsize=7, color=COLORS['ink_faint'],
            ha='center', fontfamily='monospace')

    # === NOTATION ===
    ax.plot([0.8, 1.6], [1.2, 1.2], color=COLORS['ink'], linewidth=0.8)
    ax.plot([0.8, 0.8], [1.15, 1.25], color=COLORS['ink'], linewidth=0.8)
    ax.plot([1.6, 1.6], [1.15, 1.25], color=COLORS['ink'], linewidth=0.8)
    ax.text(1.2, 1.0, '100 μm / 100 Mpc', fontsize=7, color=COLORS['ink_faint'],
            ha='center', fontfamily='monospace')

    ax.text(13.3, 0.6, 'Plate I', fontsize=8, color=COLORS['ink_light'],
            ha='right', style='italic')

    # Quote
    ax.text(0.8, 8.0, 'The same optimization mathematics', fontsize=9,
            color=COLORS['ink_light'], style='italic')
    ax.text(0.8, 7.65, 'across twenty orders of magnitude.', fontsize=9,
            color=COLORS['ink_light'], style='italic')

    # Border
    ax.add_patch(FancyBboxPatch((0.3, 0.3), 13.4, 9.4, boxstyle="round,pad=0",
                                facecolor='none', edgecolor=COLORS['wash'],
                                linewidth=0.5, alpha=0.5))

    plt.savefig(Path(__file__).parent / 'laboratory-mode.png', format='png',
                bbox_inches='tight', facecolor=fig.get_facecolor(), dpi=300)
    plt.close()
    print("Created laboratory-mode.png")


if __name__ == '__main__':
    create_canvas()
