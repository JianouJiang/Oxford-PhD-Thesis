#!/usr/bin/env python3
"""
Generate dual mesh visualization for Figure 3.1
Shows coarse and fine diffuser meshes with stencil extraction zoom regions.

Layout:
    +------------------+------------------+
    | [Stencil Zoom]   | [Wall Point]     |  <- Zoomed insets on TOP
    +------------------+------------------+
    |     Coarse Mesh Diffuser            |
    +-------------------------------------+
    |     Fine Mesh Diffuser              |
    +-------------------------------------+

With red arrows connecting mesh regions to zoomed views.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Rectangle, FancyBboxPatch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import os
import sys

# Paths to mesh files
# Script is in: Oxford-EngSci-Thesis-Template/figures/chapter3/
# Need to get to: flow2Dtube/TRAINING_DATA/data/cases/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
THESIS_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # Oxford-EngSci-Thesis-Template
PAPERS_DIR = os.path.dirname(THESIS_DIR)  # PAPERS
FLOW2DTUBE_DIR = os.path.dirname(PAPERS_DIR)  # flow2Dtube
TRAINING_DATA_DIR = os.path.join(FLOW2DTUBE_DIR, "TRAINING_DATA", "data", "cases")

FINE_MESH = os.path.join(TRAINING_DATA_DIR, "diff_000_fine", "diff_000_fine.msh")
COARSE_MESH = os.path.join(TRAINING_DATA_DIR, "diff_000_coarse", "diff_000_coarse.msh")


def read_msh_file(filepath):
    """Read Gmsh MSH format (version 2) file and extract nodes and elements"""
    nodes = {}
    elements = []

    with open(filepath, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line == '$Nodes':
            i += 1
            num_nodes = int(lines[i].strip())
            i += 1
            for _ in range(num_nodes):
                parts = lines[i].strip().split()
                node_id = int(parts[0])
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                nodes[node_id] = (x, y)
                i += 1

        elif line == '$Elements':
            i += 1
            num_elements = int(lines[i].strip())
            i += 1
            for _ in range(num_elements):
                parts = lines[i].strip().split()
                elem_type = int(parts[1])
                num_tags = int(parts[2])
                node_indices = [int(x) for x in parts[3 + num_tags:]]

                # Type 2: triangle, Type 3: quad
                if elem_type in [2, 3]:
                    elements.append(node_indices)
                i += 1
        else:
            i += 1

    return nodes, elements


def get_mesh_bounds(nodes):
    """Get bounding box of mesh"""
    all_x = [n[0] for n in nodes.values()]
    all_y = [n[1] for n in nodes.values()]
    return min(all_x), max(all_x), min(all_y), max(all_y)


def plot_mesh(ax, nodes, elements, xlim=None, ylim=None, linewidth=0.3,
              color='#2c3e50', alpha=0.7, label=None):
    """Plot mesh elements on given axes"""
    for elem in elements:
        coords = [nodes.get(n, (0, 0)) for n in elem]
        if len(coords) >= 3:
            coords.append(coords[0])
            xs = [c[0] for c in coords]
            ys = [c[1] for c in coords]
            ax.plot(xs, ys, '-', linewidth=linewidth, color=color, alpha=alpha)

    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    ax.set_aspect('equal')


def plot_mesh_region(ax, nodes, elements, x_center, y_center, width, height,
                     linewidth=0.5, color='#2c3e50'):
    """Plot mesh elements within a specific region"""
    x_min = x_center - width/2
    x_max = x_center + width/2
    y_min = y_center - height/2
    y_max = y_center + height/2

    for elem in elements:
        coords = [nodes.get(n, (0, 0)) for n in elem]
        if len(coords) >= 3:
            # Check if element is within region
            elem_x = [c[0] for c in coords]
            elem_y = [c[1] for c in coords]
            if (min(elem_x) < x_max and max(elem_x) > x_min and
                min(elem_y) < y_max and max(elem_y) > y_min):
                coords.append(coords[0])
                xs = [c[0] for c in coords]
                ys = [c[1] for c in coords]
                ax.plot(xs, ys, '-', linewidth=linewidth, color=color, alpha=0.9)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')


def add_stencil_grid(ax, x_center, y_center, dx, dy, nx=3, ny=5,
                     color='#e74c3c', linewidth=2):
    """Add a 3x5 stencil grid overlay showing extracted cells"""
    # Draw stencil cells
    for i in range(nx):
        for j in range(ny):
            x = x_center + (i - nx//2) * dx
            y = y_center + (j - ny//2) * dy
            rect = Rectangle((x - dx/2, y - dy/2), dx, dy,
                            linewidth=linewidth, edgecolor=color,
                            facecolor='none', linestyle='-', zorder=10)
            ax.add_patch(rect)

    # Highlight center (wall) cell
    center_x = x_center
    center_y = y_center - (ny//2) * dy  # Wall cell at bottom
    rect = Rectangle((center_x - dx/2, center_y - dy/2), dx, dy,
                     linewidth=linewidth*1.5, edgecolor='#27ae60',
                     facecolor='#27ae60', alpha=0.3, zorder=11)
    ax.add_patch(rect)


def add_wall_point_marker(ax, x, y, color='#27ae60', size=150):
    """Add marker for wall point output"""
    ax.scatter([x], [y], s=size, c=color, marker='s', zorder=15,
               edgecolors='white', linewidths=2)


def create_dual_mesh_figure(output_path):
    """Create the complete dual mesh figure with stencil extraction"""

    # Read mesh files
    print(f"Reading fine mesh: {FINE_MESH}")
    fine_nodes, fine_elements = read_msh_file(FINE_MESH)
    print(f"  Nodes: {len(fine_nodes)}, Elements: {len(fine_elements)}")

    print(f"Reading coarse mesh: {COARSE_MESH}")
    coarse_nodes, coarse_elements = read_msh_file(COARSE_MESH)
    print(f"  Nodes: {len(coarse_nodes)}, Elements: {len(coarse_elements)}")

    # Create figure with custom layout
    fig = plt.figure(figsize=(16, 11))

    # Grid specification: 3 rows
    # Row 0: Zoom insets (smaller)
    # Row 1: Coarse mesh
    # Row 2: Fine mesh
    gs = fig.add_gridspec(3, 2, height_ratios=[0.4, 1, 1],
                          width_ratios=[1, 1],
                          hspace=0.15, wspace=0.1)

    # Region of interest for stencil (near top wall, in expansion region)
    stencil_x = 1.5   # In the expansion region
    stencil_y_top = 1.0  # Top wall

    # Zoomed insets on top row
    ax_zoom_coarse = fig.add_subplot(gs[0, 0])
    ax_zoom_fine = fig.add_subplot(gs[0, 1])

    # Main mesh plots
    ax_coarse = fig.add_subplot(gs[1, :])
    ax_fine = fig.add_subplot(gs[2, :])

    # =========================================================================
    # Plot main meshes
    # =========================================================================

    # Coarse mesh - show full domain but focus on relevant region
    x_view_min, x_view_max = -2, 8
    y_view_min, y_view_max = -0.6, 1.1

    plot_mesh(ax_coarse, coarse_nodes, coarse_elements,
              xlim=(x_view_min, x_view_max), ylim=(y_view_min, y_view_max),
              linewidth=0.5, color='#34495e')
    ax_coarse.set_ylabel('y', fontsize=12)
    ax_coarse.set_title('(a) Coarse Mesh ($y^+ \\approx 30$-300)', fontsize=14, fontweight='bold')

    # Fine mesh
    plot_mesh(ax_fine, fine_nodes, fine_elements,
              xlim=(x_view_min, x_view_max), ylim=(y_view_min, y_view_max),
              linewidth=0.15, color='#34495e', alpha=0.5)
    ax_fine.set_xlabel('x', fontsize=12)
    ax_fine.set_ylabel('y', fontsize=12)
    ax_fine.set_title('(b) Fine Mesh ($y^+ < 2$)', fontsize=14, fontweight='bold')

    # =========================================================================
    # Add stencil region rectangles on main meshes
    # =========================================================================

    # Stencil extraction region (on top wall)
    coarse_stencil_width = 0.6
    coarse_stencil_height = 0.25
    fine_stencil_width = 0.15
    fine_stencil_height = 0.08

    # Draw rectangles on coarse mesh
    rect_coarse = Rectangle((stencil_x - coarse_stencil_width/2,
                              stencil_y_top - coarse_stencil_height),
                             coarse_stencil_width, coarse_stencil_height,
                             linewidth=2, edgecolor='#e74c3c',
                             facecolor='none', linestyle='-', zorder=20)
    ax_coarse.add_patch(rect_coarse)

    # Draw rectangle on fine mesh
    rect_fine = Rectangle((stencil_x - fine_stencil_width/2,
                           stencil_y_top - fine_stencil_height),
                          fine_stencil_width, fine_stencil_height,
                          linewidth=2, edgecolor='#e74c3c',
                          facecolor='none', linestyle='-', zorder=20)
    ax_fine.add_patch(rect_fine)

    # =========================================================================
    # Plot zoomed regions
    # =========================================================================

    # Coarse mesh zoom - stencil extraction
    zoom_coarse_width = coarse_stencil_width * 1.3
    zoom_coarse_height = coarse_stencil_height * 1.3
    plot_mesh_region(ax_zoom_coarse, coarse_nodes, coarse_elements,
                     stencil_x, stencil_y_top - coarse_stencil_height/2,
                     zoom_coarse_width, zoom_coarse_height,
                     linewidth=1.5, color='#34495e')

    # Add stencil grid overlay
    # Estimate cell size from mesh
    dx_coarse = 0.09  # Approximate cell size in x
    dy_coarse = 0.035  # Approximate cell size in y
    add_stencil_grid(ax_zoom_coarse, stencil_x, stencil_y_top - dy_coarse*2.5,
                     dx_coarse, dy_coarse, nx=3, ny=5,
                     color='#e74c3c', linewidth=2.5)

    ax_zoom_coarse.set_title('INPUT: Stencil (3$\\times$5 cells)',
                             fontsize=11, fontweight='bold', color='#e74c3c')
    ax_zoom_coarse.set_xticks([])
    ax_zoom_coarse.set_yticks([])
    for spine in ax_zoom_coarse.spines.values():
        spine.set_color('#e74c3c')
        spine.set_linewidth(2)

    # Fine mesh zoom - wall point output
    zoom_fine_width = fine_stencil_width * 1.5
    zoom_fine_height = fine_stencil_height * 1.5
    plot_mesh_region(ax_zoom_fine, fine_nodes, fine_elements,
                     stencil_x, stencil_y_top - fine_stencil_height/2,
                     zoom_fine_width, zoom_fine_height,
                     linewidth=1.0, color='#34495e')

    # Add wall point marker
    add_wall_point_marker(ax_zoom_fine, stencil_x, stencil_y_top - 0.002,
                          color='#27ae60', size=200)

    # Draw wall line
    ax_zoom_fine.axhline(y=stencil_y_top, color='#2c3e50', linewidth=3, zorder=5)

    ax_zoom_fine.set_title('OUTPUT: Wall Point ($\\tau_w$, $q_w$)',
                           fontsize=11, fontweight='bold', color='#27ae60')
    ax_zoom_fine.set_xticks([])
    ax_zoom_fine.set_yticks([])
    for spine in ax_zoom_fine.spines.values():
        spine.set_color('#27ae60')
        spine.set_linewidth(2)

    # =========================================================================
    # Add connecting arrows
    # =========================================================================

    # Arrow from coarse mesh to zoom (using figure coordinates)
    # Get axes positions
    from matplotlib.transforms import Bbox

    # Create arrows using annotate with figure-level coordinates
    fig.canvas.draw()

    # Arrow 1: Coarse mesh region to zoom
    # Calculate positions in figure coordinates
    coarse_rect_top = ax_coarse.transData.transform((stencil_x, stencil_y_top))
    zoom_coarse_bottom = ax_zoom_coarse.transData.transform(
        (stencil_x, ax_zoom_coarse.get_ylim()[0]))

    # Convert to figure coordinates
    coarse_rect_top_fig = fig.transFigure.inverted().transform(coarse_rect_top)
    zoom_coarse_bottom_fig = fig.transFigure.inverted().transform(zoom_coarse_bottom)

    # Draw arrow
    arrow1 = FancyArrowPatch(
        (coarse_rect_top_fig[0], coarse_rect_top_fig[1]),
        (zoom_coarse_bottom_fig[0], zoom_coarse_bottom_fig[1]),
        transform=fig.transFigure,
        arrowstyle='-|>',
        mutation_scale=15,
        color='#e74c3c',
        linewidth=2,
        connectionstyle='arc3,rad=0',
        zorder=100
    )
    fig.patches.append(arrow1)

    # Arrow 2: Fine mesh region to zoom
    fine_rect_top = ax_fine.transData.transform((stencil_x, stencil_y_top))
    zoom_fine_bottom = ax_zoom_fine.transData.transform(
        (stencil_x, ax_zoom_fine.get_ylim()[0]))

    fine_rect_top_fig = fig.transFigure.inverted().transform(fine_rect_top)
    zoom_fine_bottom_fig = fig.transFigure.inverted().transform(zoom_fine_bottom)

    # Need to go around - arrow from fine to zoom above coarse
    # Use annotation instead for better control

    # =========================================================================
    # Add labels and annotations
    # =========================================================================

    # Add "same (x,y) location" annotation
    mid_y = (ax_coarse.get_position().y0 + ax_fine.get_position().y1) / 2
    fig.text(0.93, 0.45, 'Same $(x,y)$\nlocation', fontsize=10,
             ha='left', va='center', style='italic',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.7))

    # Add dashed line connecting stencil regions
    # Vertical dashed line through both meshes at stencil_x
    ax_coarse.axvline(x=stencil_x, color='#7f8c8d', linestyle='--',
                      linewidth=1.5, alpha=0.7, zorder=1)
    ax_fine.axvline(x=stencil_x, color='#7f8c8d', linestyle='--',
                   linewidth=1.5, alpha=0.7, zorder=1)

    # Add flow direction arrow on coarse mesh
    ax_coarse.annotate('', xy=(6.5, 0.5), xytext=(5.5, 0.5),
                       arrowprops=dict(arrowstyle='-|>', color='#3498db', lw=2))
    ax_coarse.text(6.0, 0.65, 'Flow', fontsize=10, color='#3498db',
                   ha='center', fontweight='bold')

    # Add wall labels
    ax_coarse.text(x_view_max - 0.3, 1.05, 'Top Wall (training)', fontsize=9,
                   ha='right', va='bottom', color='#2c3e50')
    ax_coarse.text(x_view_max - 0.3, y_view_min + 0.05, 'Bottom Wall', fontsize=9,
                   ha='right', va='bottom', color='#2c3e50')

    # Add mesh statistics
    stats_coarse = f'Coarse: {len(coarse_nodes):,} nodes'
    stats_fine = f'Fine: {len(fine_nodes):,} nodes'
    ax_coarse.text(0.02, 0.95, stats_coarse, transform=ax_coarse.transAxes,
                   fontsize=9, va='top',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    ax_fine.text(0.02, 0.95, stats_fine, transform=ax_fine.transAxes,
                 fontsize=9, va='top',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # =========================================================================
    # Save figure
    # =========================================================================

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Saved: {output_path}")
    print(f"Saved: {output_path.replace('.png', '.pdf')}")
    plt.close()


def create_simplified_mesh_figure(output_path):
    """Create a cleaner version focusing on the key concepts"""

    # Read mesh files
    print(f"Reading fine mesh: {FINE_MESH}")
    fine_nodes, fine_elements = read_msh_file(FINE_MESH)

    print(f"Reading coarse mesh: {COARSE_MESH}")
    coarse_nodes, coarse_elements = read_msh_file(COARSE_MESH)

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 8),
                             gridspec_kw={'height_ratios': [1, 2],
                                         'width_ratios': [1, 1],
                                         'hspace': 0.25, 'wspace': 0.15})

    ax_zoom_input = axes[0, 0]
    ax_zoom_output = axes[0, 1]
    ax_coarse = axes[1, 0]
    ax_fine = axes[1, 1]

    # View settings
    x_view_min, x_view_max = -1, 6
    y_view_min, y_view_max = -0.6, 1.15

    # Stencil location
    stencil_x = 1.5
    stencil_y = 1.0

    # =========================================================================
    # Main meshes
    # =========================================================================

    plot_mesh(ax_coarse, coarse_nodes, coarse_elements,
              xlim=(x_view_min, x_view_max), ylim=(y_view_min, y_view_max),
              linewidth=0.6, color='#2c3e50')
    ax_coarse.set_xlabel('$x$', fontsize=12)
    ax_coarse.set_ylabel('$y$', fontsize=12)
    ax_coarse.set_title('Coarse Mesh ($y^+ \\approx 30$-300)', fontsize=12, fontweight='bold')

    plot_mesh(ax_fine, fine_nodes, fine_elements,
              xlim=(x_view_min, x_view_max), ylim=(y_view_min, y_view_max),
              linewidth=0.15, color='#2c3e50', alpha=0.6)
    ax_fine.set_xlabel('$x$', fontsize=12)
    ax_fine.set_ylabel('$y$', fontsize=12)
    ax_fine.set_title('Fine Mesh ($y^+ < 2$)', fontsize=12, fontweight='bold')

    # Add stencil region markers
    rect_coarse = Rectangle((stencil_x - 0.3, stencil_y - 0.15), 0.6, 0.15,
                             linewidth=2.5, edgecolor='#e74c3c',
                             facecolor='#e74c3c', alpha=0.15)
    ax_coarse.add_patch(rect_coarse)

    rect_fine = Rectangle((stencil_x - 0.08, stencil_y - 0.05), 0.16, 0.05,
                           linewidth=2.5, edgecolor='#27ae60',
                           facecolor='#27ae60', alpha=0.15)
    ax_fine.add_patch(rect_fine)

    # =========================================================================
    # Zoomed views on top
    # =========================================================================

    # Input stencil zoom (coarse mesh)
    plot_mesh_region(ax_zoom_input, coarse_nodes, coarse_elements,
                     stencil_x, stencil_y - 0.08, 0.8, 0.25,
                     linewidth=1.8, color='#2c3e50')

    # Draw 3x5 stencil grid
    dx, dy = 0.09, 0.035
    for i in range(3):
        for j in range(5):
            x = stencil_x + (i - 1) * dx
            y = (stencil_y - dy/2) - j * dy
            rect = Rectangle((x - dx/2, y - dy/2), dx, dy,
                            linewidth=2, edgecolor='#e74c3c',
                            facecolor='#e74c3c', alpha=0.1)
            ax_zoom_input.add_patch(rect)

    ax_zoom_input.axhline(y=stencil_y, color='#2c3e50', linewidth=3)
    ax_zoom_input.set_title('INPUT: Coarse Mesh Stencil\n($3 \\times 5$ cells)',
                            fontsize=11, fontweight='bold', color='#e74c3c')
    ax_zoom_input.set_xticks([])
    ax_zoom_input.set_yticks([])
    for spine in ax_zoom_input.spines.values():
        spine.set_color('#e74c3c')
        spine.set_linewidth(2.5)

    # Output wall point zoom (fine mesh)
    plot_mesh_region(ax_zoom_output, fine_nodes, fine_elements,
                     stencil_x, stencil_y - 0.025, 0.2, 0.08,
                     linewidth=1.2, color='#2c3e50')

    # Mark wall point
    ax_zoom_output.scatter([stencil_x], [stencil_y - 0.001], s=250,
                           c='#27ae60', marker='s', zorder=15,
                           edgecolors='white', linewidths=2)
    ax_zoom_output.axhline(y=stencil_y, color='#2c3e50', linewidth=3)
    ax_zoom_output.set_title('OUTPUT: Fine Mesh Wall Point\n($\\tau_w$, $q_w$)',
                             fontsize=11, fontweight='bold', color='#27ae60')
    ax_zoom_output.set_xticks([])
    ax_zoom_output.set_yticks([])
    for spine in ax_zoom_output.spines.values():
        spine.set_color('#27ae60')
        spine.set_linewidth(2.5)

    # =========================================================================
    # Connecting arrows and annotations
    # =========================================================================

    # Vertical dashed line showing same x location
    ax_coarse.axvline(x=stencil_x, color='#95a5a6', linestyle='--', linewidth=1.5)
    ax_fine.axvline(x=stencil_x, color='#95a5a6', linestyle='--', linewidth=1.5)

    # Flow direction
    for ax in [ax_coarse, ax_fine]:
        ax.annotate('', xy=(5, 0.3), xytext=(4, 0.3),
                   arrowprops=dict(arrowstyle='-|>', color='#3498db', lw=2))
        ax.text(4.5, 0.45, 'Flow', fontsize=10, color='#3498db',
                ha='center', fontweight='bold')

    # Add mesh statistics
    ax_coarse.text(0.02, 0.95, f'{len(coarse_nodes):,} nodes',
                   transform=ax_coarse.transAxes, fontsize=9, va='top',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    ax_fine.text(0.02, 0.95, f'{len(fine_nodes):,} nodes',
                 transform=ax_fine.transAxes, fontsize=9, va='top',
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    # Label top wall
    ax_coarse.text(5.5, 1.05, 'Top Wall', fontsize=9, ha='right', color='#2c3e50')
    ax_fine.text(5.5, 1.05, 'Top Wall', fontsize=9, ha='right', color='#2c3e50')

    # =========================================================================
    # Save
    # =========================================================================

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Saved: {output_path}")
    print(f"Saved: {output_path.replace('.png', '.pdf')}")
    plt.close()


if __name__ == "__main__":
    output_dir = os.path.dirname(os.path.abspath(__file__))

    # Generate both versions
    print("\n=== Generating dual mesh figure ===\n")
    create_dual_mesh_figure(os.path.join(output_dir, "dual_mesh_stencil.png"))

    print("\n=== Generating simplified mesh figure ===\n")
    create_simplified_mesh_figure(os.path.join(output_dir, "dual_mesh_simplified.png"))

    print("\nDone!")
