#!/usr/bin/env python3
"""
Generate Figure 3.1: Integrated dual-mesh methodology visualization.

Layout:
- Top: Zoomed schematic views (stencil left, wall gradient right)
- Middle: Full diffuser geometry meshes
- Bottom: Flowchart (matching thesis figure 3.1b exactly)

Author: ML Wall Function Project
Date: 2026-01-31
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, ConnectionPatch
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


def create_integrated_figure():
    """Create integrated dual-mesh methodology figure."""
    print("Generating integrated mesh methodology figure...")

    # Geometry parameters
    L_inlet, L_expand, L_outlet = 1.0, 2.0, 1.0
    L_total = L_inlet + L_expand + L_outlet
    Y_top, Y_bot_inlet, Y_bot_outlet = 1.0, 0.5, 0.0

    def get_bottom_wall(x):
        if x <= L_inlet:
            return Y_bot_inlet
        elif x <= L_inlet + L_expand:
            t = (x - L_inlet) / L_expand
            return Y_bot_inlet - t * (Y_bot_inlet - Y_bot_outlet)
        else:
            return Y_bot_outlet

    def get_graded_y(ny, grading, y_bot, y_top):
        if grading > 1 and ny > 1:
            r = grading ** (1.0 / (ny - 1))
            y_norm = np.array([(r**i - 1) / (r**ny - 1) for i in range(ny + 1)])
        else:
            y_norm = np.linspace(0, 1, ny + 1)
        y_norm_inverted = 1 - y_norm[::-1]
        return y_bot + y_norm_inverted * (y_top - y_bot)

    # Create figure with 3 rows
    fig = plt.figure(figsize=(14, 14))
    gs = fig.add_gridspec(3, 2, height_ratios=[0.9, 0.9, 1.2], hspace=0.25, wspace=0.25)

    ax_stencil = fig.add_subplot(gs[0, 0])    # Top left: stencil schematic
    ax_gradient = fig.add_subplot(gs[0, 1])   # Top right: wall gradient schematic
    ax_coarse = fig.add_subplot(gs[1, 0])     # Middle left: coarse mesh
    ax_fine = fig.add_subplot(gs[1, 1])       # Middle right: fine mesh
    ax_flow = fig.add_subplot(gs[2, :])       # Bottom: flowchart

    # ==================== ROW 1: ZOOMED SCHEMATICS ====================

    def draw_stencil_schematic(ax):
        """Draw stencil extraction schematic."""
        nx, ny = 6, 8
        grading = 5.0
        x_coords = np.linspace(0, 1.5, nx + 1)
        y_coords = get_graded_y(ny, grading, 0, 1)

        for i in range(nx):
            for j in range(ny):
                x_cell = [x_coords[i], x_coords[i+1], x_coords[i+1], x_coords[i], x_coords[i]]
                y_cell = [y_coords[j], y_coords[j], y_coords[j+1], y_coords[j+1], y_coords[j]]
                ax.plot(x_cell, y_cell, 'b-', linewidth=0.8, alpha=0.6)

        # Highlight 3×5 stencil
        stencil_i, stencil_j = 2, ny - 5
        for i in range(3):
            for j in range(5):
                ii, jj = stencil_i + i, stencil_j + j
                if ii < nx and jj < ny:
                    rect = Rectangle((x_coords[ii], y_coords[jj]),
                                     x_coords[ii+1] - x_coords[ii],
                                     y_coords[jj+1] - y_coords[jj],
                                     facecolor='royalblue', edgecolor='darkblue',
                                     linewidth=1.5, alpha=0.4)
                    ax.add_patch(rect)
                    cx = (x_coords[ii] + x_coords[ii+1]) / 2
                    cy = (y_coords[jj] + y_coords[jj+1]) / 2
                    ax.plot(cx, cy, 'o', color='red', markersize=5, zorder=10)

        ax.plot([-0.05, 1.55], [1, 1], 'b-', linewidth=4)
        ax.text(0.75, 1.04, 'Wall', ha='center', va='bottom', fontsize=11, color='blue', fontweight='bold')

        ax.annotate('', xy=(1.65, 1.0), xytext=(1.65, y_coords[-2]),
                   arrowprops=dict(arrowstyle='<->', color='darkred', lw=1.5))
        ax.text(1.75, (1.0 + y_coords[-2])/2, '$\\Delta y_1$\n$y^+ \\approx 30$',
               fontsize=9, va='center', color='darkred')

        ax.text(0.75, y_coords[stencil_j] - 0.06, '$3 \\times 5 = 15$ cells',
               ha='center', fontsize=10, color='darkblue', fontweight='bold')

        ax.set_xlim(-0.15, 2.0)
        ax.set_ylim(-0.15, 1.12)
        ax.set_aspect('equal')
        ax.set_xlabel('Streamwise direction')
        ax.set_ylabel('Wall-normal direction')
        ax.set_title('Stencil Extraction (Input)', fontsize=12, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])

    def draw_gradient_schematic(ax):
        """Draw wall gradient schematic."""
        nx, ny = 10, 16
        grading = 20.0
        x_coords = np.linspace(0, 1.5, nx + 1)
        y_coords = get_graded_y(ny, grading, 0, 1)

        for i in range(nx):
            for j in range(ny):
                x_cell = [x_coords[i], x_coords[i+1], x_coords[i+1], x_coords[i], x_coords[i]]
                y_cell = [y_coords[j], y_coords[j], y_coords[j+1], y_coords[j+1], y_coords[j]]
                ax.plot(x_cell, y_cell, 'g-', linewidth=0.5, alpha=0.6)

        ax.plot([-0.05, 1.55], [1, 1], 'b-', linewidth=4)
        ax.text(0.75, 1.04, 'Wall', ha='center', va='bottom', fontsize=11, color='blue', fontweight='bold')

        ax.plot(0.75, 1.0, 'o', color='red', markersize=10, zorder=10)

        ax.annotate('', xy=(1.65, 1.0), xytext=(1.65, y_coords[-2]),
                   arrowprops=dict(arrowstyle='<->', color='darkred', lw=1.5))
        ax.text(1.75, (1.0 + y_coords[-2])/2, '$\\Delta y_1$\n$y^+ < 2$',
               fontsize=9, va='center', color='darkred')

        ax.text(0.75, 0.45, r'$\tau_w = \mu \left.\frac{\partial U}{\partial y}\right|_{wall}$',
                ha='center', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='orange', alpha=0.9))
        ax.text(0.75, 0.25, r'$q_w = -k \left.\frac{\partial T}{\partial y}\right|_{wall}$',
                ha='center', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='orange', alpha=0.9))

        ax.set_xlim(-0.15, 2.0)
        ax.set_ylim(-0.15, 1.12)
        ax.set_aspect('equal')
        ax.set_xlabel('Streamwise direction')
        ax.set_ylabel('Wall-normal direction')
        ax.set_title('Wall Gradient (Target)', fontsize=12, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])

    draw_stencil_schematic(ax_stencil)
    draw_gradient_schematic(ax_gradient)

    # ==================== ROW 2: FULL DIFFUSER MESHES ====================

    def draw_diffuser_mesh(ax, nx_inlet, nx_expand, nx_outlet, ny, grading, color, lw, title):
        x_inlet = np.linspace(0, L_inlet, nx_inlet + 1)
        x_expand = np.linspace(L_inlet, L_inlet + L_expand, nx_expand + 1)
        x_outlet = np.linspace(L_inlet + L_expand, L_total, nx_outlet + 1)
        x_all = np.concatenate([x_inlet, x_expand[1:], x_outlet[1:]])

        for i in range(len(x_all) - 1):
            x_left, x_right = x_all[i], x_all[i + 1]
            y_bot_left, y_bot_right = get_bottom_wall(x_left), get_bottom_wall(x_right)
            y_left = get_graded_y(ny, grading, y_bot_left, Y_top)
            y_right = get_graded_y(ny, grading, y_bot_right, Y_top)

            for j in range(ny):
                x_coords = [x_left, x_right, x_right, x_left, x_left]
                y_coords = [y_left[j], y_right[j], y_right[j+1], y_left[j+1], y_left[j]]
                ax.plot(x_coords, y_coords, color=color, linewidth=lw, alpha=0.8)

        ax.axhline(y=Y_top, color='blue', linewidth=2.5, label='Wall')
        x_bot = np.linspace(0, L_total, 100)
        y_bot = [get_bottom_wall(x) for x in x_bot]
        ax.plot(x_bot, y_bot, color='gray', linewidth=1.5, linestyle='--', label='Symmetry')

        ax.set_xlim(-0.1, L_total + 0.1)
        ax.set_ylim(-0.15, Y_top + 0.15)
        ax.set_aspect('equal')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(loc='lower right', fontsize=8)

        ax.annotate('', xy=(0.8, -0.08), xytext=(0.2, -0.08),
                   arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
        ax.text(0.5, -0.12, 'Flow', ha='center', fontsize=9)

    draw_diffuser_mesh(ax_coarse, 3, 6, 3, 8, 5.0, 'blue', 0.7,
                       'Coarse Mesh ($y^+ \\approx 30$)')
    draw_diffuser_mesh(ax_fine, 5, 10, 5, 16, 20.0, 'green', 0.5,
                       'Fine Mesh ($y^+ < 2$)')

    # Zoom boxes and arrows
    zoom_x, zoom_y = (3.1, 3.9), (0.65, 1.02)
    for ax in [ax_coarse, ax_fine]:
        rect = Rectangle((zoom_x[0], zoom_y[0]), zoom_x[1] - zoom_x[0], zoom_y[1] - zoom_y[0],
                         linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)

    # Arrows from zoom boxes to schematics
    arrow1 = ConnectionPatch(xyA=(0.5, 0.0), coordsA=ax_stencil.transAxes,
                             xyB=((zoom_x[0]+zoom_x[1])/2, zoom_y[1]), coordsB=ax_coarse.transData,
                             arrowstyle='<-', color='red', lw=2)
    fig.add_artist(arrow1)
    arrow2 = ConnectionPatch(xyA=(0.5, 0.0), coordsA=ax_gradient.transAxes,
                             xyB=((zoom_x[0]+zoom_x[1])/2, zoom_y[1]), coordsB=ax_fine.transData,
                             arrowstyle='<-', color='red', lw=2)
    fig.add_artist(arrow2)

    # ==================== ROW 3: FLOWCHART (matching thesis 3.1b) ====================

    def draw_flowchart(ax):
        ax.set_xlim(0, 12)
        ax.set_ylim(-1, 8)
        ax.axis('off')

        # Box styles matching thesis
        input_style = dict(boxstyle='round,pad=0.4', facecolor='#cce5ff', edgecolor='#004080', linewidth=1.5)
        process_style = dict(boxstyle='round,pad=0.4', facecolor='#d4edda', edgecolor='#155724', linewidth=1.5)
        output_style = dict(boxstyle='round,pad=0.4', facecolor='#ffe5cc', edgecolor='#cc5500', linewidth=1.5)
        result_style = dict(boxstyle='round,pad=0.4', facecolor='#fff3cd', edgecolor='#856404', linewidth=1.5)
        nn_style = dict(boxstyle='round,pad=0.4', facecolor='#f8d7da', edgecolor='#721c24', linewidth=1.5)

        # Row 1: Geometry
        ax.text(6, 7.2, 'Geometry\n(Diffuser/Nozzle/Channel)', ha='center', va='center',
                fontsize=10, bbox=input_style)

        # Row 2: Two mesh branches
        ax.text(3, 5.8, 'Coarse Mesh\n($y^+ \\approx 5$--$300$)', ha='center', va='center',
                fontsize=9, bbox=process_style)
        ax.text(9, 5.8, 'Fine Mesh\n($y^+ < 2$)', ha='center', va='center',
                fontsize=9, bbox=process_style)

        # Row 3: Simulations
        ax.text(3, 4.4, 'RANS Simulation\n(input source)', ha='center', va='center',
                fontsize=9, bbox=process_style)
        ax.text(9, 4.4, 'RANS Simulation\n(wall-resolved)', ha='center', va='center',
                fontsize=9, bbox=process_style)

        # Row 4: Extraction
        ax.text(3, 3.0, 'Stencil Extraction\n(wall-adjacent & neighbouring)', ha='center', va='center',
                fontsize=9, bbox=output_style)
        ax.text(9, 3.0, 'Wall Quantities\n($\\tau_w$, $q_w$ from gradients)', ha='center', va='center',
                fontsize=9, bbox=output_style)

        # Row 5: Training pair
        ax.text(6, 1.5, 'Training Pair: $(\\mathbf{X}_{stencil},\\; \\tau_w,\\; q_w)$',
                ha='center', va='center', fontsize=10, bbox=result_style)

        # Row 6: Neural Network
        ax.text(6, 0.2, 'Neural Network: $f_\\theta(\\mathbf{X}) \\rightarrow (\\hat{\\tau}_w, \\hat{q}_w)$',
                ha='center', va='center', fontsize=10, bbox=nn_style)

        # Arrows
        # Geometry to meshes
        ax.annotate('', xy=(3, 6.35), xytext=(5.2, 6.85),
                   arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
        ax.annotate('', xy=(9, 6.35), xytext=(6.8, 6.85),
                   arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

        # Meshes to simulations
        ax.annotate('', xy=(3, 5.25), xytext=(3, 4.95),
                   arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
        ax.annotate('', xy=(9, 5.25), xytext=(9, 4.95),
                   arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

        # Simulations to extraction
        ax.annotate('', xy=(3, 3.85), xytext=(3, 3.55),
                   arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
        ax.annotate('', xy=(9, 3.85), xytext=(9, 3.55),
                   arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

        # Extraction to training pair
        ax.annotate('', xy=(4.5, 1.75), xytext=(3, 2.45),
                   arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
        ax.annotate('', xy=(7.5, 1.75), xytext=(9, 2.45),
                   arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

        # Training pair to NN
        ax.annotate('', xy=(6, 1.05), xytext=(6, 0.65),
                   arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

        # Branch labels
        ax.text(1.2, 5.8, 'Input\nbranch', ha='center', va='center', fontsize=8, color='gray')
        ax.text(10.8, 5.8, 'Ground truth\nbranch', ha='center', va='center', fontsize=8, color='gray')

        # Dashed line connecting same location
        ax.plot([4.8, 7.2], [3.0, 3.0], 'k--', linewidth=1.5, alpha=0.5)
        ax.text(6, 2.6, 'same $(x,y)$ location', ha='center', va='center', fontsize=8, color='gray',
                bbox=dict(facecolor='white', edgecolor='none', pad=1))

    draw_flowchart(ax_flow)

    plt.tight_layout()

    output_path = os.path.join(SCRIPT_DIR, 'fig_3_1_mesh_comparison.pdf')
    plt.savefig(output_path, bbox_inches='tight')
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    print(f"Saved: {output_path.replace('.pdf', '.png')}")

    plt.close()
    return output_path


def main():
    create_integrated_figure()


if __name__ == "__main__":
    main()
