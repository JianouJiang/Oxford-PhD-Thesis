#!/usr/bin/env python3
"""
Generate boundary conditions figure for thesis Chapter 3.

Shows diffuser geometry with:
- Flat TOP wall (training data source)
- Curved BOTTOM wall (expansion)
- Clear labeling of all boundary conditions
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'mathtext.fontset': 'dejavuserif',
})


def create_boundary_conditions_figure():
    """Create boundary conditions figure with correct diffuser orientation."""

    fig, ax = plt.subplots(figsize=(14, 7))

    # Geometry parameters (flat top, curved bottom - CORRECT orientation)
    L_in = 2.0      # Inlet length
    L_trans = 4.0   # Transition length
    L_out = 3.0     # Outlet length
    L_total = L_in + L_trans + L_out

    H_in = 1.0      # Inlet height
    H_out = 2.5     # Outlet height
    y_top = H_in    # Top wall position (FLAT)

    # X coordinates
    x_inlet_start = 0
    x_trans_start = L_in
    x_trans_end = L_in + L_trans
    x_outlet_end = L_total

    # Draw the geometry
    n_pts = 100

    # Top wall - FLAT (training data source)
    x_top = np.array([x_inlet_start, x_outlet_end])
    y_top_wall = np.array([y_top, y_top])
    ax.plot(x_top, y_top_wall, 'k-', linewidth=3)

    # Bottom wall - VARYING (creates pressure gradient)
    x_bottom = np.linspace(x_inlet_start, x_outlet_end, n_pts)
    y_bottom = np.zeros_like(x_bottom)

    # Inlet section (flat at 0)
    inlet_mask = x_bottom <= x_trans_start
    y_bottom[inlet_mask] = 0

    # Transition section (linear drop from 0 to -(H_out - H_in))
    trans_mask = (x_bottom > x_trans_start) & (x_bottom <= x_trans_end)
    t = (x_bottom[trans_mask] - x_trans_start) / L_trans
    y_bottom[trans_mask] = -t * (H_out - H_in)

    # Outlet section (flat at bottom)
    outlet_mask = x_bottom > x_trans_end
    y_bottom[outlet_mask] = -(H_out - H_in)

    ax.plot(x_bottom, y_bottom, 'k-', linewidth=3)

    # Draw inlet and outlet boundaries
    ax.plot([x_inlet_start, x_inlet_start], [0, y_top], 'b-', linewidth=3)
    ax.plot([x_outlet_end, x_outlet_end], [-(H_out - H_in), y_top], 'r-', linewidth=3)

    # Fill the domain
    polygon_x = np.concatenate([[x_inlet_start], x_bottom, [x_outlet_end, x_inlet_start]])
    polygon_y = np.concatenate([[y_top], y_bottom, [y_top, y_top]])
    ax.fill(polygon_x, polygon_y, color='lightblue', alpha=0.3)

    # Add flow arrows inside domain
    arrow_y = 0.5
    for x_pos in [0.8, 2.5, 5.0, 7.5]:
        ax.annotate('', xy=(x_pos + 0.6, arrow_y), xytext=(x_pos, arrow_y),
                   arrowprops=dict(arrowstyle='->', color='blue', lw=1.5, alpha=0.6))

    # Add section dividers (dashed lines)
    ax.axvline(x_trans_start, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.axvline(x_trans_end, color='gray', linestyle='--', alpha=0.5, linewidth=1)

    # Section labels at top
    ax.text(L_in/2, y_top + 0.4, 'Inlet\nSection', ha='center', va='bottom', fontsize=10, style='italic')
    ax.text(L_in + L_trans/2, y_top + 0.4, 'Transition\nSection', ha='center', va='bottom', fontsize=10, style='italic')
    ax.text(L_in + L_trans + L_out/2, y_top + 0.4, 'Outlet\nSection', ha='center', va='bottom', fontsize=10, style='italic')

    # =========================================================================
    # Boundary condition labels with boxes
    # =========================================================================

    # INLET (left boundary) - Blue
    inlet_box = FancyBboxPatch((-2.8, -0.3), 2.6, 1.6,
                                boxstyle="round,pad=0.05,rounding_size=0.1",
                                facecolor='lightsteelblue', edgecolor='blue',
                                linewidth=2, alpha=0.9)
    ax.add_patch(inlet_box)
    inlet_text = (
        r"$\mathbf{Inlet}$" + "\n"
        r"$U = U_{in}$" + "\n"
        r"$k = \frac{3}{2}(U_{in} \cdot TI)^2$" + "\n"
        r"$T = T_{in} = 300\,\mathrm{K}$"
    )
    ax.text(-1.5, 0.5, inlet_text, ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='none', edgecolor='none'))

    # Arrow from inlet box to inlet boundary
    ax.annotate('', xy=(0, 0.5), xytext=(-0.3, 0.5),
               arrowprops=dict(arrowstyle='->', color='blue', lw=2))

    # OUTLET (right boundary) - Red
    outlet_box = FancyBboxPatch((L_total + 0.3, -0.8), 2.5, 1.6,
                                 boxstyle="round,pad=0.05,rounding_size=0.1",
                                 facecolor='mistyrose', edgecolor='red',
                                 linewidth=2, alpha=0.9)
    ax.add_patch(outlet_box)
    outlet_text = (
        r"$\mathbf{Outlet}$" + "\n"
        r"$p = 0\,\mathrm{(gauge)}$" + "\n"
        r"$\frac{\partial \phi}{\partial n} = 0$" + "\n"
        r"(zero gradient)"
    )
    ax.text(L_total + 1.55, 0, outlet_text, ha='center', va='center', fontsize=10)

    # Arrow from outlet box to outlet boundary
    ax.annotate('', xy=(L_total, 0), xytext=(L_total + 0.3, 0),
               arrowprops=dict(arrowstyle='->', color='red', lw=2))

    # TOP WALL (flat - training data source)
    top_box = FancyBboxPatch((3.0, y_top + 1.0), 3.2, 1.4,
                              boxstyle="round,pad=0.05,rounding_size=0.1",
                              facecolor='lightyellow', edgecolor='darkgoldenrod',
                              linewidth=2, alpha=0.9)
    ax.add_patch(top_box)
    top_text = (
        r"$\mathbf{Top\;Wall\;(Flat)}$" + "\n"
        r"$\mathbf{U} = 0$ (no-slip)" + "\n"
        r"$T_w = 330\,\mathrm{K}$" + "\n"
        r"Training data source"
    )
    ax.text(4.6, y_top + 1.7, top_text, ha='center', va='center', fontsize=10)

    # Arrow from top box to top wall
    ax.annotate('', xy=(4.6, y_top + 0.05), xytext=(4.6, y_top + 0.95),
               arrowprops=dict(arrowstyle='->', color='darkgoldenrod', lw=2))

    # BOTTOM WALL (curved/inclined)
    bottom_box = FancyBboxPatch((4.5, -(H_out - H_in) - 1.8), 3.0, 1.4,
                                 boxstyle="round,pad=0.05,rounding_size=0.1",
                                 facecolor='honeydew', edgecolor='darkgreen',
                                 linewidth=2, alpha=0.9)
    ax.add_patch(bottom_box)
    bottom_text = (
        r"$\mathbf{Bottom\;Wall}$" + "\n"
        r"$\mathbf{U} = 0$ (no-slip)" + "\n"
        r"$T_w = 330\,\mathrm{K}$" + "\n"
        r"Creates APG"
    )
    ax.text(6.0, -(H_out - H_in) - 1.1, bottom_text, ha='center', va='center', fontsize=10)

    # Arrow from bottom box to bottom wall
    ax.annotate('', xy=(6.0, -(H_out - H_in) - 0.05), xytext=(6.0, -(H_out - H_in) - 0.35),
               arrowprops=dict(arrowstyle='->', color='darkgreen', lw=2))

    # =========================================================================
    # Dimension annotations
    # =========================================================================

    # Height annotations
    dim_x = -0.3
    ax.annotate('', xy=(dim_x, 0), xytext=(dim_x, y_top),
               arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
    ax.text(dim_x - 0.2, y_top/2, r'$H_{in}$', ha='right', va='center', fontsize=11)

    dim_x = L_total + 0.1
    ax.annotate('', xy=(dim_x, -(H_out - H_in)), xytext=(dim_x, y_top),
               arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
    ax.text(dim_x + 0.15, (y_top - (H_out - H_in))/2, r'$H_{out}$', ha='left', va='center', fontsize=11)

    # Length annotations at bottom
    y_dim = -(H_out - H_in) - 0.3

    # L_in
    ax.annotate('', xy=(0, y_dim), xytext=(L_in, y_dim),
               arrowprops=dict(arrowstyle='<->', color='black', lw=1))
    ax.text(L_in/2, y_dim - 0.15, r'$L_{in}$', ha='center', va='top', fontsize=10)

    # L_trans
    ax.annotate('', xy=(L_in, y_dim), xytext=(L_in + L_trans, y_dim),
               arrowprops=dict(arrowstyle='<->', color='black', lw=1))
    ax.text(L_in + L_trans/2, y_dim - 0.15, r'$L_{trans}$', ha='center', va='top', fontsize=10)

    # L_out
    ax.annotate('', xy=(L_in + L_trans, y_dim), xytext=(L_total, y_dim),
               arrowprops=dict(arrowstyle='<->', color='black', lw=1))
    ax.text(L_in + L_trans + L_out/2, y_dim - 0.15, r'$L_{out}$', ha='center', va='top', fontsize=10)

    # =========================================================================
    # Final adjustments
    # =========================================================================

    ax.set_xlim(-3.5, L_total + 3.5)
    ax.set_ylim(-(H_out - H_in) - 2.5, y_top + 3.0)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    ax.set_title('Boundary Conditions for Diffuser Configuration\n(Flat top wall for training data, inclined bottom wall creates adverse pressure gradient)',
                fontsize=12, fontweight='bold', pad=10)

    plt.tight_layout()

    # Save
    output_path = os.path.join(SCRIPT_DIR, '..', 'Images', 'chapter3', 'boundary_conditions.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")

    plt.close()
    return output_path


if __name__ == "__main__":
    create_boundary_conditions_figure()
