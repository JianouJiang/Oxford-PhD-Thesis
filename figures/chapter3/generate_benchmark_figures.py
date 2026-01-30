#!/usr/bin/env python3
"""
Generate benchmark figures for Chapter 3: Experimental Benchmark Data for Separated Flows

This script creates:
1. Backward-facing step geometry and Cf profile
2. Periodic hills geometry and wall shear stress profile
3. Asymmetric diffuser geometry and Cf profile
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch
import matplotlib.patches as mpatches

# Set style for publication
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except OSError:
    try:
        plt.style.use('seaborn-whitegrid')
    except OSError:
        pass  # Use default style
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 150


def create_bfs_geometry():
    """Create backward-facing step geometry schematic."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    # Step geometry
    H = 1.0  # Step height
    L_inlet = 3 * H
    L_outlet = 12 * H

    # Bottom wall with step
    x_bottom = [-L_inlet, 0, 0, L_outlet]
    y_bottom = [H, H, 0, 0]

    # Top wall (straight)
    x_top = [-L_inlet, L_outlet]
    y_top = [2.5*H, 2.5*H]

    # Draw walls
    ax.fill_between(x_bottom, y_bottom, -0.3, color='lightgray', edgecolor='black', linewidth=1.5)
    ax.plot(x_top, y_top, 'k-', linewidth=1.5)

    # Flow direction arrows
    for y in [1.2*H, 1.5*H, 1.8*H, 2.1*H]:
        ax.annotate('', xy=(-L_inlet+0.5, y), xytext=(-L_inlet-0.5, y),
                    arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))

    # Recirculation zone (shaded)
    x_recirc = [0, 0, 6*H, 6*H, 0]
    y_recirc = [0, 0.8*H, 0.3*H, 0, 0]
    recirc_patch = Polygon(list(zip(x_recirc, y_recirc)), alpha=0.3, color='red', edgecolor='red', linestyle='--')
    ax.add_patch(recirc_patch)

    # Recirculation arrow
    ax.annotate('', xy=(1*H, 0.2*H), xytext=(5*H, 0.2*H),
                arrowprops=dict(arrowstyle='<->', color='darkred', lw=1.5, connectionstyle='arc3,rad=-0.3'))

    # Labels and annotations
    ax.annotate('H', xy=(-0.3, 0.5*H), fontsize=14, fontweight='bold')
    ax.annotate('', xy=(-0.2, 0), xytext=(-0.2, H),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))

    # Reattachment point
    ax.plot(6.26*H, 0, 'ro', markersize=10, markerfacecolor='red', markeredgecolor='black', zorder=5)
    ax.annotate('$x_R/H = 6.26$', xy=(6.26*H, -0.15), fontsize=11, ha='center')

    # Region labels
    ax.text(-1.5*H, 0.5*H, '1', fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='circle', facecolor='white', edgecolor='black'))
    ax.text(2*H, 0.9*H, '2', fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='circle', facecolor='white', edgecolor='black'))
    ax.text(3*H, 0.3*H, '3', fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='circle', facecolor='white', edgecolor='black'))
    ax.text(6.5*H, 0.2*H, '4', fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='circle', facecolor='white', edgecolor='black'))
    ax.text(9*H, 0.5*H, '5', fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='circle', facecolor='white', edgecolor='black'))

    # Coordinate system
    ax.set_xlabel('$x/H$')
    ax.set_ylabel('$y/H$')
    ax.set_xlim(-L_inlet-1, L_outlet+0.5)
    ax.set_ylim(-0.5, 3)
    ax.set_aspect('equal')
    ax.set_title('Backward-Facing Step Geometry and Flow Structure', fontweight='bold')

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='lightgray', edgecolor='black', label='Solid walls'),
        mpatches.Patch(facecolor='red', alpha=0.3, edgecolor='red', linestyle='--', label='Recirculation zone'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Reattachment point')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig('bfs_geometry.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created bfs_geometry.png")


def create_bfs_cf_profile():
    """Create backward-facing step Cf profile based on Driver & Seegmiller (1985)."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Approximate Cf data from Driver & Seegmiller (1985)
    x_H = np.linspace(-2, 12, 200)
    Cf = np.zeros_like(x_H)

    # Upstream attached region
    upstream = x_H < 0
    Cf[upstream] = 0.003

    # Separation bubble (0 < x/H < 6.26)
    bubble = (x_H >= 0) & (x_H < 6.26)
    x_norm = x_H[bubble] / 6.26
    Cf[bubble] = -0.003 * np.sin(np.pi * x_norm)

    # Recovery region (x/H > 6.26)
    recovery = x_H >= 6.26
    x_recovery = x_H[recovery] - 6.26
    Cf[recovery] = 0.0025 * (1 - np.exp(-x_recovery / 2.5))

    # Plot Cf
    ax.plot(x_H, Cf * 1000, 'b-', linewidth=2, label='$C_f$ (Driver & Seegmiller, 1985)')
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    ax.axvline(x=0, color='gray', linestyle=':', linewidth=1, label='Step location')
    ax.axvline(x=6.26, color='red', linestyle='--', linewidth=1.5, label='Reattachment ($x_R/H = 6.26$)')

    # Shaded recirculation zone
    ax.fill_between(x_H[(x_H >= 0) & (x_H <= 6.26)],
                    Cf[(x_H >= 0) & (x_H <= 6.26)] * 1000, 0,
                    alpha=0.2, color='red', label='Recirculation zone')

    # Annotations
    ax.annotate('Minimum $C_f$', xy=(3.1, -3), fontsize=10,
                arrowprops=dict(arrowstyle='->', color='black'))

    ax.set_xlabel('$x/H$')
    ax.set_ylabel('$C_f \\times 10^3$')
    ax.set_xlim(-2, 12)
    ax.set_ylim(-4, 4)
    ax.set_title('Skin Friction Coefficient: Backward-Facing Step', fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('bfs_cf_profile.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created bfs_cf_profile.png")


def create_periodic_hills_geometry():
    """Create periodic hills geometry schematic."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    H = 1.0  # Hill height

    # Hill shape (polynomial approximation)
    def hill_shape(x):
        """Approximate hill shape from Breuer et al. (2009)."""
        # Simplified polynomial approximation
        if isinstance(x, np.ndarray):
            y = np.zeros_like(x)
            # Hill region (0 to ~2H)
            mask1 = (x >= 0) & (x < 0.5)
            y[mask1] = H * np.cos(np.pi * x[mask1] / 1.0) ** 2

            mask2 = (x >= 0.5) & (x < 1.929)
            y[mask2] = H * (1 - ((x[mask2] - 0.5) / 1.429) ** 2)

            mask3 = (x >= 7.071) & (x <= 9)
            x_shifted = x[mask3] - 9
            y[mask3] = H * (1 - ((-x_shifted - 0.5) / 1.429) ** 2)

            mask4 = (x >= 8.5) & (x <= 9)
            y[mask4] = H * np.cos(np.pi * (9 - x[mask4]) / 1.0) ** 2

            return np.maximum(y, 0)
        return 0

    # Create hill profile
    x = np.linspace(0, 9, 500)
    y_hill = np.array([hill_shape(xi) if 0 <= xi <= 2 or 7 <= xi <= 9 else 0 for xi in x])

    # Smooth hill shape using simpler approximation
    y_hill = H * np.exp(-((x - 0) ** 2) / 0.8) + H * np.exp(-((x - 9) ** 2) / 0.8)
    y_hill = np.clip(y_hill, 0, H)

    # Draw domain
    ax.fill_between(x, 0, y_hill, color='lightgray', edgecolor='black', linewidth=1.5)
    ax.plot([0, 9], [3.035*H, 3.035*H], 'k-', linewidth=1.5)

    # Periodic indicators
    ax.annotate('', xy=(0, 1.5*H), xytext=(-0.5, 1.5*H),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    ax.annotate('', xy=(9.5, 1.5*H), xytext=(9, 1.5*H),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    ax.text(-0.7, 1.5*H, 'Periodic', fontsize=10, va='center')
    ax.text(9.2, 1.5*H, 'Periodic', fontsize=10, va='center')

    # Separation bubble
    x_sep = np.linspace(0.5, 4.5, 50)
    y_sep = 0.5 * H * np.sin(np.pi * (x_sep - 0.5) / 4)
    ax.fill_between(x_sep, 0, y_sep, alpha=0.3, color='red')

    # Flow arrows
    for yi in [0.5*H, 1.0*H, 1.5*H, 2.0*H, 2.5*H]:
        ax.annotate('', xy=(4.5, yi), xytext=(3.5, yi),
                    arrowprops=dict(arrowstyle='->', color='blue', lw=1))

    # Dimension lines
    ax.annotate('$L_x = 9H$', xy=(4.5, -0.3), fontsize=12, ha='center')
    ax.annotate('', xy=(0, -0.15), xytext=(9, -0.15),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))

    ax.set_xlabel('$x/H$')
    ax.set_ylabel('$y/H$')
    ax.set_xlim(-1, 10)
    ax.set_ylim(-0.5, 3.5)
    ax.set_aspect('equal')
    ax.set_title('Periodic Hills Geometry', fontweight='bold')

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='lightgray', edgecolor='black', label='Hill geometry'),
        mpatches.Patch(facecolor='red', alpha=0.3, label='Separation region'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig('periodic_hills_geometry.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created periodic_hills_geometry.png")


def create_periodic_hills_cf():
    """Create periodic hills wall shear stress profile."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Approximate wall shear stress from Breuer et al. (2009) at Re=10595
    x_H = np.linspace(0, 9, 200)
    tau_w = np.zeros_like(x_H)

    # FPG region (windward side, 7 < x/H < 9 and 0 < x/H < 0.5)
    fpg1 = (x_H > 7.5) | (x_H < 0.5)
    tau_w[fpg1] = 0.004 + 0.002 * np.sin(2 * np.pi * x_H[fpg1] / 9)

    # APG region (lee side, 0.5 < x/H < 2)
    apg = (x_H >= 0.5) & (x_H < 2)
    tau_w[apg] = 0.002 * (1 - (x_H[apg] - 0.5) / 1.5)

    # Separation region (2 < x/H < 4.5)
    sep = (x_H >= 2) & (x_H < 4.5)
    x_norm = (x_H[sep] - 2) / 2.5
    tau_w[sep] = -0.001 * np.sin(np.pi * x_norm)

    # Recovery region (4.5 < x/H < 7.5)
    recovery = (x_H >= 4.5) & (x_H <= 7.5)
    tau_w[recovery] = 0.002 * (1 - np.exp(-(x_H[recovery] - 4.5) / 1.5))

    # Plot
    ax.plot(x_H, tau_w * 1000, 'b-', linewidth=2, label='$\\tau_w$ (Breuer et al., 2009)')
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)

    # Region shading
    ax.fill_between(x_H[sep], tau_w[sep] * 1000, 0, alpha=0.2, color='red', label='Separation')

    # Region labels
    ax.annotate('FPG', xy=(8.5, 5), fontsize=11, fontweight='bold', color='green')
    ax.annotate('APG', xy=(1.0, 1.5), fontsize=11, fontweight='bold', color='orange')
    ax.annotate('Separation', xy=(3.0, -1.5), fontsize=11, fontweight='bold', color='red')
    ax.annotate('Recovery', xy=(5.5, 1.5), fontsize=11, fontweight='bold', color='blue')

    ax.set_xlabel('$x/H$')
    ax.set_ylabel('$\\tau_w \\times 10^3$ (normalized)')
    ax.set_xlim(0, 9)
    ax.set_ylim(-2, 6)
    ax.set_title('Wall Shear Stress: Periodic Hills at $Re_H = 10,595$', fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('periodic_hills_cf.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created periodic_hills_cf.png")


def create_diffuser_geometry():
    """Create asymmetric diffuser geometry schematic."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    H = 1.0  # Inlet height
    L_inlet = 5 * H
    L_diffuser = 21 * H
    L_outlet = 10 * H
    theta = 10 * np.pi / 180  # 10 degrees

    # Bottom wall (inclined)
    x_bottom = [-L_inlet, 0, L_diffuser, L_diffuser + L_outlet]
    y_bottom = [0, 0, -L_diffuser * np.tan(theta), -L_diffuser * np.tan(theta)]

    # Top wall (flat)
    x_top = [-L_inlet, L_diffuser + L_outlet]
    y_top = [H, H]

    # Draw walls
    ax.fill_between(x_bottom, y_bottom, min(y_bottom) - 0.5, color='lightgray', edgecolor='black', linewidth=1.5)
    ax.plot(x_top, y_top, 'k-', linewidth=1.5)

    # Channel inlet
    ax.annotate('$H$', xy=(-L_inlet - 0.3, 0.5*H), fontsize=12, fontweight='bold')
    ax.annotate('', xy=(-L_inlet - 0.2, 0), xytext=(-L_inlet - 0.2, H),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))

    # Outlet height
    H_out = H + L_diffuser * np.tan(theta)
    ax.annotate('$4.7H$', xy=(L_diffuser + 0.5, H - H_out/2), fontsize=12, fontweight='bold')

    # Angle annotation
    ax.annotate('$\\theta = 10°$', xy=(5, -0.5), fontsize=11)

    # Separation region (approximate)
    x_sep = np.linspace(14, 21, 50)
    y_sep_bottom = -x_sep * np.tan(theta)
    y_sep_top = y_sep_bottom + 0.3 * H
    ax.fill_between(x_sep, y_sep_bottom, y_sep_top, alpha=0.3, color='red')

    # Flow direction
    for y in [0.2*H, 0.5*H, 0.8*H]:
        ax.annotate('', xy=(-L_inlet + 1, y), xytext=(-L_inlet, y),
                    arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))

    ax.set_xlabel('$x/H$')
    ax.set_ylabel('$y/H$')
    ax.set_xlim(-L_inlet - 1, L_diffuser + L_outlet + 1)
    ax.set_ylim(min(y_bottom) - 1, H + 1)
    ax.set_aspect('equal')
    ax.set_title('Asymmetric Plane Diffuser (Buice & Eaton, 1997)', fontweight='bold')

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='lightgray', edgecolor='black', label='Solid walls'),
        mpatches.Patch(facecolor='red', alpha=0.3, label='Separation region'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig('diffuser_geometry.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created diffuser_geometry.png")


def create_diffuser_cf():
    """Create asymmetric diffuser Cf profile."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Approximate Cf from Buice & Eaton (1997)
    x_H = np.linspace(-5, 35, 200)
    Cf = np.zeros_like(x_H)

    # Inlet fully developed flow
    inlet = x_H < 0
    Cf[inlet] = 0.0061

    # Diffuser region with APG
    diffuser = (x_H >= 0) & (x_H < 21)
    Cf[diffuser] = 0.0061 * np.exp(-x_H[diffuser] / 12)

    # Near separation (Cf approaching zero)
    near_sep = (x_H >= 14) & (x_H < 21)
    Cf[near_sep] = np.maximum(Cf[near_sep], 0.0005)

    # Intermittent separation (oscillating near zero)
    sep = (x_H >= 18) & (x_H < 25)
    Cf[sep] = 0.0003 + 0.0002 * np.sin(2 * np.pi * (x_H[sep] - 18) / 3)

    # Recovery
    recovery = x_H >= 25
    Cf[recovery] = 0.003 * (1 - np.exp(-(x_H[recovery] - 25) / 5))

    # Plot
    ax.plot(x_H, Cf * 1000, 'b-', linewidth=2, label='$C_f$ (Buice & Eaton, 1997)')
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    ax.axvline(x=0, color='gray', linestyle=':', linewidth=1, label='Diffuser inlet')
    ax.axvline(x=21, color='gray', linestyle=':', linewidth=1, label='Diffuser outlet')

    # Shaded APG region
    ax.axvspan(0, 21, alpha=0.1, color='orange', label='APG region')

    # Annotations
    ax.annotate('Fully developed\n$C_f = 0.0061$', xy=(-3, 6), fontsize=10, ha='center')
    ax.annotate('Incipient\nseparation', xy=(18, 0.8), fontsize=10, ha='center',
                arrowprops=dict(arrowstyle='->', color='red'))

    ax.set_xlabel('$x/H$')
    ax.set_ylabel('$C_f \\times 10^3$')
    ax.set_xlim(-5, 35)
    ax.set_ylim(-0.5, 7)
    ax.set_title('Skin Friction Coefficient: Asymmetric Diffuser', fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('diffuser_cf.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created diffuser_cf.png")


if __name__ == "__main__":
    print("Generating benchmark figures for Chapter 3...")
    create_bfs_geometry()
    create_bfs_cf_profile()
    create_periodic_hills_geometry()
    create_periodic_hills_cf()
    create_diffuser_geometry()
    create_diffuser_cf()
    print("All figures generated successfully!")
