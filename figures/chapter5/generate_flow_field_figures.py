#!/usr/bin/env python3
"""
Generate flow field figures for Chapter 5: Physics-Based Feature Variables.

This script creates flow field visualizations to complement the existing
bar charts and scatter plots with actual CFD flow field data.

New figures:
1. velocity_profiles.png - u+ vs y+ profiles showing log-law behavior
2. flow_field_contours.png - Velocity and pressure contours for different regimes
3. tau_w_distribution.png - Wall shear stress distribution along the wall
4. pressure_gradient_map.png - Visualization of pressure gradient regions
5. boundary_layer_comparison.png - BL profiles at different streamwise locations
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import matplotlib.patches as mpatches

# Try to import PyVista for VTK visualization
try:
    import pyvista as pv
    pv.set_plot_theme('document')
    HAS_PYVISTA = True
except ImportError:
    print("Warning: PyVista not available. Some visualizations will be limited.")
    HAS_PYVISTA = False

# Style settings for thesis figures
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR))))
TRAINING_DATA_DIR = os.path.join(PROJECT_ROOT, "TRAINING_DATA", "data")
CASES_DIR = os.path.join(TRAINING_DATA_DIR, "cases")
OUTPUT_DIR = SCRIPT_DIR

# Physics constants
RHO = 1.225  # kg/m³
NU = 5e-5   # m²/s
KAPPA = 0.41  # von Karman constant
B = 5.0  # Log-law additive constant


def get_available_cases():
    """Get list of available simulation cases."""
    if not os.path.exists(CASES_DIR):
        print(f"Warning: Cases directory not found: {CASES_DIR}")
        return []

    cases = []
    for d in os.listdir(CASES_DIR):
        case_dir = os.path.join(CASES_DIR, d)
        if os.path.isdir(case_dir):
            meta_file = os.path.join(case_dir, "case_metadata.json")
            if os.path.exists(meta_file):
                with open(meta_file, 'r') as f:
                    meta = json.load(f)
                meta['case_dir'] = case_dir
                meta['case_name'] = d
                cases.append(meta)
    return cases


def load_vtk_data(case_dir):
    """Load domain VTK data from a case."""
    if not HAS_PYVISTA:
        return None

    vtk_dir = os.path.join(case_dir, "VTK")
    if not os.path.exists(vtk_dir):
        return None

    # Find domain VTK file
    vtk_files = []
    for f in os.listdir(vtk_dir):
        fpath = os.path.join(vtk_dir, f)
        if f.endswith(".vtk") and not os.path.isdir(fpath):
            if "Wall" not in f and "inlet" not in f and "outlet" not in f:
                vtk_files.append(f)

    if not vtk_files:
        vtk_files = [f for f in os.listdir(vtk_dir)
                     if f.endswith(".vtk") and not os.path.isdir(os.path.join(vtk_dir, f))]

    if not vtk_files:
        return None

    vtk_path = os.path.join(vtk_dir, sorted(vtk_files)[-1])
    try:
        return pv.read(vtk_path)
    except Exception as e:
        print(f"  Error reading {vtk_path}: {e}")
        return None


def load_wall_data(case_dir, wall_name="bottomWall"):
    """Load wall patch VTK data."""
    if not HAS_PYVISTA:
        return None

    vtk_dir = os.path.join(case_dir, "VTK", wall_name)
    if not os.path.exists(vtk_dir):
        return None

    vtk_files = [f for f in os.listdir(vtk_dir) if f.endswith(".vtk")]
    if not vtk_files:
        return None

    vtk_path = os.path.join(vtk_dir, sorted(vtk_files)[-1])
    try:
        return pv.read(vtk_path)
    except Exception as e:
        print(f"  Error reading {vtk_path}: {e}")
        return None


def extract_wall_shear_stress(wall_mesh):
    """Extract wall shear stress magnitude along the wall."""
    if wall_mesh is None or 'wallShearStress' not in wall_mesh.array_names:
        return None, None

    wss = wall_mesh['wallShearStress']
    # For 2D, use x-component (streamwise)
    wss_x = wss[:, 0]

    # Get cell centers for wall data
    try:
        cell_centers = wall_mesh.cell_centers()
        x = cell_centers.points[:, 0]
    except:
        points = wall_mesh.points
        x = points[:, 0]

    # Ensure arrays have same length
    min_len = min(len(x), len(wss_x))
    x = x[:min_len]
    wss_x = wss_x[:min_len]

    # Sort by x
    idx = np.argsort(x)
    return x[idx], wss_x[idx]


def create_velocity_profiles_figure(cases, output_path):
    """
    Create u+ vs y+ velocity profile figure showing log-law behavior.

    Shows profiles at different streamwise locations for attached and
    separating flow conditions.
    """
    print("Generating velocity_profiles.png...")

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Find cases with different flow conditions
    attached_case = None
    mild_apg_case = None
    strong_apg_case = None

    for case in cases:
        if case['mesh_resolution'] != 'fine':
            continue
        er = case.get('expansion_ratio', 1.0)
        if er <= 1.1 and attached_case is None:
            attached_case = case
        elif 1.5 <= er <= 2.5 and mild_apg_case is None:
            mild_apg_case = case
        elif er >= 3.5 and strong_apg_case is None:
            strong_apg_case = case

    selected_cases = [
        (attached_case, 'Attached Flow (ER ≈ 1.0)', 'Favorable/Zero PG'),
        (mild_apg_case, 'Mild APG (ER ≈ 2.0)', 'Adverse PG'),
        (strong_apg_case, 'Strong APG (ER ≈ 4.0)', 'Near Separation'),
    ]

    # Log-law reference
    y_plus_ref = np.logspace(0, 2.5, 100)
    u_plus_linear = y_plus_ref  # Viscous sublayer
    u_plus_log = (1/KAPPA) * np.log(y_plus_ref) + B  # Log law

    for ax, (case, title, regime) in zip(axes, selected_cases):
        # Plot log-law reference
        ax.semilogx(y_plus_ref[y_plus_ref < 11], u_plus_linear[y_plus_ref < 11],
                   'k--', linewidth=1.5, label=r'$u^+ = y^+$')
        ax.semilogx(y_plus_ref[y_plus_ref > 30], u_plus_log[y_plus_ref > 30],
                   'k-', linewidth=1.5, label=r'$u^+ = \frac{1}{\kappa}\ln(y^+) + B$')

        if case is not None:
            mesh = load_vtk_data(case['case_dir'])
            if mesh is not None and 'U' in mesh.array_names:
                # Get wall data to compute u_tau
                wall = load_wall_data(case['case_dir'], "topWall")
                x_wall, tau_wall = extract_wall_shear_stress(wall)

                if x_wall is not None:
                    # Extract profiles at different x locations
                    cell_centers = mesh.cell_centers()
                    points = cell_centers.points

                    # Get z-mid plane
                    bounds = mesh.bounds
                    z_mid = (bounds[4] + bounds[5]) / 2
                    z_tol = 0.01

                    if 'U' in mesh.cell_data:
                        U = mesh.cell_data['U']
                    else:
                        U = mesh.point_data['U']

                    # Sample at 3 streamwise locations
                    x_locs = [0.1, 0.3, 0.6]  # Normalized positions
                    x_range = bounds[1] - bounds[0]
                    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

                    for x_frac, color in zip(x_locs, colors):
                        x_target = bounds[0] + x_frac * x_range
                        x_tol = x_range * 0.02

                        # Find cells near this x location
                        mask = (np.abs(points[:, 0] - x_target) < x_tol) & \
                               (np.abs(points[:, 2] - z_mid) < z_tol)

                        if np.sum(mask) > 5:
                            y_local = points[mask, 1]
                            U_local = U[mask, 0]

                            # Find local wall shear stress
                            idx_wall = np.argmin(np.abs(x_wall - x_target))
                            tau_local = np.abs(tau_wall[idx_wall])

                            if tau_local > 1e-10:
                                u_tau = np.sqrt(tau_local / RHO)

                                # Compute wall distance (from top wall)
                                y_wall = bounds[3]  # Top wall y
                                y_dist = y_wall - y_local
                                y_dist = np.abs(y_dist)

                                # Compute y+ and u+
                                y_plus = y_dist * u_tau / NU
                                u_plus = np.abs(U_local) / u_tau

                                # Sort and plot
                                idx_sort = np.argsort(y_plus)
                                y_plus_sorted = y_plus[idx_sort]
                                u_plus_sorted = u_plus[idx_sort]

                                # Filter valid range
                                valid = (y_plus_sorted > 1) & (y_plus_sorted < 300)
                                ax.semilogx(y_plus_sorted[valid], u_plus_sorted[valid],
                                          'o', color=color, markersize=3, alpha=0.7,
                                          label=f'x/L = {x_frac:.1f}')

        ax.set_xlabel(r'$y^+$', fontsize=12)
        ax.set_ylabel(r'$u^+$', fontsize=12)
        ax.set_title(f'{title}\n({regime})', fontsize=11)
        ax.set_xlim([1, 300])
        ax.set_ylim([0, 25])
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.3)

        # Add regime annotation
        if 'Attached' in title:
            ax.fill_between([30, 300], [0, 0], [25, 25], alpha=0.1, color='green')
            ax.text(100, 2, 'Log-law valid', fontsize=9, color='green', style='italic')
        elif 'Strong' in title:
            ax.fill_between([30, 300], [0, 0], [25, 25], alpha=0.1, color='red')
            ax.text(50, 2, 'Log-law deviation', fontsize=9, color='red', style='italic')

    plt.suptitle('Boundary Layer Velocity Profiles: Physics Feature $u^+$ vs $y^+$',
                fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def create_flow_field_contours_figure(cases, output_path):
    """
    Create velocity and pressure contour plots showing different flow regimes.
    """
    print("Generating flow_field_contours.png...")

    # Find representative cases
    selected_cases = []
    target_ers = [1.0, 2.5, 4.5]

    for target_er in target_ers:
        best_case = None
        best_diff = float('inf')
        for case in cases:
            if case['mesh_resolution'] != 'fine':
                continue
            er = case.get('expansion_ratio', 1.0)
            diff = abs(er - target_er)
            if diff < best_diff:
                best_diff = diff
                best_case = case
        if best_case:
            selected_cases.append(best_case)

    if len(selected_cases) < 2:
        print("  Not enough cases for flow field contours")
        return

    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(len(selected_cases), 2, figure=fig, hspace=0.3, wspace=0.25)

    for row, case in enumerate(selected_cases):
        mesh = load_vtk_data(case['case_dir'])
        if mesh is None:
            continue

        er = case.get('expansion_ratio', 1.0)

        # Get cell centers and extract 2D slice
        cell_centers = mesh.cell_centers()
        points = cell_centers.points
        bounds = mesh.bounds
        z_mid = (bounds[4] + bounds[5]) / 2
        z_tol = 0.01
        mask = np.abs(points[:, 2] - z_mid) < z_tol

        x = points[mask, 0]
        y = points[mask, 1]

        # Left column: Velocity magnitude
        ax_vel = fig.add_subplot(gs[row, 0])

        if 'U' in mesh.array_names:
            if 'U' in mesh.cell_data:
                U = mesh.cell_data['U'][mask]
            else:
                U = mesh.point_data['U'][:len(mask)][mask]

            U_mag = np.sqrt(U[:, 0]**2 + U[:, 1]**2)

            scatter = ax_vel.tricontourf(x, y, U_mag, levels=30, cmap='jet')
            cbar = plt.colorbar(scatter, ax=ax_vel, shrink=0.8)
            cbar.set_label('|U| [m/s]', fontsize=10)

            # Mark separation (where Ux < 0)
            if np.any(U[:, 0] < 0):
                sep_mask = U[:, 0] < 0
                ax_vel.scatter(x[sep_mask], y[sep_mask], c='white', s=1, alpha=0.5)

        ax_vel.set_xlabel('x [m]', fontsize=10)
        ax_vel.set_ylabel('y [m]', fontsize=10)
        ax_vel.set_aspect('equal')

        regime = 'ZPG' if er <= 1.1 else ('Mild APG' if er <= 3.0 else 'Strong APG')
        ax_vel.set_title(f'ER = {er:.1f} ({regime}) - Velocity Field', fontsize=11)

        # Right column: Pressure gradient
        ax_pg = fig.add_subplot(gs[row, 1])

        if 'p' in mesh.array_names:
            if 'p' in mesh.cell_data:
                p = mesh.cell_data['p'][mask]
            else:
                p = mesh.point_data['p'][:len(mask)][mask]

            # Use pressure for visualization (gradient would require differentiation)
            # Normalize by inlet pressure
            p_norm = (p - np.min(p)) / (np.max(p) - np.min(p) + 1e-10)

            scatter = ax_pg.tricontourf(x, y, p_norm, levels=30, cmap='RdBu_r')
            cbar = plt.colorbar(scatter, ax=ax_pg, shrink=0.8)
            cbar.set_label('p (normalized)', fontsize=10)

        ax_pg.set_xlabel('x [m]', fontsize=10)
        ax_pg.set_ylabel('y [m]', fontsize=10)
        ax_pg.set_aspect('equal')
        ax_pg.set_title(f'ER = {er:.1f} ({regime}) - Pressure Field', fontsize=11)

    plt.suptitle('Flow Field Visualization: Velocity and Pressure for Different Expansion Ratios',
                fontsize=13, fontweight='bold', y=0.98)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def create_tau_w_distribution_figure(cases, output_path):
    """
    Create wall shear stress distribution figure along the wall.
    Shows how τ_w varies with streamwise distance for different flow conditions.
    """
    print("Generating tau_w_distribution.png...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel (a): τ_w distribution for different ERs
    ax = axes[0, 0]

    # Group by ER
    er_groups = {}
    for case in cases:
        if case['mesh_resolution'] != 'fine':
            continue
        er = case.get('expansion_ratio', 1.0)
        er_rounded = round(er * 2) / 2
        if er_rounded not in er_groups:
            er_groups[er_rounded] = case

    ers = sorted(er_groups.keys())
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(ers)))

    for er, color in zip(ers[:6], colors[:6]):
        case = er_groups[er]
        wall = load_wall_data(case['case_dir'], "topWall")
        x, tau = extract_wall_shear_stress(wall)

        if x is not None:
            H_in = case.get('H_in', 1.0)
            x_norm = x / H_in
            ax.plot(x_norm, tau, color=color, linewidth=1.5, label=f'ER={er:.1f}')

    ax.set_xlabel('x / H_inlet', fontsize=11)
    ax.set_ylabel(r'$\tau_w$ [Pa]', fontsize=11)
    ax.set_title('(a) Wall Shear Stress Distribution by Expansion Ratio', fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

    # Panel (b): Skin friction coefficient
    ax = axes[0, 1]

    for er, color in zip(ers[:6], colors[:6]):
        case = er_groups[er]
        wall = load_wall_data(case['case_dir'], "topWall")
        x, tau = extract_wall_shear_stress(wall)

        if x is not None:
            U_inlet = case.get('U_inlet', 0.4)
            Cf = tau / (0.5 * RHO * U_inlet**2)
            H_in = case.get('H_in', 1.0)
            x_norm = x / H_in
            ax.plot(x_norm, Cf * 1000, color=color, linewidth=1.5, label=f'ER={er:.1f}')

    ax.set_xlabel('x / H_inlet', fontsize=11)
    ax.set_ylabel(r'$C_f \times 10^3$', fontsize=11)
    ax.set_title('(b) Skin Friction Coefficient Distribution', fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

    # Panel (c): τ_w with flow regime annotations
    ax = axes[1, 0]

    # Use a high-ER case to show different flow regimes
    high_er_case = er_groups.get(4.0) or er_groups.get(3.5) or er_groups.get(3.0)
    if high_er_case:
        wall = load_wall_data(high_er_case['case_dir'], "topWall")
        x, tau = extract_wall_shear_stress(wall)

        if x is not None:
            H_in = high_er_case.get('H_in', 1.0)
            x_norm = x / H_in

            ax.plot(x_norm, tau, 'b-', linewidth=2, label=r'$\tau_w$')
            ax.fill_between(x_norm, tau, 0, where=tau >= 0,
                          color='green', alpha=0.3, label='Attached')
            ax.fill_between(x_norm, tau, 0, where=tau < 0,
                          color='red', alpha=0.3, label='Separated')
            ax.axhline(y=0, color='k', linestyle='-', linewidth=1)

            # Mark flow regions
            # Inlet region (FPG)
            ax.axvspan(x_norm.min(), 0, alpha=0.1, color='blue')
            ax.text((x_norm.min() + 0)/2, ax.get_ylim()[1]*0.8, 'Inlet\n(FPG)',
                   fontsize=9, ha='center', color='blue')

            # Expansion region (APG)
            ax.axvspan(0, 5, alpha=0.1, color='orange')
            ax.text(2.5, ax.get_ylim()[1]*0.8, 'Expansion\n(APG)',
                   fontsize=9, ha='center', color='darkorange')

            er = high_er_case.get('expansion_ratio', 1.0)
            ax.set_title(f'(c) Flow Regime Identification (ER = {er:.1f})', fontsize=11, fontweight='bold')

    ax.set_xlabel('x / H_inlet', fontsize=11)
    ax.set_ylabel(r'$\tau_w$ [Pa]', fontsize=11)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel (d): τ_w statistics by region
    ax = axes[1, 1]

    regions = ['Inlet\n(FPG)', 'Transition', 'Expansion\n(APG)', 'Recovery']
    tau_means = []
    tau_stds = []

    # Compute regional statistics from all cases
    for case in cases:
        if case['mesh_resolution'] != 'fine':
            continue
        wall = load_wall_data(case['case_dir'], "topWall")
        x, tau = extract_wall_shear_stress(wall)
        if x is not None:
            H_in = case.get('H_in', 1.0)
            x_norm = x / H_in

            # Define regions
            inlet_mask = x_norm < 0
            trans_mask = (x_norm >= 0) & (x_norm < 2)
            exp_mask = (x_norm >= 2) & (x_norm < 10)
            rec_mask = x_norm >= 10

            for i, mask in enumerate([inlet_mask, trans_mask, exp_mask, rec_mask]):
                if len(tau_means) <= i:
                    tau_means.append([])
                    tau_stds.append([])
                if np.any(mask):
                    tau_means[i].append(np.mean(np.abs(tau[mask])))

    # Compute overall means
    region_means = [np.mean(vals) if vals else 0 for vals in tau_means]
    region_stds = [np.std(vals) if vals else 0 for vals in tau_means]

    x_pos = np.arange(len(regions))
    bars = ax.bar(x_pos, region_means, yerr=region_stds, capsize=5,
                 color=['blue', 'purple', 'orange', 'green'], alpha=0.7, edgecolor='black')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(regions, fontsize=10)
    ax.set_ylabel(r'Mean $|\tau_w|$ [Pa]', fontsize=11)
    ax.set_title('(d) Mean Wall Shear Stress by Flow Region', fontsize=11, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def create_pressure_gradient_map_figure(cases, output_path):
    """
    Create visualization of pressure gradient regions (FPG, ZPG, APG).
    Shows how dp/dx varies spatially in diffuser flows.
    """
    print("Generating pressure_gradient_map.png...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Find cases with different ERs
    target_ers = [1.0, 2.0, 3.5, 5.0]
    selected_cases = []

    for target_er in target_ers:
        best_case = None
        best_diff = float('inf')
        for case in cases:
            if case['mesh_resolution'] != 'fine':
                continue
            er = case.get('expansion_ratio', 1.0)
            diff = abs(er - target_er)
            if diff < best_diff:
                best_diff = diff
                best_case = case
        if best_case:
            selected_cases.append(best_case)

    for ax, case in zip(axes.flatten(), selected_cases):
        if case is None:
            continue

        mesh = load_vtk_data(case['case_dir'])
        if mesh is None:
            continue

        er = case.get('expansion_ratio', 1.0)

        # Get cell centers and extract 2D slice
        cell_centers = mesh.cell_centers()
        points = cell_centers.points
        bounds = mesh.bounds
        z_mid = (bounds[4] + bounds[5]) / 2
        z_tol = 0.01
        mask = np.abs(points[:, 2] - z_mid) < z_tol

        x = points[mask, 0]
        y = points[mask, 1]

        if 'p' in mesh.array_names:
            if 'p' in mesh.cell_data:
                p = mesh.cell_data['p'][mask]
            else:
                p = mesh.point_data['p'][:len(mask)][mask]

            # Estimate dp/dx using pressure values
            # (simplified - actual gradient would need proper differentiation)
            # Use pressure variation as proxy

            # Create triangulation-based visualization
            from scipy.interpolate import griddata

            # Create regular grid
            xi = np.linspace(x.min(), x.max(), 100)
            yi = np.linspace(y.min(), y.max(), 50)
            Xi, Yi = np.meshgrid(xi, yi)

            # Interpolate pressure to grid
            Pi = griddata((x, y), p, (Xi, Yi), method='linear')

            # Compute dp/dx
            dx = xi[1] - xi[0]
            dpdx = np.gradient(Pi, dx, axis=1)

            # Plot with safe normalization
            vmin = np.nanpercentile(dpdx, 5)
            vmax = np.nanpercentile(dpdx, 95)

            # Ensure valid norm values
            if vmin >= 0:
                vmin = -abs(vmax) * 0.1 if vmax > 0 else -1
            if vmax <= 0:
                vmax = abs(vmin) * 0.1 if vmin < 0 else 1

            try:
                norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
            except ValueError:
                norm = None  # Fall back to default normalization

            contour = ax.contourf(Xi, Yi, dpdx, levels=30, cmap='RdBu_r', norm=norm)
            cbar = plt.colorbar(contour, ax=ax, shrink=0.8)
            cbar.set_label(r'$\partial p/\partial x$ [Pa/m]', fontsize=10)

            # Annotate regions
            if er > 1.5:
                ax.annotate('APG\nregion', xy=(0.4, 0.7), xycoords='axes fraction',
                           fontsize=10, color='red', fontweight='bold', ha='center')
            if er >= 1.0:
                ax.annotate('FPG', xy=(0.1, 0.5), xycoords='axes fraction',
                           fontsize=10, color='blue', fontweight='bold', ha='center')

        regime = 'ZPG' if er <= 1.1 else ('Mild APG' if er <= 3.0 else 'Strong APG')
        ax.set_xlabel('x [m]', fontsize=10)
        ax.set_ylabel('y [m]', fontsize=10)
        ax.set_aspect('equal')
        ax.set_title(f'ER = {er:.1f} ({regime})', fontsize=11, fontweight='bold')

    plt.suptitle(r'Streamwise Pressure Gradient $\partial p/\partial x$: Key Physics Feature',
                fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def create_boundary_layer_comparison_figure(cases, output_path):
    """
    Create boundary layer profile comparison at different streamwise locations.
    Shows how the BL develops through the diffuser.
    """
    print("Generating boundary_layer_comparison.png...")

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    # Find a case with mild APG for clear visualization
    target_er = 2.5
    selected_case = None
    best_diff = float('inf')

    for case in cases:
        if case['mesh_resolution'] != 'fine':
            continue
        er = case.get('expansion_ratio', 1.0)
        diff = abs(er - target_er)
        if diff < best_diff:
            best_diff = diff
            selected_case = case

    if selected_case is None:
        print("  No suitable case found")
        return

    mesh = load_vtk_data(selected_case['case_dir'])
    if mesh is None:
        print("  Could not load mesh")
        return

    er = selected_case.get('expansion_ratio', 1.0)

    # Get cell centers
    cell_centers = mesh.cell_centers()
    points = cell_centers.points
    bounds = mesh.bounds
    z_mid = (bounds[4] + bounds[5]) / 2
    z_tol = 0.01

    if 'U' in mesh.cell_data:
        U = mesh.cell_data['U']
    else:
        U = mesh.point_data['U']

    # Get wall data
    wall = load_wall_data(selected_case['case_dir'], "topWall")
    x_wall, tau_wall = extract_wall_shear_stress(wall)

    # Sample at 6 streamwise locations
    x_fracs = [0.05, 0.15, 0.30, 0.45, 0.60, 0.80]
    x_range = bounds[1] - bounds[0]

    for ax, x_frac in zip(axes.flatten(), x_fracs):
        x_target = bounds[0] + x_frac * x_range
        x_tol = x_range * 0.015

        # Find cells near this x location
        mask = (np.abs(points[:, 0] - x_target) < x_tol) & \
               (np.abs(points[:, 2] - z_mid) < z_tol)

        if np.sum(mask) < 5:
            continue

        y_local = points[mask, 1]
        Ux_local = U[mask, 0]

        # Sort by y
        idx_sort = np.argsort(y_local)
        y_sorted = y_local[idx_sort]
        Ux_sorted = Ux_local[idx_sort]

        # Normalize
        y_norm = (y_sorted - y_sorted.min()) / (y_sorted.max() - y_sorted.min() + 1e-10)
        U_norm = Ux_sorted / (np.max(np.abs(Ux_sorted)) + 1e-10)

        ax.plot(U_norm, y_norm, 'b-', linewidth=2)
        ax.fill_betweenx(y_norm, 0, U_norm, alpha=0.3, color='blue')

        # Mark separation
        if np.any(Ux_sorted < 0):
            ax.axvline(x=0, color='red', linestyle='--', linewidth=1)
            ax.fill_betweenx(y_norm, U_norm, 0, where=U_norm < 0,
                           alpha=0.3, color='red')
            ax.annotate('Reversed flow', xy=(0.05, 0.1), fontsize=8, color='red')

        # Determine local regime
        if x_wall is not None:
            idx_wall = np.argmin(np.abs(x_wall - x_target))
            tau_local = tau_wall[idx_wall]
            if tau_local > 0.002:
                regime = 'FPG'
                color = 'green'
            elif tau_local > 0:
                regime = 'APG'
                color = 'orange'
            else:
                regime = 'Sep.'
                color = 'red'
        else:
            regime = '?'
            color = 'gray'

        H_in = selected_case.get('H_in', 1.0)
        x_norm = (x_target - bounds[0]) / H_in

        ax.set_xlabel('U / U_max', fontsize=10)
        ax.set_ylabel('y / H_local', fontsize=10)
        ax.set_title(f'x/H_in = {x_norm:.1f} ({regime})', fontsize=10, fontweight='bold',
                    color=color)
        ax.set_xlim([-0.2, 1.1])
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)

    plt.suptitle(f'Boundary Layer Development Through Diffuser (ER = {er:.1f})',
                fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def create_feature_physics_connection_figure(output_path):
    """
    Create a conceptual figure showing how physics features connect to flow physics.
    """
    print("Generating feature_physics_connection.png...")

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Colors
    physics_color = '#98d8aa'
    feature_color = '#a8d8ea'
    prediction_color = '#f1a7a7'

    # Draw flow field schematic (left side)
    ax.add_patch(plt.Rectangle((0.5, 2), 4, 4, facecolor='lightgray', edgecolor='black', linewidth=2))
    ax.text(2.5, 6.5, 'Flow Field', fontsize=12, fontweight='bold', ha='center')

    # Draw flow streamlines
    for y_off in [0.5, 1.5, 2.5, 3.5]:
        x_stream = np.linspace(0.6, 4.4, 20)
        y_stream = 2.5 + y_off + 0.1 * np.sin(x_stream * 2)
        ax.plot(x_stream, y_stream, 'b-', alpha=0.5, linewidth=0.8)
    ax.annotate('', xy=(4, 4), xytext=(1, 4),
               arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    ax.text(2.5, 4.3, 'Flow', fontsize=10, color='blue', ha='center')

    # Wall and boundary layer
    ax.plot([0.5, 4.5], [2, 2], 'k-', linewidth=3)
    ax.text(2.5, 1.7, 'Wall', fontsize=10, ha='center')

    # Draw physics features (middle)
    features = [
        (r'$y^+$', 'Wall distance\nscaling'),
        (r'$u^+$', 'Velocity\nscaling'),
        (r'$\partial p/\partial x$', 'Pressure\ngradient'),
        (r'$Re_y$', 'Local\nReynolds'),
        (r'$\sqrt{S_{ij}S_{ij}}$', 'Strain rate'),
    ]

    for i, (symbol, desc) in enumerate(features):
        y_pos = 7 - i * 1.2
        # Feature box
        rect = plt.Rectangle((6, y_pos - 0.4), 2.5, 0.8,
                            facecolor=feature_color, edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(7.25, y_pos, f'{symbol}', fontsize=11, ha='center', va='center', fontweight='bold')
        ax.text(9, y_pos, desc, fontsize=9, ha='left', va='center')

        # Arrow from flow field
        ax.annotate('', xy=(6, y_pos), xytext=(4.5, 4),
                   arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5,
                                  connectionstyle='arc3,rad=0.2'))

    ax.text(7.25, 8, 'Physics-Based\nFeatures', fontsize=12, fontweight='bold', ha='center')

    # Draw neural network (right side)
    nn_x = 12
    ax.add_patch(plt.Rectangle((11, 2.5), 2, 4, facecolor=prediction_color,
                               edgecolor='black', linewidth=2, alpha=0.7))
    ax.text(12, 7, 'Neural\nNetwork', fontsize=12, fontweight='bold', ha='center')

    # Hidden layers representation
    for j in range(3):
        for i in range(4):
            circle = plt.Circle((11.3 + j*0.6, 3 + i*0.9), 0.15,
                               facecolor='white', edgecolor='black')
            ax.add_patch(circle)

    # Arrows to NN
    for i in range(5):
        y_pos = 7 - i * 1.2
        ax.annotate('', xy=(11, 4 + (2-i)*0.5), xytext=(8.5, y_pos),
                   arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    # Output
    ax.add_patch(plt.Rectangle((12.5, 0.5), 1.3, 1.2, facecolor='#fff3b0',
                               edgecolor='black', linewidth=1.5))
    ax.text(13.15, 1.1, r'$\tau_w, q_w$', fontsize=11, ha='center', va='center', fontweight='bold')
    ax.annotate('', xy=(13.15, 0.5), xytext=(12, 2.5),
               arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.text(13.15, 0.2, 'Predictions', fontsize=10, ha='center')

    # Add key insight annotation
    ax.annotate('Physics features encode:\n• Scale-invariant relationships\n• Pressure gradient effects\n• Turbulence production',
               xy=(7.25, 0.8), fontsize=10, ha='center',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.set_title('Physics-Based Features: Bridging Flow Physics and Machine Learning',
                fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    """Generate all Chapter 5 flow field figures."""
    print("=" * 60)
    print("Generating Chapter 5 Flow Field Figures")
    print("=" * 60)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load available cases
    print("\nLoading simulation cases...")
    cases = get_available_cases()
    print(f"  Found {len(cases)} cases")

    if not cases:
        print("\nWarning: No simulation cases found!")
        print(f"  Expected location: {CASES_DIR}")
        # Still generate conceptual figure
        create_feature_physics_connection_figure(
            os.path.join(OUTPUT_DIR, "feature_physics_connection.png")
        )
        return

    # Print case summary
    fine_cases = [c for c in cases if c['mesh_resolution'] == 'fine']
    print(f"  Fine mesh cases: {len(fine_cases)}")

    # Generate figures
    print("\n1. Generating velocity profiles figure...")
    create_velocity_profiles_figure(
        cases, os.path.join(OUTPUT_DIR, "velocity_profiles.png")
    )

    print("\n2. Generating flow field contours figure...")
    create_flow_field_contours_figure(
        cases, os.path.join(OUTPUT_DIR, "flow_field_contours.png")
    )

    print("\n3. Generating tau_w distribution figure...")
    create_tau_w_distribution_figure(
        cases, os.path.join(OUTPUT_DIR, "tau_w_distribution.png")
    )

    print("\n4. Generating pressure gradient map figure...")
    create_pressure_gradient_map_figure(
        cases, os.path.join(OUTPUT_DIR, "pressure_gradient_map.png")
    )

    print("\n5. Generating boundary layer comparison figure...")
    create_boundary_layer_comparison_figure(
        cases, os.path.join(OUTPUT_DIR, "boundary_layer_comparison.png")
    )

    print("\n6. Generating feature physics connection figure...")
    create_feature_physics_connection_figure(
        os.path.join(OUTPUT_DIR, "feature_physics_connection.png")
    )

    print("\n" + "=" * 60)
    print("Figure generation complete!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
