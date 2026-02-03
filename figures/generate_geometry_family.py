#!/usr/bin/env python3
"""
Generate figure showing the full family of training geometries.

Shows all diffuser, channel, and nozzle variations stacked to illustrate
how parameters are varied to generate diverse training data.

Author: ML Wall Function Project
Date: 2026-01-31
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


def load_geometry_config():
    """Load geometry configuration from JSON file."""
    config_path = os.path.join(PROJECT_ROOT, "TRAINING_DATA", "data", "full_geometry_config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return None


def draw_geometry(ax, geom, y_offset, color, alpha=0.6, label=None):
    """Draw a single geometry profile."""
    H_in = geom['H_in']
    H_out = geom['H_out']
    L_in = geom['L_in']
    L_trans = geom['L_trans']
    L_out = geom['L_out']

    # Normalize lengths for visualization
    scale = 1.0 / (L_in + L_trans + L_out) * 10  # Scale to fit

    L_in_s = L_in * scale
    L_trans_s = L_trans * scale
    L_out_s = L_out * scale

    # X coordinates
    x0 = 0
    x1 = L_in_s
    x2 = L_in_s + L_trans_s
    x3 = L_in_s + L_trans_s + L_out_s

    # Top wall (flat at y=1)
    y_top = y_offset + 1

    # Bottom wall
    y_bot_in = y_offset + (1 - H_in)
    y_bot_out = y_offset + (1 - H_out)

    # Draw geometry
    # Top wall
    ax.plot([x0, x3], [y_top, y_top], color=color, linewidth=1.5, alpha=alpha)
    # Bottom wall
    ax.plot([x0, x1], [y_bot_in, y_bot_in], color=color, linewidth=1.5, alpha=alpha)
    ax.plot([x1, x2], [y_bot_in, y_bot_out], color=color, linewidth=1.5, alpha=alpha)
    ax.plot([x2, x3], [y_bot_out, y_bot_out], color=color, linewidth=1.5, alpha=alpha)
    # Inlet/outlet
    ax.plot([x0, x0], [y_bot_in, y_top], color=color, linewidth=1.5, alpha=alpha)
    ax.plot([x3, x3], [y_bot_out, y_top], color=color, linewidth=1.5, alpha=alpha)

    # Fill
    xs = [x0, x1, x2, x3, x3, x0]
    ys = [y_bot_in, y_bot_in, y_bot_out, y_bot_out, y_top, y_top]
    ax.fill(xs, ys, color=color, alpha=0.15)

    if label:
        ax.text(x3 + 0.3, y_offset + 0.5, label, fontsize=8, va='center')


def create_geometry_family_figure():
    """Create figure showing all geometry families."""
    print("Generating geometry family figure...")

    config = load_geometry_config()
    if config is None:
        print("Could not load geometry config, creating synthetic data")
        # Create synthetic geometry data
        geometries = []
        # Diffusers
        for i, ratio in enumerate([1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]):
            for angle in [8, 15]:
                H_out = ratio
                L_trans = (H_out - 1.0) / np.tan(np.radians(angle))
                geometries.append({
                    'case_id': f'diff_{i:03d}',
                    'expansion_ratio': ratio,
                    'transition_angle': angle,
                    'H_in': 1.0,
                    'H_out': H_out,
                    'L_in': 10.0,
                    'L_trans': L_trans,
                    'L_out': 20.0
                })
        # Channels
        geometries.append({'case_id': 'chan_000', 'expansion_ratio': 1.0, 'transition_angle': 0,
                          'H_in': 1.0, 'H_out': 1.0, 'L_in': 10.0, 'L_trans': 20.0, 'L_out': 20.0})
        # Nozzles
        for ratio in [0.5, 0.6, 0.7, 0.8]:
            for angle in [-8, -15]:
                H_out = ratio
                L_trans = abs(H_out - 1.0) / np.tan(np.radians(abs(angle)))
                geometries.append({
                    'case_id': f'nozz_{len(geometries):03d}',
                    'expansion_ratio': ratio,
                    'transition_angle': angle,
                    'H_in': 1.0,
                    'H_out': H_out,
                    'L_in': 10.0,
                    'L_trans': L_trans,
                    'L_out': 20.0
                })
    else:
        geometries = config['geometries']

    # Separate by type
    diffusers = [g for g in geometries if g['case_id'].startswith('diff')]
    channels = [g for g in geometries if g['case_id'].startswith('chan')]
    nozzles = [g for g in geometries if g['case_id'].startswith('nozz')]

    print(f"  Diffusers: {len(diffusers)}, Channels: {len(channels)}, Nozzles: {len(nozzles)}")

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(14, 6))

    # Color maps
    diff_colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(diffusers)))
    nozz_colors = plt.cm.Greens(np.linspace(0.3, 0.9, len(nozzles)))
    chan_color = 'orange'

    # Plot diffusers
    ax = axes[0]
    for i, geom in enumerate(diffusers):
        y_offset = i * 0.15
        draw_geometry(ax, geom, y_offset, diff_colors[i], alpha=0.8)

    ax.set_xlim(-0.5, 12)
    ax.set_ylim(-0.5, len(diffusers) * 0.15 + 6)
    ax.set_aspect('equal')
    ax.set_xlabel('Normalized streamwise position')
    ax.set_ylabel('Stacked geometries')
    ax.set_title(f'Diffusers ({len(diffusers)} variations)\nExpansion ratio: 1.5--5.5, Angle: 8°--15°', fontsize=11)
    ax.set_yticks([])

    # Add annotation
    ax.annotate('', xy=(5, 0.5), xytext=(5, len(diffusers)*0.15 + 4),
               arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax.text(5.3, len(diffusers)*0.15/2 + 2, 'Varying\nexpansion\nratio', fontsize=9, color='red', va='center')

    # Plot channels
    ax = axes[1]
    for i, geom in enumerate(channels):
        y_offset = i * 1.5
        draw_geometry(ax, geom, y_offset, chan_color, alpha=0.8)

    ax.set_xlim(-0.5, 12)
    ax.set_ylim(-0.5, max(2, len(channels) * 1.5 + 1))
    ax.set_aspect('equal')
    ax.set_xlabel('Normalized streamwise position')
    ax.set_title(f'Channels ({len(channels)} variations)\nZero pressure gradient baseline', fontsize=11)
    ax.set_yticks([])

    # Plot nozzles
    ax = axes[2]
    for i, geom in enumerate(nozzles):
        y_offset = i * 0.2
        draw_geometry(ax, geom, y_offset, nozz_colors[i], alpha=0.8)

    ax.set_xlim(-0.5, 12)
    ax.set_ylim(-0.5, len(nozzles) * 0.2 + 2)
    ax.set_aspect('equal')
    ax.set_xlabel('Normalized streamwise position')
    ax.set_title(f'Nozzles ({len(nozzles)} variations)\nContraction ratio: 0.5--0.7, Angle: 8°--15°', fontsize=11)
    ax.set_yticks([])

    # Add annotation
    ax.annotate('', xy=(5, 0.3), xytext=(5, len(nozzles)*0.2 + 1),
               arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax.text(5.3, len(nozzles)*0.2/2 + 0.5, 'Varying\ncontraction\nratio', fontsize=9, color='red', va='center')

    # Add overall title
    total = len(diffusers) + len(channels) + len(nozzles)
    fig.suptitle(f'Training Geometry Family: {total} Unique Configurations\n'
                 f'(×2 Reynolds numbers = {total*2} total cases, ×2 mesh resolutions = {total*4} simulations)',
                 fontsize=12, fontweight='bold', y=1.02)

    plt.tight_layout()

    # Save
    output_path = os.path.join(SCRIPT_DIR, 'fig_geometry_family.pdf')
    plt.savefig(output_path, bbox_inches='tight')
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    print(f"Saved: {output_path.replace('.pdf', '.png')}")

    plt.close()
    return output_path


if __name__ == "__main__":
    create_geometry_family_figure()
