#!/usr/bin/env python3
"""
Generate missing Chapter 6 figures for the thesis.
Chapter 6: Physics-Based Feature Variables as Hidden Layer Neurons

Missing figures:
1. architecture_invariance.png - Cross-architecture feature discovery
2. hybrid_architecture.png - Hybrid network architecture diagram  
3. hybrid_comparison.png - Pure learned vs hybrid network comparison
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.gridspec import GridSpec
import os

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

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def generate_architecture_invariance():
    """
    Figure: Architecture invariance - features discovered across different network sizes.
    """
    print("Generating architecture_invariance.png...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Data from Chapter 6 experiments
    architectures = ['L1_8\n(8 neurons)', 'L1_16\n(16 neurons)', 'L1_32\n(32 neurons)']
    
    # Features and their discovery across architectures
    # 1 = discovered (moderate-strong correlation), 0 = not discovered
    features = [
        'wall_distance_y_plus',
        'log_thermal_y', 
        'log_y_plus_over_y',
        'pressure_gradient_x',
        'u2_y2_over_viscosity',
        'velocity_x_friction_normalized',
        'local_re',
        'y_T_plus',
    ]
    
    feature_labels = [
        r'$y^+$ (wall distance)',
        r'$\log(y_T^+)$ (thermal)',
        r'$\log(y^+)/y$ (log-law)',
        r'$\partial p/\partial x$ (pressure)',
        r'$u^2y^2/\nu$ (viscous)',
        r'$u^+$ (velocity)',
        r'$Re_y$ (local Re)',
        r'$y_T^+$ (thermal dist)',
    ]
    
    # Discovery matrix (1 = discovered, 0.5 = weak, 0 = not found)
    # Based on thesis Chapter 6 results
    discovery = np.array([
        [1, 1, 1],      # y+
        [1, 1, 1],      # log_thermal_y
        [1, 1, 1],      # log_y_plus_over_y
        [0.5, 1, 1],    # pressure_gradient_x
        [0.5, 0.5, 1],  # u2_y2_over_viscosity
        [1, 1, 1],      # velocity_x
        [0, 0.5, 1],    # local_re
        [0.5, 1, 1],    # y_T_plus
    ])
    
    # Panel A: Heatmap of discovery
    ax = axes[0]
    im = ax.imshow(discovery, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    ax.set_xticks(np.arange(len(architectures)))
    ax.set_yticks(np.arange(len(features)))
    ax.set_xticklabels(architectures)
    ax.set_yticklabels(feature_labels)
    
    # Add text annotations
    for i in range(len(features)):
        for j in range(len(architectures)):
            val = discovery[i, j]
            text = '✓' if val == 1 else ('~' if val == 0.5 else '✗')
            color = 'white' if val < 0.5 else 'black'
            ax.text(j, i, text, ha='center', va='center', fontsize=14, color=color)
    
    ax.set_title('(a) Feature Discovery Across Architectures')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Correlation Strength')
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(['Not found', 'Weak', 'Strong'])
    
    # Panel B: Invariant features bar chart
    ax = axes[1]
    
    # Count discoveries per feature
    discovery_count = np.sum(discovery >= 0.5, axis=1)
    colors = ['green' if c == 3 else 'orange' if c == 2 else 'red' for c in discovery_count]
    
    y_pos = np.arange(len(features))
    bars = ax.barh(y_pos, discovery_count, color=colors, alpha=0.8, edgecolor='black')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_labels)
    ax.set_xlabel('Number of Architectures (out of 3)')
    ax.set_xlim([0, 3.5])
    ax.axvline(x=3, color='green', linestyle='--', alpha=0.5, linewidth=2)
    ax.set_title('(b) Architecture Invariance')
    
    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor='green', edgecolor='black', label='Invariant (3/3)'),
        mpatches.Patch(facecolor='orange', edgecolor='black', label='Partially (2/3)'),
        mpatches.Patch(facecolor='red', edgecolor='black', label='Architecture-specific (1/3)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    ax.grid(axis='x', alpha=0.3)
    
    # Annotation
    ax.annotate('Architecture-\ninvariant\nfeatures', xy=(3.2, 1), fontsize=10, 
               style='italic', color='green', ha='left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'architecture_invariance.png'))
    plt.close()
    print("  ✓ Saved architecture_invariance.png")


def generate_hybrid_architecture():
    """
    Figure: Hybrid network architecture diagram.
    Shows physics neurons (green) vs learned neurons (yellow).
    """
    print("Generating hybrid_architecture.png...")
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Colors
    input_color = '#a8d8ea'      # Light blue
    physics_color = '#98d8aa'    # Green
    learned_color = '#fff3b0'    # Yellow
    output_color = '#f1a7a7'     # Light red
    
    # Input layer (6 basic normalized variables)
    input_labels = [r'$y^+$', r'$u^+$', r'$v^+$', r'$\partial p/\partial x$',
                   r'$\partial p/\partial y$', r'$y_T^+$']
    input_x = 1.5
    input_ys = np.linspace(2, 8, 6)

    for i, (y, label) in enumerate(zip(input_ys, input_labels)):
        circle = Circle((input_x, y), 0.35, facecolor=input_color, edgecolor='black', linewidth=1.5)
        ax.add_patch(circle)
        ax.text(input_x, y, label, ha='center', va='center', fontsize=9)

    ax.text(input_x, 9, 'Input Layer\n(6 basic variables)', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Hidden layer - Physics neurons (top) and Learned neurons (bottom)
    hidden_x = 6
    
    # Physics neurons (13 neurons as per thesis)
    physics_labels = [r'$\partial p/\partial x$', r'$u^2y^2/\nu$', r'$\log(y_T^+)$', 
                     r'$\log(y^+)/y$', '...', r'$\partial p/\partial x$']
    physics_ys = np.linspace(6, 8.5, 6)
    
    for y, label in zip(physics_ys, physics_labels):
        rect = FancyBboxPatch((hidden_x - 0.5, y - 0.25), 1.0, 0.5, 
                              boxstyle="round,pad=0.05", facecolor=physics_color, 
                              edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(hidden_x, y, label, ha='center', va='center', fontsize=8)
    
    # Learned neurons (19 neurons)
    learned_ys = np.linspace(1.5, 5, 8)
    for y in learned_ys:
        circle = Circle((hidden_x, y), 0.35, facecolor=learned_color, edgecolor='black', linewidth=1.5)
        ax.add_patch(circle)
        ax.text(hidden_x, y, '?', ha='center', va='center', fontsize=10)
    
    ax.text(hidden_x, 9, 'Hidden Layer\n(32 neurons)', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Annotations for hidden layer sections
    ax.annotate('Physics Neurons\n(13 explicit formulas)', xy=(hidden_x + 1.2, 7.25), fontsize=9,
               color='green', ha='left', style='italic')
    ax.annotate('Learned Neurons\n(19 trained weights)', xy=(hidden_x + 1.2, 3.25), fontsize=9,
               color='orange', ha='left', style='italic')
    
    # Output layer
    output_x = 10.5
    output_labels = [r'$\tau_w$', r'$q_w$']
    output_ys = [4, 6]
    
    for y, label in zip(output_ys, output_labels):
        circle = Circle((output_x, y), 0.4, facecolor=output_color, edgecolor='black', linewidth=1.5)
        ax.add_patch(circle)
        ax.text(output_x, y, label, ha='center', va='center', fontsize=10)
    
    ax.text(output_x, 9, 'Output Layer\n(2 targets)', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Draw connections (simplified)
    # Input to hidden
    for iy in input_ys:
        for hy in list(physics_ys) + list(learned_ys):
            ax.plot([input_x + 0.35, hidden_x - 0.5], [iy, hy], 'gray', alpha=0.15, linewidth=0.5)
    
    # Hidden to output
    for hy in list(physics_ys) + list(learned_ys):
        for oy in output_ys:
            ax.plot([hidden_x + 0.5, output_x - 0.4], [hy, oy], 'gray', alpha=0.15, linewidth=0.5)
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=input_color, edgecolor='black', label='Input (basic variables)'),
        mpatches.Patch(facecolor=physics_color, edgecolor='black', label='Physics neurons (explicit)'),
        mpatches.Patch(facecolor=learned_color, edgecolor='black', label='Learned neurons (trained)'),
        mpatches.Patch(facecolor=output_color, edgecolor='black', label='Output (wall quantities)'),
    ]
    ax.legend(handles=legend_elements, loc='lower center', ncol=4, framealpha=0.95, fontsize=9)
    
    ax.set_title('Hybrid Neural Network Architecture with Neuron Replacement', fontsize=14, fontweight='bold', pad=20)
    
    plt.savefig(os.path.join(OUTPUT_DIR, 'hybrid_architecture.png'))
    plt.close()
    print("  ✓ Saved hybrid_architecture.png")


def generate_hybrid_comparison():
    """
    Figure: Comparison of pure learned vs hybrid network.
    """
    print("Generating hybrid_comparison.png...")
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    # Data from Chapter 6 hybrid network results
    models = ['Pure Learned', 'Hybrid']
    
    # Panel A: Accuracy comparison
    ax = axes[0]
    tau_r2 = [73.5, 72.9]  # From Table 6 in thesis
    qw_r2 = [0.4, 0.4]
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, tau_r2, width, label=r'$\tau_w$ $R^2$ (%)', color='steelblue')
    bars2 = ax.bar(x + width/2, qw_r2, width, label=r'$q_w$ $R^2$ (%)', color='coral')
    
    ax.set_ylabel(r'$R^2$ Score (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim([0, 100])
    ax.legend(loc='upper right')
    ax.set_title('(a) Prediction Accuracy')
    ax.grid(axis='y', alpha=0.3)
    
    # Add values on bars
    for bar, val in zip(bars1, tau_r2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
               f'{val:.1f}%', ha='center', va='bottom', fontsize=10)
    
    # Annotation
    ax.annotate('Only 0.8% drop!', xy=(0.5, 73), fontsize=10, color='green', 
               ha='center', style='italic')
    
    # Panel B: Parameter count
    ax = axes[1]
    params = [290, 225]
    colors = ['orange', 'green']
    
    bars = ax.bar(models, params, color=colors, alpha=0.8, edgecolor='black')
    ax.set_ylabel('Number of Parameters')
    ax.set_title('(b) Model Complexity')
    ax.grid(axis='y', alpha=0.3)
    
    # Add reduction annotation
    reduction = (290 - 225) / 290 * 100
    ax.annotate(f'-{reduction:.0f}%', xy=(1, 225), xytext=(1.3, 260),
               arrowprops=dict(arrowstyle='->', color='green'),
               fontsize=12, color='green', fontweight='bold')
    
    # Panel C: Network composition
    ax = axes[2]
    
    # Pure learned
    sizes_pure = [32]
    labels_pure = ['Learned (32)']
    colors_pure = ['#fff3b0']
    
    # Hybrid
    sizes_hybrid = [13, 19]
    labels_hybrid = ['Physics (13)', 'Learned (19)']
    colors_hybrid = ['#98d8aa', '#fff3b0']
    
    # Create side-by-side pie charts
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # Pure learned pie (left)
    ax_pie1 = fig.add_axes([0.68, 0.35, 0.12, 0.4])
    ax_pie1.pie(sizes_pure, labels=None, colors=colors_pure, autopct='', startangle=90)
    ax_pie1.set_title('Pure Learned', fontsize=10)
    
    # Hybrid pie (right)
    ax_pie2 = fig.add_axes([0.82, 0.35, 0.12, 0.4])
    wedges, texts = ax_pie2.pie(sizes_hybrid, labels=None, colors=colors_hybrid, startangle=90)
    ax_pie2.set_title('Hybrid', fontsize=10)
    
    # Legend for pies
    ax.text(5, 5.5, '(c) Network Composition', ha='center', fontsize=12, fontweight='bold')
    
    legend_elements = [
        mpatches.Patch(facecolor='#98d8aa', edgecolor='black', label='Physics neurons (41%)'),
        mpatches.Patch(facecolor='#fff3b0', edgecolor='black', label='Learned neurons (59%)'),
    ]
    ax.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=9)
    
    ax.text(5, 1.5, '41% of neurons replaced with\nexplicit physics formulas', 
           ha='center', fontsize=11, style='italic', color='green')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'hybrid_comparison.png'))
    plt.close()
    print("  ✓ Saved hybrid_comparison.png")


def main():
    """Generate all Chapter 6 figures."""
    print("=" * 60)
    print("Generating Chapter 6 Figures")
    print("=" * 60)
    
    generate_architecture_invariance()
    generate_hybrid_architecture()
    generate_hybrid_comparison()
    
    print("\n" + "=" * 60)
    print("All Chapter 6 figures generated successfully!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
