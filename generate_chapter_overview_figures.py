#!/usr/bin/env python3
"""
Generate systematic overview figures for thesis chapters 4-8.
Each figure provides a visual summary of the chapter's key approach.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, Polygon
import numpy as np
import os

# Set up matplotlib for publication-quality figures
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'text.usetex': False,
    'figure.dpi': 150,
})

# Color scheme
COLORS = {
    'input': '#E8D5B7',      # Beige for inputs
    'hidden': '#B8D4E8',     # Light blue for hidden layers
    'output': '#F5B7B1',     # Light red/pink for outputs
    'physics': '#ABEBC6',    # Light green for physics
    'box_blue': '#3498DB',   # Blue for boxes
    'box_orange': '#E67E22', # Orange for boxes
    'box_green': '#27AE60',  # Green for boxes
    'box_purple': '#9B59B6', # Purple for boxes
    'arrow': '#2C3E50',      # Dark for arrows
    'text': '#2C3E50',       # Dark for text
}

def draw_stencil(ax, x_center, y_center, scale=1.0):
    """Draw a simplified stencil mesh representation."""
    # Draw wall (curved line)
    wall_x = np.linspace(x_center - 0.8*scale, x_center + 0.8*scale, 50)
    wall_y = y_center - 0.6*scale + 0.05*scale * np.sin(4*np.pi*(wall_x - x_center)/(1.6*scale))
    ax.plot(wall_x, wall_y, 'k-', linewidth=2)

    # Draw wall hatching
    for i in range(0, len(wall_x)-1, 3):
        ax.plot([wall_x[i], wall_x[i]-0.05*scale], [wall_y[i], wall_y[i]-0.1*scale], 'k-', linewidth=0.5)

    # Draw cell grid (3x5 stencil)
    cell_positions = []
    for j in range(3):  # Wall-normal direction
        for i in range(5):  # Streamwise direction
            cx = x_center + (i - 2) * 0.25 * scale
            cy = y_center - 0.3*scale + j * 0.35 * scale
            cell_positions.append((cx, cy))
            # Draw cell center
            if j == 0:  # Wall-adjacent cells
                ax.plot(cx, cy, 'ko', markersize=6)
            else:
                ax.plot(cx, cy, 'o', color='gray', markersize=5, markerfacecolor='white')

    # Draw grid lines (dashed)
    for j in range(4):
        y = y_center - 0.5*scale + j * 0.35 * scale
        ax.plot([x_center - 0.7*scale, x_center + 0.7*scale], [y, y], 'k--', linewidth=0.5, alpha=0.5)
    for i in range(6):
        x = x_center - 0.625*scale + i * 0.25 * scale
        ax.plot([x, x], [y_center - 0.5*scale, y_center + 0.55*scale], 'k--', linewidth=0.5, alpha=0.5)

    return cell_positions

def draw_neural_network(ax, x_center, y_center, layers, colors, scale=1.0, label=None):
    """Draw a neural network with specified layer sizes."""
    layer_spacing = 0.6 * scale
    x_positions = [x_center + (i - len(layers)/2 + 0.5) * layer_spacing for i in range(len(layers))]

    all_neurons = []
    for layer_idx, (n_neurons, x_pos) in enumerate(zip(layers, x_positions)):
        layer_neurons = []
        for i in range(n_neurons):
            y = y_center + (i - n_neurons/2 + 0.5) * 0.25 * scale
            circle = Circle((x_pos, y), 0.08*scale, facecolor=colors[layer_idx],
                           edgecolor='black', linewidth=1)
            ax.add_patch(circle)
            layer_neurons.append((x_pos, y))
        all_neurons.append(layer_neurons)

    # Draw connections
    for layer_idx in range(len(all_neurons) - 1):
        for neuron1 in all_neurons[layer_idx]:
            for neuron2 in all_neurons[layer_idx + 1]:
                ax.plot([neuron1[0], neuron2[0]], [neuron1[1], neuron2[1]],
                       'k-', linewidth=0.3, alpha=0.3)

    if label:
        ax.text(x_center, y_center + max(layers)/2 * 0.25 * scale + 0.2*scale,
               label, ha='center', va='bottom', fontsize=10, fontweight='bold')

    return all_neurons

def draw_box(ax, x, y, width, height, label, color, text_color='black'):
    """Draw a labeled box."""
    box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                         boxstyle="round,pad=0.02,rounding_size=0.1",
                         facecolor='white', edgecolor=color, linewidth=2)
    ax.add_patch(box)
    ax.text(x, y, label, ha='center', va='center', fontsize=10, color=text_color, fontweight='bold')

def draw_arrow(ax, start, end, color='black', style='->', linewidth=1.5):
    """Draw an arrow between two points."""
    ax.annotate('', xy=end, xytext=start,
               arrowprops=dict(arrowstyle=style, color=color, lw=linewidth))

# ============================================================================
# Chapter 4: Data-Driven Baseline
# ============================================================================
def generate_chapter4_overview():
    """Chapter 4: Data-Driven Velocity and Thermal Wall Functions (Baseline MLP)"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    ax.set_xlim(-1, 11)
    ax.set_ylim(-1, 4)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    ax.text(5, 3.7, 'Chapter 4: Data-Driven Wall Function Baseline',
           ha='center', va='top', fontsize=14, fontweight='bold')

    # Left: Input stencil
    draw_stencil(ax, 1.5, 1.8, scale=1.2)
    ax.text(1.5, 3.2, r'$\mathbf{Inputs}$', ha='center', fontsize=12, fontweight='bold')
    ax.text(1.5, 2.9, r'$(x, y, p, U, V, T)$', ha='center', fontsize=10, style='italic')
    ax.text(1.5, 0.4, '3×5 Stencil\n(90 features)', ha='center', fontsize=9)

    # Arrow to preprocessing
    draw_arrow(ax, (2.5, 1.8), (3.3, 1.8), color=COLORS['arrow'])

    # Preprocessing box
    draw_box(ax, 4, 1.8, 1.2, 0.8, 'Normalize\n(z-score)', COLORS['box_blue'])

    # Arrow to network
    draw_arrow(ax, (4.7, 1.8), (5.5, 1.8), color=COLORS['arrow'])

    # Neural network
    layers = [4, 6, 6, 6, 2]
    colors = [COLORS['input'], COLORS['hidden'], COLORS['hidden'], COLORS['hidden'], COLORS['output']]
    draw_neural_network(ax, 7, 1.8, layers, colors, scale=1.3)
    ax.text(7, 3.0, 'Multilayer Perceptron', ha='center', fontsize=11, fontweight='bold', color=COLORS['box_blue'])
    ax.text(7, 0.3, '3 hidden layers\n64 neurons each', ha='center', fontsize=9)

    # Arrow to outputs
    draw_arrow(ax, (8.5, 1.8), (9.3, 1.8), color=COLORS['arrow'])

    # Outputs
    ax.text(10, 2.2, r'$\mathbf{Outputs}$', ha='center', fontsize=12, fontweight='bold')
    ax.text(10, 1.8, r'$\tau_w$ (wall shear)', ha='center', fontsize=11)
    ax.text(10, 1.4, r'$q_w$ (wall heat flux)', ha='center', fontsize=11)

    # Bottom annotation
    ax.text(5, -0.5, 'Supervised learning: minimize MSE between predictions and fine-mesh ground truth',
           ha='center', fontsize=10, style='italic', color='gray')

    plt.tight_layout()
    plt.savefig('Images/chapter4/chapter4_overview.png', dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    plt.savefig('Images/chapter4/chapter4_overview.pdf', dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: chapter4_overview.png/pdf")

# ============================================================================
# Chapter 5: Physics-Based Features as Inputs
# ============================================================================
def generate_chapter5_overview():
    """Chapter 5: Physics-Based Feature Variables as Network Inputs"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 5))
    ax.set_xlim(-1, 13)
    ax.set_ylim(-1, 4)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    ax.text(6, 3.7, 'Chapter 5: Physics-Based Features as Network Inputs',
           ha='center', va='top', fontsize=14, fontweight='bold')

    # Left: Input stencil
    draw_stencil(ax, 1, 1.8, scale=1.0)
    ax.text(1, 3.0, r'$\mathbf{Raw\ Data}$', ha='center', fontsize=11, fontweight='bold')
    ax.text(1, 2.7, r'$(x, y, p, U, V, T)$', ha='center', fontsize=9, style='italic')

    # Arrow to feature engineering
    draw_arrow(ax, (2.0, 1.8), (2.8, 1.8), color=COLORS['arrow'])

    # Feature engineering box (larger, with physics features listed)
    box_x, box_y = 4, 1.8
    box = FancyBboxPatch((box_x - 1.3, box_y - 1.2), 2.6, 2.4,
                         boxstyle="round,pad=0.02,rounding_size=0.1",
                         facecolor=COLORS['physics'], edgecolor=COLORS['box_green'], linewidth=2)
    ax.add_patch(box)
    ax.text(box_x, box_y + 0.9, 'Physics Feature\nEngineering', ha='center', fontsize=10, fontweight='bold')

    # List key physics features
    features = [r'$y^+$, $u^+$', r'$\partial p/\partial x$', r'$Re_y$', r'$\sqrt{S_{ij}S_{ij}}$', r'$T^+$, $y_T^+$']
    for i, feat in enumerate(features):
        ax.text(box_x, box_y + 0.4 - i*0.35, feat, ha='center', fontsize=9)

    ax.text(box_x, box_y - 1.5, '58 → 11 Core', ha='center', fontsize=9, style='italic')

    # Arrow to network
    draw_arrow(ax, (5.5, 1.8), (6.3, 1.8), color=COLORS['arrow'])

    # Neural network (smaller input layer)
    layers = [3, 5, 5, 5, 2]
    colors = [COLORS['physics'], COLORS['hidden'], COLORS['hidden'], COLORS['hidden'], COLORS['output']]
    draw_neural_network(ax, 8, 1.8, layers, colors, scale=1.2)
    ax.text(8, 2.9, 'MLP', ha='center', fontsize=11, fontweight='bold', color=COLORS['box_blue'])
    ax.text(8, 0.4, '11 physics inputs\n(vs 90 primitive)', ha='center', fontsize=9)

    # Arrow to outputs
    draw_arrow(ax, (9.5, 1.8), (10.3, 1.8), color=COLORS['arrow'])

    # Outputs
    ax.text(11.2, 2.2, r'$\mathbf{Outputs}$', ha='center', fontsize=12, fontweight='bold')
    ax.text(11.2, 1.8, r'$\tau_w$', ha='center', fontsize=11)
    ax.text(11.2, 1.4, r'$q_w$', ha='center', fontsize=11)

    # Key insight annotation
    ax.text(6, -0.5, 'Key insight: 11 physics features achieve same accuracy as 90 primitive features (8× reduction)',
           ha='center', fontsize=10, style='italic', color='gray')

    # Add improvement annotation
    ax.annotate('', xy=(11.2, 0.9), xytext=(11.2, 0.5),
               arrowprops=dict(arrowstyle='->', color=COLORS['box_green'], lw=2))
    ax.text(11.8, 0.7, r'$R^2 = 0.989$', fontsize=9, color=COLORS['box_green'], fontweight='bold')

    plt.tight_layout()
    plt.savefig('Images/chapter5/chapter5_overview.png', dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    plt.savefig('Images/chapter5/chapter5_overview.pdf', dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: chapter5_overview.png/pdf")

# ============================================================================
# Chapter 6: Physics Features as Hidden Layer Neurons
# ============================================================================
def generate_chapter6_overview():
    """Chapter 6: Physics-Based Feature Variables as Hidden Layer Neurons"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    ax.set_xlim(-1, 13)
    ax.set_ylim(-1.5, 5)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    ax.text(6, 4.7, 'Chapter 6: Do Networks Discover Physics Internally?',
           ha='center', va='top', fontsize=14, fontweight='bold')

    # Left: Basic inputs (6 features)
    ax.text(0.5, 3.8, r'$\mathbf{Basic\ Inputs}$', ha='center', fontsize=11, fontweight='bold')
    basic_inputs = [r'$y^+$', r'$u^+$', r'$v^+$', r'$\partial p/\partial x$', r'$\partial p/\partial y$', r'$y_T^+$']
    for i, inp in enumerate(basic_inputs):
        y = 3.2 - i * 0.4
        circle = Circle((0.5, y), 0.15, facecolor=COLORS['input'], edgecolor='black', linewidth=1)
        ax.add_patch(circle)
        ax.text(1.1, y, inp, ha='left', va='center', fontsize=9)

    # Arrow to network
    draw_arrow(ax, (1.8, 2.0), (2.8, 2.0), color=COLORS['arrow'])

    # Neural network with highlighted neurons
    ax.text(5, 4.0, 'Single Hidden Layer Network', ha='center', fontsize=11, fontweight='bold', color=COLORS['box_blue'])

    # Input layer
    for i in range(6):
        y = 3.2 - i * 0.4
        circle = Circle((3.5, y), 0.12, facecolor=COLORS['input'], edgecolor='black', linewidth=1)
        ax.add_patch(circle)

    # Hidden layer with some neurons highlighted
    hidden_colors = [COLORS['physics'], COLORS['hidden'], COLORS['physics'], COLORS['hidden'],
                    COLORS['physics'], COLORS['hidden'], COLORS['hidden'], COLORS['physics']]
    for i in range(8):
        y = 3.4 - i * 0.35
        circle = Circle((5, y), 0.15, facecolor=hidden_colors[i], edgecolor='black', linewidth=1.5 if hidden_colors[i] == COLORS['physics'] else 1)
        ax.add_patch(circle)

    # Output layer
    for i in range(2):
        y = 2.3 - i * 0.6
        circle = Circle((6.5, y), 0.12, facecolor=COLORS['output'], edgecolor='black', linewidth=1)
        ax.add_patch(circle)

    # Draw connections (simplified)
    for i in range(6):
        for j in range(8):
            ax.plot([3.5, 5], [3.2 - i*0.4, 3.4 - j*0.35], 'k-', linewidth=0.2, alpha=0.2)
    for i in range(8):
        for j in range(2):
            ax.plot([5, 6.5], [3.4 - i*0.35, 2.3 - j*0.6], 'k-', linewidth=0.2, alpha=0.2)

    # Correlation analysis arrow
    draw_arrow(ax, (5.3, 3.4), (7.5, 3.4), color=COLORS['box_green'], style='->')
    ax.text(6.4, 3.65, 'Correlation\nAnalysis', ha='center', fontsize=9, color=COLORS['box_green'])

    # Physics features discovered
    box = FancyBboxPatch((7.3, 1.5), 3.0, 2.2,
                         boxstyle="round,pad=0.02,rounding_size=0.1",
                         facecolor='white', edgecolor=COLORS['box_green'], linewidth=2)
    ax.add_patch(box)
    ax.text(8.8, 3.5, 'Discovered Physics', ha='center', fontsize=10, fontweight='bold', color=COLORS['box_green'])

    discovered = [
        (r'$y^+$', '|r| = 0.90'),
        (r'$u^2 y^2 / \nu$', '|r| = 0.85'),
        (r'$\log(y_T^+)$', '|r| = 0.87'),
        (r'$u^+_{friction}$', '|r| = 0.80'),
    ]
    for i, (feat, corr) in enumerate(discovered):
        y = 3.0 - i * 0.4
        circle = Circle((7.7, y), 0.12, facecolor=COLORS['physics'], edgecolor='black', linewidth=1)
        ax.add_patch(circle)
        ax.text(8.0, y, feat, ha='left', va='center', fontsize=9)
        ax.text(9.8, y, corr, ha='right', va='center', fontsize=8, color='gray')

    # Outputs
    ax.text(6.5, 1.3, r'$\tau_w$', ha='center', fontsize=10)
    ax.text(6.5, 0.7, r'$q_w$', ha='center', fontsize=10)

    # Key insight
    ax.text(6, -1.0, 'Key insight: Neurons learn physics features that are invariant across architectures',
           ha='center', fontsize=10, style='italic', color='gray')

    # Architecture invariance note
    ax.text(8.8, 1.3, 'Architecture\nInvariant!', ha='center', fontsize=9, fontweight='bold',
           color=COLORS['box_green'], style='italic')

    plt.tight_layout()
    plt.savefig('Images/chapter6/chapter6_overview.png', dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    plt.savefig('Images/chapter6/chapter6_overview.pdf', dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: chapter6_overview.png/pdf")

# ============================================================================
# Chapter 7: Physics-Informed Neural Networks
# ============================================================================
def generate_chapter7_overview():
    """Chapter 7: Physics-Constrained Learning (PINN)"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    ax.set_xlim(-1, 13)
    ax.set_ylim(-1.5, 5)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    ax.text(6, 4.7, 'Chapter 7: Physics-Constrained Learning (PINN)',
           ha='center', va='top', fontsize=14, fontweight='bold')

    # Left: Input stencil
    draw_stencil(ax, 1, 2.5, scale=1.0)
    ax.text(1, 4.0, r'$\mathbf{Stencil\ Data}$', ha='center', fontsize=11, fontweight='bold')

    # Arrow to network
    draw_arrow(ax, (2.0, 2.5), (3.0, 2.5), color=COLORS['arrow'])

    # Neural network
    layers = [3, 5, 5, 2]
    colors = [COLORS['input'], COLORS['hidden'], COLORS['hidden'], COLORS['output']]
    draw_neural_network(ax, 4.5, 2.5, layers, colors, scale=1.2)
    ax.text(4.5, 4.0, 'Neural Network', ha='center', fontsize=11, fontweight='bold', color=COLORS['box_blue'])

    # Predictions arrow
    draw_arrow(ax, (5.8, 2.5), (6.8, 2.5), color=COLORS['arrow'])

    # Predictions box
    pred_box = FancyBboxPatch((6.8, 1.8), 1.8, 1.4,
                              boxstyle="round,pad=0.02,rounding_size=0.1",
                              facecolor='white', edgecolor=COLORS['box_blue'], linewidth=2)
    ax.add_patch(pred_box)
    ax.text(7.7, 2.9, 'Predictions', ha='center', fontsize=10, fontweight='bold', color=COLORS['box_blue'])
    ax.text(7.7, 2.5, r'$\hat{\tau}_w$', ha='center', fontsize=11)
    ax.text(7.7, 2.1, r'$\hat{q}_w$', ha='center', fontsize=11)

    # Data loss
    draw_arrow(ax, (8.7, 2.8), (9.7, 3.3), color=COLORS['box_blue'])
    data_box = FancyBboxPatch((9.5, 2.9), 2.2, 0.9,
                              boxstyle="round,pad=0.02,rounding_size=0.1",
                              facecolor='white', edgecolor=COLORS['box_blue'], linewidth=2)
    ax.add_patch(data_box)
    ax.text(10.6, 3.35, r'$\mathcal{L}_{data} = MSE$', ha='center', fontsize=10, color=COLORS['box_blue'])

    # Physics loss branch
    # Physics residuals box
    phys_box = FancyBboxPatch((6.8, -0.2), 3.5, 1.8,
                              boxstyle="round,pad=0.02,rounding_size=0.1",
                              facecolor=COLORS['physics'], edgecolor=COLORS['box_green'], linewidth=2)
    ax.add_patch(phys_box)
    ax.text(8.55, 1.35, 'Physics Residuals', ha='center', fontsize=10, fontweight='bold', color=COLORS['box_green'])

    residuals = [
        r'$R_u$: Momentum',
        r'$R_T$: Energy',
        r'$R_{div}$: Continuity',
        r'$R_\tau$: Wall stress'
    ]
    for i, res in enumerate(residuals):
        ax.text(8.55, 0.9 - i*0.3, res, ha='center', fontsize=9)

    # Arrow from stencil to physics
    ax.annotate('', xy=(6.8, 0.7), xytext=(2.0, 1.5),
               arrowprops=dict(arrowstyle='->', color=COLORS['box_green'], lw=1.5,
                              connectionstyle='arc3,rad=-0.2'))
    ax.text(4.0, 0.5, 'Finite\ndifferences', ha='center', fontsize=8, color=COLORS['box_green'])

    # Physics loss
    draw_arrow(ax, (10.3, 0.7), (10.6, 1.8), color=COLORS['box_green'])
    phys_loss_box = FancyBboxPatch((9.5, 1.7), 2.2, 0.9,
                                   boxstyle="round,pad=0.02,rounding_size=0.1",
                                   facecolor='white', edgecolor=COLORS['box_green'], linewidth=2)
    ax.add_patch(phys_loss_box)
    ax.text(10.6, 2.15, r'$\mathcal{L}_{physics}$', ha='center', fontsize=10, color=COLORS['box_green'])

    # Combined loss
    draw_arrow(ax, (10.6, 2.9), (10.6, 2.65), color=COLORS['arrow'])
    total_box = FancyBboxPatch((9.3, 3.9), 2.6, 0.7,
                               boxstyle="round,pad=0.02,rounding_size=0.1",
                               facecolor='white', edgecolor=COLORS['box_purple'], linewidth=2)
    ax.add_patch(total_box)
    ax.text(10.6, 4.25, r'$\mathcal{L} = \mathcal{L}_{data} + \lambda \mathcal{L}_{physics}$',
           ha='center', fontsize=10, color=COLORS['box_purple'], fontweight='bold')

    # Key insight
    ax.text(6, -1.2, 'Key insight: Physics constraints provide regularization for extrapolation beyond training data',
           ha='center', fontsize=10, style='italic', color='gray')

    plt.tight_layout()
    plt.savefig('Images/chapter7/chapter7_overview.png', dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    plt.savefig('Images/chapter7/chapter7_overview.pdf', dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: chapter7_overview.png/pdf")

# ============================================================================
# Chapter 8: Flow Separation Identification
# ============================================================================
def generate_chapter8_overview():
    """Chapter 8: Identification of Flow Separation"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    ax.set_xlim(-1, 13)
    ax.set_ylim(-1, 5)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    ax.text(6, 4.7, 'Chapter 8: Flow Separation Detection and Hybrid Strategy',
           ha='center', va='top', fontsize=14, fontweight='bold')

    # Left: Input stencil
    draw_stencil(ax, 1, 2.5, scale=1.0)
    ax.text(1, 4.0, r'$\mathbf{Flow\ Data}$', ha='center', fontsize=11, fontweight='bold')

    # Arrow to classifier
    draw_arrow(ax, (2.0, 2.5), (3.0, 2.5), color=COLORS['arrow'])

    # Classifier box
    class_box = FancyBboxPatch((3.0, 1.7), 2.4, 1.6,
                               boxstyle="round,pad=0.02,rounding_size=0.1",
                               facecolor='white', edgecolor=COLORS['box_orange'], linewidth=2)
    ax.add_patch(class_box)
    ax.text(4.2, 3.05, 'Separation', ha='center', fontsize=10, fontweight='bold', color=COLORS['box_orange'])
    ax.text(4.2, 2.7, 'Classifier', ha='center', fontsize=10, fontweight='bold', color=COLORS['box_orange'])

    # Small classifier network
    for i in range(3):
        y = 2.4 - i * 0.3
        circle = Circle((3.5, y), 0.1, facecolor=COLORS['input'], edgecolor='black', linewidth=1)
        ax.add_patch(circle)
    for i in range(4):
        y = 2.5 - i * 0.25
        circle = Circle((4.2, y), 0.1, facecolor=COLORS['hidden'], edgecolor='black', linewidth=1)
        ax.add_patch(circle)
    for i in range(2):
        y = 2.35 - i * 0.4
        circle = Circle((4.9, y), 0.1, facecolor=COLORS['output'], edgecolor='black', linewidth=1)
        ax.add_patch(circle)

    # Decision diamond
    diamond_x, diamond_y = 6.5, 2.5
    diamond = Polygon([(diamond_x, diamond_y+0.6), (diamond_x+0.8, diamond_y),
                       (diamond_x, diamond_y-0.6), (diamond_x-0.8, diamond_y)],
                     facecolor='lightyellow', edgecolor='black', linewidth=1.5)
    ax.add_patch(diamond)
    ax.text(diamond_x, diamond_y, 'Separated?', ha='center', va='center', fontsize=9, fontweight='bold')

    # Arrow from classifier to decision
    draw_arrow(ax, (5.4, 2.5), (5.7, 2.5), color=COLORS['arrow'])

    # Yes branch (ML Wall Function)
    draw_arrow(ax, (diamond_x, diamond_y + 0.6), (diamond_x, 3.8), color=COLORS['box_green'])
    ax.text(diamond_x + 0.3, 3.3, 'Yes', fontsize=9, color=COLORS['box_green'])

    ml_box = FancyBboxPatch((5.5, 3.8), 2.0, 0.8,
                            boxstyle="round,pad=0.02,rounding_size=0.1",
                            facecolor='white', edgecolor=COLORS['box_green'], linewidth=2)
    ax.add_patch(ml_box)
    ax.text(6.5, 4.2, 'ML Wall Function', ha='center', fontsize=10, fontweight='bold', color=COLORS['box_green'])

    # No branch (Traditional Wall Function)
    draw_arrow(ax, (diamond_x, diamond_y - 0.6), (diamond_x, 0.8), color=COLORS['box_blue'])
    ax.text(diamond_x + 0.3, 1.3, 'No', fontsize=9, color=COLORS['box_blue'])

    trad_box = FancyBboxPatch((5.5, 0.0), 2.0, 0.8,
                              boxstyle="round,pad=0.02,rounding_size=0.1",
                              facecolor='white', edgecolor=COLORS['box_blue'], linewidth=2)
    ax.add_patch(trad_box)
    ax.text(6.5, 0.4, 'Traditional WF', ha='center', fontsize=10, fontweight='bold', color=COLORS['box_blue'])
    ax.text(6.5, 0.1, '(Spalding)', ha='center', fontsize=8, color='gray')

    # Merge arrows
    draw_arrow(ax, (7.5, 4.2), (8.5, 2.5), color=COLORS['box_green'])
    draw_arrow(ax, (7.5, 0.4), (8.5, 2.5), color=COLORS['box_blue'])

    # Final output
    output_box = FancyBboxPatch((8.5, 1.9), 2.2, 1.2,
                                boxstyle="round,pad=0.02,rounding_size=0.1",
                                facecolor='white', edgecolor=COLORS['box_purple'], linewidth=2)
    ax.add_patch(output_box)
    ax.text(9.6, 2.9, r'$\mathbf{Outputs}$', ha='center', fontsize=10, fontweight='bold')
    ax.text(9.6, 2.5, r'$\tau_w$', ha='center', fontsize=11)
    ax.text(9.6, 2.1, r'$q_w$', ha='center', fontsize=11)

    # Arrow to CFD
    draw_arrow(ax, (10.7, 2.5), (11.5, 2.5), color=COLORS['arrow'])

    # CFD solver
    cfd_box = FancyBboxPatch((11.5, 2.0), 1.2, 1.0,
                             boxstyle="round,pad=0.02,rounding_size=0.1",
                             facecolor='lightgray', edgecolor='black', linewidth=2)
    ax.add_patch(cfd_box)
    ax.text(12.1, 2.5, 'CFD\nSolver', ha='center', fontsize=10, fontweight='bold')

    # Key insight
    ax.text(6, -0.7, 'Key insight: Hybrid strategy uses ML where needed (separation) and robust classical methods elsewhere',
           ha='center', fontsize=10, style='italic', color='gray')

    # Accuracy annotations
    ax.text(4.2, 1.5, 'Accuracy: 94%', ha='center', fontsize=8, color=COLORS['box_orange'])

    plt.tight_layout()
    plt.savefig('Images/chapter8/chapter8_overview.png', dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    plt.savefig('Images/chapter8/chapter8_overview.pdf', dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: chapter8_overview.png/pdf")

# ============================================================================
# Main execution
# ============================================================================
if __name__ == '__main__':
    # Change to the thesis directory
    os.chdir('/home/jianoujiang/Desktop/openfoam/run/flow2Dtube/PAPERS/Oxford-EngSci-Thesis-Template')

    print("Generating chapter overview figures...")
    print("=" * 50)

    generate_chapter4_overview()
    generate_chapter5_overview()
    generate_chapter6_overview()
    generate_chapter7_overview()
    generate_chapter8_overview()

    print("=" * 50)
    print("All overview figures generated successfully!")
    print("\nFiles created:")
    print("  - Images/chapter4/chapter4_overview.png/pdf")
    print("  - Images/chapter5/chapter5_overview.png/pdf")
    print("  - Images/chapter6/chapter6_overview.png/pdf")
    print("  - Images/chapter7/chapter7_overview.png/pdf")
    print("  - Images/chapter8/chapter8_overview.png/pdf")
