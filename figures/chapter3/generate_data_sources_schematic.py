#!/usr/bin/env python3
"""
Generate schematic figure showing training data sources strategy for ML wall functions.

This figure illustrates:
- Different flow regions (attached, near-separation, separated)
- Corresponding data sources (RANS, DNS, Experimental)
- Benchmark cases and their characteristics
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Polygon
import numpy as np

# Try different style options
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except OSError:
    try:
        plt.style.use('seaborn-whitegrid')
    except OSError:
        pass

def create_data_sources_schematic():
    """Create comprehensive schematic of training data sources."""

    fig = plt.figure(figsize=(14, 10))

    # Create grid layout
    gs = fig.add_gridspec(3, 2, height_ratios=[1.5, 1, 1], width_ratios=[1.2, 1],
                          hspace=0.35, wspace=0.25)

    # Main flow schematic (top, spanning both columns)
    ax_main = fig.add_subplot(gs[0, :])

    # Draw diffuser geometry
    x_inlet = 0
    x_sep = 3.5
    x_reattach = 7
    x_outlet = 10

    h_inlet = 1.0
    h_outlet = 2.5

    # Upper wall (straight)
    ax_main.plot([x_inlet, x_outlet], [h_inlet, h_inlet], 'k-', linewidth=2)

    # Lower wall (with expansion)
    lower_x = [x_inlet, x_sep - 0.5, x_outlet]
    lower_y = [0, 0, -(h_outlet - h_inlet)]
    ax_main.plot([x_inlet, x_sep], [0, 0], 'k-', linewidth=2)
    ax_main.plot([x_sep, x_outlet], [0, -(h_outlet - h_inlet)], 'k-', linewidth=2)

    # Flow regions with color coding
    # Attached region (green)
    attached = FancyBboxPatch((x_inlet + 0.1, 0.1), x_sep - x_inlet - 0.3, h_inlet - 0.2,
                               boxstyle="round,pad=0.02", facecolor='#2ecc71',
                               alpha=0.3, edgecolor='#27ae60', linewidth=2)
    ax_main.add_patch(attached)

    # Near-separation region (orange)
    near_sep_x = [x_sep - 0.2, x_sep + 0.8, x_sep + 0.8, x_sep - 0.2]
    near_sep_y = [0.1, 0.1, h_inlet - 0.1, h_inlet - 0.1]
    near_sep = Polygon(list(zip(near_sep_x, near_sep_y)), facecolor='#f39c12',
                       alpha=0.3, edgecolor='#d35400', linewidth=2)
    ax_main.add_patch(near_sep)

    # Separated region (red)
    sep_x = np.linspace(x_sep + 0.8, x_reattach + 0.5, 50)
    sep_y_lower = np.interp(sep_x, [x_sep, x_outlet], [0, -(h_outlet - h_inlet)])
    sep_y_upper = np.minimum(sep_y_lower + 0.8, h_inlet - 0.1)

    sep_poly_x = list(sep_x) + list(sep_x[::-1])
    sep_poly_y = list(sep_y_lower + 0.1) + list(sep_y_upper[::-1])
    sep_region = Polygon(list(zip(sep_poly_x, sep_poly_y)), facecolor='#e74c3c',
                         alpha=0.3, edgecolor='#c0392b', linewidth=2)
    ax_main.add_patch(sep_region)

    # Recovery region (blue)
    rec_x = [x_reattach + 0.5, x_outlet - 0.1, x_outlet - 0.1, x_reattach + 0.5]
    rec_y_lower = np.interp([rec_x[0], rec_x[1]], [x_sep, x_outlet], [0, -(h_outlet - h_inlet)])
    rec_y = [rec_y_lower[0] + 0.1, rec_y_lower[1] + 0.1, h_inlet - 0.1, h_inlet - 0.1]
    rec_region = Polygon(list(zip(rec_x, rec_y)), facecolor='#3498db',
                         alpha=0.3, edgecolor='#2980b9', linewidth=2)
    ax_main.add_patch(rec_region)

    # Streamlines indicating flow
    for y_start in [0.2, 0.5, 0.8]:
        x_stream = np.linspace(x_inlet + 0.5, x_outlet - 0.5, 100)
        # Create varying streamlines
        if y_start > 0.5:
            y_stream = np.ones_like(x_stream) * y_start
        else:
            # Lower streamlines curve down in separated region
            y_stream = np.where(x_stream < x_sep,
                               y_start * np.ones_like(x_stream),
                               np.interp(x_stream, [x_sep, x_outlet],
                                        [y_start, y_start - (h_outlet - h_inlet) * 0.3]))
        ax_main.plot(x_stream, y_stream, 'b-', alpha=0.4, linewidth=0.8)
        # Arrow at the end
        ax_main.annotate('', xy=(x_stream[-1], y_stream[-1]),
                        xytext=(x_stream[-5], y_stream[-5]),
                        arrowprops=dict(arrowstyle='->', color='blue', alpha=0.4))

    # Recirculation bubble
    theta = np.linspace(0, np.pi, 50)
    bubble_x = x_sep + 0.5 + 1.5 * np.cos(theta)
    bubble_y_base = np.interp(bubble_x, [x_sep, x_outlet], [0, -(h_outlet - h_inlet)])
    bubble_y = bubble_y_base + 0.3 * np.sin(theta)
    ax_main.plot(bubble_x, bubble_y, 'r--', linewidth=1.5, alpha=0.7)
    ax_main.annotate('', xy=(x_sep + 0.3, bubble_y_base[len(bubble_x)//4] + 0.1),
                    xytext=(x_sep + 0.6, bubble_y_base[len(bubble_x)//4] + 0.2),
                    arrowprops=dict(arrowstyle='->', color='red', alpha=0.7))

    # Labels for regions
    ax_main.text((x_inlet + x_sep)/2, h_inlet + 0.15, 'Attached Flow',
                 ha='center', fontsize=11, fontweight='bold', color='#27ae60')
    ax_main.text(x_sep + 0.3, h_inlet + 0.15, 'Transition',
                 ha='center', fontsize=10, fontweight='bold', color='#d35400')
    ax_main.text((x_sep + x_reattach)/2 + 0.5, h_inlet + 0.15, 'Separated Flow',
                 ha='center', fontsize=11, fontweight='bold', color='#c0392b')
    ax_main.text((x_reattach + x_outlet)/2 + 0.3, h_inlet + 0.15, 'Recovery',
                 ha='center', fontsize=10, fontweight='bold', color='#2980b9')

    # Data source labels below
    ax_main.text((x_inlet + x_sep)/2, -0.6, 'RANS\n(with/without WF)',
                 ha='center', fontsize=10, bbox=dict(boxstyle='round',
                 facecolor='#2ecc71', alpha=0.5))
    ax_main.text((x_sep + x_reattach)/2 + 0.5, -(h_outlet - h_inlet)/2 - 0.8,
                 'DNS / Experimental\nBenchmarks',
                 ha='center', fontsize=10, bbox=dict(boxstyle='round',
                 facecolor='#e74c3c', alpha=0.5))
    ax_main.text((x_reattach + x_outlet)/2 + 0.3, -(h_outlet - h_inlet) - 0.3,
                 'RANS\n(recovery)',
                 ha='center', fontsize=10, bbox=dict(boxstyle='round',
                 facecolor='#3498db', alpha=0.5))

    # Markers for key points
    ax_main.plot(x_sep, 0, 'ko', markersize=8)
    ax_main.text(x_sep, -0.25, 'Separation', ha='center', fontsize=9, style='italic')

    ax_main.plot(x_reattach, np.interp(x_reattach, [x_sep, x_outlet], [0, -(h_outlet - h_inlet)]),
                 'ko', markersize=8)
    ax_main.text(x_reattach, np.interp(x_reattach, [x_sep, x_outlet],
                [0, -(h_outlet - h_inlet)]) - 0.25, 'Reattachment', ha='center', fontsize=9, style='italic')

    ax_main.set_xlim(-0.5, x_outlet + 0.5)
    ax_main.set_ylim(-(h_outlet - h_inlet) - 1.5, h_inlet + 0.5)
    ax_main.set_aspect('equal')
    ax_main.axis('off')
    ax_main.set_title('(a) Flow Regions and Corresponding Training Data Sources',
                      fontsize=12, fontweight='bold', pad=10)

    # Benchmark cases table (bottom left)
    ax_table = fig.add_subplot(gs[1, 0])
    ax_table.axis('off')

    # Table data
    table_data = [
        ['Benchmark', 'Re', 'Separation Type', 'Data Type'],
        ['Backward-Facing Step', '37,500', 'Sudden expansion', 'Experimental'],
        ['Periodic Hills', '10,595', 'Cyclic sep/reattach', 'DNS/LES'],
        ['Buice-Eaton Diffuser', '20,000', 'Gradual APG', 'Experimental'],
        ['Channel Flow (DNS)', '5,200', 'None (reference)', 'DNS'],
    ]

    table = ax_table.table(cellText=table_data[1:], colLabels=table_data[0],
                           loc='center', cellLoc='center',
                           colWidths=[0.35, 0.15, 0.25, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    # Color header row
    for i in range(4):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    ax_table.set_title('(b) Experimental and DNS Benchmark Cases',
                       fontsize=11, fontweight='bold', y=0.95)

    # Data source flowchart (bottom right)
    ax_flow = fig.add_subplot(gs[1, 1])
    ax_flow.axis('off')

    # Flow chart boxes
    boxes = {
        'input': {'pos': (0.5, 0.85), 'text': 'Flow Region\nClassification', 'color': '#ecf0f1'},
        'attached': {'pos': (0.2, 0.55), 'text': 'Attached Flow\n(dP/dx ≤ 0 or\nlow APG)', 'color': '#2ecc71'},
        'separated': {'pos': (0.8, 0.55), 'text': 'Separated Flow\n(high APG or\nτ_w < 0)', 'color': '#e74c3c'},
        'rans': {'pos': (0.2, 0.2), 'text': 'RANS Data\n(This work)', 'color': '#3498db'},
        'dns': {'pos': (0.8, 0.2), 'text': 'DNS/Exp Data\n(Benchmarks)', 'color': '#9b59b6'},
    }

    for key, box in boxes.items():
        bbox = FancyBboxPatch((box['pos'][0] - 0.15, box['pos'][1] - 0.1), 0.3, 0.2,
                              boxstyle="round,pad=0.02", facecolor=box['color'],
                              alpha=0.7, edgecolor='black', linewidth=1.5,
                              transform=ax_flow.transAxes)
        ax_flow.add_patch(bbox)
        ax_flow.text(box['pos'][0], box['pos'][1], box['text'],
                    ha='center', va='center', fontsize=9, fontweight='bold',
                    transform=ax_flow.transAxes)

    # Arrows
    arrow_style = dict(arrowstyle='->', color='black', linewidth=1.5)
    ax_flow.annotate('', xy=(0.2, 0.65), xytext=(0.4, 0.75),
                    arrowprops=arrow_style, xycoords='axes fraction')
    ax_flow.annotate('', xy=(0.8, 0.65), xytext=(0.6, 0.75),
                    arrowprops=arrow_style, xycoords='axes fraction')
    ax_flow.annotate('', xy=(0.2, 0.35), xytext=(0.2, 0.45),
                    arrowprops=arrow_style, xycoords='axes fraction')
    ax_flow.annotate('', xy=(0.8, 0.35), xytext=(0.8, 0.45),
                    arrowprops=arrow_style, xycoords='axes fraction')

    ax_flow.set_xlim(0, 1)
    ax_flow.set_ylim(0, 1)
    ax_flow.set_title('(c) Data Source Selection Strategy',
                      fontsize=11, fontweight='bold', y=0.98)

    # Key characteristics comparison (bottom row)
    ax_char = fig.add_subplot(gs[2, :])
    ax_char.axis('off')

    # Create comparison boxes
    rans_text = """RANS-Generated Data (This Work)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ Fast generation (minutes per case)
✓ Controllable parameters (Re, geometry)
✓ Consistent boundary conditions
✓ Good for attached/mild APG flows
✗ May fail in strong separation
✗ Wall function assumptions embedded"""

    dns_text = """DNS/Experimental Benchmarks
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ High-fidelity ground truth
✓ Accurate in separation regions
✓ No wall function assumptions
✗ Expensive (weeks of computation)
✗ Limited parameter range
✗ Fixed geometries/conditions"""

    combined_text = """Combined Training Strategy
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
→ Use RANS data for attached flows
→ Use DNS/Exp for separated regions
→ Transfer learning from benchmarks
→ Physics-informed augmentation
→ Robust predictions across regimes"""

    ax_char.text(0.02, 0.95, rans_text, ha='left', va='top', fontsize=9,
                family='monospace', bbox=dict(boxstyle='round', facecolor='#d5f4e6',
                alpha=0.8), transform=ax_char.transAxes)
    ax_char.text(0.36, 0.95, dns_text, ha='left', va='top', fontsize=9,
                family='monospace', bbox=dict(boxstyle='round', facecolor='#fce4ec',
                alpha=0.8), transform=ax_char.transAxes)
    ax_char.text(0.70, 0.95, combined_text, ha='left', va='top', fontsize=9,
                family='monospace', bbox=dict(boxstyle='round', facecolor='#e8eaf6',
                alpha=0.8), transform=ax_char.transAxes)

    ax_char.set_title('(d) Comparison of Data Sources and Combined Strategy',
                      fontsize=11, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig('data_sources_strategy.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig('data_sources_strategy.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("Generated: data_sources_strategy.png and .pdf")
    plt.close()


if __name__ == "__main__":
    create_data_sources_schematic()
    print("\nFigure generation complete!")
