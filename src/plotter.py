import matplotlib.pyplot as plt
import os
import numpy as np

# Hydrophobicity Scale (Kyte-Doolittle)
KYTE_DOOLITTLE = {
    'ALA': 1.8, 'ARG': -4.5, 'ASN': -3.5, 'ASP': -3.5, 'CYS': 2.5,
    'GLN': -3.5, 'GLU': -3.5, 'GLY': -0.4, 'HIS': -3.2, 'ILE': 4.5,
    'LEU': 3.8, 'LYS': -3.9, 'MET': 1.9, 'PHE': 2.8, 'PRO': -1.6,
    'SER': -0.8, 'THR': -0.7, 'TRP': -0.9, 'TYR': -1.3, 'VAL': 4.2
}

def plot_hydrophobicity(pdb_id, sequence, rmsf_values, output_dir):
    """
    Generates a Scatter Plot: Hydrophobicity vs RMSF.
    Annotates regions for Rigid Core and Mobile Loops.
    """
    hydr_vals = []
    plot_rmsf = []
    
    # Align sequence and data (filtering non-standard residues)
    for i, res in enumerate(sequence):
        if res in KYTE_DOOLITTLE and i < len(rmsf_values):
            hydr_vals.append(KYTE_DOOLITTLE[res])
            plot_rmsf.append(rmsf_values[i])
    
    if not hydr_vals: return

    # Create Plot
    plt.figure(figsize=(8, 6))
    
    # Scatter plot with colormap (Blue -> Red) and black edges
    plt.scatter(hydr_vals, plot_rmsf, c=hydr_vals, cmap='coolwarm', 
                edgecolors='black', alpha=0.7, s=40)
    
    # Guidelines (Logical Cartesian Axes)
    plt.axvline(0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    plt.axhline(1.0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7) # Flexibility Threshold
    
    # Detailed Axis Labels
    plt.xlabel('Hydrophobicity (Kyte-Doolittle)\n<-- Hydrophilic (Surface) | Hydrophobic (Core) -->')
    plt.ylabel('RMSF Flexibility (A)')
    plt.title(f'Structure-Sequence Correlation: {pdb_id}')
    
    # Light Grid
    plt.grid(True, linestyle=':', alpha=0.5)
    
    # --- TEXT ANNOTATIONS ---
    # Static coordinates, could be made dynamic
    plt.text(3.5, 0.2, "Rigid Core\n(Expected)", fontsize=9, color='darkred', ha='center', weight='bold')
    plt.text(-3.5, max(plot_rmsf)*0.8, "Mobile Loops\n(Expected)", fontsize=9, color='darkblue', ha='center', weight='bold')
    
    # Save
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{pdb_id}_scatter.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_rg_hist(pdb_id, rg_values, output_dir):
    """Generates the Radius of Gyration Histogram."""
    plt.figure(figsize=(8, 5))
    plt.hist(rg_values, bins=15, color='purple', edgecolor='black', alpha=0.7)
    
    mean_rg = np.mean(rg_values)
    plt.axvline(mean_rg, color='k', linestyle='dashed', linewidth=1, label=f'Mean: {mean_rg:.2f} A')
    
    plt.xlabel('Radius of Gyration (Rg) [A]')
    plt.ylabel('Frequency (Number of Models)')
    plt.title(f'Structural Compactness: {pdb_id}')
    plt.legend()
    plt.grid(axis='y', alpha=0.5)
    
    plt.savefig(os.path.join(output_dir, f"{pdb_id}_rg_hist.png"), dpi=300)
    plt.close()

def plot_rmsf_standard(pdb_id, rmsf_values, output_dir):
    """Standard RMSF Bar Chart (Green/Red/Orange)."""
    residui_idx = range(1, len(rmsf_values) + 1)
    
    colors = ['green' if x <= 0.8 else 'red' if x > 2.0 else 'orange' for x in rmsf_values]
    
    plt.figure(figsize=(10, 5))
    plt.bar(residui_idx, rmsf_values, color=colors, edgecolor='black', linewidth=0.5)
    
    plt.axhline(0.8, color='green', ls='--', label='Rigid (<0.8)')
    plt.axhline(2.0, color='red', ls='--', label='Flexible (>2.0)')
    
    plt.title(f'Per-Residue RMSF: {pdb_id}')
    plt.xlabel('Residue Number')
    plt.ylabel('RMSF (A)')
    plt.legend()
    
    plt.savefig(os.path.join(output_dir, f"{pdb_id}_rmsf_bars.png"), dpi=300)
    plt.close()

def plot_sasa_profile(pdb_id, sasa_values, output_dir):
    """Solvent Accessible Surface Area (SASA) Profile."""
    residui_idx = range(1, len(sasa_values) + 1)
    
    plt.figure(figsize=(10, 5))
    
    # Area Plot
    plt.fill_between(residui_idx, sasa_values, color='skyblue', alpha=0.4)
    plt.plot(residui_idx, sasa_values, color='dodgerblue', linewidth=2)
    
    # Empirical Threshold Line (Below 10 A^2 is often "Buried")
    plt.axhline(10, color='gray', linestyle=':', label='Buried Threshold (Approx)')
    
    plt.title(f'Solvent Accessibility (Mean SASA): {pdb_id}')
    plt.xlabel('Residue Number')
    plt.ylabel('Mean SASA ($A^2$)')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.savefig(os.path.join(output_dir, f"{pdb_id}_sasa.png"), dpi=300)
    plt.close()

def plot_multi_physics_profile(pdb_id, rmsf_values, sasa_values, output_dir):
    """
    Generates a Multi-Physics Profile (Nature-style).
    Overlays RMSF and SASA on dual Y-axes.
    """
    residui_idx = range(1, len(rmsf_values) + 1)
    
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    # --- LEFT Y-AXIS: RMSF (Black/Gray) ---
    color_rmsf = 'black'
    ax1.set_xlabel('Residue')
    ax1.set_ylabel('RMSF (A)', color=color_rmsf, fontweight='bold')
    # Plot line
    ax1.plot(residui_idx, rmsf_values, color=color_rmsf, linewidth=1.5, label='RMSF (Flexibility)')
    # Shaded area under curve
    ax1.fill_between(residui_idx, rmsf_values, color='gray', alpha=0.1)
    ax1.tick_params(axis='y', labelcolor=color_rmsf)
    ax1.grid(True, linestyle=':', alpha=0.5)

    # --- RIGHT Y-AXIS: SASA (Blue/Cyan) ---
    ax2 = ax1.twinx()  # Create a second axis sharing the same X
    color_sasa = 'dodgerblue'
    ax2.set_ylabel('SASA ($A^2$)', color=color_sasa, fontweight='bold')
    ax2.plot(residui_idx, sasa_values, color=color_sasa, linewidth=1.5, linestyle='--', label='SASA (Exposure)')
    ax2.tick_params(axis='y', labelcolor=color_sasa)

    # Title and Unified Legend
    plt.title(f'Multi-Physics Profile: {pdb_id} (Dynamics vs Exposure)')
    
    # Hack to have a single legend for two axes
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

    # Automatic Peak Annotation
    # Find residue with max RMSF
    max_rmsf_idx = np.argmax(rmsf_values)
    max_res = residui_idx[max_rmsf_idx]
    max_val = rmsf_values[max_rmsf_idx]
    
    # Add arrow to max peak
    ax1.annotate('Peak Mobility', xy=(max_res, max_val), xytext=(max_res+5, max_val+0.5),
                 arrowprops=dict(facecolor='red', shrink=0.05, width=1, headwidth=8),
                 color='red', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{pdb_id}_multiphysics.png"), dpi=300)
    plt.close()