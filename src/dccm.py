import numpy as np
import matplotlib.pyplot as plt
import os

def compute_dccm_matrix(coords):
    """
    Computes the Dynamic Cross-Correlation Matrix (DCCM).
    
    Args:
        coords: NumPy array of shape (Models, Residues, 3).
    Returns:
        dccm: Normalized covariance matrix (Residues, Residues) with values [-1, 1].
    """
    # 1. Compute mean structure
    mean_structure = np.mean(coords, axis=0)
    
    # 2. Compute fluctuations (displacement vectors)
    diff = coords - mean_structure

    # 3. Compute Covariance Matrix
    # Using Einstein summation: 'm i c, m j c -> i j'
    # m=models, i=row_res, j=col_res, c=coords(x,y,z)
    covariance = np.einsum('mic,mjc->ij', diff, diff) / coords.shape[0]

    # 4. Normalize (Pearson correlation coefficient)
    diag = np.diag(covariance)
    # Outer product to get sqrt(var_i * var_j)
    norm_factor = np.sqrt(np.outer(diag, diag))
    
    # Numerical stability check
    norm_factor[norm_factor == 0] = 1.0

    # 5. Final Matrix
    dccm = covariance / norm_factor
    
    # Ensure diagonal is exactly 1.0
    np.fill_diagonal(dccm, 1.0)
    
    return dccm

def plot_dccm_heatmap(pdb_id, dccm_matrix, output_dir):
    """Generates and saves the DCCM heatmap visualization."""
    plt.figure(figsize=(8, 7))
    
    # Use 'bwr' (Blue-White-Red) colormap: Blue=-1 (anti-correlated), Red=+1 (correlated)
    im = plt.imshow(dccm_matrix, cmap='bwr', vmin=-1, vmax=1, origin='lower')
    
    plt.colorbar(im, label="Correlation Coefficient")
    plt.title(f"DCCM - Dynamic Map: {pdb_id}")
    plt.xlabel("Residue J")
    plt.ylabel("Residue I")
    
    filename = os.path.join(output_dir, f"{pdb_id}_dccm.png")
    plt.savefig(filename, dpi=300)
    plt.close()