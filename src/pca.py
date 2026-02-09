import numpy as np
import matplotlib.pyplot as plt
import os

def compute_pca(coords):
    """
    Performs Principal Component Analysis (PCA) on atomic coordinates via SVD.
    
    Args:
        coords: NumPy array of shape (n_models, n_residues, 3).
        
    Returns:
        tuple: (
            projection: Array (n_models, 2) containing PC1 and PC2 coordinates.
            explained_variance: Array (2,) containing variance ratio of PC1 and PC2.
        )
    """
    # 1. Reshape Data: Flatten (Models, Residues, 3) -> (Models, Residues*3)
    n_models, n_res, _ = coords.shape
    data = coords.reshape(n_models, n_res * 3)
    
    # 2. Mean Centering
    # Subtract the mean structure from all models to analyze fluctuations
    mean_structure = np.mean(data, axis=0)
    centered_data = data - mean_structure
    
    # 3. Singular Value Decomposition (SVD)
    # U: Left singular vectors (Projections/Scores)
    # S: Singular values (related to Variance)
    # Vt: Right singular vectors (Principal Components/Eigenvectors)
    try:
        u, s, vt = np.linalg.svd(centered_data, full_matrices=False)
    except np.linalg.LinAlgError:
        print("  [PCA ERROR] SVD convergence failed.")
        return None, None

    # 4. Calculate Explained Variance
    # Eigenvalues = (s^2) / (n - 1)
    eigvals = s**2 / (n_models - 1)
    total_variance = np.sum(eigvals)
    explained_variance_ratio = eigvals / total_variance
    
    # Project data onto the first two principal components
    # The projection (scores) is calculated as U * S
    pc_projection = u[:, :2] * s[:2]
    
    # Return projection and variance ratio for PC1 and PC2
    return pc_projection, explained_variance_ratio[:2]

def plot_pca_space(pdb_id, projection, variance, output_dir):
    """
    Generates and saves the 2D Scatter Plot of the Conformational Space.
    """
    if projection is None: return

    pc1 = projection[:, 0]
    pc2 = projection[:, 1]
    var1 = variance[0] * 100
    var2 = variance[1] * 100

    plt.figure(figsize=(8, 7))
    
    # Scatter plot: Color points by model index to visualize time evolution
    sc = plt.scatter(pc1, pc2, c=range(len(pc1)), cmap='viridis', 
                     edgecolors='k', s=80, alpha=0.8)
    
    # Aesthetics
    plt.xlabel(f"Principal Component 1 ({var1:.1f}%)")
    plt.ylabel(f"Principal Component 2 ({var2:.1f}%)")
    plt.title(f"PCA - Conformational Space: {pdb_id}")
    plt.colorbar(sc, label="Model Index (1 -> N)")
    plt.grid(True, linestyle='--', alpha=0.4)
    
    # Center axes on origin
    plt.axhline(0, color='gray', linestyle=':', linewidth=1)
    plt.axvline(0, color='gray', linestyle=':', linewidth=1)

    # Annotate extreme points (Start, End, Min/Max) for clarity
    indices_to_label = {0, len(pc1)-1, np.argmax(pc1), np.argmin(pc1)}
    for i in indices_to_label:
        plt.text(pc1[i], pc2[i], f"M{i+1}", fontsize=9, fontweight='bold', 
                 color='black', ha='right', va='bottom')

    plt.tight_layout()
    filename = os.path.join(output_dir, f"{pdb_id}_pca.png")
    plt.savefig(filename, dpi=300)
    plt.close()