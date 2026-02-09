import numpy as np

def compute_radius_of_gyration(coords):
    """
    Computes the Radius of Gyration (Rg) for each model in the ensemble.
    
    Args:
        coords: NumPy array of shape (Models, Residues, 3).
    Returns:
        np.array: Array containing the Rg value for each model.
    """
    # Calculate geometric center for each model
    centroids = np.mean(coords, axis=1, keepdims=True)
    
    # Sum of squared distances from centroid
    sq_dists = np.sum((coords - centroids)**2, axis=2)
    
    # Root Mean Square distance
    return np.sqrt(np.mean(sq_dists, axis=1))

def find_representative_model(coords):
    """
    Identifies the model closest to the mean structure (lowest RMSD).
    
    Returns:
        tuple: (index_of_best_model, lowest_rmsd_value)
    """
    mean_structure = np.mean(coords, axis=0)
    
    # Calculate RMSD of each model vs Mean Structure
    diff = coords - mean_structure
    rmsd_list = np.sqrt(np.mean(np.sum(diff**2, axis=2), axis=1))
    
    best_idx = np.argmin(rmsd_list)
    return best_idx, rmsd_list[best_idx]