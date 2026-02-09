import math
import numpy as np
import matplotlib.pyplot as plt
import os

def parse_backbone(pdb_file):
    """
    Extracts N, CA, C coordinates for all models.
    Required because the standard loader only extracts CA atoms.
    
    Returns a list of models, where each model is a dictionary:
    {residue_index: {'N': [x,y,z], 'CA': [x,y,z], 'C': [x,y,z]}}
    """
    if not os.path.exists(pdb_file): return []

    models = []
    current_model = {}
    
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith("MODEL"):
                current_model = {}
            elif line.startswith("ENDMDL"):
                if current_model: models.append(current_model)
                current_model = {}
            elif line.startswith("ATOM"):
                atom_name = line[12:16].strip()
                if atom_name in ['N', 'CA', 'C']:
                    try:
                        res_id = int(line[22:26])
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        
                        if res_id not in current_model:
                            current_model[res_id] = {}
                        
                        current_model[res_id][atom_name] = np.array([x, y, z])
                    except ValueError: continue
                    
    # Fallback for files without MODEL/ENDMDL
    if not models and current_model: models.append(current_model)
    return models

def calc_dihedral(p1, p2, p3, p4):
    """Calculates the dihedral angle between 4 points in space (degrees)."""
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3

    # Normalize vectors
    b1 /= np.linalg.norm(b1)
    b2 /= np.linalg.norm(b2)
    b3 /= np.linalg.norm(b3)

    # Vectors normal to the planes
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)

    # Calculate angle using atan2 for correct sign
    m1 = np.cross(n1, b2 / np.linalg.norm(b2))
    x = np.dot(n1, n2)
    y = np.dot(m1, n2)

    return -math.degrees(math.atan2(y, x))

def compute_phi_psi(models):
    """Computes Phi and Psi angle lists for all models."""
    phi_list = []
    psi_list = []

    for model in models:
        # Sort residues to ensure correct sequence order
        res_ids = sorted(model.keys())
        
        for i in range(1, len(res_ids) - 1):
            curr_id = res_ids[i]
            prev_id = res_ids[i-1]
            next_id = res_ids[i+1]

            # Connectivity check: residues must be consecutive
            if (curr_id - prev_id != 1) or (next_id - curr_id != 1):
                continue

            try:
                # Required coordinates
                # Phi: C(prev) - N(curr) - CA(curr) - C(curr)
                c_prev = model[prev_id]['C']
                n_curr = model[curr_id]['N']
                ca_curr = model[curr_id]['CA']
                c_curr = model[curr_id]['C']
                
                # Psi: N(curr) - CA(curr) - C(curr) - N(next)
                n_next = model[next_id]['N']

                phi = calc_dihedral(c_prev, n_curr, ca_curr, c_curr)
                psi = calc_dihedral(n_curr, ca_curr, c_curr, n_next)

                phi_list.append(phi)
                psi_list.append(psi)

            except KeyError:
                # Missing atoms in this residue
                continue
                
    return phi_list, psi_list

def plot_ramachandran(pdb_id, phi, psi, output_dir):
    """Generates the Ramachandran Scatter Plot."""
    plt.figure(figsize=(7, 7))
    
    # Scatter plot
    plt.scatter(phi, psi, alpha=0.5, s=10, color='black', marker='.')
    
    # Axis Setup
    plt.xlim(-180, 180)
    plt.ylim(-180, 180)
    plt.xlabel('Phi ($\phi$)')
    plt.ylabel('Psi ($\psi$)')
    plt.title(f'Ramachandran Plot: {pdb_id}')
    
    # Center lines
    plt.axhline(0, color='gray', lw=0.5)
    plt.axvline(0, color='gray', lw=0.5)
    
    # Favorable Regions (Simplified)
    # Alpha Helix (approx -60, -50) - Bottom Left Quadrant
    rect_alpha = plt.Rectangle((-100, -100), 60, 60, fill=True, color='red', alpha=0.1, label='Alpha-Helix')
    plt.gca().add_patch(rect_alpha)
    
    # Beta Sheet (approx -120, 120) - Top Left Quadrant
    rect_beta = plt.Rectangle((-180, 90), 100, 80, fill=True, color='blue', alpha=0.1, label='Beta-Sheet')
    plt.gca().add_patch(rect_beta)

    plt.legend(loc='upper right')
    plt.grid(alpha=0.3)
    
    filename = os.path.join(output_dir, f"{pdb_id}_ramachandran.png")
    plt.savefig(filename, dpi=300)
    plt.close()