import os
import numpy as np

def parse_pdb(pdb_id):
    """
    Parses a PDB file to extract Alpha-Carbon (CA) coordinates and the sequence.
    
    Args:
        pdb_id (str): The PDB identifier (filename without extension).
        
    Returns:
        tuple: (
            np.ndarray: Coordinates array of shape (n_models, n_residues, 3),
            list: Sequence of residue names
        )
        Returns (None, None) if the file is missing or parsing fails.
    """
    filename = f"{pdb_id}.pdb"
    if not os.path.exists(filename):
        return None, None

    models_list = []
    current_model = []
    sequence = []
    sequence_read = False

    try:
        with open(filename, 'r') as f:
            for line in f:
                # Detect start of a new model (for NMR ensembles)
                if line.startswith("MODEL"):
                    current_model = []
                
                # Detect end of a model
                elif line.startswith("ENDMDL"):
                    if current_model: 
                        models_list.append(current_model)
                    current_model = []
                    # Sequence is identical for all models; read it only once
                    sequence_read = True
                
                # Parse Atom coordinates (CA only)
                # Columns 13-16: Atom Name, Columns 31-54: Coordinates
                elif line.startswith("ATOM") and line[12:16].strip() == "CA":
                    try:
                        res_name = line[17:20].strip()
                        xyz = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
                        
                        current_model.append(xyz)
                        
                        if not sequence_read: 
                            sequence.append(res_name)
                    except ValueError: 
                        continue
        
        # Handle files with a single model (no MODEL/ENDMDL tags)
        if not models_list and current_model: 
            models_list.append(current_model)

        return np.array(models_list, dtype=np.float64), sequence

    except Exception as e:
        print(f"  [LOADER ERROR] Failed to parse {filename}: {e}")
        return None, None