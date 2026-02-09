import csv
import os

def save_to_csv(pdb_id, sequence, rmsf_values, output_dir, sasa_values=None):
    """
    Saves analysis metrics (RMSF, SASA) to a structured CSV file.
    
    Args:
        pdb_id (str): Protein identifier.
        sequence (list): List of residue names.
        rmsf_values (list/array): RMSF values.
        output_dir (str): Destination directory.
        sasa_values (list/array, optional): SASA values. Defaults to None.
        
    Returns:
        str: Path to the generated CSV file.
    """
    filename = os.path.join(output_dir, f"{pdb_id}_analysis.csv")
    
    # Validate SASA input
    has_sasa = sasa_values is not None and len(sasa_values) == len(rmsf_values)
    
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # 1. Write Header
        header = ["Residue_ID", "Residue_Name", "RMSF_Angstrom"]
        if has_sasa:
            header.append("SASA_Mean_A2") # A2 = Angstrom Squared
        
        writer.writerow(header)
        
        # 2. Write Data Rows
        if has_sasa:
            for i, (res, rmsf, sasa) in enumerate(zip(sequence, rmsf_values, sasa_values)):
                writer.writerow([i + 1, res, f"{rmsf:.4f}", f"{sasa:.4f}"])
        else:
            for i, (res, rmsf) in enumerate(zip(sequence, rmsf_values)):
                writer.writerow([i + 1, res, f"{rmsf:.4f}"])
            
    return filename