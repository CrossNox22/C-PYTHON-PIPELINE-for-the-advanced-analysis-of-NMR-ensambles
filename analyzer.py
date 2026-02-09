import os
import subprocess
import numpy as np

# Import local modules
from src import loader, geometry, plotter, exporter, dccm, ramachandran, pca

PROTEIN_IDS = ["1D3Z"]

# Determine executable path based on OS
if os.name == 'nt':
    EXECUTABLE = "analyzer.exe"
else:
    EXECUTABLE = "./analyzer"

def run_cpp_engine(pdb_id, coords):
    """
    Invokes the C++ engine to compute RMSF and SASA.
    Returns two NumPy arrays: rmsf_values and sasa_values.
    """
    xyz_filename = f"temp_{pdb_id}.xyz"
    out_rmsf = f"rmsf_{pdb_id}.txt"
    out_sasa = f"sasa_{pdb_id}.txt" 
    
    # 1. Export temporary XYZ trajectory for the C++ engine
    with open(xyz_filename, 'w') as f:
        for i, model in enumerate(coords):
            f.write(f"{coords.shape[1]}\nFrame {i+1}\n")
            for atom in model:
                # passing CA coordinates; C++ engine uses generic radius
                f.write(f"CA {atom[0]:.3f} {atom[1]:.3f} {atom[2]:.3f}\n")
    
    rmsf, sasa = [], []
    
    try:
        # 2. Execute C++ binary
        subprocess.run(
            [os.path.abspath(EXECUTABLE), xyz_filename, out_rmsf, out_sasa], 
            check=True
        )
        
        # 3. Load results
        if os.path.exists(out_rmsf):
            rmsf = np.loadtxt(out_rmsf)
        
        if os.path.exists(out_sasa):
            sasa = np.loadtxt(out_sasa)
            
    except Exception as e:
        print(f"  [C++ ENGINE ERROR] {e}")
    finally:
        # 4. Cleanup temporary files
        for f in [xyz_filename, out_rmsf, out_sasa]:
            if os.path.exists(f): os.remove(f)
            
    return rmsf, sasa

def main():
    results_dir = "results"
    if not os.path.exists(results_dir): os.makedirs(results_dir)

    print("--- STARTING PIPELINE ---")

    for pid in PROTEIN_IDS:
        print(f"\n> Analyzing: {pid}")
        
        # --- PHASE 1: DATA LOADING ---
        coords, seq = loader.parse_pdb(pid)
        if coords is None: 
            print(f"  [SKIP] Invalid or missing PDB: {pid}")
            continue

        # --- PHASE 2: GEOMETRIC ANALYSIS (PYTHON) ---
        # CORRETTO: Uso dei nuovi nomi inglesi
        rg_vals = geometry.compute_radius_of_gyration(coords)
        best_model_idx, best_rmsd = geometry.find_representative_model(coords)
        
        print(f"  Avg Rg: {np.mean(rg_vals):.2f} A | Best Model: {best_model_idx} (RMSD: {best_rmsd:.2f})")

        # --- PHASE 3: DYNAMICS & CONFORMATIONAL SPACE (PYTHON) ---
        # DCCM
        print("  Computing Dynamic Cross-Correlation (DCCM)...")
        # CORRETTO: compute_dccm_matrix invece di calcola_dccm
        dccm_matrix = dccm.compute_dccm_matrix(coords)
        dccm.plot_dccm_heatmap(pid, dccm_matrix, results_dir)

        # PCA
        print("  Computing Principal Component Analysis (PCA)...")
        # CORRETTO: compute_pca invece di run_pca
        pc_proj, pc_var = pca.compute_pca(coords)
        if pc_proj is not None:
            pca.plot_pca_space(pid, pc_proj, pc_var, results_dir)
            print(f"  PCA: PC1 explains {pc_var[0]*100:.1f}%, PC2 explains {pc_var[1]*100:.1f}%")

        # Ramachandran Plot
        print("  Computing Torsional Angles (Ramachandran)...")
        try:
            full_models = ramachandran.parse_backbone(f"{pid}.pdb")
            phi, psi = ramachandran.compute_phi_psi(full_models)
            if phi:
                ramachandran.plot_ramachandran(pid, phi, psi, results_dir)
        except Exception as e:
            print(f"  [RAMACHANDRAN ERROR] {e}")

        # --- PHASE 4: HIGH-PERFORMANCE CALCULATION (C++) ---
        print("  Executing C++ Engine (RMSF + SASA)...")
        rmsf_vals, sasa_vals = run_cpp_engine(pid, coords)
        
        # --- PHASE 5: VISUALIZATION & EXPORT ---
        if len(rmsf_vals) > 0:
            # Standard Plots - CORRETTO: plot_hydrophobicity invece di plot_idrofobicita
            plotter.plot_hydrophobicity(pid, seq, rmsf_vals, results_dir)
            plotter.plot_rg_hist(pid, rg_vals, results_dir)
            plotter.plot_rmsf_standard(pid, rmsf_vals, results_dir)
            
            # Advanced Plots (requiring SASA)
            if len(sasa_vals) > 0:
                plotter.plot_sasa_profile(pid, sasa_vals, results_dir)
                plotter.plot_multi_physics_profile(pid, rmsf_vals, sasa_vals, results_dir)

            # Data Export
            exporter.save_to_csv(pid, seq, rmsf_vals, results_dir, sasa_values=sasa_vals)
            print(f"  Output saved to: {results_dir}")
        else:
            print("  [WARNING] No RMSF data generated.")

        print(f"  Analysis complete for {pid}")

if __name__ == "__main__":
    main()