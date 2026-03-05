import arviz as az
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
import pickle
import argparse

# --- CONFIGURATION ---
GP_FIGS = Path("reports/figures/gp_figures")
GP_FIGS.mkdir(parents=True, exist_ok=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_pkl", type=str, default="outputs/samples_gp_v4.pkl")
    parser.add_argument("--metadata", type=str, default="data/processed/v4_scaling_metadata.pkl")
    args = parser.parse_args()

    # 1. Load Data
    with open(args.input_pkl, "rb") as f:
        samples = pickle.load(f)
    with open(args.metadata, "rb") as f:
        smoke_sd = pickle.load(f)['smoking_std']
    
    # Convert samples to ArviZ (Assuming 4 chains)
    idata = az.from_dict(posterior={k: v.reshape(4, -1, *v.shape[1:]) for k, v in samples.items()})

    # 2. Stats & Translation
    var_names = ["b0", "beta_smoke", "beta_men", "beta_interaction", "kernel_ls", "kernel_var"]
    stats = az.summary(idata, var_names=var_names)
    
    # 3. Save Technical PDF Report
    with PdfPages("reports/gp_v4_technical_report.pdf") as pdf:
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        header = f"GP V4 STRATIFIED MODEL AUDIT\n{'='*40}\n"
        diagnostics = f"Max R-hat: {stats['r_hat'].max():.3f} | Min ESS: {stats['ess_bulk'].min():.0f}\n\n"
        
        ax.text(0.05, 0.95, header + diagnostics + stats.to_string(), 
                family='monospace', fontsize=8, va='top')
        pdf.savefig(); plt.close()

    # 4. Forest Plot & 6-Map Suite
    # (Labels and mapping logic exactly as before, saving to GP_FIGS)
    labels = [f"Smoking (per {smoke_sd:.1f}%)", "Gender (Male)", "Interaction"]
    axes = az.plot_forest(idata, var_names=["beta_smoke", "beta_men", "beta_interaction"], combined=True)
    axes[0].set_yticklabels(labels[::-1])
    plt.savefig(GP_FIGS / "gp_forest_plot.png", dpi=300); plt.close()

    # ... [Mapping code for the 6 PNGs] ...