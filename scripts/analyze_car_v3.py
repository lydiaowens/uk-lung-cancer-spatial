import arviz as az
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
import warnings

# --- PATH CONFIGURATION ---
INPUT_NC = "outputs/idata_car_population_v3.nc"
INPUT_NPZ = "data/processed/inputs_car_population.npz"
REPORTS_DIR = Path("/Users/alydiaowens/Projects/uk-lung-cancer-spatial/reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Suppress terminal warnings for a clean execution
warnings.filterwarnings("ignore")

def calculate_morans_i(residuals, A):
    """Calculates Global Moran's I for model residuals."""
    n = len(residuals)
    z = residuals - np.mean(residuals)
    sum_w = np.sum(A)
    num = n * (z.T @ A @ z)
    den = sum_w * (z.T @ z)
    return num / den

def main():
    print("Generating V3 Full Statistical Report...")
    
    # 1. Load Data
    idata = az.from_netcdf(INPUT_NC)
    inputs = np.load(INPUT_NPZ)
    
    y_obs = inputs['y']
    E = inputs['E'] 
    A = inputs['A']
    
    # 2. Research Metrics Calculation
    # Using elpd_waic and elpd_loo for deviance-scale consistency
    waic = az.waic(idata, scale="deviance")
    loo = az.loo(idata, scale="deviance")
    
    # RMSE Calculation
    rr_samples = idata.posterior["rr"].values
    mu_mean = (rr_samples * E[None, None, :]).mean(axis=(0, 1))
    rmse = np.sqrt(np.mean((y_obs - mu_mean)**2))
    
    # Moran's I on Residuals
    moran_i = calculate_morans_i(y_obs - mu_mean, A)
    
    # 3. Technical Audit Stats
    stats = az.summary(idata, var_names=["b0", "sigma", "rho", "alpha"])
    max_rhat = stats["r_hat"].max()
    divergences = int(idata.sample_stats.diverging.sum())

    # 4. Construct Report Text
    report_lines = [
        "="*70,
        f"CAR MODEL FULL STATISTICAL REPORT | {Path(INPUT_NC).name}",
        f"Execution: {idata.posterior.dims['chain']} chains | {idata.posterior.dims['draw']} samples",
        "="*70,
        "\n1. CONVERGENCE & STABILITY",
        f"{'-'*30}",
        f"Max R-Hat:    {max_rhat:.3f}",
        f"Divergences:  {divergences}",
        f"Stability:    {'PASS' if max_rhat < 1.05 and divergences == 0 else 'WARNING'}",
        "\n2. PARAMETER ESTIMATES (Global)",
        f"{'-'*30}",
        stats[["mean", "sd", "hdi_3%", "hdi_97%", "r_hat", "ess_bulk"]].to_string(),
        "\n3. RESEARCH & COMPARISON METRICS",
        f"{'-'*30}",
        f"WAIC (Deviance):   {waic.elpd_waic:.2f}",
        f"LOO-CV (Deviance): {loo.elpd_loo:.2f}",
        f"RMSE:              {rmse:.4f}",
        f"Moran's I:         {moran_i:.4f} (Residual Autocorrelation)",
        "\n4. SPATIAL INSIGHT",
        f"{'-'*30}",
        f"Spatial Fraction (rho): {stats.loc['rho', 'mean']:.2%}",
        "="*70
    ]
    report_text = "\n".join(report_lines)

    # 5. Save to PDF
    output_pdf = REPORTS_DIR / "car_model_report_v3_full.pdf"
    with PdfPages(output_pdf) as pdf:
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        ax.text(0.05, 0.95, report_text, family='monospace', fontsize=9, va='top')
        plt.title("V3 Comprehensive Statistical Summary", loc='left', pad=20, fontweight='bold')
        pdf.savefig()
        plt.close()
        
        # Add primary parameter distributions for visual confirmation
        az.plot_posterior(idata, var_names=["rho", "sigma", "alpha"])
        plt.tight_layout(pad=3.0)
        pdf.savefig()
        plt.close()

    print(f"Full statistical report saved to: {output_pdf}")

if __name__ == "__main__":
    main()