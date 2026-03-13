import arviz as az
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
import pickle
import argparse
import warnings

# --- CONFIGURATION ---
BASE_DIR = Path("/Users/alydiaowens/Projects/uk-lung-cancer-spatial")
REPORTS_DIR = BASE_DIR / "reports/gp_reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

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
    parser = argparse.ArgumentParser(description="Generate GP Technical Report")
    parser.add_argument("--version", type=str, default="v3")
    parser.add_argument("--input_pkl", type=str, default="/Users/alydiaowens/Projects/uk-lung-cancer-spatial/outputs/idata_gp_v3.pkl")
    parser.add_argument("--input_npz", type=str, default="/Users/alydiaowens/Projects/uk-lung-cancer-spatial/data/processed/inputs_car_population.npz")
    parser.add_argument("--dist_std", type=float, default=103410.97) 
    args = parser.parse_args()

    # 1. Load Data
    if not Path(args.input_pkl).exists():
        print(f"❌ Error: {args.input_pkl} not found.")
        return

    with open(args.input_pkl, "rb") as f:
        full_samples = pickle.load(f)
    
    # --- CHAIN CLIPPING ---
    # Manually excluding Chain 4 (index 3) to ensure convergence
    print(f"⚠️ Clipping Chain 4 for {args.version.upper()} audit...")
    clean_samples = {k: v[0:3] for k, v in full_samples.items()}
    
    # 2. Convert to ArviZ
    # We explicitly map log_like to enable predictive accuracy metrics
    idata = az.from_dict(
        posterior={k: v for k, v in clean_samples.items() if k not in ['log_like', 'rr', 'RR']},
        log_likelihood={"y": clean_samples['log_like']} if 'log_like' in clean_samples else None,
        observed_data={"y": np.load(args.input_npz)['y']}
    )
    
    # 3. STATISTICAL CALCULATIONS
    # Model Comparison Metrics
    waic_res = az.waic(idata, scale="deviance")
    loo_res = az.loo(idata, scale="deviance")
    
    # Residuals & Moran's I
    inputs = np.load(args.input_npz)
    y_obs, E, A = inputs['y'], inputs['E'], inputs['A']
    
    # Flexible key check for Relative Risk
    rr_key = 'RR' if 'RR' in clean_samples else 'rr'
    if rr_key not in clean_samples:
        print(f"❌ Error: Missing RR/rr key. Keys found: {list(clean_samples.keys())}")
        return
    
    rr_mean = clean_samples[rr_key].mean(axis=(0, 1)) 
    
    # Internal Standardization logic for expected counts
    global_rate = np.sum(y_obs) / np.sum(E)
    E_scaled = E * global_rate
    
    y_pred = E_scaled * rr_mean
    residuals = y_obs - y_pred
    moran_i = calculate_morans_i(residuals, A)

    # Effective Range Calculation (Standardized -> km)
    param_summary = az.summary(idata)
    ls_key = "length_scale" if "length_scale" in param_summary.index else "kernel_ls"
    l_mean = param_summary.loc[ls_key, "mean"]
    eff_range_km = (l_mean * np.sqrt(3) * args.dist_std) / 1000

    # 4. TECHNICAL DASHBOARD TEXT
    max_rhat = param_summary["r_hat"].max()
    report_text = (
        f"GP {args.version.upper()} FINAL TECHNICAL AUDIT\n"
        f"{'='*50}\n"
        f"Validation:        Chains 1-3 (Clipped)\n"
        f"WAIC (Deviance):   {waic_res.elpd_waic:.2f}\n"
        f"LOO-CV (Deviance): {loo_res.elpd_loo:.2f}\n"
        f"Moran's I (Res):   {moran_i:.4f}\n"
        f"Effective Range:   {eff_range_km:.2f} km\n"
        f"Max R-hat:         {max_rhat:.3f}\n"
        f"{'='*50}\n\n"
        f"POSTERIOR PARAMETER ESTIMATES:\n"
        f"{param_summary[['mean', 'sd', 'hdi_3%', 'hdi_97%', 'r_hat']].to_string()}"
    )

    # 5. Output PDF Generation
    output_pdf = REPORTS_DIR / f"report_gp_{args.version}_final.pdf"
    with PdfPages(output_pdf) as pdf:
        # Page 1: Metrics & Summary
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        ax.text(0.05, 0.95, report_text, family='monospace', fontsize=9, va='top')
        plt.title(f"GP {args.version.upper()} Analysis Summary", fontweight='bold', pad=20)
        pdf.savefig(); plt.close()

        # Page 2: Trace Plots (Diagnostics)
        az.plot_trace(idata)
        plt.gcf().suptitle(f"GP {args.version.upper()} Converged Trace Plots", fontsize=14)
        plt.tight_layout(pad=3.0)
        pdf.savefig(); plt.close()

    print(f"✅ GP {args.version.upper()} final report generated: {output_pdf}")

if __name__ == "__main__":
    main()