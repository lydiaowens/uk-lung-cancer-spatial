import arviz as az
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
import pickle
import argparse
import warnings

# --- PATH CONFIGURATION ---
REPORTS_DIR = Path("/Users/alydiaowens/Projects/uk-lung-cancer-spatial/reports")
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
    parser = argparse.ArgumentParser(description="Generate GP Model Technical Report")
    parser.add_argument("--version", type=str, default="v3", help="Model version (e.g., v2, v3)")
    parser.add_argument("--input_pkl", type=str, help="Path to samples pkl file")
    parser.add_argument("--input_npz", type=str, default="data/processed/inputs_car_population.npz")
    parser.add_argument("--drop_chain_4", action="store_true", help="Manually exclude chain 4")
    args = parser.parse_args()

    # Determine paths dynamically
    input_pkl = args.input_pkl if args.input_pkl else f"outputs/idata_gp_{args.version}.pkl"
    ver_upper = args.version.upper()
    
    print(f"Generating Technical Report for GP {ver_upper}...")
    
    # 1. Load Data
    if not Path(input_pkl).exists():
        print(f"❌ Error: {input_pkl} not found.")
        return

    with open(input_pkl, "rb") as f:
        full_samples = pickle.load(f)
    
    # --- CHAIN SELECTION LOGIC ---
    if args.drop_chain_4:
        print("⚠️ Dropping Chain 4 (Validated Subset mode)")
        clean_samples = {k: v[0:3] for k, v in full_samples.items()}
        config_desc = "3 Chains (Validated Subset)"
    else:
        clean_samples = full_samples
        num_chains = full_samples['b0'].shape[0]
        config_desc = f"{num_chains} Chains (Full Run)"
    
    idata_clean = az.from_dict(posterior=clean_samples)
    
    inputs = np.load(args.input_npz)
    y_obs, E, A = inputs['y'], inputs['E'], inputs['A']
    
    # 2. Statistical Calculations
    stats = az.summary(idata_clean, var_names=["length_scale", "variance", "b0"])
    
    # Residuals and Moran's I
    f_mean = clean_samples['RR'].mean(axis=(0, 1))
    y_pred = E * f_mean * np.exp(stats.loc["b0", "mean"])
    residuals = y_obs - y_pred
    moran_i = calculate_morans_i(residuals, A)

    # Effective Range for Matérn 3/2
    l_mean = stats.loc["length_scale", "mean"]
    eff_range = l_mean * np.sqrt(3)

    # 3. Report Composition
    max_rhat = stats["r_hat"].max()
    
    report_lines = [
        "="*70,
        f"GP MODEL TECHNICAL AUDIT ({ver_upper}) | {Path(input_pkl).name}",
        f"Configuration: {config_desc} | Matérn 3/2 Kernel",
        "="*70,
        "\n1. SAMPLING SPECIFICATIONS",
        f"{'-'*30}",
        f"Warmup Iterations:  2000",
        f"Sample Iterations:  2000",
        f"Total Post-Warmup:  {int(idata_clean.posterior.dims['chain'] * idata_clean.posterior.dims['draw'])}",
        "\n2. MCMC DIAGNOSTICS",
        f"{'-'*30}",
        f"Max R-hat:          {max_rhat:.3f}",
        f"Numerical Status:   {'STABLE' if max_rhat < 1.05 else 'UNSTABLE'}",
    ]
    
    if args.drop_chain_4:
        report_lines.append("Note: Chain 4 excluded due to numerical singularity.")

    report_lines.extend([
        "\n3. PARAMETER ESTIMATES",
        f"{'-'*30}",
        stats[["mean", "sd", "hdi_3%", "hdi_97%", "r_hat", "ess_bulk"]].to_string(),
        "\n4. SPATIAL CHARACTERISTICS",
        f"{'-'*30}",
        f"Mean Length Scale:  {l_mean:.4f}",
        f"Effective Range:    {eff_range:.4f} (Standardized units)",
        f"Spatial Variance:   {stats.loc['variance', 'mean']:.4f}",
        f"Moran's I (Resid):  {moran_i:.4f}",
        "="*70
    ])
    
    report_text = "\n".join(report_lines)

    # 4. Save to PDF (Added missing save/close commands)
    output_pdf = REPORTS_DIR / f"gp_model_report_{args.version}.pdf"
    with PdfPages(output_pdf) as pdf:
        # Page 1: Dashboard
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        ax.text(0.05, 0.95, report_text, family='monospace', fontsize=10, va='top')
        plt.title(f"GP Model {ver_upper}: Statistical Audit", loc='left', pad=20, fontweight='bold')
        pdf.savefig(fig)
        plt.close()

        # Page 2: Trace Plots
        az.plot_trace(idata_clean, var_names=["length_scale", "variance", "b0"])
        plt.gcf().suptitle(f"Posterior Trace Plots ({ver_upper})", fontsize=14)
        plt.tight_layout(pad=3.0)
        pdf.savefig()
        plt.close()

    print(f"✅ GP {ver_upper} report successfully generated at: {output_pdf}")

if __name__ == "__main__":
    main()