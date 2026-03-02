import arviz as az
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
from pathlib import Path
import argparse

# --- PATH CONFIGURATION ---
REPORTS_DIR = Path("/Users/alydiaowens/Projects/uk-lung-cancer-spatial/reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def generate_car_report(input_nc, output_name, warmup_val):
    if not Path(input_nc).exists():
        print(f"❌ Error: {input_nc} not found.")
        return
    
    # Define final PDF path
    output_pdf = REPORTS_DIR / output_name
    
    # Load InferenceData
    idata = az.from_netcdf(input_nc)
    
    # Extract Metadata & Stats
    n_chains = idata.posterior.dims['chain']
    n_samples = idata.posterior.dims['draw']
    divergences = int(idata.sample_stats.diverging.sum())
    
    # Summary of global parameters
    stats = az.summary(idata, var_names=["b0", "sigma", "rho", "alpha"])
    max_rhat = stats["r_hat"].max()
    rho_mean = stats.loc["rho", "mean"]
    
    # Construct Dashboard Text
    dash = []
    dash.append("="*70)
    dash.append(f"CAR MODEL EXECUTION REPORT | {Path(input_nc).name}")
    dash.append(f"Structure: {n_chains} chains | {warmup_val} warmup | {n_samples} samples")
    dash.append("="*70)
    dash.append(f"{'PARAMETER':<12} | {'MEAN':>8} | {'R-HAT':>8} | {'ESS_BULK':>10}")
    dash.append("-" * 70)
    
    for param in ["rho", "sigma", "alpha", "b0"]:
        row = stats.loc[param]
        dash.append(f"{param:<12} | {row['mean']:>8.3f} | {row['r_hat']:>8.3f} | {int(row['ess_bulk']):>10}")

    dash.append("-" * 70)
    status = "PASS" if (max_rhat < 1.05 and divergences == 0) else "WARNING"
    dash.append(f"STABILITY CHECK: {status}")
    dash.append(f"MAX R-HAT:       {max_rhat:.3f}")
    dash.append(f"DIVERGENCES:     {divergences}")
    dash.append(f"SPATIAL FRACTION (ρ): {rho_mean:.2%}")
    dash.append("="*70)

    report_text = "\n".join(dash)
    print(report_text)

    # Save to Multi-page PDF
    with PdfPages(output_pdf) as pdf:
        # Page 1: Text Dashboard
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        ax.text(0.05, 0.95, report_text, family='monospace', fontsize=10, va='top')
        plt.title("Bayesian CAR Model Summary", loc='left', pad=20, fontweight='bold')
        pdf.savefig(fig)
        plt.close()

        # Page 2: Trace Plots
        az.plot_trace(idata, var_names=["rho", "sigma", "alpha"])
        plt.gcf().suptitle("MCMC Chain Traces", fontsize=16)
        pdf.savefig()
        plt.close()
        
        # Page 3: Posterior Distributions
        az.plot_posterior(idata, var_names=["rho", "sigma"])
        plt.gcf().suptitle("Posterior Belief Distributions", fontsize=16)
        pdf.savefig()
        plt.close()

    print(f"\n✅ PDF archived at: {output_pdf}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="outputs/idata_car_population_v3.nc")
    parser.add_argument("--filename", type=str, default="car_model_report_v3.pdf")
    parser.add_argument("--warmup", type=int, default=1500)
    args = parser.parse_args()
    
    generate_car_report(args.input, args.filename, args.warmup)