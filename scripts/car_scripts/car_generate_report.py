import arviz as az
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
from pathlib import Path
import argparse

REPORTS_DIR = Path("/Users/alydiaowens/Projects/uk-lung-cancer-spatial/reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def generate_car_report(input_nc, output_name, warmup_val):
    if not Path(input_nc).exists():
        print(f"Error: {input_nc} not found.")
        return
    
    output_pdf = REPORTS_DIR / output_name
    idata = az.from_netcdf(input_nc)
    
    stats = az.summary(idata, var_names=["b0", "sigma", "rho", "alpha"])
    max_rhat = stats["r_hat"].max()
    divergences = int(idata.sample_stats.diverging.sum())
    
    dash = [
        "="*70,
        f"CAR MODEL TECHNICAL AUDIT | {Path(input_nc).name}",
        f"Structure: {idata.posterior.dims['chain']} chains | {warmup_val} warmup",
        "="*70,
        f"{'PARAMETER':<12} | {'MEAN':>8} | {'R-HAT':>8} | {'ESS_BULK':>10}",
        "-" * 70
    ]
    for param in ["rho", "sigma", "alpha", "b0"]:
        row = stats.loc[param]
        dash.append(f"{param:<12} | {row['mean']:>8.3f} | {row['r_hat']:>8.3f} | {int(row['ess_bulk']):>10}")

    dash.append("-" * 70)
    dash.append(f"STABILITY CHECK: {'PASS' if max_rhat < 1.05 and divergences == 0 else 'WARNING'}")
    dash.append(f"MAX R-HAT:       {max_rhat:.3f}")
    dash.append(f"DIVERGENCES:     {divergences}")
    dash.append("="*70)

    with PdfPages(output_pdf) as pdf:
        # Page 1: Dashboard
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        ax.text(0.05, 0.95, "\n".join(dash), family='monospace', fontsize=10, va='top')
        plt.title("MCMC Diagnostic Summary", loc='left', pad=20, fontweight='bold')
        pdf.savefig(fig)
        plt.close()

        # Page 2: Trace Plots (Improved Spacing)
        az.plot_trace(idata, var_names=["alpha", "sigma", "rho"])
        plt.gcf().suptitle("MCMC Chain Traces", fontsize=14)
        plt.tight_layout(pad=3.0) # Added significant padding
        pdf.savefig()
        plt.close()
        
        # Page 3: Posterior Distributions (Improved Spacing)
        az.plot_posterior(idata, var_names=["alpha", "sigma", "rho"])
        plt.gcf().suptitle("Posterior Distributions", fontsize=14)
        plt.tight_layout(pad=3.0) # Added significant padding
        pdf.savefig()
        plt.close()

    print(f"Technical report archived at: {output_pdf}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="outputs/idata_car_population_v3.nc")
    parser.add_argument("--filename", type=str, default="car_model_report_v3.pdf")
    parser.add_argument("--warmup", type=int, default=1500)
    args = parser.parse_args()
    generate_car_report(args.input, args.filename, args.warmup)