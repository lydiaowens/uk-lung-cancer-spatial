import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import arviz as az
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import wkb
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
import pickle
import argparse
import warnings

# --- GLOBAL STYLING ---
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9
})
warnings.filterwarnings("ignore")

BASE_DIR = Path("/Users/alydiaowens/Projects/uk-lung-cancer-spatial")
REPORTS_DIR = BASE_DIR / "reports"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_nc", type=str, required=True)
    parser.add_argument("--metadata", type=str, default=str(BASE_DIR / "data/processed/v4_scaling_metadata.pkl"))
    parser.add_argument("--areas", type=str, default=str(BASE_DIR / "data/processed/areas.parquet"))
    args = parser.parse_args()

    # 1. Load Data
    input_path = Path(args.input_nc)
    version = input_path.stem.replace("idata_car_", "")
    idata = az.from_netcdf(args.input_nc)
    
    areas = pd.read_parquet(args.areas)
    gdf = gpd.GeoDataFrame(areas, geometry=areas['geometry'].apply(lambda x: wkb.loads(x) if isinstance(x, bytes) else x))

    # 2. Setup Output Directories
    output_pdf = REPORTS_DIR / f"report_{input_path.stem}.pdf"
    FIGURES_DIR = REPORTS_DIR / f"figures/car_figures/car_figures_{version}"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    var_names = [v for v in ["b0", "beta_smoke", "beta_men", "beta_interaction", "sigma", "rho", "alpha"] if v in idata.posterior]
    stats = az.summary(idata, var_names=var_names)
    waic = az.waic(idata, scale="deviance")

    with PdfPages(output_pdf) as pdf:
        # Dashboard
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        header = f"CAR {version.upper()} MODEL AUDIT\n{'='*65}\n"
        diag = f"WAIC: {waic.elpd_waic:.2f} | Max R-hat: {stats['r_hat'].max():.3f}\n\n"
        ax.text(0.05, 0.95, header + diag + stats.to_string(), family='monospace', fontsize=8, va='top')
        plt.title(f"Statistical Summary: {version}", fontweight='bold')
        pdf.savefig(); plt.close()

        # Posterior Plots
        for var in var_names:
            fig, ax = plt.subplots(figsize=(8, 4))
            az.plot_posterior(idata, var_names=[var], ax=ax, hdi_prob=0.95, round_to=3)
            pdf.savefig(); plt.close()

        # Trace Plots (FIXED Spacing)
        az.plot_trace(
            idata, 
            var_names=var_names, 
            compact=True, 
            combined=True, 
            backend_kwargs={"constrained_layout": True}
        )
        fig = plt.gcf()
        fig.set_size_inches(12, len(var_names) * 2.5) 
        plt.suptitle(f"MCMC Chain Diagnostics: {version.upper()}", fontsize=16, fontweight='bold', y=1.02)
        pdf.savefig(bbox_inches='tight'); plt.close()

        # 3. SHARED GEOGRAPHIC RISK MAPPING (REWRITTEN)
        # Isolate the log-risk and subtract the global mean to get spatial deviations
        log_rr = np.log(idata.posterior["rr"].values)
        spatial_deviations = log_rr[:, :, 0:318] - log_rr.mean(axis=(0, 1, 2))
        shared_rr = np.exp(spatial_deviations)
        
        gdf["shared_rr"] = shared_rr.mean(axis=(0, 1))
        gdf["shared_prob"] = (shared_rr > 1.0).mean(axis=(0, 1))
        hdi = az.hdi(shared_rr, hdi_prob=0.95)
        gdf["shared_unc"] = hdi[:, 1] - hdi[:, 0]

        for metric, cmap, label in [
            ("shared_rr", "coolwarm", "Geographic Relative Risk (Centered)"),
            ("shared_prob", "RdYlBu_r", "Exceedance Prob P(RR > 1)"),
            ("shared_unc", "Purples", "Uncertainty (95% HDI Width)")
        ]:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            
            if "rr" in metric:
                limit = max(abs(1 - gdf[metric].min()), abs(gdf[metric].max() - 1), 0.05)
                norm = mcolors.TwoSlopeNorm(vcenter=1.0, vmin=1-limit, vmax=1+limit)
            elif "prob" in metric:
                norm = mcolors.Normalize(vmin=0, vmax=1)
            else:
                norm = mcolors.Normalize(vmin=gdf[metric].min(), vmax=gdf[metric].max())

            gdf.plot(column=metric, cmap=cmap, norm=norm, legend=True, ax=ax,
                     legend_kwds={"label": label, "orientation": "horizontal", "pad": 0.02, "shrink": 0.8})
            
            ax.set_title(f"V{version.upper()}: {label}", fontsize=15, fontweight="bold")
            ax.axis("off")
            
            plt.savefig(FIGURES_DIR / f"shared_{metric.split('_')[1]}.png", dpi=300, bbox_inches='tight')
            pdf.savefig(); plt.close()

    print(f"✅ Polished audit finalized: {output_pdf}")

if __name__ == "__main__":
    main()