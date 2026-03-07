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

# --- GLOBAL STYLING (Standardized to V3) ---
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9
})

# --- ABSOLUTE PATH CONFIGURATION ---
BASE_DIR = Path("/Users/alydiaowens/Projects/uk-lung-cancer-spatial")
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures" / "car_figures"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

warnings.filterwarnings("ignore")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_nc", type=str, default=str(BASE_DIR / "outputs/idata_car_v4.nc"))
    parser.add_argument("--metadata", type=str, default=str(BASE_DIR / "data/processed/v4_scaling_metadata.pkl"))
    parser.add_argument("--areas", type=str, default=str(BASE_DIR / "data/processed/areas.parquet"))
    args = parser.parse_args()

    # 1. Load Data
    if not Path(args.input_nc).exists():
        print(f"Error: {args.input_nc} not found.")
        return
        
    idata = az.from_netcdf(args.input_nc)
    with open(args.metadata, "rb") as f:
        meta = pickle.load(f)
        smoke_sd = meta['smoking_std']
    
    areas = pd.read_parquet(args.areas)
    gdf = gpd.GeoDataFrame(areas, geometry=areas['geometry'].apply(lambda x: wkb.loads(x) if isinstance(x, bytes) else x))

    # 2. Extract Diagnostics & Stats
    var_names = ["b0", "beta_smoke", "beta_men", "beta_interaction", "sigma", "rho", "alpha"]
    stats = az.summary(idata, var_names=var_names)
    waic = az.waic(idata, scale="deviance")
    loo = az.loo(idata, scale="deviance")
    divergences = int(idata.sample_stats.diverging.sum())

    # 3. Public Health Translation
    smoke_pct = (np.exp(stats.loc["beta_smoke", "mean"]) - 1) * 100
    int_pct = (np.exp(stats.loc["beta_interaction", "mean"]) - 1) * 100

    # 4. Save Technical PDF Report
    output_pdf = REPORTS_DIR / "car_v4_technical_report_final.pdf"
    with PdfPages(output_pdf) as pdf:
        # Page 1: Technical Summary Dashboard
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        header = f"CAR V4 STRATIFIED MODEL AUDIT | {Path(args.input_nc).name}\n{'='*65}\n"
        diagnostics = (f"WAIC (Deviance): {waic.elpd_waic:.2f} | LOO-CV: {loo.elpd_loo:.2f}\n"
                       f"Divergences: {divergences} | Max R-hat: {stats['r_hat'].max():.3f}\n\n")
        translation = (f"PUBLIC HEALTH INTERPRETATION:\n- Smoking: {smoke_pct:+.2f}% risk per {smoke_sd:.1f}% increase\n"
                       f"- Male Interaction: {int_pct:+.2f}% additional risk\n\n")
        
        ax.text(0.05, 0.95, header + diagnostics + translation + stats.to_string(), 
                family='monospace', fontsize=9, va='top')
        plt.title("V4 Comprehensive Statistical Summary", loc='left', pad=20, fontweight='bold')
        pdf.savefig(); plt.close()

        # Page 2: Forest Plot (Landscape Layout for labels)
        fig, ax = plt.subplots(figsize=(12, 6))
        labels = [f"Smoking (per {smoke_sd:.1f}%)", "Gender (Male)", "Interaction (Smoke*Male)"]
        az.plot_forest(idata, var_names=["beta_smoke", "beta_men", "beta_interaction"], combined=True, ax=ax)
        ax.set_yticklabels(labels[::-1])
        plt.title("CAR V4: Covariate Effects", fontweight='bold')
        plt.tight_layout()
        pdf.savefig(); plt.close()

        # Page 3: Posterior Distributions (Combined with spacing)
        # --- Pages 3-9: Individual Posterior Distributions ---
        # This loop creates a dedicated page for each parameter
        for var in var_names:
            fig, ax = plt.subplots(figsize=(10, 6))
            az.plot_posterior(idata, var_names=[var], ax=ax)
            ax.set_title(f"Posterior Distribution: {var}", fontsize=14, fontweight='bold')
            plt.tight_layout()
            pdf.savefig(); plt.close()

        # Page 4: Trace Plots (Combined onto one sheet)
        az.plot_trace(idata, var_names=var_names)
        plt.gcf().suptitle("MCMC Chain Traces", fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        pdf.savefig(); plt.close()

    # 5. Save Individual Forest Plot PNG
    fig, ax = plt.subplots(figsize=(12, 6))
    az.plot_forest(idata, var_names=["beta_smoke", "beta_men", "beta_interaction"], combined=True, ax=ax)
    ax.set_yticklabels(labels[::-1])
    plt.title("CAR V4: Covariate Effects", fontweight='bold')
    plt.savefig(FIGURES_DIR / "car_forest_plot.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 6. 6-Map Geospatial Suite (Stratified)
    rr_samples = idata.posterior["rr"].values 
    
    for gender in ["men", "women"]:
        start, end = (0, 318) if gender == "men" else (318, 636)
        subset = rr_samples[:, :, start:end]
        
        gdf[f"rr_{gender}"] = subset.mean(axis=(0, 1))
        gdf[f"prob_{gender}"] = (subset > 1.0).mean(axis=(0, 1))
        hdi = az.hdi(subset, hdi_prob=0.95)
        gdf[f"unc_{gender}"] = hdi[:, 1] - hdi[:, 0]

        for metric, cmap, label in [
            (f"rr_{gender}", "coolwarm", "Relative Risk"),
            (f"prob_{gender}", "YlOrRd", "Exceedance Prob P(RR > 1)"),
            (f"unc_{gender}", "Purples", "Uncertainty (95% HDI Width)")
        ]:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            norm = mcolors.TwoSlopeNorm(vcenter=1.0) if "rr" in metric else None
            
            gdf.plot(
                column=metric, 
                cmap=cmap, 
                norm=norm, 
                legend=True, 
                ax=ax,
                legend_kwds={'label': label, 'orientation': "horizontal", 'pad': 0.02, 'shrink': 0.8}
            )
            ax.set_title(f"CAR V4: {gender.capitalize()} {label}", fontsize=14, fontweight='bold', pad=15)
            ax.axis("off")
            
            plt.savefig(FIGURES_DIR / f"car_{gender}_{metric.split('_')[0]}.png", dpi=300, bbox_inches='tight')
            plt.close()

    print(f"✅ Technical report finalized: {output_pdf}")
    print(f"✅ Stratified maps finalized in: {FIGURES_DIR}")

if __name__ == "__main__":
    main()