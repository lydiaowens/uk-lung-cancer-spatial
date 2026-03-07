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

# --- GLOBAL STYLING (Standardized to V3/V4) ---
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9
})

# --- ABSOLUTE PATH CONFIGURATION V4.5 ---
BASE_DIR = Path("/Users/alydiaowens/Projects/uk-lung-cancer-spatial")
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures/car_figures/car_figures_v4.5"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

warnings.filterwarnings("ignore")

def main():
    parser = argparse.ArgumentParser()
    # Updated default to V4.5 output
    parser.add_argument("--input_nc", type=str, default=str(BASE_DIR / "outputs/idata_car_v4.5.nc"))
    parser.add_argument("--metadata", type=str, default=str(BASE_DIR / "data/processed/v4_scaling_metadata.pkl"))
    parser.add_argument("--areas", type=str, default=str(BASE_DIR / "data/processed/areas.parquet"))
    args = parser.parse_args()

    # 1. Load Data
    if not Path(args.input_nc).exists():
        print(f"Error: {args.input_nc} not found.")
        return
        
    idata = az.from_netcdf(args.input_nc)
    
    # Extract MCMC metadata for the header
    n_chains = idata.posterior.dims['chain']
    n_samples = idata.posterior.dims['draw']
    
    with open(args.metadata, "rb") as f:
        meta = pickle.load(f)
        smoke_sd = meta['smoking_std']
    
    # PRESERVED V4 GEOMETRY LOGIC
    areas = pd.read_parquet(args.areas)
    gdf = gpd.GeoDataFrame(areas, geometry=areas['geometry'].apply(lambda x: wkb.loads(x) if isinstance(x, bytes) else x))

    # 2. Extract Diagnostics & Stats
    var_names = ["b0", "beta_smoke", "beta_men", "beta_interaction", "sigma", "rho", "alpha"]
    stats = az.summary(idata, var_names=var_names)
    waic = az.waic(idata, scale="deviance")
    loo = az.loo(idata, scale="deviance")
    divergences = int(idata.sample_stats.diverging.sum())

    # 3. Public Health Translation (Adjusted for V4.5 Centered Interpretation)
    smoke_pct = (np.exp(stats.loc["beta_smoke", "mean"]) - 1) * 100
    # Note: in V4.5, beta_interaction is the gendered modifier of the smoke effect
    int_pct = (np.exp(stats.loc["beta_interaction", "mean"]) - 1) * 100

    # 4. Save Technical PDF Report
    output_pdf = REPORTS_DIR / "car_v4.5_technical_report.pdf"
    with PdfPages(output_pdf) as pdf:
        # Page 1: V4.5 Centered Model Audit
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        header = f"CAR V4.5 CENTERED MODEL AUDIT | {Path(args.input_nc).name}\n{'='*65}\n"
        diagnostics = (f"WAIC (Deviance): {waic.elpd_waic:.2f} | LOO-CV: {loo.elpd_loo:.2f}\n"
                       f"Chains: {n_chains} | Samples: {n_samples} | Divergences: {divergences}\n"
                       f"Max R-hat: {stats['r_hat'].max():.3f}\n\n")
        
        methodology = ("V4.5 SPECIFICATION:\n"
                       "- Effect Coding: Men (+0.5) / Women (-0.5)\n"
                       "- Centered Interaction: Smoke * (Gender - 0.5)\n"
                       "- Variance Anchor: InverseGamma(3.0, 1.0)\n\n")

        translation = (f"PUBLIC HEALTH INTERPRETATION:\n"
                       f"- Population Smoking Effect: {smoke_pct:+.2f}% risk per {smoke_sd:.1f}% increase\n"
                       f"- Gender Modifier (Men): {int_pct:+.2f}% additional risk adjustment\n\n")
        
        ax.text(0.05, 0.95, header + diagnostics + methodology + translation + stats.to_string(), 
                family='monospace', fontsize=9, va='top')
        plt.title("V4.5 Comprehensive Statistical Summary", loc='left', pad=20, fontweight='bold')
        pdf.savefig(); plt.close()

        # Page 2: Forest Plot (V4 Style)
        fig, ax = plt.subplots(figsize=(12, 6))
        # Updated labels to reflect population average (centered) effects
        labels = [f"Pop. Smoking (per {smoke_sd:.1f}%)", "Gender Deviation", "Interaction (Modifier)"]
        az.plot_forest(idata, var_names=["beta_smoke", "beta_men", "beta_interaction"], combined=True, ax=ax)
        ax.set_yticklabels(labels[::-1])
        plt.title("CAR V4.5: Centered Covariate Effects", fontweight='bold')
        plt.tight_layout()
        pdf.savefig(); plt.close()

        # Pages 3-9: Individual Posterior Distributions
        for var in var_names:
            fig, ax = plt.subplots(figsize=(10, 6))
            az.plot_posterior(idata, var_names=[var], ax=ax)
            ax.set_title(f"Posterior Distribution: {var} (V4.5 Centered)", fontsize=14, fontweight='bold')
            plt.tight_layout()
            pdf.savefig(); plt.close()

        # Page 10: Trace Plots
        az.plot_trace(idata, var_names=var_names)
        plt.gcf().suptitle("V4.5 MCMC Chain Traces", fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        pdf.savefig(); plt.close()

    # 5. 6-Map Geospatial Suite (PRESERVED V4 MAPPING LOGIC)
    rr_samples = idata.posterior["rr"].values 
    
    for gender in ["men", "women"]:
        # 0:318 are Men (+0.5 effect), 318:636 are Women (-0.5 effect)
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
            ax.set_title(f"CAR V4.5: {gender.capitalize()} {label}", fontsize=14, fontweight='bold', pad=15)
            ax.axis("off")
            
            # Save PNGs into the V4.5 specific figure folder
            plt.savefig(FIGURES_DIR / f"car_v4.5_{gender}_{metric.split('_')[0]}.png", dpi=300, bbox_inches='tight')
            plt.close()

    print(f"✅ V4.5 Technical report finalized: {output_pdf}")
    print(f"✅ V4.5 Stratified maps finalized in: {FIGURES_DIR}")

if __name__ == "__main__":
    main()