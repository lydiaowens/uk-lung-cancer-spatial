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
REPORTS_DIR = BASE_DIR / "reports/car_reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def calculate_morans_i(residuals, A):
    """Calculates Global Moran's I using provided adjacency matrix."""
    n = len(residuals)
    z = residuals - np.mean(residuals)
    sum_w = np.sum(A)
    num = n * (z.T @ A @ z)
    den = sum_w * (z.T @ z)
    return num / den

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_nc", type=str, required=True)
    parser.add_argument("--areas", type=str, default=str(BASE_DIR / "data/processed/areas.parquet"))
    parser.add_argument("--adj_path", type=str, default=str(BASE_DIR / "data/processed/inputs_car_population.npz"))
    args = parser.parse_args()

    # 1. Load Data
    input_path = Path(args.input_nc)
    version = input_path.stem.replace("idata_car_", "")
    idata = az.from_netcdf(args.input_nc)
    
    # Ensure areas are sorted to match the Adjacency Matrix
    areas = pd.read_parquet(args.areas)
    if 'code' in areas.columns:
        areas = areas.sort_values('code').reset_index(drop=True)
    gdf = gpd.GeoDataFrame(areas, geometry=areas['geometry'].apply(lambda x: wkb.loads(x) if isinstance(x, bytes) else x))

    # 2. Setup Output Directories
    output_pdf = REPORTS_DIR / f"report_{input_path.stem}.pdf"
    FIGURES_DIR = REPORTS_DIR / f"figures/car_figures_{version}"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 3. PERFORMANCE METRICS (WAIC + LOO-CV)
    var_names = [v for v in ["b0", "beta_smoke", "beta_men", "beta_interaction", "sigma", "rho", "alpha"] if v in idata.posterior]
    stats = az.summary(idata, var_names=var_names)
    
    waic = az.waic(idata, scale="deviance")
    loo = az.loo(idata, scale="deviance")

    # 4. MORAN'S I CALCULATION (Explicit Slice & Average)
    print(f"🌍 Running Slice-and-Average Moran's I for {version.upper()}...")
    adj_data = np.load(args.adj_path)
    A = adj_data['A']
    Pop = adj_data['E']  # 318 geographic population counts

    y_obs = idata.observed_data.obs.values
    rr_mean = idata.posterior["rr"].mean(axis=(0, 1)).values
    n_obs = len(y_obs)

    # Magnitude Correction (Global Rate scaling)
    global_rate = np.sum(y_obs) / (np.sum(Pop) * (2 if n_obs == 636 else 1))
    E_true = Pop * global_rate

    if n_obs == 636:
        # Explicitly slice the 636 vectors into two 318 geographic layers
        y_m, y_f = y_obs[:318], y_obs[318:]
        rr_m, rr_f = rr_mean[:318], rr_mean[318:]
        
        # Calculate residuals per gender layer
        res_m = y_m - (E_true * rr_m)
        res_f = y_f - (E_true * rr_f)
        
        # Average residuals back to the 318 geographic LADs
        res_geog = (res_m + res_f) / 2
    else:
        # Non-stratified (V3) logic
        res_geog = y_obs - (E_true * rr_mean)
    
    morans_i_val = calculate_morans_i(res_geog, A)

    with PdfPages(output_pdf) as pdf:
        # Dashboard
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        header = f"CAR {version.upper()} MODEL AUDIT\n{'='*65}\n"
        diag = (f"WAIC (Dev): {waic.elpd_waic:.2f} | "
                f"LOO-CV (Dev): {loo.elpd_loo:.2f} | "
                f"Moran's I: {morans_i_val:.4f}\n"
                f"Max R-hat: {stats['r_hat'].max():.3f}\n\n")
        
        ax.text(0.05, 0.95, header + diag + stats.to_string(), family='monospace', fontsize=8, va='top')
        plt.title(f"Statistical Summary: {version}", fontweight='bold')
        pdf.savefig(); plt.close()

        # Posterior Plots
        for var in var_names:
            fig, ax = plt.subplots(figsize=(8, 4))
            az.plot_posterior(idata, var_names=[var], ax=ax, hdi_prob=0.95, round_to=3)
            pdf.savefig(); plt.close()

        # Trace Plots
        az.plot_trace(idata, var_names=var_names, compact=True, combined=True)
        pdf.savefig(bbox_inches='tight'); plt.close()

        # 5. SHARED GEOGRAPHIC RISK MAPPING
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

            gdf.plot(column=metric, cmap=cmap, norm=norm, legend=True, ax=ax)
            ax.set_title(f"V{version.upper()}: {label}", fontsize=15, fontweight="bold")
            ax.axis("off")
            pdf.savefig(); plt.close()

    print(f"✅ Clean report with explicit slicing generated: {output_pdf}")

if __name__ == "__main__":
    main()