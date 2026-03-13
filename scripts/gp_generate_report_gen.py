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

warnings.filterwarnings("ignore")

def calculate_morans_i(residuals, A):
    """Calculates Global Moran's I."""
    n = len(residuals)
    z = residuals - np.mean(residuals)
    sum_w = np.sum(A)
    num = n * (z.T @ A @ z)
    den = sum_w * (z.T @ z)
    return num / den

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_pkl", type=str, default="/Users/alydiaowens/Projects/uk-lung-cancer-spatial/outputs/samples_gp_v4.pkl")
    parser.add_argument("--car_nc", type=str, default="/Users/alydiaowens/Projects/uk-lung-cancer-spatial/outputs/idata_car_v4_8.nc")
    parser.add_argument("--adj_path", type=str, default="/Users/alydiaowens/Projects/uk-lung-cancer-spatial/data/processed/inputs_car_population.npz")
    parser.add_argument("--metadata", type=str, default="/Users/alydiaowens/Projects/uk-lung-cancer-spatial/data/processed/v4_scaling_metadata.pkl")
    args = parser.parse_args()

    BASE_DIR = Path("/Users/alydiaowens/Projects/uk-lung-cancer-spatial")
    REPORTS_DIR = BASE_DIR / "reports/gp_reports"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Samples & Metadata
    with open(args.input_pkl, "rb") as f:
        samples = pickle.load(f)
    
    dist_std = 1.0
    if Path(args.metadata).exists():
        with open(args.metadata, "rb") as f:
            meta = pickle.load(f)
            dist_std = meta.get('dist_std', 1.0)
            print(f"✅ Loaded distance scaling factor: {dist_std:.2f}")

    # 2. FETCH OBSERVATIONS
    if not Path(args.car_nc).exists():
        print(f"❌ Error: {args.car_nc} not found.")
        return
        
    car_idata = az.from_netcdf(args.car_nc)
    y_obs = car_idata.observed_data.obs.values 

    # 3. ArviZ Conversion
    posterior_dict = {k: v.reshape(4, -1, *v.shape[1:]) for k, v in samples.items() 
                     if k not in ['log_like', 'y', 'obs', 'log_likelihood']}
    
    idata = az.from_dict(
        posterior=posterior_dict,
        log_likelihood={"y": samples['log_like'].reshape(4, -1, *samples['log_like'].shape[1:])} if 'log_like' in samples else None,
        observed_data={"y": y_obs}
    )
    
    # 4. IC Metrics & Effective Range
    waic_res = az.waic(idata, scale="deviance")
    loo_res = az.loo(idata, scale="deviance")
    
    # Range = sqrt(3) * length_scale * scaling_factor
    ls_mean = idata.posterior["kernel_ls"].mean().values
    eff_range_km = np.sqrt(3) * ls_mean * dist_std/1000

    # 5. MORAN'S I
    adj_data = np.load(args.adj_path)
    A = adj_data['A']
    Pop = adj_data['E'] 
    
    rr_mean = samples['rr'].mean(axis=0) 
    global_rate = np.sum(y_obs) / (np.sum(Pop) * 2)
    E_true = Pop * global_rate

    y_m, y_f = y_obs[:318], y_obs[318:]
    rr_m, rr_f = rr_mean[:318], rr_mean[318:]
    res_m = y_m - (E_true * rr_m)
    res_f = y_f - (E_true * rr_f)
    res_geog = (res_m + res_f) / 2
    
    morans_i_val = calculate_morans_i(res_geog, A)

    # 6. Geography
    areas = pd.read_parquet(BASE_DIR / "data/processed/areas.parquet")
    if 'code' in areas.columns:
        areas = areas.sort_values('code').reset_index(drop=True)
    gdf = gpd.GeoDataFrame(areas, geometry=areas['geometry'].apply(lambda x: wkb.loads(x) if isinstance(x, bytes) else x))

    output_pdf = REPORTS_DIR / "report_gp_v4_audit.pdf"
    
    with PdfPages(output_pdf) as pdf:
        available_vars = [v for v in ["b0", "beta_smoke", "beta_men", "beta_interaction", "kernel_ls", "kernel_var"] if v in idata.posterior]
        stats = az.summary(idata, var_names=available_vars)
        
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        header = (f"GP V4 AUDIT SUMMARY\n{'='*30}\n"
                  f"WAIC (Deviance): {waic_res.elpd_waic:.2f}\n"
                  f"LOO-CV (Deviance): {loo_res.elpd_loo:.2f}\n"
                  f"Moran's I (Residuals): {morans_i_val:.4f}\n"
                  f"Effective Range: {eff_range_km:.2f} km\n"
                  f"Max R-hat: {stats['r_hat'].max():.3f}\n\n")
        
        ax.text(0.05, 0.95, header + stats.to_string(), family='monospace', fontsize=8, va='top')
        pdf.savefig(); plt.close()

        # Mapping
        log_rr = np.log(samples["rr"].reshape(-1, 636))
        spatial_deviations = log_rr[:, 0:318] - log_rr.mean()
        shared_rr = np.exp(spatial_deviations)
        gdf["shared_rr"] = shared_rr.mean(axis=0)
        gdf["shared_prob"] = (shared_rr > 1.0).mean(axis=0)

        for metric, cmap, label in [("shared_rr", "coolwarm", "Geographic RR"), ("shared_prob", "RdYlBu_r", "P(RR > 1)")]:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            norm = mcolors.TwoSlopeNorm(vcenter=1.0) if "rr" in metric else mcolors.Normalize(0, 1)
            gdf.plot(column=metric, cmap=cmap, norm=norm, legend=True, ax=ax)
            ax.set_title(label, fontweight="bold"); ax.axis("off")
            pdf.savefig(); plt.close()

    print(f"✅ GP Audit finalized with Effective Range: {output_pdf}")

if __name__ == "__main__":
    main()