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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_pkl", type=str, default="outputs/samples_gp_v4.pkl")
    args = parser.parse_args()

    BASE_DIR = Path("/Users/alydiaowens/Projects/uk-lung-cancer-spatial")
    with open(args.input_pkl, "rb") as f:
        samples = pickle.load(f)
    
    # Convert to ArviZ (4 chains)
    idata = az.from_dict(posterior={k: v.reshape(4, -1, *v.shape[1:]) for k, v in samples.items()})
    
    areas = pd.read_parquet(BASE_DIR / "data/processed/areas.parquet")
    gdf = gpd.GeoDataFrame(areas, geometry=areas['geometry'].apply(lambda x: wkb.loads(x) if isinstance(x, bytes) else x))

    output_pdf = BASE_DIR / "reports/report_gp_v4_audit.pdf"
    
    with PdfPages(output_pdf) as pdf:
        # Diagnostics
        stats = az.summary(idata, var_names=["b0", "beta_smoke", "beta_men", "beta_interaction", "kernel_ls", "kernel_var"])
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        ax.text(0.05, 0.95, f"GP V4 AUDIT\nMax R-hat: {stats['r_hat'].max():.3f}\n\n" + stats.to_string(), family='monospace', fontsize=8, va='top')
        pdf.savefig(); plt.close()

        # Shared Geographic Risk (Centered Logic)
        log_rr = np.log(samples["rr"].reshape(-1, 636))
        # Center the geographic signal by subtracting the global mean
        spatial_deviations = log_rr[:, 0:318] - log_rr.mean()
        shared_rr = np.exp(spatial_deviations)

        gdf["shared_rr"] = shared_rr.mean(axis=0)
        gdf["shared_prob"] = (shared_rr > 1.0).mean(axis=0)

        for metric, cmap, label in [
            ("shared_rr", "coolwarm", "Geographic RR (Centered)"),
            ("shared_prob", "RdYlBu_r", "Exceedance Prob P(RR > 1)")
        ]:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            if "rr" in metric:
                limit = max(abs(1 - gdf[metric].min()), abs(gdf[metric].max() - 1), 0.05)
                norm = mcolors.TwoSlopeNorm(vcenter=1.0, vmin=1-limit, vmax=1+limit)
            else:
                norm = mcolors.Normalize(vmin=0, vmax=1)
            
            gdf.plot(column=metric, cmap=cmap, norm=norm, legend=True, ax=ax)
            ax.set_title(f"GP V4: {label}", fontweight="bold")
            ax.axis("off")
            pdf.savefig(); plt.close()

    print(f"✅ GP Audit Report Generated: {output_pdf}")

if __name__ == "__main__":
    main()