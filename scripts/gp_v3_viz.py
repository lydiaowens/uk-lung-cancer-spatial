import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pickle
import numpy as np
import pandas as pd
import geopandas as gpd
import argparse
from shapely import wkb
from pathlib import Path

# --- GLOBAL STYLING (Matched to CAR V3) ---
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 10})

# --- PATH CONFIGURATION ---
BASE_DIR = Path("/Users/alydiaowens/Projects/uk-lung-cancer-spatial")
AREAS_PARQUET = BASE_DIR / "data/processed/areas.parquet"
FIGURES_DIR = BASE_DIR / "reports/figures/gp_figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def main():
    parser = argparse.ArgumentParser(description="Generate Standardized GP Spatial Maps")
    parser.add_argument("--version", type=str, default="v3", help="Model version (v3)")
    args = parser.parse_args()

    input_pkl = BASE_DIR / f"outputs/idata_gp_{args.version}.pkl"
    print(f"🎨 Generating Standardized Visualizations for GP {args.version.upper()}...")

    if not input_pkl.exists():
        print(f"❌ Error: {input_pkl} not found.")
        return

    with open(input_pkl, 'rb') as f:
        samples = pickle.load(f)
    
    # rr_samples shape: (total_samples, regions)
    rr_samples = samples['RR'].reshape(-1, samples['RR'].shape[-1])
    
    df = pd.read_parquet(AREAS_PARQUET)
    df['geometry'] = df['geometry'].apply(lambda x: wkb.loads(x) if isinstance(x, bytes) else x)
    gdf = gpd.GeoDataFrame(df, geometry='geometry')

    # --- METRIC CALCULATION ---
    gdf["mean_rr"] = rr_samples.mean(axis=0)
    gdf["exceedance_prob"] = (rr_samples > 1.0).mean(axis=0)
    gdf["rr_sd"] = rr_samples.std(axis=0)

    # --- STANDARDIZED PLOTTING FUNCTION ---
    def save_styled_map(column, cmap, title, label, suffix, norm=None, vmin=None, vmax=None):
        fig, ax = plt.subplots(figsize=(8, 10))
        gdf.plot(
            column=column, 
            cmap=cmap, 
            norm=norm, 
            vmin=vmin, 
            vmax=vmax, 
            legend=True, 
            ax=ax,
            legend_kwds={'label': label, 'orientation': "horizontal", 'pad': 0.02, 'shrink': 0.8}
        )
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax.axis("off")
        
        # Consistent Filename Schema
        out_path = FIGURES_DIR / f"gp_map_{args.version}_{suffix}.png"
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Generated: {out_path.name}")

    # 1. Relative Risk (Magnitude) - Matched to CAR 'coolwarm'
    vmin, vmax = gdf["mean_rr"].min(), gdf["mean_rr"].max()
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)
    save_styled_map("mean_rr", "coolwarm", f"GP {args.version.upper()}: Estimated Relative Risk", 
                    "Relative Risk (1.0 = National Average)", "rr_final", norm=norm)

    # 2. Exceedance Probability (Confidence) - Matched to CAR 'YlOrRd'
    save_styled_map("exceedance_prob", "YlOrRd", f"GP {args.version.upper()}: Hotspot Probability", 
                    "P(Relative Risk > 1.0)", "exceedance_final", vmin=0, vmax=1)

    # 3. Uncertainty (Precision) - Matched to CAR 'Purples'
    save_styled_map("rr_sd", "Purples", f"GP {args.version.upper()}: Model Uncertainty", 
                    "Posterior Standard Deviation", "uncertainty_final")

if __name__ == "__main__":
    main()