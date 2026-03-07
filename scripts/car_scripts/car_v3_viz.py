import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors 
import arviz as az
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import wkb
from pathlib import Path

# --- GLOBAL STYLING ---
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 10})

# --- PATH CONFIGURATION ---
INPUT_NC = "outputs/idata_car_population_v3.nc"
AREAS_PARQUET = "data/processed/areas.parquet"
FIGURES_DIR = Path("/Users/alydiaowens/Projects/uk-lung-cancer-spatial/reports/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("--- Starting Standardized Visualization Process ---")
    
    # 1. Load Data
    idata = az.from_netcdf(INPUT_NC)
    df = pd.read_parquet(AREAS_PARQUET)
    
    # 2. Decode Geometry
    df['geometry'] = df['geometry'].apply(lambda x: wkb.loads(x) if isinstance(x, bytes) else x)
    gdf = gpd.GeoDataFrame(df, geometry='geometry')

    # 3. Corrected Relative Risk Calculation (Centered at 1.0)
    u_samples = idata.posterior["car_effect"].values
    rr_corrected = np.exp(u_samples)
    
    gdf["mean_rr"] = rr_corrected.mean(axis=(0, 1))
    gdf["exceedance_prob"] = (rr_corrected > 1.0).mean(axis=(0, 1))
    gdf["rr_sd"] = rr_corrected.std(axis=(0, 1))

    # --- PLOTTING FUNCTION ---
    def save_styled_map(column, cmap, title, label, filename, norm=None, vmin=None, vmax=None):
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
        
        save_path = FIGURES_DIR / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Generated: {filename}")

    # --- EXECUTION ---

    # 1. Relative Risk (Magnitude)
    # Uses TwoSlopeNorm to ensure 1.0 is exactly the middle/white color
    vmin, vmax = gdf["mean_rr"].min(), gdf["mean_rr"].max()
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)
    save_styled_map("mean_rr", "coolwarm", "Estimated Relative Risk", 
                    "Relative Risk (1.0 = National Average)", "v3_rr_final.png", norm=norm)

    # 2. Exceedance Probability (Confidence)
    save_styled_map("exceedance_prob", "YlOrRd", "Hotspot Exceedance Probability", 
                    "P(Relative Risk > 1.0)", "v3_exceedance_final.png", vmin=0, vmax=1)

    # 3. Posterior Standard Deviation (Precision)
    save_styled_map("rr_sd", "Purples", "Model Uncertainty", 
                    "Posterior Standard Deviation", "v3_uncertainty_final.png")

    print(f"\nAll final standardized maps are in: {FIGURES_DIR}")

if __name__ == "__main__":
    main()