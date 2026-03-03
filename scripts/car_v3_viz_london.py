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

# --- PATH CONFIGURATION ---
INPUT_NC = "outputs/idata_car_population_v3.nc"
AREAS_PARQUET = "data/processed/areas.parquet"
FIGURES_DIR = Path("/Users/alydiaowens/Projects/uk-lung-cancer-spatial/reports/figures/london_zoom")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def main():
    idata = az.from_netcdf(INPUT_NC)
    df = pd.read_parquet(AREAS_PARQUET)
    
    # Decode Geometry
    df['geometry'] = df['geometry'].apply(lambda x: wkb.loads(x) if isinstance(x, bytes) else x)
    gdf = gpd.GeoDataFrame(df, geometry='geometry')

    # --- CALCULATION (Relative Risk Correction) ---
    u_samples = idata.posterior["car_effect"].values
    rr_corrected = np.exp(u_samples)
    
    gdf["mean_rr"] = rr_corrected.mean(axis=(0, 1))
    gdf["exceedance_prob"] = (rr_corrected > 1.0).mean(axis=(0, 1))
    gdf["rr_sd"] = rr_corrected.std(axis=(0, 1))

    # Identify London Boroughs (Codes starting with E09)
    # We use this to set the map limits for the zoom-in
    london_gdf = gdf[gdf['LAD_code'].str.startswith('E09', na=False)]
    minx, miny, maxx, maxy = london_gdf.total_bounds
    
    # Add a small buffer (5-10km) so the edges aren't cut off
    buffer = 5000 
    london_xlim = (minx - buffer, maxx + buffer)
    london_ylim = (miny - buffer, maxy + buffer)

    # --- PLOTTING FUNCTION ---
    def create_plots(column, cmap, title_prefix, filename_prefix, norm=None, vmin=None, vmax=None):
        # 1. Full UK Map
        fig, ax = plt.subplots(figsize=(10, 12))
        gdf.plot(column=column, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, legend=True, ax=ax,
                 legend_kwds={'orientation': "horizontal", 'pad': 0.01})
        ax.set_title(f"UK: {title_prefix}", fontsize=16)
        ax.axis("off")
        plt.savefig(FIGURES_DIR.parent / f"uk_{filename_prefix}.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 2. London Zoom Map
        fig, ax = plt.subplots(figsize=(10, 10))
        gdf.plot(column=column, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, legend=True, ax=ax,
                 edgecolor='black', linewidth=0.2, # Add lines to distinguish boroughs
                 legend_kwds={'orientation': "horizontal", 'pad': 0.02})
        
        ax.set_xlim(london_xlim)
        ax.set_ylim(london_ylim)
        ax.set_title(f"London Detail: {title_prefix}", fontsize=18)
        ax.axis("off")
        plt.savefig(FIGURES_DIR / f"london_{filename_prefix}.png", dpi=300, bbox_inches='tight')
        plt.close()

    # --- EXECUTE MAPS ---
    print("Generating Mean RR Maps...")
    vmin_rr, vmax_rr = gdf["mean_rr"].min(), gdf["mean_rr"].max()
    rr_norm = mcolors.TwoSlopeNorm(vmin=vmin_rr, vcenter=1.0, vmax=vmax_rr) if vmin_rr < 1.0 < vmax_rr else None
    create_plots("mean_rr", "coolwarm", "Relative Risk (exp(u))", "mean_rr", norm=rr_norm)

    print("Generating Exceedance Maps...")
    create_plots("exceedance_prob", "YlOrRd", "Hotspot Probability", "exceedance", vmin=0, vmax=1)

    print("Generating Uncertainty Maps...")
    create_plots("rr_sd", "Purples", "Posterior Std Dev", "uncertainty")

    print(f"Success! National maps in /figures and London crops in {FIGURES_DIR}")

if __name__ == "__main__":
    main()