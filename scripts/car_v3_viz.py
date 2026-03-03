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
FIGURES_DIR = Path("/Users/alydiaowens/Projects/uk-lung-cancer-spatial/reports/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def main():
    idata = az.from_netcdf(INPUT_NC)
    df = pd.read_parquet(AREAS_PARQUET)
    
    # Decode WKB Geometry
    df['geometry'] = df['geometry'].apply(lambda x: wkb.loads(x) if isinstance(x, bytes) else x)
    gdf = gpd.GeoDataFrame(df, geometry='geometry')

    # --- THE CORRECTED CALCULATION ---
    # Derive Relative Risk (RR) from the spatial effect u, ignoring the global b0
    u_samples = idata.posterior["car_effect"].values
    rr_corrected = np.exp(u_samples)
    
    # Assign values to GeoDataFrame
    gdf["mean_rr"] = rr_corrected.mean(axis=(0, 1))
    gdf["exceedance_prob"] = (rr_corrected > 1.0).mean(axis=(0, 1))
    gdf["rr_sd"] = rr_corrected.std(axis=(0, 1))
    
    print(f"Stats -> RR Mean: {gdf['mean_rr'].mean():.4f} | Exceedance Max: {gdf['exceedance_prob'].max():.2f}")

    # --- MAP 1: MEAN RELATIVE RISK (Coolwarm, centered at 1.0) ---
    fig, ax = plt.subplots(figsize=(10, 12))
    vmin, vmax = gdf["mean_rr"].min(), gdf["mean_rr"].max()
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax) if vmin < 1.0 < vmax else None
    
    gdf.plot(column="mean_rr", cmap="coolwarm", norm=norm, legend=True, ax=ax,
             legend_kwds={'label': "Relative Risk (1.0 = UK Average)", 'orientation': "horizontal", 'pad': 0.01})
    ax.set_title("Estimated Relative Risk [exp(u)]", fontsize=16)
    ax.axis("off")
    plt.savefig(FIGURES_DIR / "v3_map_mean_rr.png", dpi=300, bbox_inches='tight')
    plt.close()

    # --- MAP 2: EXCEEDANCE PROBABILITY (YlOrRd, 0 to 1) ---
    fig, ax = plt.subplots(figsize=(10, 12))
    gdf.plot(column="exceedance_prob", cmap="YlOrRd", vmin=0, vmax=1, legend=True, ax=ax,
             legend_kwds={'label': "P(Relative Risk > 1.0)", 'orientation': "horizontal", 'pad': 0.01})
    ax.set_title("Hotspot Probability", fontsize=16)
    ax.axis("off")
    plt.savefig(FIGURES_DIR / "v3_map_exceedance.png", dpi=300, bbox_inches='tight')
    plt.close()

    # --- MAP 3: UNCERTAINTY (Purples) ---
    fig, ax = plt.subplots(figsize=(10, 12))
    gdf.plot(column="rr_sd", cmap="Purples", legend=True, ax=ax,
             legend_kwds={'label': "Posterior SD of Relative Risk", 'orientation': "horizontal", 'pad': 0.01})
    ax.set_title("Model Uncertainty (Standard Deviation)", fontsize=16)
    ax.axis("off")
    plt.savefig(FIGURES_DIR / "v3_map_uncertainty.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"All 3 maps successfully saved to: {FIGURES_DIR}")

if __name__ == "__main__":
    main()