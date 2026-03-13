import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import arviz as az
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import wkb
from pathlib import Path
import pickle
import warnings

# --- CUSTOM PALETTES ---
# Risk/Exceedance: High-Contrast Blue to Vermillion (Colorblind Safe)
risk_palette = [
    (0.0, "#004974"), (0.2, "#0072B2"), (0.48, "#f7f7f7"), 
    (0.5, "#ffffff"), (0.52, "#f7f7f7"), (0.8, "#D55E00"), (1.0, "#8b3d00")
]
high_contrast_map = mcolors.LinearSegmentedColormap.from_list("HighContrast", risk_palette)

# Uncertainty: Teal-Green Gradient
teal_green_map = mcolors.LinearSegmentedColormap.from_list("TealGreen", ["#f7f7f7", "#008080", "#004d4d"])

# --- CONFIGURATION ---
# Setting Helvetica with a sans-serif fallback
import matplotlib.font_manager as fm

# Force matplotlib to look for Helvetica
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Helvetica Neue", "Arial", "Nimbus Sans", "Liberation Sans"],
    "pdf.fonttype": 42,  # Ensures text is editable/vector in PDFs
    "ps.fonttype": 42,
    "font.size": 14,
    "axes.titlesize": 24,
    "axes.titleweight": "bold"
})

# Optional: Print out if Helvetica was actually found
fonts = [f.name for f in fm.fontManager.ttflist]
if "Helvetica" in fonts:
    print("✅ Helvetica found and loaded.")
else:
    print("⚠️ Helvetica not found. Falling back to Arial/Sans-Serif.")
warnings.filterwarnings("ignore")

BASE_DIR = Path("/Users/alydiaowens/Projects/uk-lung-cancer-spatial")
CAR_PATH = BASE_DIR / "outputs/idata_car_v4_8.nc"
GP_PATH = BASE_DIR / "outputs/samples_gp_v4.pkl"
AREAS_PATH = BASE_DIR / "data/processed/areas.parquet"
POSTER_DIR = BASE_DIR / "enar_poster"
POSTER_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = POSTER_DIR / "poster_comparison_3x2.png"

def get_gdf():
    areas = pd.read_parquet(AREAS_PATH)
    return gpd.GeoDataFrame(areas, geometry=areas['geometry'].apply(lambda x: wkb.loads(x) if isinstance(x, bytes) else x))

def main():
    print(f"🎨 Generating Final Figure 1 for Poster...")
    gdf_car, gdf_gp = get_gdf(), get_gdf()

    # --- 1. CAR V4.8 PROCESSING ---
    idata_car = az.from_netcdf(CAR_PATH)
    rr_car_raw = idata_car.posterior["rr"].values 
    c, d, _ = rr_car_raw.shape
    # Fold to shared LAD risk
    rr_car_folded = rr_car_raw.reshape(c, d, 2, 318).mean(axis=2) 
    log_shared_car = np.log(rr_car_folded)
    centered_log_car = log_shared_car - log_shared_car.mean()
    final_rr_car = np.exp(centered_log_car)

    gdf_car["rr"] = final_rr_car.mean(axis=(0, 1))
    gdf_car["prob"] = (final_rr_car > 1.0).mean(axis=(0, 1))
    hdi_car = az.hdi(final_rr_car, hdi_prob=0.95)
    gdf_car["unc"] = hdi_car[:, 1] - hdi_car[:, 0]

    # --- 2. GP V4 PROCESSING ---
    with open(GP_PATH, "rb") as f:
        samples_gp = pickle.load(f)
    rr_gp_raw = samples_gp["rr"].reshape(-1, 636)[:, 0:318]
    log_rr_gp = np.log(rr_gp_raw)
    centered_log_gp = log_rr_gp - log_rr_gp.mean()
    final_rr_gp = np.exp(centered_log_gp)

    gdf_gp["rr"] = final_rr_gp.mean(axis=0)
    gdf_gp["prob"] = (final_rr_gp > 1.0).mean(axis=0)
    hdi_gp = az.hdi(final_rr_gp, hdi_prob=0.95)
    gdf_gp["unc"] = hdi_gp[:, 1] - hdi_gp[:, 0]

    # --- 3. PLOTTING ---
    fig, axes = plt.subplots(3, 2, figsize=(20, 30))
    
    # Global Title


    rr_limit = max(abs(1 - gdf_car["rr"].min()), abs(gdf_car["rr"].max() - 1),
                   abs(1 - gdf_gp["rr"].min()), abs(gdf_gp["rr"].max() - 1))
    
    rr_norm = mcolors.TwoSlopeNorm(vcenter=1.0, vmin=1-rr_limit, vmax=1+rr_limit)
    prob_norm = mcolors.TwoSlopeNorm(vcenter=0.5, vmin=0.0, vmax=1.0)
    unc_norm = mcolors.Normalize(vmin=0, vmax=1.0)

    metrics = [
        ("rr", high_contrast_map, "Residual Relative Risk (Blue < 1 < Orange)", rr_norm),
        ("prob", high_contrast_map, "Exceedance Probability (Blue < 0.5 < Orange)", prob_norm),
        ("unc", teal_green_map, "Posterior Uncertainty (95% HDI Scale 0-1)", unc_norm)
    ]

    for row, (attr, cmap, label, norm) in enumerate(metrics):
        for col, gdf in enumerate([gdf_car, gdf_gp]):
            gdf.plot(column=attr, cmap=cmap, norm=norm, ax=axes[row, col], legend=True,
                     legend_kwds={'label': label, 'orientation': "horizontal", 'pad': 0.01, 'shrink': 0.8})

    axes[0, 0].set_title("CAR (Discrete Neighborhoods)", pad=20)
    axes[0, 1].set_title("GP (Continuous Risk Surface)", pad=20)
    for ax in axes.flatten(): ax.axis("off")

    plt.subplots_adjust(top=0.92, wspace=0.05, hspace=0.1)
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
    print(f"✅ Figure 1 successfully generated: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()