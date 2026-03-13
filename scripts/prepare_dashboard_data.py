import pandas as pd
import geopandas as gpd
import xarray as xr
import pickle
import numpy as np
import os
import arviz as az

def prepare_dashboard_data():
    # --- 1. FILE PATHS ---
    CAR_PATH = '/Users/alydiaowens/Projects/uk-lung-cancer-spatial/outputs/idata_car_v4_8.nc'
    GP_PATH = '/Users/alydiaowens/Projects/uk-lung-cancer-spatial/outputs/samples_gp_v4.pkl'
    DATA_DIR = '/Users/alydiaowens/Projects/uk-lung-cancer-spatial/data/raw/Local_Authority_Districts_December_2023_Boundaries_UK_BFC_9042356933902664268'
    SHP_FILE = 'LAD_DEC_2023_UK_BFC.shp' 
    SHP_PATH = os.path.join(DATA_DIR, SHP_FILE)
    OUTPUT_PATH = '/Users/alydiaowens/Projects/uk-lung-cancer-spatial/dashboard_data.geojson'

    # --- 2. CAR V4.8: Geographic Centering Logic ---
    print("--- Processing CAR Model (Geographic RR Centering) ---")
    idata_car = az.from_netcdf(CAR_PATH)
    # Exactly as in your reference code:
    log_rr_car = np.log(idata_car.posterior["rr"].values)
    mean_all_car = log_rr_car.mean(axis=(0, 1, 2))
    
    # Stratified (636) Logic: Select the first 318 (Male) to isolate geographic signal
    # This centers the male risk against the global mean, unmasking the paradox
    spatial_dev_car = log_rr_car[:, :, 0:318] - mean_all_car
    rr_car_samples = np.exp(spatial_dev_car)
    
    rr_car_mean = rr_car_samples.mean(axis=(0, 1))
    ep_car = (rr_car_samples > 1.0).mean(axis=(0, 1))
    sd_car = rr_car_samples.std(axis=(0, 1))

    # --- 3. GP V4.0: Geographic Centering Logic ---
    print("--- Processing GP Model (Geographic RR Centering) ---")
    with open(GP_PATH, "rb") as f:
        samples_gp = pickle.load(f)
    
    # Reshape if necessary (4 chains)
    rr_gp_raw = samples_gp['rr']
    log_rr_gp = np.log(rr_gp_raw)
    mean_all_gp = np.mean(log_rr_gp)
    
    # Handle stratification (636) for GP similarly
    if log_rr_gp.shape[-1] == 636:
        # Use first 318 for geographic consistency with CAR
        spatial_dev_gp = log_rr_gp[..., 0:318] - mean_all_gp
    else:
        spatial_dev_gp = log_rr_gp - mean_all_gp
        
    rr_gp_samples = np.exp(spatial_dev_gp)
    rr_gp_mean = np.mean(rr_gp_samples, axis=tuple(range(rr_gp_samples.ndim - 1)))
    ep_gp = np.mean(rr_gp_samples > 1.0, axis=tuple(range(rr_gp_samples.ndim - 1)))
    sd_gp = np.std(rr_gp_samples, axis=tuple(range(rr_gp_samples.ndim - 1)))

    # --- 4. VALIDATION ---
    print("\n📊 GEOGRAPHIC RISK VALIDATION (Centered at 1.0)")
    print(f"CAR Geographic RR Range: {rr_car_mean.min():.2f} - {rr_car_mean.max():.2f}")
    print(f"GP Geographic RR Range:  {rr_gp_mean.min():.2f} - {rr_gp_mean.max():.2f}")

    # --- 5. MERGE & SAVE ---
    df_results = pd.DataFrame({
        'rr_car': rr_car_mean, 'ep_car': ep_car, 'sd_car': sd_car,
        'rr_gp': rr_gp_mean, 'ep_gp': ep_gp, 'sd_gp': sd_gp
    })
    
    uk_lads = gpd.read_file(SHP_PATH)
    merged_gdf = uk_lads.merge(df_results, left_index=True, right_index=True)
    merged_gdf = merged_gdf.to_crs(epsg=4326)
    merged_gdf['geometry'] = merged_gdf.simplify(0.002)
    
    essential_cols = ['LAD23NM', 'geometry', 'rr_car', 'ep_car', 'sd_car', 'rr_gp', 'ep_gp', 'sd_gp']
    merged_gdf[essential_cols].to_file(OUTPUT_PATH, driver='GeoJSON')
    print(f"✅ Success! Data synced to reference report logic.")

if __name__ == "__main__":
    prepare_dashboard_data()