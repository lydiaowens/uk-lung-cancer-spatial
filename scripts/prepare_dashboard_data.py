import pandas as pd
import geopandas as gpd
import xarray as xr
import pickle
import numpy as np
import os

def prepare_dashboard_data():
    # --- 1. FILE PATHS (UPDATED) ---
    CAR_PATH = '/Users/alydiaowens/Projects/uk-lung-cancer-spatial/outputs/idata_car_v4_8.nc'
    GP_PATH = '/Users/alydiaowens/Projects/uk-lung-cancer-spatial/outputs/samples_gp_v4.pkl'
    
    # Updated to your specific raw data directory
    DATA_DIR = '/Users/alydiaowens/Projects/uk-lung-cancer-spatial/data/raw/Local_Authority_Districts_December_2023_Boundaries_UK_BFC_9042356933902664268'
    SHP_FILE = 'LAD_DEC_2023_UK_BFC.shp' # Ensure this filename matches the one inside your folder
    SHP_PATH = os.path.join(DATA_DIR, SHP_FILE)
    
    OUTPUT_PATH = '/Users/alydiaowens/Projects/uk-lung-cancer-spatial/dashboard_data.geojson'

    print("--- Loading CAR Model (NetCDF) ---")
    with xr.open_dataset(CAR_PATH, group='posterior') as ds_car:
        # Note: Using 'rr'—verify if this is the correct variable name in your NetCDF
        rr_car_samples = ds_car['rr']
        rr_car_mean = rr_car_samples.mean(dim=['chain', 'draw']).values
        ep_car = (rr_car_samples > 1.0).mean(dim=['chain', 'draw']).values
        sd_car = rr_car_samples.std(dim=['chain', 'draw']).values

    print("--- Loading GP Model (Pickle) ---")
    with open(GP_PATH, 'rb') as f:
        gp_samples = pickle.load(f)
        rr_gp_samples = gp_samples['rr']
        rr_gp_mean = np.mean(rr_gp_samples, axis=0)
        ep_gp = np.mean(rr_gp_samples > 1.0, axis=0)
        sd_gp = np.std(rr_gp_samples, axis=0)

    # --- 2. CONSOLIDATE RESULTS ---
    # Helper to handle stratified vs aggregate dimensions
    def collapse_strata(arr):
        if len(arr) == 636:
            return (arr[:318] + arr[318:]) / 2
        return arr

    df_results = pd.DataFrame({
        'rr_car': collapse_strata(rr_car_mean),
        'ep_car': collapse_strata(ep_car),
        'sd_car': collapse_strata(sd_car),
        'rr_gp': collapse_strata(rr_gp_mean),
        'ep_gp': collapse_strata(ep_gp),
        'sd_gp': collapse_strata(sd_gp)
    })

    # --- 3. GEOSPATIAL MERGE ---
    print(f"--- Loading Shapefile from: {SHP_PATH} ---")
    if not os.path.exists(SHP_PATH):
        print(f"❌ ERROR: Shapefile not found at {SHP_PATH}. Check the filename.")
        return

    uk_lads = gpd.read_file(SHP_PATH)
    
    # Merging based on index assuming 0-317 order matches
    # If the merge is messy, we should switch to: uk_lads.merge(df_results, on='LAD23CD')
    merged_gdf = uk_lads.merge(df_results, left_index=True, right_index=True)

    # --- 4. OPTIMIZE FOR WEB ---
    print("--- Optimizing for Web Deployment ---")
    merged_gdf = merged_gdf.to_crs(epsg=4326) # Required for Leafmap/Folium
    merged_gdf['geometry'] = merged_gdf.simplify(0.002, preserve_topology=True)
    
    # Subset columns to keep GeoJSON size small for GitHub
    essential_cols = ['LAD23NM', 'geometry', 'rr_car', 'ep_car', 'sd_car', 'rr_gp', 'ep_gp', 'sd_gp']
    merged_gdf[essential_cols].to_file(OUTPUT_PATH, driver='GeoJSON')
    
    print(f"✅ Success! Dashboard GeoJSON saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    prepare_dashboard_data()