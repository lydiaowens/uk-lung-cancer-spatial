"""
UK LUNG CANCER SPATIAL: DATA ALIGNMENT AUDITOR
==============================================
Diagnoses whether N=636 data is Tiled or Interleaved.
"""
import numpy as np
import arviz as az
from pathlib import Path

BASE_DIR = Path("/Users/alydiaowens/Projects/uk-lung-cancer-spatial")

def main():
    # 1. Load Data
    idata_path = BASE_DIR / "outputs/idata_car_v4_8.nc"
    inputs_path = BASE_DIR / "data/processed/inputs_car_population.npz"
    
    if not idata_path.exists():
        print(f"❌ Cannot find idata at {idata_path}")
        return

    idata = az.from_netcdf(str(idata_path))
    inputs = np.load(str(inputs_path))

    # 2. Extract Values
    y = idata.observed_data.obs.values
    E = inputs['E']
    
    # Check if rr exists to help calculate a 'quick' residual
    rr_mean = idata.posterior["rr"].mean(axis=(0,1)).values

    print("--- RAW DATA SNAPSHOT ---")
    print(f"First 5 y (Counts):      {y[:5]}")
    print(f"Middle 5 y (Counts):     {y[318:323]}")
    print(f"First 5 E (Expected):   {E[:5]}")
    
    print("\n--- RATIO CHECK (y/E) ---")
    # This should roughly equal the Relative Risk (RR)
    # If the alignment is Tiled, y[0]/E[0] should be reasonable (e.g., 0.5 to 2.0)
    print(f"LAD 0 Tiled Ratio (y[0]/E[0]):      {y[0]/E[0]:.4f}")
    
    # If the alignment is Interleaved, y[1]/E[0] should be reasonable
    if len(y) > 1:
        print(f"LAD 0 Interleaved Ratio (y[1]/E[0]): {y[1]/E[0]:.4f}")

    print("\n--- MAGNITUDE CHECK ---")
    print(f"Max y: {np.max(y)} | Max E: {np.max(E)}")
    if np.max(E) > 10000 and np.max(y) < 1000:
        print("⚠️ ALERT: E looks like Population, not Expected Counts. This will break Moran's I.")

if __name__ == "__main__":
    main()