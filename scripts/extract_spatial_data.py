import numpy as np
import pickle
from pathlib import Path

def extract_spatial_data():
    # Use your specific V3 filename
    v3_path = Path("data/processed/inputs_car_population.npz")
    
    if not v3_path.exists():
        print(f"❌ Error: {v3_path} not found.")
        return

    arrays = np.load(v3_path, allow_pickle=True)
    
    # Extract keys
    A = arrays['A']
    alpha_max = float(arrays['alpha_max'])
    lad_order = arrays['LAD_code'].tolist() # This is the "Master Order"

    # Save as a single spatial dictionary
    spatial_data = {
        'A': A,
        'alpha_max': alpha_max,
        'lad_order': lad_order
    }

    output_path = "data/processed/spatial_structure.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(spatial_data, f)

    print(f"✅ Spatial structure saved to {output_path}")
    print(f"Number of districts: {len(lad_order)}")

if __name__ == "__main__":
    extract_spatial_data()