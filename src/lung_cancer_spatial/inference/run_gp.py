import os
import sys
import pickle
import argparse
import numpyro
import jax
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from numpyro.infer import MCMC, NUTS

# Ensure the root project directory is in the path
root_dir = Path("/Users/alydiaowens/Projects/uk-lung-cancer-spatial")
sys.path.insert(0, str(root_dir))

from src.lung_cancer_spatial.models.gp import gp_model

def main():
    # --- INTERACTIVE BASH ARGUMENTS ---
    parser = argparse.ArgumentParser(description="Interactive GP Spatial Model Runner")
    parser.add_argument("--version", type=str, default="v3", help="Model version (e.g., v3)")
    parser.add_argument("--warmup", type=int, default=2000, help="Number of warmup samples")
    parser.add_argument("--samples", type=int, default=2000, help="Number of posterior samples")
    parser.add_argument("--chains", type=int, default=4, help="Number of MCMC chains")
    parser.add_argument("--target_accept", type=float, default=0.95, help="Target acceptance rate for NUTS")
    parser.add_argument("--output_name", type=str, help="Custom name for output pkl (optional)")
    args = parser.parse_args()

    # Dynamic Version Branding
    ver_label = args.version.upper()
    
    # 1. DATA LOADING
    input_path = root_dir / "data/processed/inputs_car_population.npz"
    if not input_path.exists():
        print(f"❌ Error: Data file {input_path} not found.")
        return
        
    arrays = np.load(input_path, allow_pickle=True)
    y = jax.numpy.array(arrays["y"])
    E = jax.numpy.array(arrays["E"])

    # 2. COORDINATE EXTRACTION
    areas_path = root_dir / "data/processed/areas.parquet"
    gdf = gpd.read_parquet(areas_path)
    centroids = gdf.geometry.centroid
    X = np.stack([centroids.x, centroids.y], axis=1)
    
    # Standardize X for the Matérn length-scale prior
    X_mean, X_std = X.mean(axis=0), X.std(axis=0)
    X_standardized = jax.numpy.array((X - X_mean) / X_std)

    # 3. MCMC CONFIGURATION
    # Set output path based on version or custom name
    fname = args.output_name if args.output_name else f"idata_gp_{args.version}.pkl"
    output_path = root_dir / f"outputs/{fname}"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Note: Sequential method used to bypass Mac JAX device initialization issues
    kernel = NUTS(gp_model, target_accept_prob=args.target_accept)
    mcmc = MCMC(
        kernel, 
        num_warmup=args.warmup, 
        num_samples=args.samples, 
        num_chains=args.chains,
        chain_method="sequential",
        progress_bar=True
    )

    print(f"🚀 Running GP {ver_label} | Warmup: {args.warmup} | Samples: {args.samples} | Chains: {args.chains}")
    
    mcmc.run(jax.random.PRNGKey(42), y=y, X=X_standardized, E=E)
    
    # 4. SAVE SAMPLES
    samples = mcmc.get_samples(group_by_chain=True)
    with open(output_path, "wb") as f:
        pickle.dump(samples, f)
    
    print(f"✅ GP {ver_label} Complete. Samples saved to {output_path}")

if __name__ == "__main__":
    main()