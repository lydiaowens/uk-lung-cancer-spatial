import argparse
import os
import pickle
import jax
import jax.numpy as jnp
import pandas as pd
import geopandas as gpd
import numpyro
from numpyro.infer import MCMC, NUTS, init_to_median
from pathlib import Path

# Assuming your new model is in models/gp_v4.py
from lung_cancer_spatial.models.gp_v4 import gp_model_v4

def main():
    parser = argparse.ArgumentParser(description="Universal GP Covariate Runner")
    parser.add_argument("--warmup", type=int, default=2000)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--target_accept", type=float, default=0.98)
    args = parser.parse_args()

    BASE_DIR = Path("/Users/alydiaowens/Projects/uk-lung-cancer-spatial")
    
    # 1. Load Centered Covariates
    df = pd.read_csv(BASE_DIR / "data/processed/inputs_v4_stratified.csv")
    y = jnp.array(df['Lung_Cancer'].values)
    E = jnp.array(df['All_Causes'].values)
    bsmoke = jnp.array(df['bsmokecentered'].values)
    bmen = jnp.array(df['bmen'].values)
    # Re-calculate interaction for safety
    bint = bsmoke * (bmen - 0.5)

    # 2. Extract and Standardize Coordinates
    gdf = gpd.read_parquet(BASE_DIR / "data/processed/areas.parquet")
    centroids = gdf.geometry.centroid
    coords = jnp.stack([centroids.x.values, centroids.y.values], axis=1)
    X = (coords - coords.mean(axis=0)) / coords.std(axis=0)

    # 3. NUTS Configuration (Using high acceptance target for GP stability)
    kernel = NUTS(gp_model_v4, target_accept_prob=args.target_accept, init_strategy=init_to_median)
    mcmc = MCMC(kernel, num_warmup=args.warmup, num_samples=args.samples, num_chains=args.chains)

    print(f"🚀 Launching GP V4 | {args.warmup}/{args.samples} | Target: {args.target_accept}")
    mcmc.run(jax.random.PRNGKey(42), X=X, y=y, E=E, bsmoke=bsmoke, bmen=bmen, binteraction=bint)

    # 4. Save Outputs
    samples = mcmc.get_samples()
    out_path = BASE_DIR / "outputs/samples_gp_v4.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(samples, f)
    
    print(f"✅ GP Inference Complete: {out_path}")

if __name__ == "__main__":
    main()