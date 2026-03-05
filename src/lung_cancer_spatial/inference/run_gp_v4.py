import argparse
from pathlib import Path
import jax
import jax.numpy as jnp
import pandas as pd
import numpyro
from numpyro.infer import MCMC, NUTS
import pickle
from lung_cancer_spatial.models.gp_v4 import gp_model

def run_gp_v4(inputs_csv, coords_csv, out_dir, warmup, samples, chains, seed):
    out_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(inputs_csv)
    # Ensure coords match the 318 unique LADs in master order
    df_coords = pd.read_csv(coords_csv).iloc[:318]
    coords = jnp.array(df_coords[['lat', 'lon']].values)

    y = jnp.array(df['Lung_Cancer'].values)
    E = jnp.array(df['All_Causes'].values)
    bs = jnp.array(df['bsmoke'].values)
    bm = jnp.array(df['bmen'].values)
    bi = jnp.array(df['binteraction'].values)

    numpyro.set_host_device_count(chains)
    mcmc = MCMC(NUTS(gp_model, target_accept_prob=0.95), 
                num_warmup=warmup, num_samples=samples, num_chains=chains)
    
    print(f"🚀 Running GP V4 | Samples: {samples} | Chains: {chains}")
    mcmc.run(jax.random.PRNGKey(seed), y=y, E=E, coords=coords, bsmoke=bs, bmen=bm, binteraction=bi)

    # Save as .pkl for consistency with prior GP runs
    out_path = out_dir / "samples_gp_v4.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(mcmc.get_samples(), f)
    print(f"✅ Samples saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, default="data/processed/inputs_v4_stratified.csv")
    parser.add_argument("--coords", type=Path, default="data/processed/lad_coordinates.csv")
    parser.add_argument("--out_dir", type=Path, default="outputs")
    parser.add_argument("--warmup", type=int, default=2000)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_gp_v4(args.inputs, args.coords, args.out_dir, args.warmup, args.samples, args.chains, args.seed)