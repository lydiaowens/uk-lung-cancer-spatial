import argparse
from pathlib import Path
import jax
import jax.numpy as jnp
import pandas as pd
import numpyro
from numpyro.infer import MCMC, NUTS, init_to_median
import arviz as az
import pickle

# Import the V4 model from your models directory
from lung_cancer_spatial.models.car_v4 import car_model

def run_car_v4(
    inputs_csv: Path,
    adj_pkl: Path,
    out_dir: Path,
    seed: int = 42,
    num_warmup: int = 2000,
    num_samples: int = 2000,
    num_chains: int = 4,
    target_accept: float = 0.85
):
    # 1. Create output directory
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Load the Stratified Data (636 observations)
    df = pd.read_csv(inputs_csv)
    y = jnp.array(df['Lung_Cancer'].values)
    E = jnp.array(df['All_Causes'].values)
    bsmoke = jnp.array(df['bsmoke'].values)
    bmen = jnp.array(df['bmen'].values)
    binteraction = jnp.array(df['binteraction'].values)

    # 3. Load Adjacency Data (318 nodes)
    with open(adj_pkl, "rb") as f:
        adj_data = pickle.load(f)
        A = jnp.array(adj_data['A'])
        alpha_max = float(adj_data['alpha_max'])

    # 4. Configure MCMC
    # Set host device count to num_chains to run in parallel
    numpyro.set_host_device_count(num_chains)
    kernel = NUTS(car_model, 
                  target_accept_prob=target_accept,
                    max_tree_depth=8, dense_mass=True,
                      init_strategy=init_to_median)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=True
    )

    # 5. Run Inference
    print(f"🚀 Running CAR V4 | Warmup: {num_warmup} | Samples: {num_samples} | Chains: {num_chains}")
    rng_key = jax.random.PRNGKey(seed)
    mcmc.run(
        rng_key,
        y=y,
        E=E,
        A=A,
        alpha_max=alpha_max,
        bsmoke=bsmoke,
        bmen=bmen,
        binteraction=binteraction
    )

    # 6. Diagnostics & Saving
    mcmc.print_summary()
    
    # Convert to ArviZ for posterior analysis
    # We save only the .nc file to match your project standards
    idata = az.from_numpyro(mcmc)
    out_path = out_dir / "idata_car_v4.nc"
    idata.to_netcdf(out_path)
    
    print(f"✅ Inference Complete. Results saved to: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Stratified CAR V4 Model")
    parser.add_argument("--inputs", type=Path, default="data/processed/inputs_v4_stratified.csv")
    parser.add_argument("--adj", type=Path, default="data/processed/spatial_structure.pkl")
    parser.add_argument("--out_dir", type=Path, required=True) # Now required to be explicit
    parser.add_argument("--warmup", type=int, default=2000)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--target_accept", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    run_car_v4(
        inputs_csv=args.inputs,
        adj_pkl=args.adj,
        out_dir=args.out_dir,
        seed=args.seed,
        num_warmup=args.warmup,
        num_samples=args.samples,
        num_chains=args.chains,
        target_accept=args.target_accept
    )