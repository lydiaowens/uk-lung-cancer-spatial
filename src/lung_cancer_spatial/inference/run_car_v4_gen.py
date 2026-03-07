import argparse
from pathlib import Path
import jax
import jax.numpy as jnp
import pandas as pd
import numpyro
from numpyro.infer import MCMC, NUTS, init_to_median
import arviz as az
import pickle
import importlib

def run_gen():
    parser = argparse.ArgumentParser(description="Universal CAR Model Runner")
    parser.add_argument("--model_ver", type=str, default="v4_8", help="Version suffix (e.g. v4_8)")
    parser.add_argument("--inputs", type=str, default="data/processed/inputs_v4_stratified.csv")
    parser.add_argument("--adj", type=str, default="data/processed/spatial_structure.pkl")
    parser.add_argument("--warmup", type=int, default=2000)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--target_accept", type=float, default=0.95)
    args = parser.parse_args()

    # 1. Dynamic Model Import
    module_name = f"lung_cancer_spatial.models.car_{args.model_ver}"
    model_func_name = f"car_model_{args.model_ver}"
    module = importlib.import_module(module_name)
    model_func = getattr(module, model_func_name)

    # 2. Data Loading
    df = pd.read_csv(args.inputs)
    y = jnp.array(df['Lung_Cancer'].values)
    E = jnp.array(df['All_Causes'].values)
    # Flexible column loading (handles both raw and centered names if they exist)
    bsmoke = jnp.array(df['bsmokecentered'].values) if 'bsmokecentered' in df.columns else jnp.array(df['bsmoke'].values)
    bmen = jnp.array(df['bmen'].values)
    bint = jnp.array(df['binteraction'].values) if 'binteraction' in df.columns else bsmoke * (bmen - 0.5)

    with open(args.adj, "rb") as f:
        adj_data = pickle.load(f)
        A = jnp.array(adj_data['A'])
        alpha_max = 1.0 # Standard bound for scaled Q

    # 3. MCMC Execution
    nuts_kernel = NUTS(model_func, target_accept_prob=args.target_accept, init_strategy=init_to_median)
    mcmc = MCMC(nuts_kernel, num_warmup=args.warmup, num_samples=args.samples, num_chains=args.chains)
    
    print(f"🚀 Running Model {args.model_ver} | {args.warmup}/{args.samples} iterations...")
    mcmc.run(jax.random.PRNGKey(42), y=y, E=E, A=A, alpha_max=alpha_max, bsmoke=bsmoke, bmen=bmen, binteraction=bint)
    
    # 4. Save Output
    idata = az.from_numpyro(mcmc)
    out_path = Path(f"outputs/idata_car_{args.model_ver}.nc")
    idata.to_netcdf(out_path)
    print(f"✅ Results saved to: {out_path}")

if __name__ == "__main__":
    run_gen()