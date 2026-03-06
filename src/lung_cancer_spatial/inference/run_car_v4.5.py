import argparse
import jax
import jax.numpy as jnp
import pandas as pd
import numpyro
from numpyro.infer import MCMC, NUTS, init_to_median
import arviz as az
from pathlib import Path
from lung_cancer_spatial.models.car_v4_5 import car_model_v4_5

def run_v4_5():
    out_dir = Path("outputs")
    df = pd.read_csv("data/processed/inputs_v4_stratified.csv")
    
    y = jnp.array(df['Lung_Cancer'].values)
    E = jnp.array(df['All_Causes'].values)
    bsmoke = jnp.array(df['bsmoke'].values)
    bmen = jnp.array(df['bmen'].values)

    nuts_kernel = NUTS(
        car_model_v4_5, 
        target_accept_prob=0.95, 
        dense_mass=True,
        init_strategy=init_to_median
    )

    mcmc = MCMC(nuts_kernel, num_warmup=3000, num_samples=3000, num_chains=4)
    mcmc.run(jax.random.PRNGKey(42), y=y, E=E, bsmoke=bsmoke, bmen=bmen)
    
    idata = az.from_numpyro(mcmc)
    idata.to_netcdf(out_dir / "idata_car_v4.5.nc")
    print("✅ V4.5 Inference Complete.")

if __name__ == "__main__":
    run_v4_5()