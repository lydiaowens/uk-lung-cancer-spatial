import jax
import jax.numpy as jnp
import pandas as pd
import numpyro
from numpyro.infer import MCMC, NUTS, init_to_median
import arviz as az
from pathlib import Path
from lung_cancer_spatial.models.car_v4_6 import car_model_v4_6

def run_v4_6():
    BASE_DIR = Path("/Users/alydiaowens/Projects/uk-lung-cancer-spatial")
    df = pd.read_csv(BASE_DIR / "data/processed/inputs_v4_stratified.csv")
    
    # Extract data from the newly built CSV columns
    y = jnp.array(df['Lung_Cancer'].values)
    E = jnp.array(df['All_Causes'].values)
    bsmokecentered = jnp.array(df['bsmokecentered'].values)
    bmen = jnp.array(df['bmen'].values)

    # NUTS Configuration (Maintaining the 3000/3000 split)
    nuts_kernel = NUTS(
        car_model_v4_6, 
        target_accept_prob=0.95, 
        dense_mass=True,
        init_strategy=init_to_median
    )

    mcmc = MCMC(nuts_kernel, num_warmup=3000, num_samples=3000, num_chains=4)
    
    print("🚀 Starting V4.6 (Intrinsic/Centered) Inference...")
    mcmc.run(
        jax.random.PRNGKey(42), 
        y=y, 
        E=E, 
        bsmokecentered=bsmokecentered, 
        bmen=bmen
    )
    
    # Save results to a dedicated V4.6 netCDF file
    idata = az.from_numpyro(mcmc)
    output_path = BASE_DIR / "outputs/idata_car_v4.6.nc"
    idata.to_netcdf(output_path)
    print(f"✅ V4.6 Inference Complete. Data saved to: {output_path}")

if __name__ == "__main__":
    run_v4_6()