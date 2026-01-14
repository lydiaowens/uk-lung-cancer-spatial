import os
import pickle
import numpyro
import jax
import sys
import os.path
import numpy as np
# Ensure the parent directory is in the path to import gp_model
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from numpyro.infer import MCMC, NUTS

from gp_model import gp_model
from CAR_model.uk_lung_car import y, E, X
# Before running the model
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_standardized = (X - X_mean) / X_std


# Set device count
numpyro.set_host_device_count(4)

def run_and_save_model():
    output_path = "gp_mcmc_samples.pkl"

    if os.path.exists(output_path):
        print(f"⚠️ {output_path} already exists. Skipping model run.")
        return

    kernel = NUTS(gp_model)
    mcmc = MCMC(kernel, num_samples=100, num_warmup=100, num_chains=4, chain_method="parallel")

    print("🚀 Running GP MCMC...")
    mcmc.run(jax.random.PRNGKey(0), y=y, X=X_standardized, E=E)
    samples = mcmc.get_samples(group_by_chain=True)
    
    print(f"📦 Saving samples to {output_path}...")
    with open(output_path, "wb") as f:
        pickle.dump(samples, f)
    
    print("✅ Done. Sample shape:", {k: v.shape for k, v in samples.items()})
    mcmc.print_summary()

if __name__ == "__main__":
    run_and_save_model()