# run_model.py

import os
import pickle
import numpyro
import jax
from numpyro.infer import MCMC, NUTS
from uk_lung_car import car_model, y, A, E

# Set device count
numpyro.set_host_device_count(4)

def run_and_save_model():
    output_path = "mcmc_samples200.pkl"

    if os.path.exists(output_path):
        print(f"⚠️ {output_path} already exists. Skipping model run.")
        return

    kernel = NUTS(car_model)
    mcmc = MCMC(kernel, num_samples=200, num_warmup=200, num_chains=4, chain_method="parallel")
    

    print("🚀 Running MCMC...")
    mcmc.run(jax.random.PRNGKey(0), y=y, A=A, E= E)
    samples = mcmc.get_samples(group_by_chain=True)
    
    print(f"📦 Saving samples to {output_path}...")
    with open(output_path, "wb") as f:
        pickle.dump(samples, f)
    
    print("✅ Done. Sample shape:", samples['alpha'].shape)
    mcmc.print_summary()

# Only run this when executed directly
if __name__ == "__main__":
    run_and_save_model()
