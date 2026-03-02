# inference/run_car.py
from __future__ import annotations

import argparse
from pathlib import Path
import os
import jax
import numpy as np
import numpyro
from numpyro.infer import MCMC, NUTS
import arviz as az

from lung_cancer_spatial.models.car import car_model

def run_car(
    inputs_npz: Path,
    out_nc: Path,
    seed: int = 0,
    num_warmup: int = 1000,
    num_samples: int = 2000,
    num_chains: int = 4,
    overwrite: bool = False,
    save_pkl: bool = False,
):
    out_nc.parent.mkdir(parents=True, exist_ok=True)

    if out_nc.exists() and not overwrite:
        print(f"⚠️ {out_nc} exists. Use --overwrite to re-run.")
        return

    arrays = np.load(inputs_npz, allow_pickle=True)
    y = jax.numpy.array(arrays["y"])
    E = jax.numpy.array(arrays["E"])
    A = jax.numpy.array(arrays["A"])
    alpha_max = float(arrays["alpha_max"])
    Z = jax.numpy.array(arrays["Z"]) if "Z" in arrays.files else None

    numpyro.set_host_device_count(num_chains)

    kernel = NUTS(car_model, target_accept_prob=0.95)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method="parallel",
        progress_bar=True,
    )

    print("🚀 Running CAR Poisson MCMC...")
    rng = jax.random.PRNGKey(seed)

    if Z is None:
        mcmc.run(rng, y=y, E=E, A=A, alpha_max=alpha_max)
    else:
        mcmc.run(rng, y=y, E=E, A=A, Z=Z, alpha_max=alpha_max)

    # --- TECHNICAL DIAGNOSTICS DASHBOARD ---
    # Convert to InferenceData to extract stats and metadata
    idata = az.from_numpyro(mcmc)
    
    # Extract metadata for the title
    n_chains = idata.posterior.dims['chain']
    n_samples = idata.posterior.dims['draw']
    # Warmup isn't stored in idata by default, so we use the argument passed to the function
    
    # Extract summary for global parameters
    stats = az.summary(idata, var_names=["b0", "sigma", "rho", "alpha"])
    max_rhat = stats["r_hat"].max()
    divergences = int(idata.sample_stats.diverging.sum())
    
    print("\n" + "="*70)
    print(f"MODEL DIAGNOSTICS | {out_nc.name}")
    print(f"Config: {num_chains} chains | {num_warmup} warmup | {num_samples} samples")
    print("="*70)
    
    # Header for the table
    print(f"{'PARAMETER':<12} | {'MEAN':>8} | {'R-HAT':>8} | {'ESS_BULK':>10}")
    print("-" * 70)
    
    for param in ["rho", "sigma", "alpha", "b0"]:
        mean = stats.loc[param, "mean"]
        rhat = stats.loc[param, "r_hat"]
        ess  = stats.loc[param, "ess_bulk"]
        print(f"{param:<12} | {mean:>8.3f} | {rhat:>8.3f} | {int(ess):>10}")

    print("-" * 70)
    
    # Stability and Divergence Checks
    status = "PASS" if (max_rhat < 1.05 and divergences == 0) else "WARNING"
    print(f"STABILITY CHECK: {status}")
    print(f"MAX R-HAT:       {max_rhat:.3f}")
    print(f"DIVERGENCES:     {divergences}")
    
    # Calculate Spatial Contribution
    rho_mean = stats.loc["rho", "mean"]
    print(f"SPATIAL FRACTION (ρ): {rho_mean:.2%} of variance is spatially structured.")
    print("="*70 + "\n")

    print(f"💾 Saving ArviZ NetCDF → {out_nc}")
    idata.to_netcdf(out_nc)

    if save_pkl:
        import pickle
        out_pkl = out_nc.with_suffix(".pkl")
        samples = mcmc.get_samples(group_by_chain=True)
        with open(out_pkl, "wb") as f:
            pickle.dump(samples, f)
        print(f"📦 Samples archived → {out_pkl}")

    print("✅ Done.")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs_npz", type=Path, default=Path("data/processed/inputs_car.npz"))
    p.add_argument("--out_nc", type=Path, default=Path("outputs/idata_car.nc"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--samples", type=int, default=2000)
    p.add_argument("--chains", type=int, default=4)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--save_pkl", action="store_true")
    args = p.parse_args()

    run_car(
        inputs_npz=args.inputs_npz,
        out_nc=args.out_nc,
        seed=args.seed,
        num_warmup=args.warmup,
        num_samples=args.samples,
        num_chains=args.chains,
        overwrite=args.overwrite,
        save_pkl=args.save_pkl,
    )

if __name__ == "__main__":
    main()