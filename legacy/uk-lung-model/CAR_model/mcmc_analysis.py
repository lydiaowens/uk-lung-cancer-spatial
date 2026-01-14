# analyze_mcmc_samples.py

import pickle
import arviz as az
import matplotlib.pyplot as plt
import numpy as np

sampforrun = "mcmc_samples200.pkl"  # Path to your saved MCMC samples

# 1. Load saved samples
with open(sampforrun, "rb") as f:
    samples = pickle.load(f)

print("✅ Samples loaded. Shape info:")
for key, value in samples.items():
    print(f"  {key}: {np.shape(value)}")

# 2. Reshape 1D arrays to (chains, draws)
n_chains = 4  # Update if you change your chain config
reshaped_samples = {}
for key, arr in samples.items():
    arr = np.asarray(arr)
    if arr.ndim == 1 and arr.size % n_chains == 0:
        reshaped_samples[key] = arr.reshape(n_chains, -1)
    else:
        reshaped_samples[key] = arr  # assume already okay


# 3. Convert to ArviZ InferenceData (flatten multivariate arrays)
flattened = {}

for k, v in reshaped_samples.items():
    if v.ndim == 2:
        flattened[k] = v  # keep 2D (chains, draws)
    elif v.ndim == 3:
        for i in range(v.shape[2]):
            new_key = f"{k}[{i}]"
            flattened[new_key] = v[:, :, i]

# 4. Filter invalid variables
valid_vars = []
for key, arr in flattened.items():
    arr_flat = arr.flatten()
    if not np.all(np.isfinite(arr_flat)):
        print(f"⚠️ Skipping '{key}' (NaN or Inf values)")
    elif np.std(arr_flat) < 1e-8:
        print(f"⚠️ Skipping '{key}' (near-zero variance)")
    else:
        valid_vars.append(key)

# 5. Convert to ArviZ InferenceData
idata = az.from_dict(posterior={k: flattened[k] for k in valid_vars})
# ...existing code...
# Save InferenceData to NetCDF for later use
az.to_netcdf(idata, "inference_data.nc")
print("✅ InferenceData saved to inference_data.nc")
# ...existing code...

# 6. Print summary statistics
summary = az.summary(idata, hdi_prob=0.95)
print("📊 Inference Summary:\n", summary)

# 7. Define subset of variables to plot (e.g., top-level + a few p[i])
# Pick a subset to plot: top-level + selected spatial values
plot_vars = [v for v in valid_vars if v in ["alpha", "b0", "sigma", "tau"]]
plot_vars += [v for v in valid_vars if v.startswith("car[") and v in [f"car[{i}]" for i in [0, 1, 5]]]
plot_vars += [v for v in valid_vars if v.startswith("p[") and v in [f"p[{i}]" for i in [10, 25, 100]]]

# 8. Trace plots
print("📈 Generating trace plots...")
az.plot_trace(idata, var_names=valid_vars)
plt.tight_layout()
plt.savefig("trace_plots.png", dpi=300)
print("🖼️ Trace plots saved to trace_plots.png")

# 9. Posterior plots
az.plot_posterior(idata, var_names=valid_vars)
plt.tight_layout()
plt.savefig("posterior_plots.png", dpi=300)
print("🖼️ Posterior plots saved to posterior_plots.png")

# 10. Prior vs Posterior comparison
# 8. Prior vs Posterior plots for alpha and b0
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# --- b0 ---
# Prior: Normal(0, 1)
prior_b0 = np.random.normal(0, 1, size=2000)
posterior_b0 = idata.posterior["b0"].values.flatten()
axes[0].hist(prior_b0, bins=50, density=True, alpha=0.5, label="Prior", color="gray")
axes[0].hist(posterior_b0, bins=50, density=True, alpha=0.5, label="Posterior", color="blue")
axes[0].set_title("b0: Prior vs Posterior")
axes[0].legend()

# --- alpha ---
# Prior: Uniform(0.01, 0.95)
prior_alpha = np.random.uniform(0.01, 0.95, size=2000)
posterior_alpha = idata.posterior["alpha"].values.flatten()
axes[1].hist(prior_alpha, bins=50, density=True, alpha=0.5, label="Prior", color="gray")
axes[1].hist(posterior_alpha, bins=50, density=True, alpha=0.5, label="Posterior", color="blue")
axes[1].set_title("alpha: Prior vs Posterior")
axes[1].legend()

plt.tight_layout()
plt.savefig("priorvspost.png", dpi=300)
print("🖼️ Prior vs Posterior plots saved to priorvspost.png")