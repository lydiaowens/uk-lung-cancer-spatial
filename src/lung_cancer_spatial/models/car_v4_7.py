"""
CAR (Conditional AutoRegressive) Model for UK Lung Cancer Spatial Data - V4.7 

This module defines the CAR model used for stratified analysis of lung cancer data.
- Incorporates Gender (Men/Women) and Smoking Prevalence as covariates.
- Uses a BYM2 spatial prior with non-centered reparameterization.

Version 4.7 Updates:
- centering the beta based off of prior runs to try to improve Rhat and ESS for the smoke interaction term.
- removing alpha constraints to see if it improves sampling efficiency.
- using non centered parameterization for the spatial effects to improve sampling efficiency and reduce divergences.

Variable Definitions:
---------------------
y : (n_obs,) array
    Observed lung cancer mortality counts. (Stacked: 318 Men, 318 Women).
E : (n_obs,) array
    Expected mortality counts (Offset).
A : (n_lads, n_lads) matrix
    Adjacency matrix for the 318 UK Local Authority Districts.
alpha_max : float
    Maximum eigenvalues-based bound for the spatial dependence parameter.
bsmoke : (n_obs,) array
    Standardized smoking prevalence per district/gender.
bmen : (n_obs,) array
    Binary indicator: 1 for Men, 0 for Women.
binteraction : (n_obs,) array
    Interaction term (bsmoke * bmen).
b0 : float
    Global intercept (log-baseline risk).
beta_smoke : float
    Effect of smoking on mortality (baseline: Women).
beta_men : float
    Baseline mortality difference for Men vs Women.
beta_interaction : float
    Additional smoking risk specific to Men.
u_lads : (n_lads,) array
    The shared geographic random effect for each district.
"""

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax.scipy.linalg import solve_triangular
import pickle

# Pre-load spatial structure
with open("data/processed/spatial_structure.pkl", "rb") as f:
    spatial_data = pickle.load(f)
A = jnp.array(spatial_data['A'])
d = jnp.sum(A, axis=1)
n_lads = A.shape[0]
lambda_max = jnp.max(jnp.linalg.eigvalsh(A))

def car_model_v4_7(y, E, bsmokecentered, bmen):
    # --- 1. FIXED EFFECTS: THE PINNED INTERCEPT ---
    # Based on V4, V4.5, and V4.6 reports, b0 is consistently -3.09.
    # We "pin" this with a tight prior to force all 4 chains to initialize 
    # in the same posterior basin.
    b0 = numpyro.sample("b0", dist.Normal(0, 1))

    beta_smoke = numpyro.sample("beta_smoke", dist.Normal(0, 0.2))
    beta_men = numpyro.sample("beta_men", dist.Normal(0, 0.2))
    beta_interaction = numpyro.sample("beta_interaction", dist.Normal(0, 0.1))

    sigma = numpyro.sample("sigma", dist.HalfNormal(0.3))
    rho = numpyro.sample("rho", dist.Beta(2,2))

    alpha_fixed = 0.99

   # --- 3. PRECISION MATRIX & NON-CENTERED RE-PARAMETERIZATION ---

    # Proper CAR precision matrix
    Q = jnp.diag(d) - alpha_fixed * A

    # Numerical jitter for stability
    Q_jitter = Q + jnp.eye(n_lads) * 1e-5

    # Cholesky factor
    L_Q = jnp.linalg.cholesky(Q_jitter)

    # --- Standard Normal innovations (Non-centered trick) ---
    z_u = numpyro.sample("z_u", dist.Normal(0, 1).expand([n_lads]))
    z_e = numpyro.sample("z_e", dist.Normal(0, 1).expand([n_lads]))

    # Solve Q^{-1/2} z_u
    u_std_raw = solve_triangular(L_Q, z_u, lower=True, trans="T")

    # Enforce ICAR sum-to-zero constraint
    u_centered = u_std_raw - jnp.mean(u_std_raw)

    # --- Fast spatial scaling (no matrix inverse) ---
    # Compute diag(Q^{-1}) using triangular solves

    I = jnp.eye(n_lads)

    # Solve L_Q * X = I  ->  X = L_Q^{-1}
    L_inv = solve_triangular(L_Q, I, lower=True)

    # diag(Q^{-1}) = row sums of (L^{-1})^2
    diag_Q_inv = jnp.sum(L_inv ** 2, axis=1)

    # scaling so spatial variance ≈ 1
    scaling_factor = jnp.sqrt(jnp.mean(diag_Q_inv))

    u_std = u_centered / scaling_factor

    # --- BYM2 mixing of spatial + iid noise ---
    u_lads = sigma * (jnp.sqrt(rho) * u_std + jnp.sqrt(1 - rho) * z_e)

    # Stratify: map 318 LAD spatial risks to 636 demographic rows
    u_stratified = jnp.concatenate([u_lads, u_lads])

    # --- 4. CENTERED INTERACTION LOGIC ---
    # Internal shift ensures bmen_centered is strictly -0.5/0.5
    bmen_centered = bmen - 0.5
    bint = bsmokecentered * bmen_centered
    
    # --- 5. LINEAR PREDICTOR ---
    eta = (jnp.log(E) + b0 + 
           beta_smoke * bsmokecentered + 
           beta_men * bmen_centered + 
           beta_interaction * bint + 
           u_stratified)
    
    expected_deaths = jnp.exp(jnp.clip(eta, a_max=15.0))
    numpyro.sample('obs', dist.Poisson(rate=expected_deaths), obs=y)
    
    # Deterministic tracking for mapping (V4 Style)
    numpyro.deterministic('rr', jnp.exp(eta - jnp.log(E)))