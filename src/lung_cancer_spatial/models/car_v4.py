"""
CAR (Conditional AutoRegressive) Model for UK Lung Cancer Spatial Data - V4

This module defines the CAR model used for stratified analysis of lung cancer data.
- Incorporates Gender (Men/Women) and Smoking Prevalence as covariates.
- Uses a BYM2 spatial prior with non-centered reparameterization.

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
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax.scipy.linalg import solve_triangular

def car_model(y, E, A, alpha_max, bsmoke, bmen, binteraction): 
    # 1. Shape Logic
    n_obs = y.shape[0]    # 636
    n_lads = A.shape[0]   # 318
    d = jnp.sum(A, axis=1) # Degree vector for the adjacency matrix
    lambda_max = jnp.max(jnp.linalg.eigvalsh(A))  # Spectral radius for standardization


    # 3. Fixed Effects (New V4 Betas)
    # Informative priors (Log-Relative Risk Scale)
    #Using SD = 0.2 to reflect more realistic effect sizes based on literature and domain knowledge. 
    # This helps regularize the estimates and prevent overfitting, especially given the limited number of districts (318) and the complexity of the spatial model.
    b0 = numpyro.sample('b0', dist.Normal(0, 0.5)) 
    beta_smoke = numpyro.sample('beta_smoke', dist.Normal(0, 0.2))  
    beta_men = numpyro.sample('beta_men', dist.Normal(0, 0.2))
    beta_interaction = numpyro.sample('beta_interaction', dist.Normal(0, 0.2))

    # 4. Spatial Hyperparameters (Matching V3)
    sigma = numpyro.sample('sigma', dist.HalfNormal(1.0))
    rho = numpyro.sample('rho', dist.Beta(2.0, 2.0)) 
    alpha = numpyro.sample('alpha', dist.Uniform(0, alpha_max))

    # 5. Spatial Components (Non-centered & Shared across Genders)
    z_u = numpyro.sample('z_u', dist.Normal(0, 1).expand([n_lads]))
    z_e = numpyro.sample('z_e', dist.Normal(0, 1).expand([n_lads]))

    # --- 3. Standardized Precision Matrix ---
    # We divide alpha by the actual lambda_max of your specific A matrix
    Q = jnp.diag(d) - (alpha / lambda_max) * A
    
    # --- 4. Numerical Stability & Sampling ---
    # 1e-5 jitter is the "sweet spot" for stratified CAR models
    L_Q = jnp.linalg.cholesky(Q + jnp.eye(n_lads) * 1e-5)
    
    # Non-centered reparameterization (The z_u trick)
    u_std_raw = solve_triangular(L_Q, z_u, lower=True, trans='T')
    u_std = u_std_raw - jnp.mean(u_std_raw)  # Centering the spatial effects 
    
    # Combined effect for 318 districts
    u_lads = sigma * (jnp.sqrt(rho) * u_std + jnp.sqrt(1 - rho) * z_e)

    # 6. Stratified Mapping
    # Maps the 318 shared geographic risks to the 636 demographic rows
    lad_indices = jnp.tile(jnp.arange(n_lads), 2)
    u_stratified = u_lads[lad_indices]

    # 7. Linear Predictor (V4 Stratified Equation)
    eta = (jnp.log(E) + b0 + 
           beta_smoke * bsmoke + 
           beta_men * bmen + 
           beta_interaction * binteraction + 
           u_stratified)
    
  # 7. Expected Count Calculation (The Poisson Rate)
    # We exponentiate the log-predictor and clip it at 15.0 (~3.2 million deaths) 
    # to prevent the sampler from crashing on 'Infinity' during warmup.
    expected_deaths = jnp.exp(jnp.clip(eta, a_max=15.0))
    
    # 8. Likelihood
    # The 'rate' argument here is the Poisson parameter lambda
    numpyro.sample('obs', dist.Poisson(rate=expected_deaths), obs=y)  
    
    # Deterministic tracking for mapping
    numpyro.deterministic('rr', jnp.exp(b0 + u_stratified))