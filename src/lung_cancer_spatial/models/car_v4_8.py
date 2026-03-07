import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax.scipy.linalg import solve_triangular

def car_model_v4_8(y, E, A, alpha_max, bsmoke, bmen, binteraction): 
    n_lads = A.shape[0]
    d = jnp.sum(A, axis=1)
    lambda_max = jnp.max(jnp.linalg.eigvalsh(A))

    # --- 1. Fixed Effects ---
    b0 = numpyro.sample('b0', dist.Normal(0, 0.5)) 
    beta_smoke = numpyro.sample('beta_smoke', dist.Normal(0, 0.2))  
    beta_men = numpyro.sample('beta_men', dist.Normal(0, 0.2))
    
    # NEW: Strict Interaction Prior to stabilize R-hat
    beta_interaction = numpyro.sample('beta_interaction', dist.Normal(0, 0.03))

    # --- 2. Spatial Hyperparameters (V4 Baseline) ---
    sigma = numpyro.sample('sigma', dist.HalfNormal(1.0))
    rho = numpyro.sample('rho', dist.Beta(2.0, 2.0)) 
    alpha = numpyro.sample('alpha', dist.Uniform(0, alpha_max))

    # --- 3. BYM2 Scaling & Non-centered Reparameterization ---
    z_u = numpyro.sample('z_u', dist.Normal(0, 1).expand([n_lads]))
    z_e = numpyro.sample('z_e', dist.Normal(0, 1).expand([n_lads]))

    Q = jnp.diag(d) - (alpha / lambda_max) * A
    L_Q = jnp.linalg.cholesky(Q + jnp.eye(n_lads) * 1e-5)
    
    # Triangular solve for unit-variance spatial scaling
    u_std_raw = solve_triangular(L_Q, z_u, lower=True, trans='T')
    u_std = u_std_raw - jnp.mean(u_std_raw) 
    
    u_lads = sigma * (jnp.sqrt(rho) * u_std + jnp.sqrt(1 - rho) * z_e)

    # --- 4. Stratified Mapping ---
    lad_indices = jnp.tile(jnp.arange(n_lads), 2)
    u_stratified = u_lads[lad_indices]

    # --- 5. Linear Predictor ---
    eta = (jnp.log(E) + b0 + 
           beta_smoke * bsmoke + 
           beta_men * bmen + 
           beta_interaction * binteraction + 
           u_stratified)
    
    expected_deaths = jnp.exp(jnp.clip(eta, a_max=15.0))
    numpyro.sample('obs', dist.Poisson(rate=expected_deaths), obs=y)  
    
    numpyro.deterministic('rr', jnp.exp(eta - jnp.log(E)))