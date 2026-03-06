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

def car_model_v4_6(y, E, bsmokecentered, bmen):
    # --- 1. Fixed Effects (V4.6 Strict Orthogonality) ---
    b0 = numpyro.sample('b0', dist.Normal(0, 1.0))
    beta_smoke = numpyro.sample('beta_smoke', dist.Normal(0, 0.2))
    beta_men = numpyro.sample('beta_men', dist.Normal(0, 0.2))
    beta_interaction = numpyro.sample('beta_interaction', dist.Normal(0, 0.1))

    # --- 2. Spatial Architecture (V4.6 ICAR Simplification) ---
    # Reverting to HalfNormal(0.5) to smooth out the patchy uncertainty
    sigma = numpyro.sample('sigma', dist.HalfNormal(0.5))
    rho = numpyro.sample('rho', dist.Beta(0.5, 0.5))
    
    # Fixing alpha at 0.99 stabilizes the precision matrix (Intrinsic CAR)
    # This prevents 'alpha' and 'rho' from fighting over the same variance
    alpha_fixed = 0.99 

    # --- 3. Precision Matrix Logic ---
    Q = jnp.diag(d) - (alpha_fixed / lambda_max) * A
    L_Q = jnp.linalg.cholesky(Q + jnp.eye(n_lads) * 1e-5)
    
    z_u = numpyro.sample('z_u', dist.Normal(0, 1).expand([n_lads]))
    z_e = numpyro.sample('z_e', dist.Normal(0, 1).expand([n_lads]))
    
    u_std_raw = solve_triangular(L_Q, z_u, lower=True, trans='T')
    u_std = u_std_raw - jnp.mean(u_std_raw) 
    
    # Combined spatial/noise effect
    u_lads = sigma * (jnp.sqrt(rho) * u_std + jnp.sqrt(1 - rho) * z_e)
    u_stratified = jnp.concatenate([u_lads, u_lads])

    # --- 4. Linear Predictor (Centered Interaction) ---
    # Effect Coding: Shift 0/1 to -0.5/0.5
    bmen_centered = bmen - 0.5
    
    # Calculate interaction internally for perfect centering
    bint = bsmokecentered * bmen_centered
    
    eta = (jnp.log(E) + b0 + 
           beta_smoke * bsmokecentered + 
           beta_men * bmen_centered + 
           beta_interaction * bint + 
           u_stratified)
    
    expected_deaths = jnp.exp(jnp.clip(eta, a_max=15.0))
    numpyro.sample('obs', dist.Poisson(rate=expected_deaths), obs=y)
    
    numpyro.deterministic('rr', jnp.exp(eta - jnp.log(E)))