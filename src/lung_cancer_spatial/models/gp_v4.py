import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

def gp_model(y, E, coords, bsmoke, bmen, binteraction):
    # 1. Fixed Effects (Aligned with CAR V4)
    b0 = numpyro.sample('b0', dist.Normal(0, 1))
    beta_smoke = numpyro.sample('beta_smoke', dist.Normal(0, 0.2))
    beta_men = numpyro.sample('beta_men', dist.Normal(0, 0.5))
    beta_interaction = numpyro.sample('beta_interaction', dist.Normal(0, 0.2))

    # 2. GP Kernel Hyperparameters
    length_scale = numpyro.sample('kernel_ls', dist.Gamma(2, 1))
    variance = numpyro.sample('kernel_var', dist.Exponential(1))
    
    # 3. Full Covariance Matrix (V3 style)
    dist_sq = jnp.sum((coords[:, None, :] - coords[None, :, :])**2, axis=-1)
    K = variance * jnp.exp(-dist_sq / (2 * length_scale**2))
    K += jnp.eye(K.shape[0]) * 1e-6 # Jitter for numerical stability
    
    # Spatial effect for 318 geographic districts
    u_spatial = numpyro.sample('u_spatial', dist.MultivariateNormal(jnp.zeros(K.shape[0]), K))
    
    # Broadcast spatial effect to 636 observations (Stacked Men/Women)
    u_stratified = jnp.concatenate([u_spatial, u_spatial])

    # 4. Predictor & Likelihood
    eta = jnp.log(E) + b0 + beta_smoke*bsmoke + beta_men*bmen + beta_interaction*binteraction + u_stratified
    
    # Stability Clip aligned with CAR V4
    expected_deaths = jnp.exp(jnp.clip(eta, a_max=10.0))
    
    numpyro.sample('obs', dist.Poisson(rate=expected_deaths), obs=y)
    numpyro.deterministic('rr', jnp.exp(eta - jnp.log(E)))