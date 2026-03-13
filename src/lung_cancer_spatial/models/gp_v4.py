import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

def matern_32_kernel(X1, X2, length_scale, variance):
    """
    Matérn 3/2 Kernel: The spatial standard for epidemiological data.
    More robust against 'jagged' transitions than the Squared Exponential.
    """
    # Compute Euclidean distance with a small epsilon for gradients
    d = jnp.sqrt(jnp.sum((X1[:, None, :] - X2[None, :, :])**2, axis=-1) + 1e-8)
    arg = jnp.sqrt(3.0) * d / length_scale
    return variance * (1.0 + arg) * jnp.exp(-arg)

def gp_model_v4(X, y, E, bsmoke, bmen, binteraction):
    """
    GP V4: Stratified Lung Cancer Model with Centered Covariates.
    X: Standardized coordinates (n_lads x 2)
    y: Observed counts (636)
    E: Expected counts (636)
    """
    n_lads = X.shape[0]
    n_obs = y.shape[0]

    # --- 1. FIXED EFFECTS (REPLICATED FROM CAR V4.8) ---
    # Centered baseline intercept
    b0 = numpyro.sample('b0', dist.Normal(0, 0.5))
    beta_smoke = numpyro.sample('beta_smoke', dist.Normal(0, 0.2))
    beta_men = numpyro.sample('beta_men', dist.Normal(0, 0.2))
    
    # Strict Interaction Prior: The 'Anchor' that saved CAR V4.8
    beta_interaction = numpyro.sample('beta_interaction', dist.Normal(0, 0.03))

    # --- 2. GP HYPERPARAMETERS ---
    # Variance of the spatial field
    kernel_var = numpyro.sample('kernel_var', dist.HalfNormal(0.5))
    
    # Length-scale: Using InverseGamma to prevent 'zero-length' pathologies
    # In standardized space (range -3 to 3), 0.5 to 2.0 is a healthy scale
    kernel_ls = numpyro.sample('kernel_ls', dist.InverseGamma(3.0, 1.0))

    # --- 3. NON-CENTERED REPARAMETERIZATION ---
    # Standard normal latent vector (one per district)
    z = numpyro.sample('z', dist.Normal(0, 1).expand([n_lads]))

    # Compute Kernel and Cholesky Factor
    # We add a 1e-5 jitter (nugget) to ensure the matrix is positive definite
    K = matern_32_kernel(X, X, kernel_ls, kernel_var)
    L = jnp.linalg.cholesky(K + jnp.eye(n_lads) * 1e-5)

    # Transform standard normal 'z' into the spatial field 'f'
    # f = L * z ensures the sampler sees a flat geometry
    f_lads = jnp.matmul(L, z)

    # --- 4. STRATIFIED MAPPING ---
    # Map the 318 geographic GP values to the 636 gender rows
    lad_indices = jnp.tile(jnp.arange(n_lads), 2)
    f_stratified = f_lads[lad_indices]

    # --- 5. LINEAR PREDICTOR ---
    eta = (jnp.log(E) + b0 + 
           beta_smoke * bsmoke + 
           beta_men * bmen + 
           beta_interaction * binteraction + 
           f_stratified)
    
    # Poisson Likelihood with clipping to prevent NaNs in early warmup
    expected_rate = jnp.exp(jnp.clip(eta, a_max=15.0))
    numpyro.sample('obs', dist.Poisson(rate=expected_rate), obs=y)

    # Deterministic for RR reporting
    numpyro.deterministic('rr', jnp.exp(eta - jnp.log(E)))

    numpyro.deterministic('log_like', dist.Poisson(expected_rate).log_prob(y))