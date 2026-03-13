import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist

def matern_32_kernel(x1, x2, length_scale=1.0, variance=1.0):
    """
    Matérn 3/2 Kernel: The spatial standard for 'jagged' epidemiological data.
    More robust against overfitting than the Squared Exponential (RBF).
    """
    # Compute Euclidean distance
    d = jnp.sqrt(jnp.sum((x1[:, None, :] - x2[None, :, :])**2, axis=-1) + 1e-8)
    arg = jnp.sqrt(3.0) * d / length_scale
    return variance * (1.0 + arg) * jnp.exp(-arg)

def gp_model(X, y=None, E=None):
    """
    Gaussian Process with Matérn kernel for UK Spatial Data (V2).
    X: standardized coordinates (centroids)
    y: observed counts
    E: expected counts (offset)
    """
    n_regions = X.shape[0]

    # PRIORS: Re-aligned for standardized space (-3 to 3 range)
    # Length-scale: We expect correlation over 0.5 to 2.0 standardized units
    length_scale = numpyro.sample("length_scale", dist.Gamma(3.0, 2.0)) 
    variance = numpyro.sample("variance", dist.HalfNormal(1.0))
    
    # Global intercept: Centered near your CAR results (-7.7)
    b0 = numpyro.sample("b0", dist.Normal(-7.7, 1.0))

    # Kernel Matrix with a small jitter for numerical stability
    K = matern_32_kernel(X, X, length_scale, variance)
    K += 2e-4 * jnp.eye(n_regions) 

    # Latent spatial effect (Non-centered parameterization for better MCMC)
    # This prevents the 'funnel' geometry that causes divergences
    z = numpyro.sample("z", dist.Normal(0, 1).expand([n_regions]))
    L = jnp.linalg.cholesky(K)
    f = jnp.dot(L, z) 

    # Linear Predictor (Log-rate)
    eta = b0 + f
    
    # Likelihood
    mu = E * jnp.exp(eta)
    numpyro.sample("obs", dist.Poisson(mu), obs=y)

    numpyro.deterministic("log_like", dist.Poisson(mu).log_prob(y))

    # Deterministic for the Viz script
    numpyro.deterministic("RR", jnp.exp(f)) # This is the Relative Risk centered at 1.0