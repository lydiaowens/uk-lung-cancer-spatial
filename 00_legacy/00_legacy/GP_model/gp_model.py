import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

numpyro.set_host_device_count(4)  # Set the device count to enable parallel sampling

def squared_exponential_kernel(x1, x2, length_scale=1.0, variance=1.0):
    """Squared Exponential Kernel (RBF Kernel)."""
    sqdist = jnp.sum(x1**2, 1).reshape(-1, 1) + jnp.sum(x2**2, 1) - 2 * jnp.dot(x1, x2.T)
    return variance * jnp.exp(-0.5 / length_scale**2 * sqdist)

def gp_model(X, y=None, E=None):
    """
    Gaussian Process regression with Poisson likelihood.
    X: input features, shape (n_samples, n_features)
    y: observed counts/rates
    E: exposure (e.g., population at risk), or None for pure counts
    """
    length_scale = numpyro.sample("length_scale", dist.LogNormal(np.log(1000.0), 0.5))  # meters
    variance = numpyro.sample("variance", dist.HalfNormal(2.0))
    noise = numpyro.sample("noise", dist.HalfNormal(1.0))


    K = squared_exponential_kernel(X, X, length_scale, variance) + noise**2 * jnp.eye(len(X))
    f = numpyro.sample("f", dist.MultivariateNormal(loc=jnp.zeros(len(X)), covariance_matrix=K))

    mu =  E * jnp.exp(f)
    if y is not None:
        numpyro.sample("obs", dist.Poisson(mu), obs=y)

def run_gp_model(X, y, E=None):
    """Run the Gaussian Process model with Poisson likelihood."""
    kernel = NUTS(lambda X, y, E=None: gp_model(X, y, E))
    mcmc = MCMC(kernel, num_samples=100, num_warmup=100)
    mcmc.run(jax.random.PRNGKey(0), X=X, y=y, E=E)
    return mcmc.get_samples()