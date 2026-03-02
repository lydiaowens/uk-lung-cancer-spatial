"""

CAR (Conditional AutoRegressive) Model for UK Lung Cancer Spatial Data

This module defines the CAR model used for spatial analysis of lung cancer data in the UK.
- A Numpyro-based implementation of the CAR model.
- Helper functions for constructing adjacency matrices and spatial precision matrices.

Author: Lydia Owens 

"""

##Imports 
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax.scipy.linalg import solve_triangular

#Model Definition
def car_model(y, E, A, alpha_max, Z = None): 
    """
    Docstring for car_model
    
    Model: 
    :param y: (n,) observed counts 
    :param E: (n,) expected counts/ offset (positive)
    :param A: (n,n) adjacency matrix
    :param alpha_max: (float) maximum allowed value for alpha (from build_inputs.py)
    :param Z: (n,p) optional covaraite matrix 

    jitter Q_std added for numerical stability 
    """

    # 1. Shape
    n = A.shape[0]

    # 2. Degree matrix 
    d = jnp.sum(A, axis=1) #number of neighbors for each region
    D = jnp.diag(d) #diagonal matrix of neighborhood counts 

    # 3. Parameters
    b0 = numpyro.sample('b0', dist.Normal(0, 1)) #global intercept on the log-risk scale

    #tau = numpyro.sample('tau', dist.Gamma(3, 2)) #precision parameter for spatial effects

    # Using sigma (standard deviation) directly is often more stable than tau (precision)
    sigma = numpyro.sample('sigma', dist.HalfNormal(1.0)) #marginal standard deviation of spatial effects

    alpha = alpha = numpyro.sample('alpha', dist.Uniform(0.0, alpha_max - 1e-4)) #car dependency parameter
    #alpha close to 1 means high spatial correlation
    
    #4. Options for covariates 
    if Z is not None: 
        p = Z.shape[1]
        beta = numpyro.sample('beta', dist.Normal(0.0,1.0).expand([p]))
    else: 
        beta = None

    #CAR Precision Matrix 
    Q_std = D - alpha * A

    #Q_std must be symmetric positive definite, add jitter for numerical stability
    Q_std = Q_std + 1e-5 * jnp.eye(n)

    # We use the Cholesky decomposition of the Precision matrix Q
    # Since Q = L @ L.T, then u_std = inv(L.T) @ z has precision Q
    L = jnp.linalg.cholesky(Q_std)

    # Sample i.i.d. standard normal noise (The "Non-Centered" part)
    z = numpyro.sample("z_u", dist.Normal(0, 1).expand([n]))

    #5. Spatial random effects
    u_std = solve_triangular(L.T, z, lower=False) #u_std ~ MVN(0, Q_std^{-1})
    

    u = numpyro.deterministic("car", sigma * u_std) #Spatial effects on the log-risk scale
    #u captures spatial dependence between neighboring regions (not explained by covariates)

    #6. Log-relative risk (Linear predictor)
    eta = b0 + u #baseline log-risk + spatial effects

     #Add covariate effects if provided
    if Z is not None:
        eta = eta + jnp.dot(Z, beta)

    # --- ADD THIS: Guardrails ---
    # We "clip" the log-rate so it can't exceed safe limits.
    # exp(12) is ~160,000, which is a massive relative risk. 
    # This prevents the computer from seeing 'Infinity'.
    eta = jnp.clip(eta, a_min=-20.0, a_max=12.0)

    #7. Poisson likelihood with offset 
    mu = E * jnp.exp(eta) #mean parameter of Poisson

    #Safety mu if mu hits NaN or Inf due to extreme eta: 
    mu = jnp.where(jnp.isfinite(mu), mu, 1e-6) #replace NaN/Inf with small number
    numpyro.sample("obs", dist.Poisson(mu), obs=y)
    rr = numpyro.deterministic("rr", jnp.exp(eta)) #relative risk (multiplicative increase in risk compared to baseline)



"""
VARIABLE DEFINITIONS (Quick Reference)
-------------------------------------

Inputs
------
y : (n,)
    Observed lung cancer case counts for each spatial region.

E : (n,)
    Expected counts / offset for each region (e.g., based on population
    and baseline rates). Must be positive.

A : (n, n)
    Binary adjacency matrix defining spatial neighborhood structure.
    A[i, j] = 1 if regions i and j are neighbors, 0 otherwise.
    Diagonal elements should be 0.

Z : (n, p), optional
    Covariate matrix. Each column represents a standardized covariate.
    If None, the model is fit without covariates.

Dimensions
----------
n : int
    Number of spatial regions.

p : int
    Number of covariates (only defined if Z is provided).

Adjacency-derived quantities
----------------------------
d : (n,)
    Vector of neighborhood counts, where d[i] is the number of neighbors
    of region i.

D : (n, n)
    Diagonal matrix with D[i, i] = d[i]. Used in construction of the
    CAR precision matrix.

Model parameters
----------------
b0 : float
    Global intercept on the log-risk scale.
    Represents baseline log-relative risk shared across all regions.

tau : float
    Precision parameter controlling the overall variability of the
    spatial random effects.
    Larger tau => smaller spatial variance.

alpha : float
    Spatial dependence parameter for the proper CAR model.
    Values closer to 1 imply stronger spatial smoothing; values closer
    to 0 imply weaker spatial dependence.

beta : (p,), optional
    Regression coefficients for covariates in Z.
    Each beta_j represents the log-relative risk associated with
    covariate j.

Spatial structure
-----------------
Q_std : (n, n)
    Standardized CAR precision matrix:
        Q_std = D - alpha * A + jitter * I
    Used as the precision matrix for the spatial random effects.
    A small diagonal jitter is added for numerical stability.

u_std : (n,)
    Standardized spatial random effects drawn from:
        MVN(0, Q_std^{-1})

sigma : float
    Marginal standard deviation of the spatial random effects.
    Defined as sigma = 1 / sqrt(tau).

u : (n,)
    Spatial random effects on the log-risk scale.
    Captures residual spatial variation not explained by covariates.

Linear predictor and likelihood
-------------------------------
eta : (n,)
    Linear predictor (log-relative risk):
        eta = b0 + u + Z @ beta  (if covariates are included)

mu : (n,)
    Mean parameter of the Poisson likelihood:
        mu = E * exp(eta)

Derived quantities
------------------
rr : (n,)
    Relative risk for each region:
        rr = exp(eta)
    Represents multiplicative risk relative to baseline.
"""

