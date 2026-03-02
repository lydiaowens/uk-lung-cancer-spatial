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

    # rho: proportion of variance that is spatial (BYM2 parameter)
    # rho = 1 is pure CAR, rho = 0 is pure random noise
    rho = numpyro.sample('rho', dist.Beta(0.5, 0.5))

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
    z_u = numpyro.sample("z_u", dist.Normal(0, 1).expand([n])) # Spatial noise
    z_e = numpyro.sample("z_e", dist.Normal(0, 1).expand([n])) # Unstructured noise

    #5. Spatial random effects
    u_std = solve_triangular(L.T, z_u, lower=False) #u_std ~ MVN(0, Q_std^{-1})
    
    #6. Sum to zero constraint 
    u_std = u_std - jnp.mean(u_std) #enforce sum-to-zero constraint for identifiability

    u = sigma * (jnp.sqrt(rho) * u_std + jnp.sqrt(1 - rho) * z_e) #Spatial effects on the log-risk scale
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

    #Deterministic variables for monitoring
    numpyro.deterministic("car_effect", u_std)
    numpyro.deterministic("rr", jnp.exp(eta))


"""
VARIABLE DEFINITIONS (Updated for BYM2 & Stability)
--------------------------------------------------

Core Inputs
-----------
y : (n,)
    Observed lung cancer mortality counts for each UK district.
E : (n,)
    Expected counts (offset). Calculated in build_inputs.py as 
    (District Population * National Baseline Rate).
A : (n, n)
    Adjacency matrix. A[i,j]=1 if districts share a border.
alpha_max : float
    The reciprocal of the maximum eigenvalue of the scaled adjacency matrix. 
    Ensures the CAR precision matrix remains positive definite.

Global Parameters
-----------------
b0 : float
    The 'Grand Mean' or global intercept. Represents the baseline 
    log-relative risk for the entire UK study area.
sigma : float
    Total marginal standard deviation. Controls the overall 'strength' 
    of the combined random effects (spatial + non-spatial).
rho : float (0 to 1)
    The BYM2 mixing parameter. 
    - Near 1: Most of the unexplained risk is spatially clustered.
    - Near 0: Most of the unexplained risk is random 'noise' (unstructured).

Spatial & Noise Components
--------------------------
alpha : float
    Spatial dependence parameter. Controls the 'smoothness' of the 
    CAR component (u_std).
z_u / z_e : (n,)
    Standard Normal 'innovation' vectors. Used for non-centered 
    re-parameterization to help the MCMC sampler avoid 'funnels.'
u_std : (n,)
    The 'Structured' spatial effect. It has been forced to sum-to-zero 
    and follows the CAR neighborhood dependency logic.
u : (n,)
    The 'Combined' random effect. This is the final result of the 
    BYM2 logic, incorporating both smooth trends and local outliers.

Outputs & Insights
------------------
eta : (n,)
    The linear predictor in log-space: b0 + u + (optional) Z*beta.
mu : (n,)
    The predicted count: E * exp(eta).
rr : (n,)
    Relative Risk. An RR of 1.2 means that district has a 20% 
    higher risk of lung cancer than the UK average after 
    adjusting for population and covariates.
"""
