## Aggregated Lung Models (By Region) - CAR Model Method 
# Importing Libraries
import numpy as np
import pandas as pd 
import geopandas as gpd 
from jax.nn import sigmoid

import numpyro
numpyro.set_host_device_count(4)
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
from numpyro.diagnostics import hpdi

import jax
import jax.numpy as jnp

import arviz as az
import matplotlib.pyplot as plt
from libpysal.weights import Queen
# Check number of devices available
num_device = jax.local_device_count()
print("Number of available devices: ", num_device)
# Loading aggrgated mortality data 
# Loading mortality data 
data = pd.read_csv("/Users/alydiaowens/uk-lung-model/ukaggregatedlung.csv", skiprows=8)  # skipping metadata
data = data.rename(columns={
    "mnemonic": "LAD_code",
    "A00-R99,U00-Y89 All causes, all ages": "All deaths",
    "C33-C34 Malignant neoplasm of trachea, bronchus and lung": "Lung Cancer Deaths"
})

