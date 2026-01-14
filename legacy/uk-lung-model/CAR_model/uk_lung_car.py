"""
LEGACY FILE — DO NOT RUN

This file is kept for reference only.
New CAR implementation lives in:
    src/lung_cancer_spatial/models/car.py
"""




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

# Loading mortality data 
data = pd.read_csv("/Users/alydiaowens/uk-lung-model/1376830934927918.csv", skiprows=8)  # skipping metadata
data = data.rename(columns={
    "mnemonic": "LAD_code",
    "A00-R99,U00-Y89 All causes, all ages": "All deaths",
    "C33-C34 Malignant neoplasm of trachea, bronchus and lung": "Lung Cancer Deaths"
})

# Loading population estimates 
pop = pd.read_csv("/Users/alydiaowens/uk-lung-model/1380710199350457.csv", skiprows=6)  # skipping metadata
pop = pop.rename(columns={"mnemonic": "LAD_code", "2023": "population"})

# Loading geoportal data
gdf = gpd.read_file("/Users/alydiaowens/uk-lung-model/Local_Authority_Districts_December_2023_Boundaries_UK_BFC_9042356933902664268/LAD_DEC_2023_UK_BFC.shp")
gdf = gdf[['LAD23CD', 'geometry']].rename(columns={'LAD23CD': "LAD_code"})

# Merging death and population data
deathandpop = data.merge(pop, on="LAD_code")
deathandpop = deathandpop.rename(columns={
    "local authority: district / unitary (as of April 2023)_x": "LAD_name"
})
deathandpop = deathandpop[["LAD_name", "LAD_code", "Lung Cancer Deaths", "population"]]

# Merging geoportal to death+pop dataset
dataset = gdf.merge(deathandpop, on="LAD_code", how="left")
dataset["population"] = dataset["population"].astype(float)

# Compute lung cancer mortality rate per 100,000 people
dataset["lung_cancer_rate"] = (
    dataset["Lung Cancer Deaths"] / dataset["population"] * 100000
)

# Remove rows with null or infinite values in the rate
dataset = dataset[dataset["lung_cancer_rate"].notna() & np.isfinite(dataset["lung_cancer_rate"])]

# Sorting and aligning geodataframe to dataset
dataset = dataset.sort_values("LAD_code").reset_index(drop=True)
gdf = gdf.sort_values("LAD_code").reset_index(drop=True)
gdf = gdf.loc[dataset.index]  # ensures row alignment

# Generating spatial weights and adjacency matrix
w = Queen.from_dataframe(gdf)
A = w.full()[0]
A = jnp.array(A)
y = jnp.array(dataset["Lung Cancer Deaths"].values)
E = jnp.array(dataset["population"].values)
# Ensure y and E are 1D arrays
if y.ndim > 1:
    y = y.flatten()
if E.ndim > 1:
    E = E.flatten()
# Ensure A is a square matrix
if A.ndim == 1:
    A = A.reshape(-1, 1)
# Ensure A is a 2D square matrix
if A.ndim == 2 and A.shape[0] != A.shape[1]:
    raise ValueError("Adjacency matrix A must be square.")
# Ensure y and E have the same length as the number of rows in A
if len(y) != A.shape[0] or len(E) != A.shape[0]:
    raise ValueError("Length of y and E must match the number of rows in A.")

#For GP Model, using centroids as coordinates 
# Add centroids to your aligned GeoDataFrame
gdf["centroid"] = gdf.geometry.centroid

# Extract coordinates from centroids
gdf["longitude"] = gdf.centroid.x
gdf["latitude"] = gdf.centroid.y

# Create the coordinate matrix X (shape: [n_regions, 2])
X = np.column_stack((gdf["latitude"].values, gdf["longitude"].values))

# Plotting lung cancer mortality rates in UK
#fig, ax = plt.subplots(1, 2, figsize=(15, 10))

# 1. Plot LAD boundaries only
#boundary_gdf = dataset.boundary
#boundary_gdf.plot(ax=ax[0], color='black', lw=0.5)
#ax[0].set_title('UK: Borders of Local Authority Districts')
#ax[0].axis('off')

# 2. Plot lung cancer mortality rate
#dataset.plot(column='lung_cancer_rate',ax=ax[1], cmap='OrRd',legend=True,edgecolor='black',linewidth=0.5)
#ax[1].set_title('UK: Lung Cancer Deaths per 100,000 People')
#ax[1].axis('off')

#plt.tight_layout()
#plt.savefig("lung_cancer_rate_map.png", dpi=300)
#print("✅ Map saved as lung_cancer_rate_map.png")

# Modified from numpyro textbook code (https://elizavetasemenova.github.io/prob-epi/20_areal_data.html)

def car_model(y, E, A):
    n = A.shape[0]
    
    # Compute degree matrix
    d = jnp.sum(A, axis=1)
    D = jnp.diag(d)

    # Parameters
    b0 = numpyro.sample('b0', dist.Normal(0, 1))
    tau = numpyro.sample('tau', dist.Gamma(3, 2))
    alpha = numpyro.sample('alpha', dist.Uniform(0.01, 0.95))

    # CAR precision matrix
    Q_std = D - alpha * A
    Q_std = Q_std + 1e-5 * jnp.eye(n)  # Numerical stability

    # Spatial random effects
    car_std = numpyro.sample('car_std', dist.MultivariateNormal(loc=jnp.zeros(n), precision_matrix=Q_std))
    sigma = numpyro.deterministic('sigma', 1. / jnp.sqrt(tau))
    car = numpyro.deterministic('car', sigma * car_std)

    # Log-relative risk
    lin_pred = b0 + car
    log_lambda = lin_pred
    lambda_ = jnp.exp(log_lambda)

    # Poisson likelihood
    mu = E * lambda_
    numpyro.sample("obs", dist.Poisson(mu), obs=y)
    rr = numpyro.deterministic("rr", lambda_)
    