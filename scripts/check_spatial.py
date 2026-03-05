"""
Spatial Structure Audit Utility
------------------------------
This script verifies the normalization status of the adjacency matrix (A) 
used in CAR (Conditional Autoregressive) models. 

In a standard Leroux or intrinsic CAR formulation, the spatial dependence 
parameter 'alpha' is only validly bounded between 0 and 1 if the adjacency 
matrix is scaled by its spectral radius (maximum eigenvalue).

Usage:
    python scripts/check_spatial.py

Outputs:
    - Stored alpha_max (from V3 metadata)
    - Calculated Max Eigenvalue of A
    - Standardization status and recommended scaling factor
"""

import pickle
import jax.numpy as jnp

path = "data/processed/spatial_structure.pkl"

with open(path, "rb") as f:
    data = pickle.load(f)

A = jnp.array(data['A'])
alpha_max = data['alpha_max']

# Calculate the actual maximum eigenvalue of your matrix A
eigenvalues = jnp.linalg.eigvalsh(A)
actual_max_eigen = jnp.max(eigenvalues)

print(f"--- Spatial Structure Audit ---")
print(f"Stored alpha_max: {alpha_max:.4f}")
print(f"Actual Max Eigenvalue of A: {actual_max_eigen:.4f}")

if jnp.isclose(actual_max_eigen, 1.0):
    print("✅ Matrix is standardized (Max Eigenvalue = 1).")
else:
    print("⚠️ Matrix is NOT standardized. alpha should be scaled by 1/actual_max_eigen.")