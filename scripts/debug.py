"""
UK Lung Cancer Spatial Model - V4 Data Integrity & Alignment Debugger

PURPOSE:
    This script serves as a diagnostic tool for the V4 stratified model (Men/Women).
    It was primarily used to resolve the 'ValueError: Poisson distribution got 
    invalid rate parameter' by identifying missing data introduced during 
    spatial alignment.

KEY TASKS PERFORMED:
    1. Integrity Scan: Checks for NaNs, Infinities, and Zeros in critical 
       model inputs (Lung_Cancer counts, All_Causes offsets, and covariates).
    2. Spatial Verification: Cross-references the 318 geographic LADs in the 
       adjacency matrix (spatial_structure.pkl) with the processed CSV.
    3. Gap Detection: Specifically identified 5 missing districts (10 rows) 
       caused by 2023 UK Unitary Authority consolidations (e.g., Buckinghamshire, 
       North Yorkshire, Somerset) which lacked direct matches in raw mortality/
       smoking files.

OUTPUT:
    - Summary of missingness per column.
    - Explicit list of LAD_CODEs requiring NaN patching to ensure MCMC 
      initialization stability.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle

# Path to your processed data
data_path = Path("data/processed/inputs_v4_stratified.csv")

if not data_path.exists():
    print(f"❌ Error: {data_path} not found. Check your directory.")
else:
    df = pd.read_csv(data_path)
    cols_to_check = ['Lung_Cancer', 'All_Causes', 'bsmoke', 'bmen', 'binteraction']
    
    print("--- Data Integrity Check ---")
    for col in cols_to_check:
        if col not in df.columns:
            print(f"⚠️ Column {col} is MISSING from the CSV!")
            continue
            
        nans = df[col].isna().sum()
        infs = np.isinf(df[col]).sum()
        print(f"{col:<15} | NaNs: {nans:<5} | Infs: {infs:<5} | Max: {df[col].max():.2f}")

    # Specific check for the Poisson 'offset'
    if 'All_Causes' in df.columns:
        zeros = (df['All_Causes'] == 0).sum()
        print(f"\nAll_Causes Zeros: {zeros}")

import pandas as pd
import numpy as np
from pathlib import Path

# Load Master Order
with open("data/processed/spatial_structure.pkl", "rb") as f:
    spatial = pickle.load(f)
    master_order = spatial['lad_order']

# Load your stratified data
df = pd.read_csv("data/processed/inputs_v4_stratified.csv")
# Look only at Men (the first 318 rows)
df_men = df.iloc[:318]

# Find the codes that are in master_order but have NaNs in the data
missing_codes = df_men[df_men['Lung_Cancer'].isna()]['LAD_CODE'].tolist()

print(f"Total LADs in Spatial Model: {len(master_order)}")
print(f"Missing LADs in Data: {len(missing_codes)}")
print("-" * 30)
for code in missing_codes:
    print(f"Code: {code}")