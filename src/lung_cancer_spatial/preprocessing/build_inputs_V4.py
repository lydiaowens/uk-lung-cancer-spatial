import pandas as pd
import numpy as np
import pickle
from pathlib import Path

def build_v4_inputs():
    # --- 1. DATA PATHS ---
    path_smoke_m = "data/raw/2023smokingprevalence_M.csv" 
    path_smoke_w = "data/raw/2023smokingprevalence_W.csv" 
    path_mort_m  = "data/raw/2023mortality_M.csv" 
    path_mort_w  = "data/raw/2023mortality_W.csv" 
    path_spatial = "data/processed/spatial_structure.pkl"
    output_path  = "data/processed/inputs_v4_stratified.csv"

    # --- 2. LOAD & CLEAN SMOKING DATA ---
    df_smoke_m = pd.read_csv(path_smoke_m)
    df_smoke_w = pd.read_csv(path_smoke_w)

    smoke_cols = ['Gender', 'LAD_CODE', 'LAD_NAME', 'Smoking_Prev']
    for df in [df_smoke_m, df_smoke_w]:
        df.columns = smoke_cols
        df['LAD_NAME'] = df['LAD_NAME'].str.strip()
        df['LAD_CODE'] = df['LAD_CODE'].str.strip()
        df['Smoking_Prev'] = pd.to_numeric(df['Smoking_Prev'], errors='coerce')
        df['Smoking_Prev'] = df['Smoking_Prev'].fillna(df['Smoking_Prev'].mean())

    df_smoke = pd.concat([df_smoke_m, df_smoke_w])

    # Create a name-to-code lookup to fix the mortality data missing codes
    name_to_code = df_smoke[['LAD_NAME', 'LAD_CODE']].drop_duplicates().set_index('LAD_NAME')['LAD_CODE']

    # --- 3. LOAD & CLEAN MORTALITY DATA ---
    df_mort_m = pd.read_csv(path_mort_m, skiprows=8)
    df_mort_w = pd.read_csv(path_mort_w, skiprows=8)

    mort_cols = ['LAD_NAME', 'All_Causes', 'Lung_Cancer']
    for df, gender_label in [(df_mort_m, 'Men'), (df_mort_w, 'Women')]:
        df.columns = mort_cols
        df['Gender'] = gender_label
        df['LAD_NAME'] = df['LAD_NAME'].str.strip()
        
        # Map the LAD_CODE from our lookup
        df['LAD_CODE'] = df['LAD_NAME'].map(name_to_code)
        
        for col in ['All_Causes', 'Lung_Cancer']:
            if df[col].dtype == object:
                df[col] = df[col].str.replace(',', '').astype(float)

    # --- 4. SPATIAL ALIGNMENT (The Fix) ---
    with open(path_spatial, "rb") as f:
        spatial = pickle.load(f)
        master_order = spatial['lad_order'] # The 318 codes in V3 order

    # Merge Mortality and Smoking
    df_mort = pd.concat([df_mort_m, df_mort_w])
    df_all = pd.merge(df_mort, df_smoke[['LAD_CODE', 'Gender', 'Smoking_Prev']], on=['LAD_CODE', 'Gender'], how='inner')

    # Force the Master Order for Men and Women separately, then stack
    df_men = df_all[df_all['Gender'] == 'Men'].set_index('LAD_CODE').reindex(master_order).reset_index()
    df_women = df_all[df_all['Gender'] == 'Women'].set_index('LAD_CODE').reindex(master_order).reset_index()
    
    df_v4 = pd.concat([df_men, df_women], ignore_index=True)

    # --- 5. NaN PATCH (The Bulletproof Fix) ---
    print(f"🛠  Patching {df_v4.isna().sum().sum()} total NaNs in the dataframe...")
    
    # 1. Fix Numeric Columns (Essential for the Math)
    df_v4['Lung_Cancer'] = df_v4['Lung_Cancer'].fillna(0)
    df_v4['All_Causes'] = df_v4['All_Causes'].fillna(df_v4['All_Causes'].median())
    
    # 2. Fix Categorical/String Columns (Essential for Alignment & Reporting)
    # This fills the 10 missing Gender slots and 10 missing Name slots
    df_v4.loc[0:317, 'Gender'] = df_v4.loc[0:317, 'Gender'].fillna('Men')
    df_v4.loc[318:, 'Gender'] = df_v4.loc[318:, 'Gender'].fillna('Women')
    df_v4['LAD_NAME'] = df_v4['LAD_NAME'].fillna("Unknown District")

    # 3. Covariate Logic (Mean-filling before Z-scoring)
    smoke_mean = df_v4['Smoking_Prev'].mean()
    smoke_std = df_v4['Smoking_Prev'].std()
    df_v4['Smoking_Prev'] = df_v4['Smoking_Prev'].fillna(smoke_mean)
    
    # --- 6. COVARIATE GENERATION ---
    df_v4['bmen'] = (df_v4['Gender'] == 'Men').astype(int)
    # 1. First Z-score
    df_v4['bsmoke'] = (df_v4['Smoking_Prev'] - smoke_mean) / smoke_std

    # 2. Fill NaNs in the base column with 0 (the population average)
    df_v4['bsmoke'] = df_v4['bsmoke'].fillna(0)

    # 3. THEN center for the V4.6 signature
    # This ensures the mean is calculated on a complete, imputed column
    df_v4['bsmokecentered'] = df_v4['bsmoke'] - df_v4['bsmoke'].mean()

    # 4. Final safety check (should already be 0, but good for catch-all)
    df_v4['bsmokecentered'] = df_v4['bsmokecentered'].fillna(0)
    # --- 7. SAVE SCALING METADATA ---
    scaling_metadata = {
        'smoking_mean': float(smoke_mean),
        'smoking_std': float(smoke_std),
        'units': 'percentage_points',
        'variable_name': 'Smoking_Prevalence_2023'
    }
    
    metadata_path = Path("data/processed/v4_scaling_metadata.pkl")
    with open(metadata_path, "wb") as f:
        pickle.dump(scaling_metadata, f)
        
    print(f"📊 Scaling metadata saved to {metadata_path}")

    # --- 8. EXPORT ---
    df_v4.to_csv(output_path, index=False)
    print(f"✅ V4 Stratified & Aligned Dataset Created (with NaN Patches): {output_path}")
    return df_v4
       

if __name__ == "__main__":
    build_v4_inputs()