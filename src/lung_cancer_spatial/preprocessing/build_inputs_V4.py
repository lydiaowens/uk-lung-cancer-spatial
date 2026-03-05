import pandas as pd
import numpy as np

def build_v4_inputs():
    # --- 1. DATA PATHS (Fill these in with your local paths) ---
    path_smoke_m = "/Users/alydiaowens/Projects/uk-lung-cancer-spatial/data/raw/2023smokingprevalence_M.csv" 
    path_smoke_w = "/Users/alydiaowens/Projects/uk-lung-cancer-spatial/data/raw/2023smokingprevalence_W.csv" 
    path_mort_m  = "/Users/alydiaowens/Projects/uk-lung-cancer-spatial/data/raw/2023mortality_M.csv" 
    path_mort_w  = "/Users/alydiaowens/Projects/uk-lung-cancer-spatial/data/raw/2023mortality_W.csv" 
    output_path  = "/Users/alydiaowens/Projects/uk-lung-cancer-spatial/data/processed/inputs_v4_stratified.csv" 

    # --- 2. LOAD & STANDARDIZE SMOKING DATA ---
    df_smoke_m = pd.read_csv(path_smoke_m)
    df_smoke_w = pd.read_csv(path_smoke_w)
    
    # Smoking headers often have newlines or spaces from ONS
    smoke_cols = ['Gender', 'LAD_CODE', 'LAD_NAME', 'Smoking_Prev']
    for df in [df_smoke_m, df_smoke_w]:
        df.columns = smoke_cols
        df['LAD_NAME'] = df['LAD_NAME'].str.strip()
        df['LAD_CODE'] = df['LAD_CODE'].str.strip()
        # Ensure we use UK ONS labels: Men and Women
        df['Gender'] = df['Gender'].replace({'Male': 'Men', 'Female': 'Women'})

    df_smoke = pd.concat([df_smoke_m, df_smoke_w])

    # --- 3. LOAD & STANDARDIZE MORTALITY DATA ---
    # Nomis mortality exports typically require skipping the first 8 metadata rows
    df_mort_m = pd.read_csv(path_mort_m, skiprows=8)
    df_mort_w = pd.read_csv(path_mort_w, skiprows=8)

    mort_cols = ['LAD_NAME', 'All_Causes', 'Lung_Cancer']
    for df, gender_label in [(df_mort_m, 'Men'), (df_mort_w, 'Women')]:
        df.columns = mort_cols
        df['Gender'] = gender_label
        df['LAD_NAME'] = df['LAD_NAME'].str.strip()
        
        # Clean numeric columns (remove commas in strings like "1,023")
        for col in ['All_Causes', 'Lung_Cancer']:
            if df[col].dtype == object:
                df[col] = df[col].str.replace(',', '').astype(float)

    df_mort = pd.concat([df_mort_m, df_mort_w])

    # --- 4. MERGE DATASETS ---
    # We join on Name and Gender to create the stratified rows
    df_v4 = pd.merge(df_mort, df_smoke, on=['LAD_NAME', 'Gender'], how='inner')

    # --- 5. CREATE MODEL COVARIATES (The "Betas") ---
    
    # bmen: Binary indicator (1 for Men, 0 for Women)
    df_v4['bmen'] = (df_v4['Gender'] == 'Men').astype(int)
    
    # bsmoke: Scaled Smoking Prevalence (Center & Scale for MCMC stability)
    # We save the raw version too for plotting later
    df_v4['bsmoke'] = (df_v4['Smoking_Prev'] - df_v4['Smoking_Prev'].mean()) / df_v4['Smoking_Prev'].std()
    
    # binteraction: Interaction term (Smoking effect specifically for Men)
    df_v4['binteraction'] = df_v4['bsmoke'] * df_v4['bmen']

    # --- 6. EXPORT ---
    print(f"✅ V4 Stratified Dataset Created.")
    print(f"Total Observations: {len(df_v4)} (Men and Women per LAD)")
    print(df_v4[['LAD_NAME', 'Gender', 'bsmoke', 'bmen', 'binteraction']].head())

    df_v4.to_csv(output_path, index=False)
    
    return df_v4

if __name__ == "__main__":
    build_v4_inputs()