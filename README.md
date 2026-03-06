# Modeling lung cancer mortality in the UK using Bayesian GP and CAR Methods

🌍 Project Overview: Stratified Spatial Modeling of UK Lung Cancer Risk

This research project develops a high-resolution, Bayesian geographic framework to investigate the drivers of lung cancer mortality across 318 Local Authority Districts (LADs) in the United Kingdom. By integrating demographic stratification and advanced spatial priors, the project quantifies how the relationship between smoking prevalence and mortality risk varies across geographic and gendered boundaries.

🔬 Methodology & Model Evolution

The project utilizes a hierarchical Bayesian framework implemented in JAX and NumPyro, comparing global spatial trends against local district-level variations.

CAR (Conditional Autoregressive) Models: 

Utilizing BYM2 priors to decompose spatial risk into structured geographic trends and unstructured "white noise" components.

GP (Gaussian Process) Models: 

Employing continuous spatial kernels (Matern/RBF) to identify long-range spatial dependencies and non-linear risk surfaces.

Version History

V3.0 (Baseline): Initial district-level spatial analysis.

V4.0 (Stratified): Transitioned to a 636-observation dataset to include gender-specific interactions.

V4.5 (Production): Current state-of-the-art. Implements Effect Coding (-0.5, 0.5) and Centered Interactions to resolve multicollinearity, and utilizes an Inverse-Gamma prior for spatial variance ($\sigma$) to ensure convergence stability.

💡 Why This Project Matters

Health Equity: By stratifying by gender, the model identifies specific regions where public health interventions (like smoking cessation programs) may yield disproportionate benefits for specific sub-populations.

Statistical Rigor: The transition to Version 4.5 addresses the "Neal's Funnel" and parameter interference common in high-dimensional spatial models, providing a blueprint for stable MCMC convergence in complex epidemiological datasets.

Policy Support: The output generates Exceedance Probability Maps—a critical tool for policymakers to visualize districts where the relative risk of mortality significantly exceeds the national average with 95% statistical certainty.


---

# CAR Model: 
## Maps: 
  Three maps were developed for the CAR model to understand estimated relative risk, hotspot probability, and model uncertainty. 
  1. Estimated Relative Risk (v3_map_mean_rr.png)
    This map shows the geographic multiplier of lung cancer risk. Values near 1 (shown in white and gray) show that the risk in that district is exactly the national average, while values above 1.0 in red show LADs with elevated risk. An RR of 1.5 means this LAD experiences 50% more cases than expected. Districts with values below 1 have a "protective" effect or lower than average risk of lung cancer mortality. 
  
  2. Hotspot Probability (v3_map_exceedance.png)
    This map shows the Bayesian "Exceedance Probability", which measures how certain the model is that a district is a true hotspot of lung cancer mortality. Specifically, it shows the probability of the relative risk being above 1 (P(RR>1.0)). Values 0.95-1.0 are highlighted in dark red while values 0.0-0.05 are in white, with yellow districts being inconclusive.

  3. Model Uncertainty (v3_map_uncertainty.png)
    This map shows the standard deviation of the Relative Risk estimates for each LAD by illustrating the posterior standard deviation. Areas with high uncertainty (usually corresponding with low population districts) are in dark purple, while areas in light purple/white are high precision areas. 

## Code: 
### 1. Build Processed Inputs
#### Disease-mapping (population exposure model)
```bash 
python -m lung_cancer_spatial.preprocessing.build_inputs \
  --shapefile data/raw/Local_Authority_Districts_December_2023_Boundaries_UK_BFC_9042356933902664268/LAD_DEC_2023_UK_BFC.shp \
  --deaths_csv data/raw/2023mortality.csv \
  --pop_csv data/raw/2023population.csv \
  --E_mode population \
  --out_dir data/processed
  ```

### 2. Running Model with Inputs 
#### Running CAR model (population exposure version)
```bash 
python -m lung_cancer_spatial.inference.run_car \
  --inputs_npz data/processed/inputs_car_population.npz \
  --out_nc outputs/idata_car_population.nc \
  --warmup 1500 \
  --samples 2000 \
  --chains 4
``` 
#### Running CAR model (Smoking and Gender Covariate Version)
```bash 
export PYTHONPATH=$PYTHONPATH:$(pwd)/src && \
python -m lung_cancer_spatial.inference.run_car_v4 \
    --inputs data/processed/inputs_v4_stratified.csv \
    --adj data/processed/spatial_structure.pkl \
    --out_dir outputs \
    --warmup 3000 \
    --samples 3000 \
    --chains 4 \
    --target_accept 0.95 && \
python /Users/alydiaowens/Projects/uk-lung-cancer-spatial/scripts/car_generate_report_v4.py
```

### 3. Reporting Model Results 
#### Generating CAR Model Report (Population Exposure version)
```bash
python scripts/car_generate_report.py \
    --input outputs/idata_car_population_v3.nc \
    --filename car_model_report_v3.pdf \
    --warmup 1500
```
#### Generating CAR Model Report (Smoking and Gender Covariate Version)
```bash 
export PYTHONPATH=$PYTHONPATH:$(pwd)/src && \
python scripts/car_generate_report_v4.py \
    --input_nc /Users/alydiaowens/Projects/uk-lung-cancer-spatial/outputs/idata_car_v4.nc \
    --metadata data/processed/v4_scaling_metadata.pkl
```

# GP Model: 
## Maps: 
Three maps were developed for the GP model to visualize the continuous spatial risk surface, hotspot probability, and posterior precision. Unlike the CAR model, these maps highlight risk that "flows" across administrative boundaries based on geographic distance.

1. Estimated Relative Risk (gp_map_v3_rr_final.png)
2. Hotspot Probability (gp_map_v3_exceedance_final.png)
3. Model Uncertainty (gp_map_v3_uncertainty_final.png)

## Code: 
### 2. Running Model with Inputs 
#### Running GP model 
```bash 
python src/lung_cancer_spatial/inference/run_gp.py \
    --version v3 \
    --warmup 2000 \
    --samples 2000 \
    --chains 3 \
    --target_accept 0.95
```

### 3. Reporting Model Results 
#### Generating GP Model Report 
```bash
python scripts/gp_generate_report.py --version v3
```


### Version Notes 
> **CAR Version 2**: Note: Version 2 of CAR model was introduced with an spatial dependence parameter (alpha) Max contsraint and we replaced the standard Multivariate Normal (MVN) sampling with a manual "Cholesky" decomposition and a standard normal noise vector z_u.
>
> **CAR Version 3**:  Note: Version 3 of CAR model uses a sum-to-zero constraint of u_spatial to ensure spatial effects don't drift and adds a rho parameter from the Besag-York-Mollie (BYM) 2 model framework to help with divergence issues. The geometric mean scaling factor is assumed to be ~1. Spatial connectivity was enforced by identifying topological islands in the UK and manually bridging them to the nearest mainlaind centroids to ensure a non-singular precision matrix and enable Cholesky factorization. 

> **GP Version 3**: Note: Version 3 of GP model transitions from RBF to a Matern 3/2 kernel to prevent overfitting seen in legacy iterations. To ensure numerical stability and convergence, the model uses an increased diagonal jitter (1x10^-4) and utilizes a dense mass matrix during NUTS sampling to account for high posterior correlation between length-scale and variance parameters. Coordinates are standardized to align with Gamma(3,2) length-scale prior, ensuring the identified spatial clusters remain physically meaningful.


<details>
    <summary> Click to expand: CAR Model (Internal Standardization Method, Not Used) </summary>

    Disease-mapping version (internal standardization) 

    ```bash
    python -m lung_cancer_spatial.preprocessing.build_inputs \
    --shapefile data/raw/Local_Authority_Districts_December_2023_Boundaries_UK_BFC_9042356933902664268/LAD_DEC_2023_UK_BFC.shp \
    --deaths_csv data/raw/2023mortality.csv \
    --pop_csv data/raw/2023population.csv \
    --E_mode expected \
     --out_dir data/processed
    ``` 


    Running CAR model (disease mapping/internal standardization) 
    ```bash 
    python -m lung_cancer_spatial.inference.run_car \
    --inputs_npz data/processed/inputs_car_expected.npz \
    --out_nc outputs/idata_car_expected.nc \
    --warmup 1500 \
    --samples 2000 \
    --chains 4
    ```
</details>

<details>
  <summary> Version History </summary>
    The Model Evolution: From Baseline to V3/V2CAR Model (Neighborhood-Discrete)V1/V2 Baseline: 
    
    Initially utilized a standard CAR prior with a generic precision matrix. While it captured broad regional trends, the mixing of spatial and unstructured noise was unconstrained, making it difficult to quantify how much of the lung cancer risk was truly "geographic."Refinement to V3: I moved to a BYM2 (Besag-York-Mollié) formulation. This allowed for the introduction of the $\rho$ (rho) parameter to explicitly partition the variance. I also enforced a sum-to-zero constraint on the spatial effects for identifiability.
    
    Result: This transition produced a much more stable model with a clear spatial fraction of 0.985, providing the statistical "green light" to move forward with covariate analysis.
    
    Gaussian Process (Distance-Continuous)V1 Legacy: The initial GP used a Squared Exponential (RBF) kernel with a LogNormal(1000, 0.5) prior on the length-scale. Because the coordinates were standardized, this prior was mismatched with the data scale, causing the model to overfit the local noise (the "nugget" effect) rather than the regional signal.
    
    Refinement to V3: I have pivoted to a Matérn 3/2 kernel. This is a more robust choice for public health data as it allows for slightly "rougher" transitions, preventing the over-oscillations seen in the RBF version. I also re-scaled the length-scale prior to align with the standardized coordinate space, and used a Poisson likelihood with exposure.
    
    Result: This V3 GP now produces a "smoothed" risk surface that captures the industrial corridors of the North without hugging every individual district's data point, making it a much more credible comparison to the CAR results.
</details>