# Modeling lung cancer mortality in the UK using Bayesian GP and CAR methods 

---
### Latest Versions: 
CAR model: Version 3, located in car.py 

GP model: Version 1, located in gp.py 

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
### 3. Reporting Model Results 
#### Generating CAR Model Report (Population Exposure version)
```bash
python scripts/car_generate_report.py \
    --input outputs/idata_car_population_v3.nc \
    --filename car_model_report_v3.pdf \
    --warmup 1500
```


### Version Notes 
> **Version 2**: Note: Version 2 of CAR model was introduced with an spatial dependence parameter (alpha) Max contsraint and we replaced the standard Multivariate Normal (MVN) sampling with a manual "Cholesky" decomposition and a standard normal noise vector z_u.
>
> **Version 3**:  Note: Version 3 of CAR model uses a sum-to-zero constraint of u_spatial to ensure spatial effects don't drift and adds a rho parameter from the Besag-York-Mollie (BYM) 2 model framework to help with divergence issues. The geometric mean scaling factor is assumed to be ~1. Spatial connectivity was enforced by identifying topological islands in the UK and manually bridging them to the nearest mainlaind centroids to ensure a non-singular precision matrix and enable Cholesky factorization. 


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
