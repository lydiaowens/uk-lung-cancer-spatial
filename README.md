# uk-lung-cancer-spatial
Modeling lung cancer mortality in the UK using Bayesian GP and CAR methods 

## Project Structure

data/
  raw/
  processed/
outputs/
src/lung_cancer_spatial/
  preprocessing/
  models/
  inference/
  viz/

---

## 1. Build Processed Inputs

### Disease-mapping version (internal standardization)

```bash
python -m lung_cancer_spatial.preprocessing.build_inputs \
  --shapefile data/raw/Local_Authority_Districts_December_2023_Boundaries_UK_BFC_9042356933902664268/LAD_DEC_2023_UK_BFC.shp \
  --deaths_csv data/raw/2023mortality.csv \
  --pop_csv data/raw/2023population.csv \
  --E_mode expected \
  --out_dir data/processed
``` 
### Disease-mapping (rate model)
```bash 
python -m lung_cancer_spatial.preprocessing.build_inputs \
  --shapefile data/raw/Local_Authority_Districts_December_2023_Boundaries_UK_BFC_9042356933902664268/LAD_DEC_2023_UK_BFC.shp \
  --deaths_csv data/raw/2023mortality.csv \
  --pop_csv data/raw/2023population.csv \
  --E_mode population \
  --out_dir data/processed
  ```

### Running CAR model (disease mapping/internal standardization) 
```bash 
python -m lung_cancer_spatial.inference.run_car \
  --inputs_npz data/processed/inputs_car_expected.npz \
  --out_nc outputs/idata_car_expected.nc \
  --warmup 1500 \
  --samples 2000 \
  --chains 4
```
# Note: Version 2 of CAR model was introduced with an spatial dependence parameter (alpha) Max contsraint and we replaced the standard Multivariate Normal (MVN) sampling with a manual "Cholesky" decomposition and a standard normal noise vector z_u.


### Running CAR model (population exposure version)
```bash 
python -m lung_cancer_spatial.inference.run_car \
  --inputs_npz data/processed/inputs_car_population.npz \
  --out_nc outputs/idata_car_population.nc \
  --warmup 1500 \
  --samples 2000 \
  --chains 4
``` 

python -m lung_cancer_spatial.inference.run_car \
  --inputs_npz data/processed/inputs_car_population.npz \
  --out_nc outputs/idata_car_population_v2.nc \
  --warmup 1500 \
  --samples 2000 \
  --chains 4