# preprocessing/build_inputs.py
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from libpysal.weights import Queen
import scipy.sparse.linalg as sla
from sklearn import neighbors

def build_inputs(
    shapefile_path: Path,
    deaths_csv: Path,
    pop_csv: Path,
    out_dir: Path,
    deaths_skiprows: int = 8,
    pop_skiprows: int = 6,
    deaths_col: str = "C33-C34 Malignant neoplasm of trachea, bronchus and lung",
    pop_col: str = "2023",
    geo_code_col: str = "LAD23CD",
    csv_code_col: str = "mnemonic",
    include_covariates_csv: Path | None = None,
    covariate_cols: list[str] | None = None,
    E_mode: str = "expected"
):
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load geometry ---
    gdf = gpd.read_file(shapefile_path)
    gdf = gdf[[geo_code_col, "geometry"]].rename(columns={geo_code_col: "LAD_code"})

    # --- Load deaths + pop ---
    deaths = pd.read_csv(deaths_csv, skiprows=deaths_skiprows).rename(
        columns={csv_code_col: "LAD_code", deaths_col: "lung_deaths"}
    )
    pop = pd.read_csv(pop_csv, skiprows=pop_skiprows).rename(
        columns={csv_code_col: "LAD_code", pop_col: "population"}
    )

    deaths = deaths[["LAD_code", "lung_deaths"]].copy()
    pop = pop[["LAD_code", "population"]].copy()

    deaths["lung_deaths"] = pd.to_numeric(deaths["lung_deaths"], errors="coerce")
    pop["population"] = pd.to_numeric(pop["population"], errors="coerce")

    df = deaths.merge(pop, on="LAD_code", how="inner")

    # --- Optional covariates ---
    if include_covariates_csv is not None:
        cov = pd.read_csv(include_covariates_csv)
        if covariate_cols is None:
            raise ValueError("If include_covariates_csv is provided, covariate_cols must be provided.")
        cov = cov.rename(columns={csv_code_col: "LAD_code"})
        keep = ["LAD_code"] + covariate_cols
        df = df.merge(cov[keep], on="LAD_code", how="left")

    # --- Merge into geometry ---
    dataset = gdf.merge(df, on="LAD_code", how="inner")

    # --- Clean + align ordering ---
    dataset = dataset.dropna(subset=["lung_deaths", "population"]).copy()
    dataset = dataset[np.isfinite(dataset["lung_deaths"]) & np.isfinite(dataset["population"])].copy()
    dataset = dataset.sort_values("LAD_code").reset_index(drop=True)

    # --- Build adjacency using aligned dataset ---
    w = Queen.from_dataframe(dataset)

    # --- FIX: Connect Islands ---
    neighbors = w.neighbors.copy()
    # Identify indices of islands (those with 0 neighbors)
    islands = [k for k, v in neighbors.items() if len(v) == 0]

    for island_idx in islands:
        # Find the nearest neighbor by distance using centroids
        curr_geom = dataset.iloc[island_idx].geometry.centroid
        distances = dataset.geometry.centroid.distance(curr_geom)
        # Set its own distance to infinity so it doesn't pick itself
        distances.iloc[island_idx] = np.inf
        nearest_idx = distances.idxmin()
    
        # Manually link them in both directions
        neighbors[island_idx] = [nearest_idx]
        neighbors[nearest_idx].append(island_idx)

        # Re-build the weight object with these manual links
    from libpysal.weights import W
    w = W(neighbors)

    A = w.full()[0].astype(np.float32)  # (n,n)

    # --- NEW: Calculate Alpha Max for Proper CAR ---

    # D_inv_sqrt = D^-1/2
    d = np.sum(A, axis=1)
    # Handle potential isolated regions (d=0) to avoid division by zero
    d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
    D_inv_sqrt = np.diag(d_inv_sqrt)

    # S = D^-1/2 * A * D^-1/2
    S = D_inv_sqrt @ A @ D_inv_sqrt

    # Find the largest eigenvalue (spectral radius)
    # For a connected Queen graph, this is usually 1.0, but calculation is safer.
    vals = sla.eigsh(S, k=1, which='LM', return_eigenvectors=False)
    alpha_max = float(1.0 / vals[0])

    # --- y and E for Poisson disease mapping ---
    y = dataset["lung_deaths"].to_numpy().astype(int)
    popn = dataset["population"].to_numpy().astype(np.float64)
    if E_mode == "expected":
        baseline_rate = y.sum() / popn.sum()
        E = (popn * baseline_rate).astype(np.float64)
    elif E_mode == "population":
        E = popn.astype(np.float64)
    else:
        raise ValueError("E_mode must be 'expected' or 'population'")

    # --- Optional Z matrix (standardized) ---
    Z = None
    if include_covariates_csv is not None and covariate_cols is not None:
        Z_df = dataset[covariate_cols].copy()
        # simple z-score standardization (skip if already standardized)
        Z = ((Z_df - Z_df.mean()) / Z_df.std(ddof=0)).to_numpy().astype(np.float32)

    # --- Save arrays ---
    npz_path = out_dir / f"inputs_car_{E_mode}.npz"
    if Z is None:
        np.savez_compressed(npz_path, y=y, E=E, A=A, alpha_max=alpha_max, LAD_code=dataset["LAD_code"].to_numpy())
    else:
        np.savez_compressed(npz_path, y=y, E=E, A=A, Z=Z, alpha_max=alpha_max, LAD_code=dataset["LAD_code"].to_numpy())

    # --- Save geometry for mapping ---
    areas_path = out_dir / "areas.parquet"
    dataset[["LAD_code", "geometry"]].to_parquet(areas_path, index=False)

    # --- Print quick summary ---
    print("✅ Saved:")
    print(f"  - {npz_path}")
    print(f"  - {areas_path}")
    if E_mode == "expected":
        print(f"n={len(dataset)} | total deaths={y.sum()} | baseline_rate={baseline_rate:.6g}")
    else:
        print(f"n={len(dataset)} | total deaths={y.sum()} | E_mode=population")
    print(f"A shape={A.shape} | E min/max=({E.min():.3g}, {E.max():.3g})")
    if Z is not None:
        print(f"Z shape={Z.shape} | covariates={covariate_cols}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shapefile", type=Path, required=True)
    p.add_argument("--deaths_csv", type=Path, required=True)
    p.add_argument("--pop_csv", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, default=Path("data/processed"))
    p.add_argument("--deaths_skiprows", type=int, default=8)
    p.add_argument("--pop_skiprows", type=int, default=6)
    p.add_argument("--deaths_col", type=str, default="C33-C34 Malignant neoplasm of trachea, bronchus and lung")
    p.add_argument("--pop_col", type=str, default="2023")
    p.add_argument("--geo_code_col", type=str, default="LAD23CD")
    p.add_argument("--csv_code_col", type=str, default="mnemonic")
    p.add_argument("--covariates_csv", type=Path, default=None)
    p.add_argument("--covariate_cols", nargs="*", default=None)
    p.add_argument(
    "--E_mode",
    type=str,
    choices=["expected", "population"],
    default="expected",
    help="How to define E: 'expected' (internal standardization) or 'population' (exposure model)"
)
    args = p.parse_args()

    build_inputs(
        shapefile_path=args.shapefile,
        deaths_csv=args.deaths_csv,
        pop_csv=args.pop_csv,
        out_dir=args.out_dir,
        deaths_skiprows=args.deaths_skiprows,
        pop_skiprows=args.pop_skiprows,
        deaths_col=args.deaths_col,
        pop_col=args.pop_col,
        geo_code_col=args.geo_code_col,
        csv_code_col=args.csv_code_col,
        include_covariates_csv=args.covariates_csv,
        covariate_cols=args.covariate_cols,
        E_mode=args.E_mode
    )

if __name__ == "__main__":
    main()