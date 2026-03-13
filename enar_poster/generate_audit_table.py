import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# --- CONFIGURATION ---
BASE_DIR = Path("/Users/alydiaowens/Projects/uk-lung-cancer-spatial")
OUTPUT_PATH = BASE_DIR / "enar_poster/model_audit_table_v2.png"

# --- DATA STRUCTURE ---
# Integrated reported numbers for GP V3, GP V4, and CAR models
data = {
    "Metric": [
        "Spatial Prior",
        "Data Strata (N)",
        "WAIC (Deviance)",
        "LOO-CV (Deviance)",
        "Moran's I (Residuals)",
        "Spatial Parameters\n(rho/lengthscale)",
        "Effective Range",
        "Max R-hat"
    ],
    "CAR V3 (Baseline)": [
        "BYM2", "318", "2420.63", "2511.64", "-0.1262", "0.9850", "N/A", "1.010"
    ],
    "GP V3 (Baseline)": [
        "Matérn 3/2", "318", "2506.49", "2585.96", "0.2034", "0.2652", "47.46 km", "1.000"
    ],
    "CAR V4.8 (Covariate)": [
        "BYM2", "636", "4431.35*", "4487.67*", "0.2616*", "0.3090*", "N/A", "1.000"
    ],
    "GP V4.0 (Covariate)": [
        "Matérn 3/2", "636", "4358.46*", "4392.09*", "0.2613*", "0.3384*", "60.59 km*", "1.000"
    ]
}

def main():
    df = pd.DataFrame(data)
    
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial"],
        "font.size": 12
    })
    
    fig, ax = plt.subplots(figsize=(15, 9))
    ax.axis('off')
    
    # Create Table
    tbl = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center',
        colColours=["#f2f2f2"] * 5
    )
    
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    tbl.scale(1.0, 4.5)
    
    # Header & Formatting
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.get_text().set_weight('bold')
            cell.set_facecolor('#e6e6e6')
        
        # Color-coding for V4 models to highlight the stratified results
        if row > 0 and col in [3, 4]:
            text = cell.get_text().get_text()
            if "*" in text:
                cell.get_text().set_color("#d62728")
                cell.get_text().set_weight('bold')

    plt.title("Table 1. Global Model Audit: Performance and Spatial Scale Diagnostics", 
              fontsize=20, fontweight='bold', pad=40)
    
    plt.figtext(0.5, 0.05, 
                "*Red/Starred values indicate N=636 (Stratified). Unstarred values indicate N=318 (Aggregate).", 
                ha="center", fontsize=12, color="#d62728", style='italic')

    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
    print(f"✅ Final model audit table saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()