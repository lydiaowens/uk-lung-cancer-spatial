import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

# --- GP V4.0 DATA (From your reported numbers) ---
# Format: {var: (mean, sd, hdi_3, hdi_97)}
gp_stats = {
    "b0": (-3.097, 0.101, -3.268, -2.902),
    "beta_smoke": (0.016, 0.009, -0.002, 0.033), 
    "beta_men": (0.033, 0.014, 0.008, 0.060),
    "beta_interaction": (-0.002, 0.014, -0.029, 0.024)
}

# --- CAR V4.8 DATA (From your audit/screenshot) ---
car_stats = {
    "b0": (-3.101, 0.016, -3.129, -3.070),
    "beta_smoke": (0.020, 0.011, -0.001, 0.040),
    "beta_men": (0.030, 0.014, 0.005, 0.058),
    "beta_interaction": (-0.002, 0.015, -0.029, 0.026)
}

vars = ["b0", "beta_smoke", "beta_men", "beta_interaction"]
titles = ["Intercept ($b_0$)", "Smoking ($\\beta_{smoke}$)", 
          "Men ($\\beta_{men}$)", "Interaction ($\\beta_{int}$)"]

def main():
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    for i, var in enumerate(vars):
        # Shared X-axis calculation for visual comparison
        combined_mu = [gp_stats[var][0], car_stats[var][0]]
        combined_sd = [gp_stats[var][1], car_stats[var][1]]
        x_min = min(combined_mu) - 4 * max(combined_sd)
        x_max = max(combined_mu) + 4 * max(combined_sd)
        x = np.linspace(x_min, x_max, 500)

        # PLOT GP (Row 0)
        mu_g, sd_g, h3_g, h97_g = gp_stats[var]
        y_g = stats.norm.pdf(x, mu_g, sd_g)
        axes[0, i].plot(x, y_g, color='steelblue', lw=3)
        axes[0, i].fill_between(x, y_g, alpha=0.3, color='steelblue')
        
        # Add "ArviZ Style" Mean and HDI line
        axes[0, i].set_title(f"GP V4: {titles[i]}", fontsize=14, fontweight='bold')
        axes[0, i].hlines(y=0, xmin=h3_g, xmax=h97_g, color='black', lw=6)
        axes[0, i].text(mu_g, max(y_g)*0.1, f"{mu_g:.3f}", ha='center', fontweight='bold')
        axes[0, i].text(h3_g, max(y_g)*0.02, f"{h3_g:.2f}", ha='right', fontsize=10)
        axes[0, i].text(h97_g, max(y_g)*0.02, f"{h97_g:.2f}", ha='left', fontsize=10)

        # PLOT CAR (Row 1)
        mu_c, sd_c, h3_c, h97_c = car_stats[var]
        y_c = stats.norm.pdf(x, mu_c, sd_c)
        axes[1, i].plot(x, y_c, color='firebrick', lw=3)
        axes[1, i].fill_between(x, y_c, alpha=0.3, color='firebrick')
        
        # Add "ArviZ Style" Mean and HDI line
        axes[1, i].set_title(f"CAR V4.8: {titles[i]}", fontsize=14, fontweight='bold')
        axes[1, i].hlines(y=0, xmin=h3_c, xmax=h97_c, color='black', lw=6)
        axes[1, i].text(mu_c, max(y_c)*0.1, f"{mu_c:.3f}", ha='center', fontweight='bold')
        axes[1, i].text(h3_c, max(y_c)*0.02, f"{h3_c:.2f}", ha='right', fontsize=10)
        axes[1, i].text(h97_c, max(y_c)*0.02, f"{h97_c:.2f}", ha='left', fontsize=10)

        # Clean up axes
        for row in [0, 1]:
            axes[row, i].set_yticks([])
            axes[row, i].spines['top'].set_visible(False)
            axes[row, i].spines['right'].set_visible(False)
            axes[row, i].spines['left'].set_visible(False)

    plt.tight_layout()
    plt.savefig("posterior_estimates.png", dpi=300)
    print("✅ 2x4 Poster-ready plot saved: posterior_estimates.png")

if __name__ == "__main__":
    main()