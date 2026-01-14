import arviz as az
import matplotlib.pyplot as plt

# Load InferenceData from NetCDF
idata = az.from_netcdf("inference_data.nc")

# Compute summary
summary = az.summary(idata, hdi_prob=0.95)
summary = summary.head(10)

# Save summary as a PNG table
fig, ax = plt.subplots(figsize=(min(20, 2 + 0.8 * len(summary)), 0.5 + 0.4 * len(summary)))
ax.axis('off')
tbl = ax.table(cellText=summary.values,
               colLabels=summary.columns,
               rowLabels=summary.index,
               loc='center',
               cellLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
tbl.scale(1.2, 1.2)
plt.tight_layout()
plt.savefig("summary_table.png", dpi=300)
plt.close(fig)
print("🖼️ Summary table saved to summary_table.png")