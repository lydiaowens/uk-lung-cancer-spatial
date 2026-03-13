import streamlit as st
import leafmap.foliumap as leafmap
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ==========================================
# 1. Page Setup & Styling
# ==========================================
st.set_page_config(layout="wide", page_title="UK Lung Cancer Mapping")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { 
        background-color: #ffffff; 
        padding: 20px; 
        border-radius: 12px; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.05); 
        border: 1px solid #e9ecef;
    }
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. Data Loading
# ==========================================
@st.cache_data
def load_spatial_data():
    data = gpd.read_file("dashboard_data.geojson")
    if data.crs != "EPSG:4326":
        data = data.to_crs(epsg=4326)
    return data

try:
    gdf = load_spatial_data()
except Exception as e:
    st.error(f"Error loading dashboard_data.geojson: {e}")
    st.stop()

# ==========================================
# 3. Sidebar: Controls & Credits
# ==========================================
with st.sidebar:
    st.title("🛡️ Control Panel")
    st.markdown("---")
    
    model_mode = st.radio(
        "Select Spatial Model Prior:",
        ["GP V4.0 (Continuous)", "CAR V4.8 (Discrete)"],
        help="Toggle between the distance-based Gaussian Process and the neighborhood-based CAR model."
    )
    
    st.markdown("---")
    st.subheader("Model Hyperparameters")
    
    if "GP" in model_mode:
        suffix = "_gp"
        st.info("**GP V4.0 (Covariate Model)**")
        st.write("**Variance ($\sigma^2$):** 0.17")
        st.write("**Length-scale ($\ell$):** 0.34")
        st.write("**Effective Range:** 60.59 km")
    else:
        suffix = "_car"
        st.info("**CAR V4.8 (Covariate Model)**")
        st.write("**Sigma ($\sigma$):** 0.26")
        st.write("**Rho ($\rho$):** 0.31")
        st.write("**Alpha ($\\alpha$):** 0.54")

    st.markdown("---")
    st.markdown("""
    **Project Credits:** **Allison Lydia Owens** University of North Carolina at Chapel Hill  
    📧 [owens23a@unc.edu](mailto:owens23a@unc.edu)
    """)

# ==========================================
# 4. Main Header
# ==========================================
st.title("UK Lung Cancer Mortality: Bayesian-Powered Interactive Mapping")
st.caption("Visualizing the 'Northern Paradox' through Discrete and Continuous Spatial Priors.")

# ==========================================
# 5. Interactive Maps (Dual Column)
# ==========================================
col_map1, col_map2 = st.columns(2)

with col_map1:
    st.subheader("📍 Relative Risk (RR)")
    m1 = leafmap.Map(center=[54.5, -2], zoom=6, draw_control=False, measure_control=False)
    
    rr_bins = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8]
    # Robust hex conversion: map the float (0-1) to RGBA, then to hex string
    cmap_rdbu = plt.get_cmap("RdBu_r")
    rr_colors = [mcolors.rgb2hex(cmap_rdbu(i / len(rr_bins))) for i in range(len(rr_bins) + 1)]

    m1.add_data(
        gdf, 
        column=f"rr{suffix}", 
        scheme="UserDefined", 
        classification_kwds={'bins': rr_bins}, 
        colors=rr_colors, 
        legend_title="Relative Risk (1.0 = Baseline)",
        layer_name="Relative Risk"
    )
    m1.to_streamlit(height=600)

with col_map2:
    st.subheader("🔥 Exceedance Probabilities")
    m2 = leafmap.Map(center=[54.5, -2], zoom=6, draw_control=False, measure_control=False)
    
    ep_bins = [0.2, 0.5, 0.8, 0.95]
    cmap_ylorrd = plt.get_cmap("YlOrRd")
    ep_colors = [mcolors.rgb2hex(cmap_ylorrd(i / len(ep_bins))) for i in range(len(ep_bins) + 1)]

    m2.add_data(
        gdf, 
        column=f"ep{suffix}", 
        scheme="UserDefined", 
        classification_kwds={'bins': ep_bins}, 
        colors=ep_colors, 
        legend_title="Pr(RR > 1.0)",
        layer_name="Exceedance"
    )
    m2.to_streamlit(height=600)

# ==========================================
# 6. LAD Inspector & Interpretation
# ==========================================
st.markdown("---")
st.header("🔍 Local Authority District (LAD) Inspector")

selected_lad_name = st.selectbox("Search District:", sorted(gdf['LAD23NM'].unique()))
lad_data = gdf[gdf['LAD23NM'] == selected_lad_name].iloc[0]

val_rr = lad_data[f'rr{suffix}']
val_ep = lad_data[f'ep{suffix}']
val_sd = lad_data[f'sd{suffix}']

metric_col1, metric_col2, metric_col3 = st.columns(3)

with metric_col1:
    st.metric("Relative Risk (RR)", f"{val_rr:.3f}")
    if val_rr > 1.05:
        interp_rr = f"Risk is **{(val_rr-1)*100:.1f}% higher** than the national average."
    elif val_rr < 0.95:
        interp_rr = f"Risk is **{(1-val_rr)*100:.1f}% lower** than the national average."
    else:
        interp_rr = "Risk is consistent with the national average baseline."
    st.markdown(f"**Interpretation:** {interp_rr}")

with metric_col2:
    is_hotspot = val_ep >= 0.95
    ep_label = "Near-Certain Hotspot" if is_hotspot else "Standard Variation"
    st.metric("Exceedance Prob.", f"{val_ep:.1%}", delta=ep_label if is_hotspot else None)
    
    if is_hotspot:
        interp_ep = "This district is a **Near-Certain Hotspot**. There is a >95% probability that elevated mortality survives adjustment for smoking."
    else:
        interp_ep = f"There is a {val_ep:.1%} probability of elevated risk in this district."
    st.markdown(f"**Interpretation:** {interp_ep}")

with metric_col3:
    st.metric("Posterior Uncertainty (SD)", f"{val_sd:.4f}")
    st.markdown("**Interpretation:** Standard Deviation of the posterior distribution. Lower values indicate higher model confidence in the local geographic risk.")

st.markdown("---")
st.caption("Data source: 2023 UK Lung Cancer Mortality. Model: Bayesian Poisson GLM. Disclaimer: AI-assisted visualization workflow.")