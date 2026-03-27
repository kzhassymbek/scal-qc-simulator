import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from scipy import stats

# ------------------------------
# Helper Functions
# ------------------------------
def generate_synthetic_data(n_samples=20, add_errors=False):
    """Generate realistic SCAL data with optional outliers."""
    np.random.seed(42)
    data = {
        "Sample_ID": [f"S{i+1:02d}" for i in range(n_samples)],
        "Porosity": np.random.uniform(0.05, 0.35, n_samples),
        "Permeability": 10 ** np.random.uniform(-1, 3, n_samples),
        "Swirr": np.random.uniform(0.1, 0.4, n_samples),
        "Sor": np.random.uniform(0.1, 0.4, n_samples),
        "Kro_swirr": np.random.uniform(0.7, 1.0, n_samples),
        "Krw_sor": np.random.uniform(0.05, 0.3, n_samples)
    }
    df = pd.DataFrame(data)
    # Add correlation
    df["Permeability"] = df["Permeability"] * (df["Porosity"] / df["Porosity"].mean())**2
    if add_errors:
        outliers = np.random.choice(n_samples, size=int(0.1*n_samples), replace=False)
        df.loc[outliers, "Porosity"] = np.random.uniform(0.01, 0.7, size=len(outliers))
        df.loc[outliers, "Permeability"] = np.random.uniform(1e-3, 1e4, size=len(outliers))
        df.loc[outliers[:2], "Kro_swirr"] = np.random.uniform(1.2, 1.5, size=2)
        df.loc[outliers[2:4], "Krw_sor"] = np.random.uniform(0.4, 0.8, size=2)
    return df

def qc_checks(df, z_threshold=2.5):
    """Apply QC checks and return DataFrame with flags."""
    df_qc = df.copy()
    flags = {}
    
    # Range checks
    range_checks = {
        "Porosity": (0, 1),
        "Permeability": (0, None),
        "Swirr": (0, 1),
        "Sor": (0, 1),
        "Kro_swirr": (0, 1),
        "Krw_sor": (0, 1)
    }
    for col, (low, high) in range_checks.items():
        if col in df_qc.columns:
            low_cond = (df_qc[col] < low) if low is not None else False
            high_cond = (df_qc[col] > high) if high is not None else False
            out_range = low_cond | high_cond
            flags[col] = [f"Range: {val:.2f} outside [{low}, {high}]" if out else "" 
                          for val, out in zip(df_qc[col], out_range)]
    
    # Statistical outliers (Z-score)
    numeric_cols = df_qc.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        mean = df_qc[col].mean()
        std = df_qc[col].std()
        if std == 0:
            continue
        z = (df_qc[col] - mean) / std
        outlier = np.abs(z) > z_threshold
        flags[f"{col}_outlier"] = [f"Outlier (|z|={z_i:.2f})" if out else "" 
                                   for out, z_i in zip(outlier, z)]
    
    # Physical consistency
    # Porosity vs Permeability trend (log-linear)
    if "Porosity" in df_qc and "Permeability" in df_qc:
        df_qc["logPerm"] = np.log10(df_qc["Permeability"].clip(lower=1e-6))
        slope, intercept, r_value, p_value, std_err = stats.linregress(df_qc["Porosity"], df_qc["logPerm"])
        pred = slope * df_qc["Porosity"] + intercept
        residual = np.abs(df_qc["logPerm"] - pred)
        residual_std = residual.std()
        outlier_res = residual > 2 * residual_std
        flags["Perm_trend"] = [f"Perm trend outlier (res={res_i:.2f})" if out else "" 
                               for out, res_i in zip(outlier_res, residual)]
    
    # Relative permeability endpoints
    if "Kro_swirr" in df_qc:
        flags["Kro_swirr_low"] = [f"Low Kro at Swirr (<0.7)" if val < 0.7 else "" for val in df_qc["Kro_swirr"]]
    if "Krw_sor" in df_qc:
        flags["Krw_sor_high"] = [f"High Krw at Sor (>0.3)" if val > 0.3 else "" for val in df_qc["Krw_sor"]]
    
    # Saturation sum
    if "Swirr" in df_qc and "Sor" in df_qc:
        sat_sum = df_qc["Swirr"] + df_qc["Sor"]
        flags["Sat_sum"] = [f"Saturation sum > 1 ({s:.2f})" if s > 1 else "" for s in sat_sum]
    
    # Combine all flags into a summary column
    all_flag_cols = [col for col in flags if isinstance(flags[col], list)]
    combined = []
    for i in range(len(df_qc)):
        msgs = [flags[col][i] for col in all_flag_cols if flags[col][i]]
        combined.append("; ".join(msgs))
    df_qc["QC_Summary"] = combined
    
    # Add individual flag columns
    for col, flag_list in flags.items():
        df_qc[f"Flag_{col}"] = flag_list
    
    # Drop temporary column
    if "logPerm" in df_qc.columns:
        df_qc.drop(columns=["logPerm"], inplace=True)
    
    return df_qc

def create_plots(df_qc):
    plots = {}
    # Porosity vs Permeability
    if "Porosity" in df_qc and "Permeability" in df_qc:
        fig = px.scatter(df_qc, x="Porosity", y="Permeability", log_y=True,
                         hover_data=["Sample_ID", "QC_Summary"],
                         title="Porosity vs Permeability (log scale)")
        flagged = df_qc[df_qc["QC_Summary"] != ""]
        if not flagged.empty:
            fig.add_trace(go.Scatter(x=flagged["Porosity"], y=flagged["Permeability"],
                                     mode="markers", marker=dict(color="red", size=12),
                                     name="Flagged"))
        plots["Porosity-Perm"] = fig
    
    # Relative permeability endpoints
    if "Kro_swirr" in df_qc and "Krw_sor" in df_qc:
        fig = px.scatter(df_qc, x="Kro_swirr", y="Krw_sor",
                         hover_data=["Sample_ID", "QC_Summary"],
                         title="Relative Permeability Endpoints")
        fig.add_shape(type="rect", x0=0.7, x1=1, y0=0, y1=0.3,
                      line=dict(color="green", width=2, dash="dash"),
                      fillcolor="LightGreen", opacity=0.2)
        plots["Kr endpoints"] = fig
    
    return plots

# ------------------------------
# Streamlit App
# ------------------------------
st.set_page_config(page_title="SCAL QC Simulator", layout="wide")
st.title("🪨 Special Core Analysis (SCAL) Quality Control Simulator")

st.markdown("""
This tool helps you QC SCAL data for **porosity**, **permeability**, **saturations**, and **relative permeability endpoints**.  
Upload your Excel/CSV file or generate synthetic data, adjust thresholds, and download a flagged report.
""")

# Sidebar
st.sidebar.header("Data Input")
data_source = st.sidebar.radio("Data source", ["Upload Excel/CSV", "Generate synthetic data"])

df = None
if data_source == "Upload Excel/CSV":
    uploaded_file = st.sidebar.file_uploader("Choose a file", type=["xlsx", "xls", "csv"])
    if uploaded_file:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.sidebar.success("File loaded")
else:
    n_samples = st.sidebar.slider("Number of samples", 5, 100, 20)
    add_errors = st.sidebar.checkbox("Add errors (outliers)", False)
    if st.sidebar.button("Generate"):
        df = generate_synthetic_data(n_samples, add_errors)
        st.sidebar.success(f"Generated {n_samples} samples")

if df is None:
    st.info("Please upload or generate data to start.")
    st.stop()

# QC settings
st.sidebar.header("QC Settings")
z_threshold = st.sidebar.slider("Z-score outlier threshold", 1.5, 3.5, 2.5, 0.1)

# Apply QC
df_qc = qc_checks(df, z_threshold)

# Tabs
tab_data, tab_qc, tab_plots = st.tabs(["📊 Data", "📋 QC Summary", "📈 Plots"])

with tab_data:
    st.subheader("Raw Data with QC Flags")
    st.dataframe(df_qc, use_container_width=True)
    
    # Download
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df_qc.to_excel(writer, sheet_name="SCAL_QC", index=False)
    st.download_button("📥 Download QC Results (Excel)", data=output.getvalue(),
                       file_name="scal_qc_results.xlsx", 
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

with tab_qc:
    st.subheader("QC Summary")
    # Count violations
    flag_cols = [col for col in df_qc.columns if col.startswith("Flag_") or col == "QC_Summary"]
    violations = {}
    for col in flag_cols:
        if col == "QC_Summary":
            violations[col] = sum(df_qc[col] != "")
        else:
            violations[col] = sum(df_qc[col] != "")
    st.dataframe(pd.DataFrame(list(violations.items()), columns=["Check", "Count"]))
    
    st.subheader("Flagged Records")
    flagged = df_qc[df_qc["QC_Summary"] != ""]
    if not flagged.empty:
        st.dataframe(flagged)
    else:
        st.success("No QC violations found!")

with tab_plots:
    st.subheader("Diagnostic Plots")
    plots = create_plots(df_qc)
    for name, fig in plots.items():
        st.plotly_chart(fig, use_container_width=True)

with st.expander("Data Statistics"):
    st.dataframe(df.describe())
