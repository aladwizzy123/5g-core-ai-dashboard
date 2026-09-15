import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import IsolationForest

# 1. Dashboard Layout Setup
st.set_page_config(page_title="5GC AI Observability", layout="wide")
st.title("🌐 5G Standalone Core (5GC) AI Validation Dashboard")
st.markdown("Automating anomaly isolation across the Service-Based Architecture (SBA) using multi-dimensional Machine Learning.")

# 2. Load Core Telemetry Logs
@st.cache_data
def load_core_data():
    df = pd.read_csv('5g_core_kpi_data.csv')
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    return df

df = load_core_data()

# 3. AI Anomaly Engine Setup
@st.cache_resource
def analyze_core_anomalies(data):
    # Features mapped out across Control Plane & User Plane performance
    features = ['PDU_Session_Success_Rate', 'SBI_Latency_ms', 'Packet_Drop_Rate_Pct']
    
    # Isolation Forest isolates data vectors that show unnatural coordinate distances
    model = IsolationForest(contamination=0.04, random_state=2026)
    data['Anomaly_Score'] = model.fit_predict(data[features])
    data['Status'] = data['Anomaly_Score'].map({1: 'Healthy Profile', -1: 'Degraded State'})
    return data

df_analyzed = analyze_core_anomalies(df.copy())

# 4. Interactive Sidebar
st.sidebar.header("5GC Topology Filter")
selected_nf = st.sidebar.selectbox("Select Network Function (NF):", df_analyzed['NF_Instance'].unique())

nf_df = df_analyzed[df_analyzed['NF_Instance'] == selected_nf]
anomalies = nf_df[nf_df['Status'] == 'Degraded State']

# 5. Core Performance Widgets
st.header(f"⚡ Performance Matrix: {selected_nf}")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Mean PDU Session Success Rate", f"{round(nf_df['PDU_Session_Success_Rate'].mean(), 2)}%")
with col2:
    st.metric("Mean SBI Interface Latency", f"{round(nf_df['SBI_Latency_ms'].mean(), 1)} ms")
with col3:
    if not anomalies.empty:
        st.error(f"🚨 {len(anomalies)} Unconfigured Anomalies Flagged by AI")
    else:
        st.success("✅ Signaling & User Planes Stable")

# 6. Multi-Variant Visualization Chart
st.subheader("Signaling / Packet Flow Deviations Over Time")
st.markdown("Red indicators flag multidimensional metric anomalies that circumvent static thresholds.")

# Dynamically change the primary tracking graph axis based on the chosen entity type
y_axis_metric = 'SBI_Latency_ms' if 'AMF' in selected_nf else 'Packet_Drop_Rate_Pct'

fig = px.scatter(nf_df, x='Timestamp', y=y_axis_metric,
                 color='Status',
                 color_discrete_map={'Healthy Profile': '#2ECC71', 'Degraded State': '#E74C3C'},
                 title=f"Time-Series Performance Diagnostics for {selected_nf}")

fig.add_scatter(x=nf_df['Timestamp'], y=nf_df[y_axis_metric], mode='lines', name='Traffic Volume Baseline', line=dict(color='rgba(150,150,150,0.2)'))

st.plotly_chart(fig, use_container_width=True)

# Display deep raw context logs for telecom engineers
if not anomalies.empty:
    st.subheader("📋 Core Failure Signatures Detected")
    st.dataframe(anomalies[['Timestamp', 'PDU_Session_Success_Rate', 'SBI_Latency_ms', 'Packet_Drop_Rate_Pct']].reset_index(drop=True))
