import streamlit as st
import pandas as pd
import plotly.express as px
import os
import re
from sklearn.ensemble import IsolationForest

# LangChain & Vector DB Packages
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 1. Page Configuration
st.set_page_config(page_title="5GC AI Observability & RAG", layout="wide")
st.title("🌐 5G Standalone Core (5GC) Intelligent Copilot Platform")
st.markdown("Combines **Isolation Forest Anomaly Detection** with **Retrieval-Augmented Generation (RAG)** to isolate faults and extract 3GPP remediation playbooks.")

# 2. Vector DB Initialization Engine (Production Pattern)
@st.cache_resource
def initialize_rag_system():
    doc_path = "documents/3gpp_runbook.txt"
    persist_directory = "vector_db"
    
    # 1. Use HuggingFace to download a free, lightweight embedding model locally
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Check if vector DB is already built to save processing time
    if os.path.exists(persist_directory) and len(os.listdir(persist_directory)) > 0:
        vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    else:
        # Load and chunk document structure
        loader = TextLoader(doc_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
        docs = text_splitter.split_documents(documents)
        
        # Build local database
        vector_store = Chroma.from_documents(docs, embeddings, persist_directory=persist_directory)
    
    return vector_store

vector_db = initialize_rag_system()

# 3. Load Telemetry Logs
@st.cache_data
def load_core_data():
    df = pd.read_csv('5g_core_kpi_data.csv')
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    return df

df = load_core_data()

# 4. AI Anomaly Engine Setup
@st.cache_resource
def analyze_core_anomalies(data):
    features = ['PDU_Session_Success_Rate', 'SBI_Latency_ms', 'Packet_Drop_Rate_Pct']
    model = IsolationForest(contamination=0.04, random_state=2026)
    data['Anomaly_Score'] = model.fit_predict(data[features])
    data['Status'] = data['Anomaly_Score'].map({1: 'Healthy Profile', -1: 'Degraded State'})
    return data

df_analyzed = analyze_core_anomalies(df.copy())

# 5. Interactive Topology Sidebar Filter
st.sidebar.header("5GC Network Topology")
selected_nf = st.sidebar.selectbox("Select Network Function (NF):", df_analyzed['NF_Instance'].unique())

nf_df = df_analyzed[df_analyzed['NF_Instance'] == selected_nf]
anomalies = nf_df[nf_df['Status'] == 'Degraded State']

# 6. Performance Layout UI Metrics
st.header(f"⚡ Performance Matrix: {selected_nf}")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Mean PDU Session Success Rate", f"{round(nf_df['PDU_Session_Success_Rate'].mean(), 2)}%")
with col2:
    st.metric("Mean SBI Interface Latency", f"{round(nf_df['SBI_Latency_ms'].mean(), 1)} ms")
with col3:
    if not anomalies.empty:
        st.error(f"🚨 {len(anomalies)} Outages Flagged by Anomaly Model")
    else:
        st.success("✅ Signaling & User Planes Stable")

# 7. Visualization Plot Chart
y_axis_metric = 'SBI_Latency_ms' if 'AMF' in selected_nf else 'Packet_Drop_Rate_Pct'
fig = px.scatter(nf_df, x='Timestamp', y=y_axis_metric, color='Status',
                 color_discrete_map={'Healthy Profile': '#2ECC71', 'Degraded State': '#E74C3C'},
                 title=f"Time-Series Diagnostics for {selected_nf}")
fig.add_scatter(x=nf_df['Timestamp'], y=nf_df[y_axis_metric], mode='lines', name='Baseline Flow', line=dict(color='rgba(150,150,150,0.2)'))
st.plotly_chart(fig, width='stretch')

# 8. PRODUCTION RAG COPILOT INTERFACE
st.markdown("---")
st.header("🤖 3GPP Knowledge Copilot (RAG System)")
st.markdown("When an outage is isolated above, query the local embedded vector store for standardized resolution guidelines.")

# Smart Context Injector: Pre-populate the user query box based on the active node state!
default_query = "What are the remediation steps for AMF signaling overload?" if "AMF" in selected_nf else "How to fix UPF packet drop rate issues?"
user_query = st.text_input("Ask Copilot for 3GPP Remediation Runbooks:", value=default_query)

if user_query:
    with st.spinner("Querying vector store embedding layers..."):
        # Perform similarity lookup inside ChromaDB vector space
        matched_docs = vector_db.similarity_search(user_query, k=1)
        
        if matched_docs:
            retrieved_text = matched_docs[0].page_content.strip()
            title = retrieved_text.splitlines()[0] if retrieved_text else "3GPP Runbook"
            error_signature = re.search(r"ERROR_SIGNATURE:\s*(.+)", retrieved_text)
            root_cause = re.search(r"ROOT_CAUSE:\s*(.+)", retrieved_text)
            remediation_steps = re.findall(r"^\d+\.\s+(.+)$", retrieved_text, re.MULTILINE)

            st.subheader(title)
            if error_signature:
                st.info(f"**Error signature:** {error_signature.group(1)}")
            if root_cause:
                st.markdown("#### Likely Root Cause")
                st.write(root_cause.group(1))
            if remediation_steps:
                st.markdown("#### Recommended Remediation")
                for step_number, step in enumerate(remediation_steps, start=1):
                    st.markdown(f"{step_number}. {step}")
            if not error_signature and not root_cause and not remediation_steps:
                st.write(retrieved_text)
        else:
            st.warning("No matching 3GPP runbook entries found in vector storage vectors.")
