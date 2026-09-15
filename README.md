# 🌐 5G Standalone Core (5GC) AI Validation Dashboard

An intelligent observability platform that utilizes **Machine Learning (Isolation Forest)** to automate multi-dimensional anomaly detection across 3GPP Service-Based Architectures (SBA). 

## 🧠 The Telecom Problem Solved
Traditional network performance management relies on rigid, static thresholds (e.g., alert if latency > 50ms). However, complex network degradation often happens below these thresholds when multiple metrics interact abnormally. This project monitors **AMF**, **SMF**, and **UPF** behaviors simultaneously to catch stealth faults like signaling loops and user-plane saturation before they trigger outages.

## 🛠️ Tech Stack & Key Features
- **Machine Learning Engine:** Scikit-learn (`Isolation Forest` algorithm for unsupervised anomaly detection)
- **Web App Dashboard:** Streamlit (Python-native frontend engine)
- **Data Visualizations:** Plotly Express (Interactive time-series charts)
- **Key KPIs Monitored:** PDU Session Success Rate, SBI Interface Latency (ms), Packet Drop Rate (%)

## ⚙️ How to Run Locally
1. Clone the repository:
   ```bash
   git clone https://github.com
   cd 5g-core-ai-dashboard
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Generate the telemetry telemetry data:
   ```bash
   python generate_data.py
   ```
4. Launch the dashboard application:
   ```bash
   streamlit run app.py
   ```
