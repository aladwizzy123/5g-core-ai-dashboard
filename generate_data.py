import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_5g_core_data():
    np.random.seed(2026)
    start_date = datetime(2026, 9, 14)
    hours = 24 * 7  # 1 week of data
    
    # We will track three different network functions / deployment instances
    nf_instances = ['AMF_01', 'SMF_01', 'UPF_01']
    data_list = []
    
    for nf in nf_instances:
        for h in range(hours):
            timestamp = start_date + timedelta(hours=h)
            hour_of_day = timestamp.hour
            
            # Baseline metrics with time-of-day traffic waves
            traffic_factor = np.sin((hour_of_day - 6) * np.pi / 12)
            
            pdu_success_rate = np.clip(99.5 - np.random.exponential(0.1), 90.0, 100.0)
            sbi_latency_ms = np.clip(12 + 5 * traffic_factor + np.random.normal(0, 2), 5, 45)
            packet_drop_rate = np.clip(np.random.exponential(0.02), 0.0, 2.0)
            
            # Inject distinct 5GC failure signatures that copycats won't understand
            if nf == 'AMF_01' and h in range(48, 54):
                # Simulated HTTP/2 SBI Signaling overload (e.g., UDM/UDR slow response)
                sbi_latency_ms = np.random.uniform(120.0, 250.0) 
                pdu_success_rate = np.random.uniform(82.0, 88.0)
                
            elif nf == 'UPF_01' and h in range(120, 126):
                # Simulated User Plane interface saturation / N3-GTP-U interface errors
                packet_drop_rate = np.random.uniform(4.5, 9.5)
                pdu_success_rate = np.random.uniform(92.0, 95.0)
                
            data_list.append({
                'Timestamp': timestamp,
                'NF_Instance': nf,
                'PDU_Session_Success_Rate': round(pdu_success_rate, 2),
                'SBI_Latency_ms': round(sbi_latency_ms, 1),
                'Packet_Drop_Rate_Pct': round(packet_drop_rate, 3)
            })
            
    df = pd.DataFrame(data_list)
    df.to_csv('5g_core_kpi_data.csv', index=False)
    print("✅ 5G Core Network KPI data saved to '5g_core_kpi_data.csv'!")

if __name__ == '__main__':
    create_5g_core_data()
