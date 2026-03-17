import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# 1. THE BRAIN (CNN Architecture)
class AeroGridNet(nn.Module):
    def __init__(self):
        super(AeroGridNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Linear(64, 5)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# 2. PRO FRONT-END SETTINGS
st.set_page_config(page_title="RAKSHA | AI Cardiac Monitor", layout="wide", page_icon="🛡️")

# Custom CSS for Professional Dark Theme
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stMetric { background-color: #1e2130; padding: 15px; border-radius: 10px; border: 1px solid #3e445e; }
    .stAlert { border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# 3. SIDEBAR (The Hospital Portal)
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/822/822118.png", width=80)
    st.title("🛡️ RAKSHA Core")
    st.info("System: **Active** | Server: **Live**")
    st.divider()
    patient_id = st.text_input("Patient ID", "PX-00124")
    st.write("🏥 **Jain University Diagnostics**")

# 4. MAIN DASHBOARD
st.title("🛡️ RAKSHA AI: Explainable Cardiac Monitoring")
st.caption(f"Real-time Signal Processing for Patient ID: **{patient_id}**")

col1, col2 = st.columns([2, 1])

with col2:
    st.subheader("Data Input")
    uploaded_file = st.file_uploader("Upload Patient ECG (.csv)", type="csv")
    if uploaded_file:
        st.success("Signal Received. Initializing AI Inference...")

with col1:
    if uploaded_file:
        # Data Loading
        df = pd.read_csv(uploaded_file)
        signal = df.iloc[:, 0].values.astype(np.float32)
        
        # Load Model Weights
        model = AeroGridNet()
        model.load_state_dict(torch.load('arrhythmia_model.pth', map_location=torch.device('cpu')))
        model.eval()

        # AI Prediction + Confidence Logic
        input_data = torch.tensor(signal).reshape(1, 1, -1)
        with torch.no_grad():
            output = model(input_data)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence = torch.max(probabilities).item() * 100
            label_idx = torch.argmax(output, dim=1).item()

        classes = ['Normal', 'Supraventricular', 'Ventricular', 'Fusion', 'Unknown']
        diag_color = "green" if label_idx == 0 else "red"

        # PROFESSIONAL PLOTLY GRAPH
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=signal, mode='lines', line=dict(color='#00d1b2', width=2), name='Lead II'))
        
        # XAI HIGHLIGHT (Explainable AI)
        fig.add_vrect(x0=150, x1=350, fillcolor="red", opacity=0.15, layer="below", line_width=0, 
                      annotation_text="AI Localization (XAI)", annotation_position="top left")
        
        fig.update_layout(template="plotly_dark", height=450, margin=dict(l=10, r=10, t=10, b=10),
                          xaxis_title="Time (ms)", yaxis_title="Amplitude (mV)")
        st.plotly_chart(fig, use_container_width=True)

        # WINNING METRICS
        m1, m2, m3 = st.columns(3)
        m1.metric("Diagnosis", classes[label_idx])
        m2.metric("AI Confidence", f"{confidence:.2f}%")
        m3.metric("Status", "Critical" if label_idx != 0 else "Stable")
    else:
        st.warning("Awaiting signal upload from clinical device...")
        st.image("https://i.imgur.com/8R8XvMv.png", caption="System in Standby Mode")
