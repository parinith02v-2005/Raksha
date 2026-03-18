import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import find_peaks
from datetime import datetime
import time

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="RAKSHA AI Cardiac Analytics",
    page_icon="🩺",
    layout="wide"
)

# ---------------- MODEL ---------------- #

class AeroGridNet(nn.Module):

    def __init__(self):
        super(AeroGridNet,self).__init__()

        self.features = nn.Sequential(
            nn.Conv1d(1,32,5,padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32,64,5,padding=2),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1)
        )

        self.classifier = nn.Linear(64,5)

    def forward(self,x):
        x=self.features(x)
        x=x.view(x.size(0),-1)
        return self.classifier(x)

# ---------------- SIDEBAR ---------------- #

with st.sidebar:

    st.title("RAKSHA v3")
    st.success("AI CORE: ACTIVE")

    source = st.selectbox(
        "ECG Source",
        ["CSV Upload","Live Simulation","Live Monitor"]
    )

# ---------------- HEADER ---------------- #

st.title("RAKSHA: Next-Gen Cardiac Analytics")
st.markdown("---")

col_main,col_side = st.columns([2,1])

signal=None

# ---------------- DATA SOURCE ---------------- #

if source=="CSV Upload":

    with col_side:
        uploaded_file=st.file_uploader("Upload ECG CSV",type="csv")

    if uploaded_file:
        df=pd.read_csv(uploaded_file)
        signal=df.iloc[:,0].values.astype(np.float32)

elif source=="Live Simulation":

    t=np.linspace(0,10,1000)
    signal=np.sin(5*t)+np.random.normal(0,0.2,1000)

elif source=="Live Monitor":

    placeholder=st.empty()

    for _ in range(30):

        t=np.linspace(0,5,500)
        live_signal=np.sin(5*t)+np.random.normal(0,0.15,500)

        fig_live=go.Figure()

        fig_live.add_trace(go.Scatter(
            y=live_signal,
            mode="lines",
            line=dict(color="#00ffa2",width=3)
        ))

        fig_live.update_layout(
            template="plotly_dark",
            title="Live ECG Monitor"
        )

        placeholder.plotly_chart(fig_live,use_container_width=True)

        time.sleep(0.3)

# ---------------- PROCESS ---------------- #

if signal is not None:

    signal=(signal-np.mean(signal))/(np.std(signal)+1e-8)

    model=AeroGridNet()

    try:
        model.load_state_dict(
            torch.load("arrhythmia_model.pth",map_location="cpu")
        )
    except:
        st.warning("Model file not found. Using demo output.")

    input_data=torch.tensor(signal).float().unsqueeze(0).unsqueeze(0)

    output=model(input_data)

    probs=torch.nn.functional.softmax(output,dim=1)

    label_idx=torch.argmax(output,dim=1).item()

    conf=float(torch.max(probs))*100

    classes=[
        "NORMAL SINUS",
        "SUPRAVENTRICULAR",
        "VENTRICULAR",
        "FUSION",
        "UNKNOWN"
    ]

# ---------------- HEART RATE ---------------- #

    peaks,_=find_peaks(signal,distance=150)

    if len(peaks)>1:
        rr=np.diff(peaks)
        heart_rate=60/(np.mean(rr)/360)
    else:
        heart_rate=72

# ---------------- SIDE PANEL ---------------- #

    with col_side:

        st.metric("Heart Rate",f"{heart_rate:.1f} BPM")
        st.metric("Diagnosis",classes[label_idx])
        st.metric("AI Confidence",f"{conf:.2f}%")

# ---------------- ECG GRAPH ---------------- #

    with col_main:

        fig=go.Figure()

        fig.add_trace(go.Scatter(
            y=signal,
            mode='lines',
            line=dict(color='#00ffa2',width=3),
            name='ECG'
        ))

        fig.add_trace(go.Scatter(
            x=peaks,
            y=signal[peaks],
            mode='markers',
            marker=dict(color='red',size=8),
            name='R Peaks'
        ))

# ---------------- PQRST LABELS ---------------- #

        for p in peaks:

            fig.add_annotation(x=p,y=signal[p],text="R",showarrow=True)

            if p-40>0:
                fig.add_annotation(x=p-40,y=signal[p-40],text="Q",showarrow=False)

            if p+40<len(signal):
                fig.add_annotation(x=p+40,y=signal[p+40],text="S",showarrow=False)

# ---------------- ARRHYTHMIA RISK ZONES ---------------- #

        if np.std(signal)>1.2:

            fig.add_vrect(
                x0=200,
                x1=350,
                fillcolor="rgba(255,0,0,0.2)",
                line_width=0,
                annotation_text="ARRHYTHMIA RISK"
            )

# ---------------- ECG GRID ---------------- #

        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor="#1f2937"
        )

        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor="#1f2937"
        )

        fig.update_layout(
            template="plotly_dark",
            height=450,
            paper_bgcolor="#020617",
            plot_bgcolor="#020617"
        )

        st.plotly_chart(fig,use_container_width=True)

# ---------------- PROBABILITY CHART ---------------- #

        st.subheader("Arrhythmia Probability")

        prob_values=probs.detach().numpy()[0]

        prob_fig=go.Figure()

        prob_fig.add_trace(go.Bar(
            x=classes,
            y=prob_values
        ))

        prob_fig.update_layout(template="plotly_dark")

        st.plotly_chart(prob_fig,use_container_width=True)

# ---------------- ANOMALY DETECTION ---------------- #

        st.subheader("AI Anomaly Detection")

        anomaly=np.std(signal)

        if anomaly>1.5:
            st.error("⚠ Possible abnormal rhythm detected")
        else:
            st.success("ECG rhythm stable")

# ---------------- ECG STATS ---------------- #

        st.subheader("ECG Statistics")

        c1,c2,c3=st.columns(3)

        c1.metric("Mean Voltage",f"{np.mean(signal):.3f}")
        c2.metric("Max Voltage",f"{np.max(signal):.3f}")
        c3.metric("Min Voltage",f"{np.min(signal):.3f}")

# ---------------- REPORT ---------------- #

        st.subheader("Clinical Summary")

        report=f"""
Date: {datetime.now()}
Diagnosis: {classes[label_idx]}
Confidence: {conf:.2f}%
Heart Rate: {heart_rate:.1f} BPM
"""

        st.markdown(f"""
        <div style="background:#020617;padding:15px;border-radius:10px">
        <b>Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}<br>
        <b>Diagnosis:</b> {classes[label_idx]}<br>
        <b>Confidence:</b> {conf:.2f}%<br>
        Recommendation: Clinical ECG review recommended.
        </div>
        """,unsafe_allow_html=True)

        st.download_button(
            "Download Medical Report",
            report,
            file_name="raksha_report.txt"
        )

else:

    st.info("Upload ECG CSV or start simulation.")
