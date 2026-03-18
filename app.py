import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import find_peaks
from datetime import datetime
import os

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="RAKSHA AI Cardiac Analytics",
    page_icon="🩺",
    layout="wide"
)

# ---------------- UI STYLE ---------------- #

st.markdown("""
<style>
body{color:#e2e8f0;}

[data-testid="stAppViewContainer"]{
background: linear-gradient(135deg,#0f172a,#1e293b,#334155);
}

[data-testid="stSidebar"]{
background:#020617;
border-right:1px solid #1e293b;
}

h1{color:#38bdf8;}

.report-box{
background:#020617;
border:1px solid #1e293b;
border-radius:12px;
padding:20px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- MODEL CLASS ---------------- #

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
        ["CSV Upload","Live Simulation"]
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
        df=pd.read_csv(uploaded_file,header=None)
        signal=df.iloc[:,0].values.astype(np.float32)

elif source=="Live Simulation":

    st.warning("Running Live ECG Simulation")

    t=np.linspace(0,10,1000)
    signal=np.sin(5*t)+np.random.normal(0,0.2,1000)

# ---------------- PROCESS ---------------- #

if signal is not None:

    # Normalize signal
    signal=(signal-np.mean(signal))/(np.std(signal)+1e-8)

    # Resize to model input size (MIT-BIH standard)
    TARGET_LENGTH=187

    if len(signal)>TARGET_LENGTH:
        signal=signal[:TARGET_LENGTH]
    else:
        signal=np.pad(signal,(0,TARGET_LENGTH-len(signal)))

    # ---------------- LOAD MODEL ---------------- #

    model_path="arrhythmia_model.pth"

    if not os.path.exists(model_path):
        st.error("Model file not found in repository")
        st.write("Files available:",os.listdir())
        st.stop()

    try:
        # load complete model safely
        model=torch.load(model_path,map_location="cpu")
        model.eval()

    except:
        # fallback if only state_dict exists
        model=AeroGridNet()
        model.load_state_dict(torch.load(model_path,map_location="cpu"))
        model.eval()

    # ---------------- PREDICTION ---------------- #

    input_data=torch.tensor(signal).float().unsqueeze(0).unsqueeze(0)

    with torch.no_grad():

        output=model(input_data)

        probs=torch.softmax(output,dim=1)

        prob_values=probs.cpu().numpy()[0]

        label_idx=int(np.argmax(prob_values))

        conf=float(np.max(prob_values))*100

    classes=[
        "NORMAL SINUS",
        "SUPRAVENTRICULAR",
        "VENTRICULAR",
        "FUSION",
        "UNKNOWN"
    ]

# ---------------- HEART RATE ---------------- #

    peaks,_=find_peaks(signal,distance=30)

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
            mode="lines",
            line=dict(color="#38bdf8",width=4),
            name="ECG"
        ))

        fig.add_trace(go.Scatter(
            x=peaks,
            y=signal[peaks],
            mode="markers",
            marker=dict(color="red",size=8),
            name="R Peaks"
        ))

        fig.update_layout(
            template="plotly_dark",
            height=450
        )

        st.plotly_chart(fig,use_container_width=True)

# ---------------- PROBABILITY CHART ---------------- #

        st.markdown("### Arrhythmia Probability")

        prob_fig=go.Figure()

        prob_fig.add_trace(go.Bar(
            x=classes,
            y=prob_values
        ))

        prob_fig.update_layout(template="plotly_dark")

        st.plotly_chart(prob_fig,use_container_width=True)

# ---------------- ECG STATS ---------------- #

        st.markdown("### ECG Statistics")

        c1,c2,c3=st.columns(3)

        c1.metric("Mean Voltage",f"{np.mean(signal):.3f}")
        c2.metric("Max Voltage",f"{np.max(signal):.3f}")
        c3.metric("Min Voltage",f"{np.min(signal):.3f}")

# ---------------- PATIENT PROFILE ---------------- #

        st.markdown("### Patient Profile")

        p1,p2,p3=st.columns(3)

        p1.metric("Age","50")
        p2.metric("Blood Pressure","128/82")
        p3.metric("SpO₂","97%")

# ---------------- REPORT ---------------- #

        st.markdown("### Clinical Summary")

        report=f"""
Date: {datetime.now()}
Diagnosis: {classes[label_idx]}
Confidence: {conf:.2f}%
Heart Rate: {heart_rate:.1f} BPM
"""

        st.markdown(f"""
        <div class="report-box">
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
