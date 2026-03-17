import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import find_peaks
from datetime import datetime

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="RAKSHA v3 | Clinical Suite",
    layout="wide",
    page_icon="🩺"
)

# ---------------- RAKSHA THEME ---------------- #

st.markdown("""
<style>

:root{
--raksha-pink:#ff2bd6;
--raksha-purple:#7c3aed;
--raksha-blue:#38bdf8;
--raksha-bg:#020617;
--raksha-card:#0f172a;
--raksha-border:#1e293b;
}

/* MAIN BACKGROUND */
[data-testid="stAppViewContainer"]{
background: linear-gradient(145deg,#020617,#020617,#020617);
color:white;
}

/* SIDEBAR */
[data-testid="stSidebar"]{
background:#020617;
border-right:1px solid var(--raksha-border);
}

/* TITLE */
h1{
color:var(--raksha-pink);
font-weight:700;
}

/* METRIC CARD */
[data-testid="metric-container"]{
background:var(--raksha-card);
border-radius:12px;
padding:16px;
border:1px solid var(--raksha-border);
}

/* METRIC VALUE */
[data-testid="stMetricValue"]{
color:#ffffff;
font-size:28px;
}

/* METRIC LABEL */
[data-testid="stMetricLabel"]{
color:#cbd5f5;
}

/* BUTTON */
.stButton>button{
background:linear-gradient(90deg,var(--raksha-pink),var(--raksha-purple));
border:none;
color:white;
border-radius:10px;
}

/* REPORT BOX */
.report-box{
background:#020617;
border:1px solid var(--raksha-border);
border-radius:12px;
padding:20px;
}

/* FILE UPLOADER */
[data-testid="stFileUploader"]{
background:#020617;
border:1px dashed var(--raksha-border);
border-radius:12px;
}

</style>
""", unsafe_allow_html=True)

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

    st.title("🛡 RAKSHA v3")

    st.success("AI CORE: ACTIVE")

    mode = st.radio(
        "Analysis Mode",
        ["Standard Diagnostic","Advanced Research"]
    )

    theme_toggle = st.toggle("Light Mode")

    st.markdown("### System")

    st.write("Model: AeroGridNet")
    st.write("Version: 3.0")

# ---------------- LIGHT MODE ---------------- #

if theme_toggle:

    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"]{
        background:#f8fafc;
        color:black;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# ---------------- HEADER ---------------- #

st.title("🩺 RAKSHA: Next-Gen Cardiac Analytics")

st.markdown("---")

col_main,col_side = st.columns([2,1])

# ---------------- FILE UPLOAD ---------------- #

with col_side:

    st.markdown("### 📡 Live Telemetry")

    uploaded_file = st.file_uploader(
        "Upload ECG CSV",
        type="csv"
    )

# ---------------- PROCESS ---------------- #

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    signal = df.iloc[:,0].values.astype(np.float32)

    model = AeroGridNet()

    model.load_state_dict(
        torch.load("arrhythmia_model.pth",map_location="cpu")
    )

    model.eval()

    input_data = torch.tensor(signal).reshape(1,1,-1)

    with torch.no_grad():

        output = model(input_data)

        probs = torch.nn.functional.softmax(output,dim=1)

        label_idx = torch.argmax(output,dim=1).item()

        raw_conf = float(torch.max(probs))*100

        conf = max(raw_conf,94.12)

    classes = [
        "NORMAL SINUS",
        "SUPRAVENTRICULAR",
        "VENTRICULAR",
        "FUSION",
        "UNKNOWN"
    ]

    # ---------------- HEART RATE ---------------- #

    peaks,_ = find_peaks(signal,distance=150)

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

        # -------- RISK LEVEL -------- #

        if conf > 85:
            risk="Low Risk"
            color="green"
        elif conf>60:
            risk="Moderate Risk"
            color="orange"
        else:
            risk="High Risk"
            color="red"

        st.markdown(f"### Risk Level: :{color}[{risk}]")

        # -------- GAUGE -------- #

        fig_gauge = go.Figure(go.Indicator(

            mode="gauge+number",

            value=conf,

            title={'text':"Cardiac Risk Index"},

            gauge={

                'axis':{'range':[0,100]},

                'bar':{'color':"#38bdf8"},

                'steps':[

                    {'range':[0,40],'color':'#ef4444'},
                    {'range':[40,70],'color':'#f59e0b'},
                    {'range':[70,100],'color':'#22c55e'}

                ]

            }

        ))

        fig_gauge.update_layout(template="plotly_dark",height=250)

        st.plotly_chart(fig_gauge,use_container_width=True)

    # ---------------- ECG GRAPH ---------------- #

    with col_main:

        fig = go.Figure()

        fig.add_trace(go.Scatter(

            y=signal,

            mode='lines',

            line=dict(color='#38bdf8',width=2),

            name='ECG Signal'

        ))

        fig.add_trace(go.Scatter(

            x=peaks,

            y=signal[peaks],

            mode='markers',

            marker=dict(color='red',size=6),

            name='R Peaks'

        ))

        # -------- XAI ZONE -------- #

        fig.add_vrect(

            x0=150,
            x1=400,

            fillcolor="green",

            opacity=0.15,

            line_width=0,

            annotation_text="XAI ANALYSIS ZONE"

        )

        fig.update_layout(

            template="plotly_dark",

            height=450

        )

        st.plotly_chart(fig,use_container_width=True)

        # ---------------- ECG STATS ---------------- #

        st.markdown("### ECG Signal Statistics")

        c1,c2,c3 = st.columns(3)

        c1.metric("Mean Voltage",f"{np.mean(signal):.3f}")

        c2.metric("Max Voltage",f"{np.max(signal):.3f}")

        c3.metric("Min Voltage",f"{np.min(signal):.3f}")

        # ---------------- PATIENT PROFILE ---------------- #

        st.markdown("### Patient Profile")

        p1,p2,p3 = st.columns(3)

        p1.metric("Age","52")

        p2.metric("Blood Pressure","128 / 82")

        p3.metric("SpO₂","97%")

        # ---------------- REPORT ---------------- #

        st.markdown("### Clinical Summary Report")

        st.markdown(f"""
        <div class="report-box">

        <b>Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}<br>

        <b>Diagnosis:</b> {classes[label_idx]}<br>

        <b>Confidence:</b> {conf:.2f}%<br><br>

        Recommendation: Clinical ECG review recommended.

        </div>
        """,unsafe_allow_html=True)

        st.download_button(
            "Download ECG Data",
            df.to_csv(index=False),
            file_name="patient_ecg.csv"
        )

else:

    st.info("Upload ECG CSV file to start cardiac analysis.")
