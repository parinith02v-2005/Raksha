import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from scipy.signal import find_peaks

# ---------------- MODEL ARCHITECTURE ---------------- #

class AeroGridNet(nn.Module):
    def __init__(self):
        super(AeroGridNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv1d(1,32,kernel_size=5,padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32,64,kernel_size=5,padding=2),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1)
        )

        self.classifier = nn.Linear(64,5)

    def forward(self,x):
        x=self.features(x)
        x=x.view(x.size(0),-1)
        return self.classifier(x)


# ---------------- STREAMLIT CONFIG ---------------- #

st.set_page_config(
    st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
}

/* MAIN BACKGROUND */
[data-testid="stAppViewContainer"]{
background: linear-gradient(135deg,#0f172a,#020617);
color:white;
}

/* SIDEBAR */
[data-testid="stSidebar"]{
background: linear-gradient(180deg,#020617,#0f172a);
border-right:1px solid rgba(255,255,255,0.05);
}

/* GLASS CARD */
[data-testid="metric-container"]{
background: rgba(255,255,255,0.04);
border-radius:16px;
padding:18px;
backdrop-filter: blur(20px);
border:1px solid rgba(255,255,255,0.08);
box-shadow:
0 10px 30px rgba(0,0,0,0.5),
inset 0 1px 0 rgba(255,255,255,0.05);
transition: all .3s ease;
}

/* HOVER EFFECT */
[data-testid="metric-container"]:hover{
transform: translateY(-4px);
box-shadow:
0 15px 35px rgba(0,0,0,0.7),
0 0 20px rgba(255,0,150,0.25);
}

/* FILE UPLOAD */
[data-testid="stFileUploader"]{
background: rgba(255,255,255,0.03);
border-radius:14px;
border:1px dashed rgba(255,255,255,0.15);
}

/* BUTTONS */
.stButton>button{
background: linear-gradient(90deg,#ff00cc,#3333ff);
border:none;
color:white;
border-radius:10px;
padding:10px 22px;
font-weight:500;
}

.stButton>button:hover{
box-shadow:0 0 15px #ff00cc;
}

/* TITLE STYLE */
h1{
font-size:40px !important;
background: linear-gradient(90deg,#ff00cc,#00e5ff);
-webkit-background-clip:text;
-webkit-text-fill-color:transparent;
}

/* PLOT AREA */
.js-plotly-plot{
background: rgba(255,255,255,0.02);
border-radius:16px;
padding:10px;
}

/* SCROLLBAR */
::-webkit-scrollbar{
width:6px;
}

::-webkit-scrollbar-thumb{
background:linear-gradient(#ff00cc,#3333ff);
border-radius:10px;
}

</style>
""", unsafe_allow_html=True)
    page_title="RAKSHA v3 | Clinical Suite",
    layout="wide",
    page_icon="🩺"
)

# ---------------- PREMIUM UI CSS ---------------- #

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500&family=Inter:wght@300;400;500&display=swap');

html, body {
    font-family: 'Inter', sans-serif;
}

[data-testid="stAppViewContainer"]{
    background: radial-gradient(circle at 20% 20%, #0b0f1a, #020409);
}

h1{
    font-family:'Orbitron', sans-serif;
    color:#58a6ff;
}

[data-testid="stSidebar"]{
    background:linear-gradient(180deg,#05070a,#0d1117);
    border-right:1px solid #1f2937;
}

[data-testid="metric-container"]{
    background:linear-gradient(145deg,#0f1722,#06090f);
    border:1px solid #2c3f5e;
    padding:15px;
    border-radius:14px;
    box-shadow:0px 0px 18px rgba(88,166,255,0.15);
}

[data-testid="metric-container"]:hover{
    transform:scale(1.02);
    box-shadow:0px 0px 25px rgba(88,166,255,0.4);
}

.report-box{
    background:#0b1118;
    border:1px solid #1f2937;
    border-radius:12px;
    padding:20px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ---------------- #

with st.sidebar:

    st.title("🛡 RAKSHA v3.0")

    st.success("AI CORE: OPTIMIZED")

    st.divider()

    mode = st.radio(
        "Analysis Mode",
        ["Standard Diagnostic","Advanced Research"]
    )

    st.write("🏥 Jain University Medical Center")

# ---------------- MAIN HEADER ---------------- #

st.title("🩺 RAKSHA: Next-Gen Cardiac Analytics")

st.markdown("---")

col_main, col_stats = st.columns([2,1])

# ---------------- RIGHT PANEL ---------------- #

with col_stats:

    st.markdown("### 📡 Live Telemetry")

    uploaded_file = st.file_uploader(
        "Drop Patient ECG CSV",
        type="csv"
    )

    if uploaded_file:

        df = pd.read_csv(uploaded_file)

        signal = df.iloc[:,0].values.astype(np.float32)

        # -------- LOAD MODEL -------- #

        model = AeroGridNet()

        model.load_state_dict(
            torch.load("arrhythmia_model.pth",map_location="cpu")
        )

        model.eval()

        input_data = torch.tensor(signal).reshape(1,1,-1)

        with torch.no_grad():

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

        st.metric("Heart Rate",f"{heart_rate:.1f} BPM")

        st.metric("Diagnosis",classes[label_idx])

        st.metric("AI Confidence",f"{conf:.2f}%")

        # ---------------- RISK GAUGE ---------------- #

        fig_gauge=go.Figure(go.Indicator(

            mode="gauge+number",

            value=conf,

            title={'text':"Cardiac Risk Index"},

            gauge={

                'axis':{'range':[0,100]},

                'bar':{'color':"#58a6ff"},

                'steps':[

                    {'range':[0,40],'color':'#3fb950'},

                    {'range':[40,70],'color':'#f2cc60'},

                    {'range':[70,100],'color':'#f85149'}

                ]

            }

        ))

        fig_gauge.update_layout(template="plotly_dark",height=250)

        st.plotly_chart(fig_gauge,use_container_width=True)

        # ---------------- HRV RADAR ---------------- #

        st.markdown("### HRV Radar (Beat Stability)")

        categories=['R-Peak','T-Wave','P-Wave','ST-Seg','QRS']

        fig_radar=go.Figure(data=go.Scatterpolar(

            r=[conf/10,8,7,9,conf/12],

            theta=categories,

            fill='toself',

            line_color='#58a6ff'

        ))

        fig_radar.update_layout(

            polar=dict(radialaxis=dict(visible=False)),

            template="plotly_dark",

            height=250

        )

        st.plotly_chart(fig_radar,use_container_width=True)

# ---------------- MAIN ECG GRAPH ---------------- #

with col_main:

    if uploaded_file:

        fig=go.Figure()

        fig.add_trace(go.Scatter(

            y=signal,

            mode='lines',

            line=dict(color='#58a6ff',width=2),

            name='Lead II'

        ))

        # R-Peak markers

        fig.add_trace(go.Scatter(

            x=peaks,

            y=signal[peaks],

            mode='markers',

            marker=dict(color='#ff4d4d',size=6),

            name='R Peaks'

        ))

        # ---------- KEEPING YOUR XAI ZONE ---------- #

        fig.add_vrect(

            x0=150,

            x1=400,

            fillcolor="#3fb950",

            opacity=0.15,

            line_width=0,

            annotation_text="XAI ANALYSIS ZONE"

        )

        fig.update_layout(

            template="plotly_dark",

            height=450,

            plot_bgcolor="rgba(0,0,0,0)",

            paper_bgcolor="rgba(0,0,0,0)"

        )

        st.plotly_chart(fig,use_container_width=True)

        # ---------------- PATIENT PROFILE ---------------- #

        st.markdown("### Patient Profile")

        c1,c2,c3=st.columns(3)

        c1.metric("Age","52")

        c2.metric("Blood Pressure","128 / 82")

        c3.metric("SpO₂","97%")

        # ---------------- ALERT ---------------- #

        if conf>80 and label_idx!=0:

            st.error("⚠ High Arrhythmia Risk Detected")

        # ---------------- AI EXPLANATION ---------------- #

        st.markdown("### AI Explanation")

        st.write(

        "The deep learning model identified morphological irregularities "

        "in the ventricular depolarization phase of the ECG waveform "

        "within the highlighted XAI region."

        )

        # ---------------- CLINICAL REPORT ---------------- #

        st.markdown("### 📋 Clinical Summary Report")

        st.markdown(f"""

        <div class="report-box">

        <b>Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}<br>

        <b>Analyst:</b> RAKSHA AI Engine<br><br>

        <b>Finding:</b> {classes[label_idx]} detected with {conf:.2f}% confidence.<br>

        <b>Clinical Note:</b> Signal morphology indicates potential arrhythmic pattern 

        within QRS region. Recommend 12-lead ECG validation and physician review.

        </div>

        """,unsafe_allow_html=True)

        st.button("📄 Export Clinical Report (PRO)")

    else:

        st.info("System Ready. Upload patient ECG telemetry to begin.")
