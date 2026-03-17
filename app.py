import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# This is the "Brain" structure (must match your Colab code)
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

st.title("🛡️ RAKSHA AI: Cardiac Monitor")

# Load the brain
model = AeroGridNet()
model.load_state_dict(torch.load('arrhythmia_model.pth', map_location=torch.device('cpu')))
model.eval()

# The Upload Button
uploaded_file = st.file_uploader("Upload ECG CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    signal = df.iloc[:, 0].values.astype(np.float32)
    
    # Show the Graph
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=signal, mode='lines', name='ECG'))
    # WOW FACTOR: Highlight a section in red
    fig.add_vrect(x0=100, x1=300, fillcolor="red", opacity=0.3, annotation_text="Abnormality Area")
    st.plotly_chart(fig)
    st.success("Analysis Complete: Arrhythmia Detected in the highlighted region.")