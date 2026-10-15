"""Synthetic lane-following and vehicle control simulation."""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

if not st.session_state.get("_embedded_ai_project"):
    st.set_page_config(page_title="Autonomous Driving Simulator",page_icon="🚗",layout="wide")
st.markdown("<style>.block-container{max-width:1100px;padding-top:2rem}</style>",unsafe_allow_html=True)
st.title("Autonomous Driving: Lane-Following Lab")
st.caption("Explore a synthetic camera lane detector and a proportional steering controller.")
curvature=st.sidebar.slider("Road curvature",-.012,.012,.004,.001)
noise=st.sidebar.slider("Sensor noise",0.0,12.0,2.0,.5)
speed=st.sidebar.slider("Vehicle speed (m/s)",5,35,18)
gain=st.sidebar.slider("Steering gain",.1,2.0,.8,.1)
steps=st.sidebar.slider("Simulation steps",30,240,100,10)
rng=np.random.default_rng(42)
y=np.linspace(0,120,steps)
center=curvature*(y-60)**2+5*np.sin(y/25)
left_lane=center-3.5;right_lane=center+3.5
detected_left=left_lane+rng.normal(0,noise/10,steps)
detected_right=right_lane+rng.normal(0,noise/10,steps)
lane_center=(detected_left+detected_right)/2
offset=lane_center-center[0]
steering=np.clip(-gain*offset,-1,1)
vehicle=np.zeros(steps);vehicle[0]=0
dt=(y[1]-y[0])/speed
for i in range(1,steps):
    vehicle[i]=vehicle[i-1]+speed*np.sin(steering[i-1]*.18)*dt
fig=go.Figure()
fig.add_trace(go.Scatter(x=left_lane,y=y,name="Left lane boundary",line=dict(color="#fbbf24",dash="dash")))
fig.add_trace(go.Scatter(x=right_lane,y=y,name="Right lane boundary",line=dict(color="#fbbf24",dash="dash")))
fig.add_trace(go.Scatter(x=vehicle,y=y,name="Simulated vehicle path",line=dict(color="#2563eb",width=4)))
fig.update_yaxes(autorange="reversed",title="Distance ahead (m)")
fig.update_xaxes(title="Lateral position (m)")
fig.update_layout(height=620,title="Lane-following trajectory",template="plotly_white")
st.plotly_chart(fig,use_container_width=True)
c1,c2,c3=st.columns(3)
c1.metric("Mean absolute lane offset",f"{np.mean(np.abs(vehicle)):.2f} m")
c2.metric("Steering saturation",f"{np.mean(np.abs(steering)>=.99):.1%}")
c3.metric("Simulated duration",f"{y[-1]/speed:.1f} s")
st.warning("Synthetic kinematic simulation only; no real vehicle, camera, safety validation, or driving instructions.")
