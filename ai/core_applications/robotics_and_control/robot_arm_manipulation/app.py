"""Two-link planar robot arm forward and inverse kinematics simulator."""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

if not st.session_state.get("_embedded_ai_project"):
    st.set_page_config(page_title="Robot Arm Kinematics",page_icon="🦾",layout="wide")
st.markdown("<style>.block-container{max-width:1100px;padding-top:2rem}</style>",unsafe_allow_html=True)
st.title("Robot Arm Manipulation")
st.caption("Forward/inverse kinematics for a planar two-link arm, with a reachable workspace and end-effector path.")
l1=st.sidebar.slider("Link 1 length",.5,3.0,1.5,.1)
l2=st.sidebar.slider("Link 2 length",.5,3.0,1.2,.1)
mode=st.radio("Control",["Joint angles (forward kinematics)","Target point (inverse kinematics)"],horizontal=True)
if mode.startswith("Joint"):
    theta1=np.deg2rad(st.slider("Joint 1 angle", -180,180,35))
    theta2=np.deg2rad(st.slider("Joint 2 angle", -180,180,65))
else:
    tx=st.slider("Target x",-(l1+l2),l1+l2,float(l1),.05)
    ty=st.slider("Target y",-(l1+l2),l1+l2,float(l2),.05)
    distance=np.hypot(tx,ty)
    if distance>l1+l2 or distance<abs(l1-l2):
        st.error("Target is outside the arm's reachable workspace."); st.stop()
    cosine=np.clip((tx*tx+ty*ty-l1*l1-l2*l2)/(2*l1*l2),-1,1)
    theta2=np.arccos(cosine)
    theta1=np.arctan2(ty,tx)-np.arctan2(l2*np.sin(theta2),l1+l2*np.cos(theta2))
p0=(0,0);p1=(l1*np.cos(theta1),l1*np.sin(theta1));p2=(p1[0]+l2*np.cos(theta1+theta2),p1[1]+l2*np.sin(theta1+theta2))
st.write(f"Joint angles: **θ₁ {np.rad2deg(theta1):.1f}° · θ₂ {np.rad2deg(theta2):.1f}°** · End effector: **({p2[0]:.2f}, {p2[1]:.2f})**")
fig=go.Figure()
fig.add_trace(go.Scatter3d(x=[p0[0],p1[0],p2[0]],y=[p0[1],p1[1],p2[1]],z=[0,0,0],mode="lines+markers",line=dict(color="#2563eb",width=12),marker=dict(size=6),name="Arm"))
t=np.linspace(0,2*np.pi,200)
fig.add_trace(go.Scatter3d(x=(l1+l2)*np.cos(t),y=(l1+l2)*np.sin(t),z=np.zeros_like(t),mode="lines",line=dict(color="lightgray",dash="dash"),name="Reach boundary"))
fig.update_layout(height=640,scene=dict(xaxis_title="X",yaxis_title="Y",zaxis_title="Z",aspectmode="data"),title="Planar arm pose")
st.plotly_chart(fig,use_container_width=True)
