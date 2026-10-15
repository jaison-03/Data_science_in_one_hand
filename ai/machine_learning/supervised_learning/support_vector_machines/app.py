"""Interactive SVM classifier and two-dimensional decision boundary."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

if not st.session_state.get("_embedded_ai_project"):
    st.set_page_config(page_title="SVM Explorer", page_icon="〰️", layout="wide")
st.markdown("<style>.block-container{max-width:1150px;padding-top:2rem}</style>", unsafe_allow_html=True)
st.title("Support Vector Machine Explorer")
st.caption("Change the kernel and regularization strength to see how the decision boundary responds.")
c1,c2,c3=st.columns(3)
kernel=c1.selectbox("Kernel",["linear","rbf","poly","sigmoid"])
C=c2.slider("Regularization C",0.05,10.0,1.0,0.05)
gamma_choice=c3.select_slider("Gamma",["scale","0.05","0.1","0.5","1.0","2.0"],value="scale")
gamma="scale" if gamma_choice=="scale" else float(gamma_choice)
noise=st.slider("Dataset noise",0.0,0.5,0.22,0.01)
X,y=make_moons(n_samples=500,noise=noise,random_state=32)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.25,random_state=32,stratify=y)
scaler=StandardScaler().fit(X_train)
X_train=scaler.transform(X_train); X_test=scaler.transform(X_test)
model=SVC(kernel=kernel,C=C,gamma=gamma,probability=True,random_state=32).fit(X_train,y_train)
accuracy=accuracy_score(y_test,model.predict(X_test))
st.metric("Holdout accuracy",f"{accuracy:.1%}")
padding=.7
gx,gy=np.meshgrid(np.linspace(X_train[:,0].min()-padding,X_train[:,0].max()+padding,180),np.linspace(X_train[:,1].min()-padding,X_train[:,1].max()+padding,180))
grid=np.c_[gx.ravel(),gy.ravel()]
Z=model.decision_function(grid).reshape(gx.shape)
fig=go.Figure()
fig.add_trace(go.Contour(x=gx[0],y=gy[:,0],z=Z,showscale=False,colorscale=[[0,"#dbeafe"],[.5,"#ffffff"],[1,"#fce7f3"]],contours=dict(start=-1,end=1,size=1,showlines=True)))
for label,color in [(0,"#2563eb"),(1,"#db2777")]:
    points=X_train[y_train==label]
    fig.add_trace(go.Scatter(x=points[:,0],y=points[:,1],mode="markers",name=f"Class {label}",marker=dict(color=color,size=7,opacity=.7)))
support=model.support_vectors_
fig.add_trace(go.Scatter(x=support[:,0],y=support[:,1],mode="markers",name="Support vectors",marker=dict(symbol="circle-open",size=13,color="#111827",line=dict(width=1.5))))
fig.update_layout(title="SVM decision regions and margins",xaxis_title="Scaled feature 1",yaxis_title="Scaled feature 2",height=620)
st.plotly_chart(fig,use_container_width=True)
st.info(f"Support vectors: {len(support)} | Kernel: {kernel} | C: {C} | Gamma: {gamma}")
