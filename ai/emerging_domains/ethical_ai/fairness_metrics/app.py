"""Compute group fairness metrics from labeled predictions."""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import confusion_matrix

if not st.session_state.get("_embedded_ai_project"):
    st.set_page_config(page_title="Fairness Metrics",page_icon="📐",layout="wide")
st.markdown("<style>.block-container{max-width:1100px;padding-top:2rem}</style>",unsafe_allow_html=True)
st.title("Fairness Metrics Dashboard")
st.caption("Upload predictions with actual labels and a group field, or inspect a clearly synthetic example.")
source=st.radio("Data",["Synthetic demo","Upload CSV"],horizontal=True)
if source=="Synthetic demo":
    rng=np.random.default_rng(29); n=700
    group=np.where(rng.random(n)<.5,"Group A","Group B")
    actual=(rng.random(n)<np.where(group=="Group A",.55,.48)).astype(int)
    prediction=(rng.random(n)<np.where(actual==1,np.where(group=="Group A",.84,.66),.17)).astype(int)
    frame=pd.DataFrame({"group":group,"actual":actual,"prediction":prediction})
else:
    uploaded=st.file_uploader("CSV file",type="csv")
    if uploaded is None: st.info("Upload a CSV to continue."); st.stop()
    frame=pd.read_csv(uploaded)
group_col=st.selectbox("Group",frame.columns)
actual_col=st.selectbox("Actual label",frame.columns)
pred_col=st.selectbox("Predicted label",frame.columns)
positive=st.selectbox("Positive class",list(pd.unique(frame[actual_col])))
minimum=st.slider("Minimum group size",1,50,10)
rows=[]
for name,part in frame.dropna(subset=[group_col,actual_col,pred_col]).groupby(group_col,observed=True):
    if len(part)<minimum: continue
    actual=(part[actual_col]==positive).astype(int)
    pred=(part[pred_col]==positive).astype(int)
    tn,fp,fn,tp=confusion_matrix(actual,pred,labels=[0,1]).ravel()
    rows.append({"Group":str(name),"n":len(part),"selection_rate":float(pred.mean()),"TPR":float(tp/(tp+fn)) if tp+fn else np.nan,"FPR":float(fp/(fp+tn)) if fp+tn else np.nan,"accuracy":float((tp+tn)/len(part))})
metrics=pd.DataFrame(rows)
if len(metrics)<2: st.error("At least two groups must meet the minimum group size."); st.stop()
selection_gap=metrics.selection_rate.max()-metrics.selection_rate.min()
disparate_impact=metrics.selection_rate.min()/metrics.selection_rate.max() if metrics.selection_rate.max()>0 else np.nan
tpr_gap=metrics.TPR.max()-metrics.TPR.min(); fpr_gap=metrics.FPR.max()-metrics.FPR.min()
a,b,c,d=st.columns(4)
a.metric("Statistical parity difference",f"{selection_gap:.3f}")
b.metric("Disparate impact (min/max)",f"{disparate_impact:.3f}" if np.isfinite(disparate_impact) else "N/A")
c.metric("Equal opportunity gap (TPR)",f"{tpr_gap:.3f}")
d.metric("False-positive-rate gap",f"{fpr_gap:.3f}")
st.plotly_chart(px.bar(metrics.melt(id_vars=["Group","n"],value_vars=["selection_rate","TPR","FPR"],var_name="Metric",value_name="Rate"),x="Group",y="Rate",color="Metric",barmode="group",title="Group rates"),use_container_width=True)
st.dataframe(metrics,hide_index=True,use_container_width=True)
st.warning("Fairness metrics depend on label choice, thresholds, sampling, and deployment context. Metrics may conflict; use this as an audit aid, not a certification.")
