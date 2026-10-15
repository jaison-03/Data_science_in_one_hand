"""Audit prediction rates across supplied demographic groups."""
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

if not st.session_state.get("_embedded_ai_project"):
    st.set_page_config(page_title="Bias Detection Audit",page_icon="⚖️",layout="wide")
st.markdown("<style>.block-container{max-width:1100px;padding-top:2rem}</style>",unsafe_allow_html=True)
st.title("Bias Detection Audit")
st.caption("Compare model outcomes across groups in a synthetic or uploaded dataset. This is a screening tool, not a fairness certification.")
mode=st.radio("Data source",["Synthetic demonstration","Upload CSV"],horizontal=True)
if mode=="Synthetic demonstration":
    X,y=make_classification(n_samples=800,n_features=5,n_informative=4,n_redundant=0,random_state=31)
    data=pd.DataFrame(X,columns=[f"feature_{i}" for i in range(X.shape[1])])
    data["group"]=["Group A" if i%2==0 else "Group B" for i in range(len(data))]
    data["outcome"]=y
    data.loc[data["group"]=="Group B","feature_0"]-=.65
else:
    file=st.file_uploader("CSV with numeric features, one outcome column, and one group column",type="csv")
    if file is None: st.info("Upload a CSV to start the audit."); st.stop()
    data=pd.read_csv(file)
    if len(data)>20000: st.error("Limit uploads to 20,000 rows."); st.stop()
outcome=st.selectbox("Actual outcome column",data.columns,index=len(data.columns)-1)
group=st.selectbox("Demographic group column",data.columns,index=max(0,len(data.columns)-2))
features=[c for c in data.select_dtypes(include="number").columns if c not in {outcome,group}]
if not features: st.error("The dataset needs at least one numeric feature."); st.stop()
if data[group].nunique()<2: st.error("The group column must contain at least two groups."); st.stop()
positive=st.selectbox("Positive outcome value",list(pd.unique(data[outcome])))
minimum=st.slider("Minimum group size",1,50,10)
try:
    usable=data.dropna(subset=features+[outcome,group]).copy()
    if usable[outcome].nunique()!=2: raise ValueError("Select data with a binary outcome.")
    X=usable[features]; y=(usable[outcome]==positive).astype(int)
    if y.value_counts().min()<2: raise ValueError("Each outcome class needs at least two samples.")
    xtr,xte,ytr,yte=train_test_split(X,y,test_size=.25,random_state=31,stratify=y)
    model=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000)).fit(xtr,ytr)
    evaluated=usable.loc[xte.index].copy()
    evaluated["predicted_positive"]=model.predict(xte)
    rates=evaluated.groupby(group,observed=True).agg(samples=("predicted_positive","size"),selection_rate=("predicted_positive","mean"),actual_positive_rate=(outcome,lambda s:(s==positive).mean())).reset_index()
    rates=rates[rates["samples"]>=minimum]
    if len(rates)<2: raise ValueError("Fewer than two groups meet the selected minimum size.")
    gap=float(rates.selection_rate.max()-rates.selection_rate.min())
    ratio=float(rates.selection_rate.min()/rates.selection_rate.max()) if rates.selection_rate.max() else 0
    a,b,c=st.columns(3); a.metric("Selection-rate gap",f"{gap:.1%}"); b.metric("Disparate-impact ratio",f"{ratio:.2f}"); c.metric("Included groups",len(rates))
    st.plotly_chart(px.bar(rates,x=group,y="selection_rate",color=group,error_y=None,text_auto=".1%",title="Predicted positive rate by group"),use_container_width=True)
    st.dataframe(rates,hide_index=True,use_container_width=True)
    st.warning("A disparity is a prompt for investigation, not proof of unlawful or causal bias. Consider data quality, sample size, context, and applicable law.")
except ValueError as error:
    st.error(str(error))
