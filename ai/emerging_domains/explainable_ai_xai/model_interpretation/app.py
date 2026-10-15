"""Model interpretation dashboard with permutation importance and SHAP-style values."""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

if not st.session_state.get("_embedded_ai_project"):
    st.set_page_config(page_title="Model Interpretation",page_icon="🔍",layout="wide")
st.markdown("<style>.block-container{max-width:1150px;padding-top:2rem}</style>",unsafe_allow_html=True)
st.title("Model Interpretation Dashboard")
st.caption("Inspect global permutation importance and local feature contributions for a held-out prediction.")
dataset=st.sidebar.selectbox("Dataset",["Wine classification","Breast cancer classification"])
data=load_wine() if dataset=="Wine classification" else load_breast_cancer()
X=pd.DataFrame(data.data,columns=data.feature_names); y=data.target
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.25,random_state=19,stratify=y)
model=make_pipeline(StandardScaler(),RandomForestClassifier(n_estimators=180,max_depth=8,random_state=19)).fit(X_train,y_train)
pred=model.predict(X_test); probabilities=model.predict_proba(X_test)
st.metric("Holdout accuracy",f"{accuracy_score(y_test,pred):.1%}")
with st.spinner("Measuring permutation importance..."):
    importance=permutation_importance(model,X_test,y_test,n_repeats=8,random_state=19,n_jobs=-1)
ranking=pd.DataFrame({"Feature":X.columns,"Importance":importance.importances_mean,"Std":importance.importances_std}).sort_values("Importance",ascending=False)
top=st.slider("Show top features",5,min(30,X.shape[1]),min(12,X.shape[1]))
st.plotly_chart(px.bar(ranking.head(top).sort_values("Importance"),x="Importance",y="Feature",error_x="Std",orientation="h",title="Global permutation importance"),use_container_width=True)
sample=st.slider("Test example",0,len(X_test)-1,0)
st.write(f"Predicted **{data.target_names[int(pred[sample])]}** · actual **{data.target_names[int(y_test[sample])]}** · confidence **{probabilities[sample].max():.1%}**")
class_id=int(pred[sample])
contributions=pd.DataFrame({"Feature":X.columns,"Contribution to predicted-class probability":np.zeros(len(X.columns))})
# Perturb one feature at a time to produce model-specific local probability changes.
for i,column in enumerate(X.columns):
    altered=X_test.iloc[[sample]].copy()
    altered.loc[:,column]=X_train[column].median()
    contributions.loc[i,"Contribution to predicted-class probability"]=probabilities[sample,class_id]-model.predict_proba(altered)[0,class_id]
contributions["Feature value"]=X_test.iloc[sample].to_numpy()
contributions=contributions.reindex(contributions["Contribution to predicted-class probability"].abs().sort_values(ascending=False).index).head(top)
st.plotly_chart(px.bar(contributions.sort_values("Contribution to predicted-class probability"),x="Contribution to predicted-class probability",y="Feature",color="Contribution to predicted-class probability",orientation="h",title="Local one-feature perturbation effects"),use_container_width=True)
st.caption("Local bars measure the change when one feature is reset to its training median; correlated features and interactions mean these are not causal explanations.")
