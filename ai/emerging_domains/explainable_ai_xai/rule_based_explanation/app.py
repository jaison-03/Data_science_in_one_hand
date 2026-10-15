"""Decision-tree surrogate rules for explaining black-box classifications."""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

if not st.session_state.get("_embedded_ai_project"):
    st.set_page_config(page_title="Rule-Based Explanation",page_icon="🌳",layout="wide")
st.markdown("<style>.block-container{max-width:1100px;padding-top:2rem}</style>",unsafe_allow_html=True)
st.title("Rule-Based Explanation")
st.caption("Fit an interpretable shallow decision tree to approximate a more complex classifier.")
data=load_wine()
X=pd.DataFrame(data.data,columns=data.feature_names); y=data.target
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.25,random_state=23,stratify=y)
blackbox=RandomForestClassifier(n_estimators=160,max_depth=7,random_state=23).fit(X_train,y_train)
depth=st.sidebar.slider("Surrogate maximum depth",1,6,3)
leaf=st.sidebar.slider("Minimum samples per leaf",2,25,6)
surrogate=DecisionTreeClassifier(max_depth=depth,min_samples_leaf=leaf,random_state=23)
surrogate.fit(X_train,blackbox.predict(X_train))
st.metric("Surrogate fidelity on holdout",f"{accuracy_score(blackbox.predict(X_test),surrogate.predict(X_test)):.1%}")
st.caption("Fidelity is agreement with the black-box model, not accuracy against ground-truth labels.")
rules=export_text(surrogate,feature_names=list(X.columns),decimals=2)
st.code(rules,language="text")
importance=pd.DataFrame({"Feature":X.columns,"Importance":surrogate.feature_importances_}).query("Importance > 0").sort_values("Importance")
st.plotly_chart(px.bar(importance,x="Importance",y="Feature",orientation="h",title="Surrogate split importance"),use_container_width=True)
sample=st.slider("Test sample",0,len(X_test)-1,0)
black_id=int(blackbox.predict(X_test.iloc[[sample]])[0]); surrogate_id=int(surrogate.predict(X_test.iloc[[sample]])[0])
st.write(f"Black box: **{data.target_names[black_id]}** · Rule surrogate: **{data.target_names[surrogate_id]}**")
st.dataframe(pd.DataFrame({"Feature":X.columns,"Value":X_test.iloc[sample].values}),hide_index=True,use_container_width=True)
