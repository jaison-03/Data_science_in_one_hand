"""Interactive K-Means and DBSCAN clustering workbench."""
import numpy as np
import plotly.express as px
import streamlit as st
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

if not st.session_state.get("_embedded_ai_project"):
    st.set_page_config(page_title="Clustering Workbench",page_icon="🫧",layout="wide")
st.markdown("<style>.block-container{max-width:1150px;padding-top:2rem}</style>",unsafe_allow_html=True)
st.title("Clustering Workbench")
st.caption("Generate a dataset, tune a clustering method, and inspect cluster assignments and noise.")
kind=st.sidebar.selectbox("Data shape",["Blobs","Interleaving moons"])
samples=st.sidebar.slider("Samples",100,1500,600,50)
noise=st.sidebar.slider("Noise",0.0,0.6,0.12,0.02)
if kind=="Blobs":
    centers=st.sidebar.slider("True blob centers",2,8,4)
    X,_=make_blobs(n_samples=samples,centers=centers,cluster_std=1.0+noise,random_state=14)
else:
    X,_=make_moons(n_samples=samples,noise=noise,random_state=14)
X=StandardScaler().fit_transform(X)
algorithm=st.sidebar.selectbox("Algorithm",["K-Means","DBSCAN"])
if algorithm=="K-Means":
    count=st.sidebar.slider("Clusters (k)",2,10,4)
    labels=KMeans(n_clusters=count,n_init=10,random_state=14).fit_predict(X)
else:
    eps=st.sidebar.slider("Neighborhood radius (eps)",0.05,1.5,0.28,0.01)
    min_samples=st.sidebar.slider("Minimum neighbors",2,30,6)
    labels=DBSCAN(eps=eps,min_samples=min_samples).fit_predict(X)
clusters=set(labels)-{-1}
noise_count=int(np.sum(labels==-1))
c1,c2,c3=st.columns(3)
c1.metric("Clusters found",len(clusters)); c2.metric("Noise points",noise_count); c3.metric("Silhouette",f"{silhouette_score(X,labels):.3f}" if len(clusters)>1 and len(clusters)<len(X) else "N/A")
st.plotly_chart(px.scatter(x=X[:,0],y=X[:,1],color=labels.astype(str),labels={"x":"Feature 1 (scaled)","y":"Feature 2 (scaled)","color":"Cluster"},title=f"{algorithm} assignments",opacity=.8),use_container_width=True)
st.caption("Silhouette is omitted for a single cluster or when every sample has its own label.")
