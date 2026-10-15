"""PCA and t-SNE dimensionality-reduction explorer."""
import numpy as np
import plotly.express as px
import streamlit as st
from sklearn.datasets import load_digits, load_wine
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

if not st.session_state.get("_embedded_ai_project"):
    st.set_page_config(page_title="Dimensionality Reduction Lab",page_icon="🧊",layout="wide")
st.markdown("<style>.block-container{max-width:1150px;padding-top:2rem}</style>",unsafe_allow_html=True)
st.title("Dimensionality Reduction Lab")
st.caption("Project high-dimensional feature sets into 2D and compare structure between PCA and t-SNE.")
dataset=st.sidebar.selectbox("Dataset",["Handwritten digits","Wine chemistry"])
method=st.sidebar.selectbox("Projection",["PCA","t-SNE"])
limit=st.sidebar.slider("Samples to project",200,1500,700,100) if dataset=="Handwritten digits" else 178
if dataset=="Handwritten digits":
    data=load_digits(); X,y=data.data[:limit],data.target[:limit]
    labels=y.astype(str)
else:
    data=load_wine(); X,y=data.data,data.target
    labels=np.array([data.target_names[i] for i in y])
X=StandardScaler().fit_transform(X)
if method=="PCA":
    projection=PCA(n_components=2,random_state=21)
    points=projection.fit_transform(X)
    variance=projection.explained_variance_ratio_
    st.metric("Variance retained in 2D",f"{variance.sum():.1%}",delta=f"PC1 {variance[0]:.1%} · PC2 {variance[1]:.1%}")
else:
    perplexity=st.sidebar.slider("Perplexity",5,50,25)
    points=TSNE(n_components=2,perplexity=min(perplexity,len(X)-1),init="pca",learning_rate="auto",random_state=21).fit_transform(X)
    st.info("t-SNE is a visualization, not a metric-preserving embedding. Distances between separate clusters can be misleading.")
st.plotly_chart(px.scatter(x=points[:,0],y=points[:,1],color=labels,labels={"x":"Component 1","y":"Component 2","color":"Class"},title=f"{method} projection of {dataset.lower()}",opacity=.8),use_container_width=True)
st.caption(f"Original feature count: {X.shape[1]} · projected dimensions: 2")
