"""Interactive PyTorch MLP classifier with a training and evaluation dashboard."""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.datasets import load_digits, make_moons
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    DataLoader = TensorDataset = None
    TORCH_AVAILABLE = False

if not st.session_state.get("_embedded_ai_project"):
    st.set_page_config(page_title="Neural Network Lab", page_icon="🧠", layout="wide")
st.markdown("<style>.block-container{max-width:1200px;padding-top:2rem} .hero{padding:1.5rem;border-radius:16px;background:linear-gradient(120deg,#172554,#0e7490);color:white}</style>", unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>Neural Network Training Lab</h1><p>Train and inspect a multilayer perceptron on a reproducible dataset.</p></div>', unsafe_allow_html=True)

dataset = st.sidebar.selectbox("Dataset", ["Digits (8×8)", "Two-moons"])
epochs = st.sidebar.slider("Epochs", 5, 100, 30, 5)
hidden = st.sidebar.slider("Hidden units", 8, 128, 48, 8)
learning_rate = st.sidebar.select_slider("Learning rate", [0.0005, 0.001, 0.005, 0.01], value=0.005)
if dataset == "Digits (8×8)":
    data = load_digits()
    features, labels = data.data.astype("float32") / 16.0, data.target.astype("int64")
    names = [str(i) for i in range(10)]
else:
    features, labels = make_moons(n_samples=900, noise=0.22, random_state=17)
    features, labels = features.astype("float32"), labels.astype("int64")
    names = ["class 0", "class 1"]
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.25, random_state=17, stratify=labels)
scaler = StandardScaler().fit(x_train)
x_train, x_test = scaler.transform(x_train).astype("float32"), scaler.transform(x_test).astype("float32")

if TORCH_AVAILABLE:
    class MLP(nn.Module):
        def __init__(self, dimensions: int, classes: int, width: int):
            super().__init__()
            self.network = nn.Sequential(nn.Linear(dimensions, width), nn.ReLU(), nn.Dropout(0.1),
                                         nn.Linear(width, width // 2), nn.ReLU(), nn.Linear(width // 2, classes))
        def forward(self, values):
            return self.network(values)

@st.cache_resource
def train_model(x_values, y_values, width, rate, rounds):
    if not TORCH_AVAILABLE:
        fallback=MLPClassifier(hidden_layer_sizes=(width,width//2),activation="relu",solver="adam",
                               learning_rate_init=rate,max_iter=rounds,batch_size=64,
                               early_stopping=True,random_state=17)
        fallback.fit(x_values,y_values)
        return fallback,list(fallback.loss_curve_)
    torch.manual_seed(17)
    model = MLP(x_values.shape[1], int(np.max(y_values)) + 1, width)
    loader = DataLoader(TensorDataset(torch.tensor(x_values), torch.tensor(y_values)), batch_size=64, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=rate)
    loss_function = nn.CrossEntropyLoss()
    losses = []
    for _ in range(rounds):
        model.train()
        batch_losses = []
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = loss_function(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        losses.append(float(np.mean(batch_losses)))
    return model, losses

with st.spinner("Training the neural network..."):
    model, history = train_model(x_train, y_train, hidden, learning_rate, epochs)
if TORCH_AVAILABLE:
    model.eval()
    with torch.no_grad():
        probabilities = torch.softmax(model(torch.tensor(x_test)), dim=1).numpy()
else:
    probabilities=model.predict_proba(x_test)
    st.info("PyTorch is not installed in this environment; using scikit-learn's MLP fallback. Install the root requirements to run the PyTorch model.")
parameter_count=sum(p.numel() for p in model.parameters()) if TORCH_AVAILABLE else sum(coef.size for coef in model.coefs_)+sum(bias.size for bias in model.intercepts_)
predictions = probabilities.argmax(axis=1)
left, right = st.columns([1, 1])
left.metric("Test accuracy", f"{accuracy_score(y_test, predictions):.1%}")
right.metric("Parameters", f"{parameter_count:,}")
tabs = st.tabs(["Training curves", "Confusion matrix", "Try an example"])
with tabs[0]:
    st.plotly_chart(px.line(x=list(range(1, len(history) + 1)), y=history, labels={"x": "Epoch", "y": "Training loss"}, title="Cross-entropy loss"), use_container_width=True)
with tabs[1]:
    cm = confusion_matrix(y_test, predictions)
    st.plotly_chart(px.imshow(cm, x=names, y=names, text_auto=True, labels={"x": "Predicted", "y": "Actual", "color": "Count"}, title="Test confusion matrix"), use_container_width=True)
with tabs[2]:
    index = st.slider("Test sample", 0, len(x_test) - 1, 0)
    st.write(f"Prediction: **{names[int(predictions[index])]}** | Actual: **{names[int(y_test[index])]}** | Confidence: **{probabilities[index].max():.1%}**")
    st.plotly_chart(px.bar(x=names, y=probabilities[index], labels={"x": "Class", "y": "Probability"}), use_container_width=True)
