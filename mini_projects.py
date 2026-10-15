"""Interactive, self-contained machine-learning teaching projects."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs, make_classification, make_regression, load_digits
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


PROJECTS = {
    "Advertising Popularity Prediction": "regression",
    "Diagnostics": "classification",
    "Customer Retention": "classification",
    "Image Classification": "classification",
    "Identity Fraud Detection": "classification",
    "Feature Elicitation": "dimensionality",
    "Feature Selection": "dimensionality",
    "Structure Discovery": "dimensionality",
    "Meaningful Compression": "dimensionality",
    "Big Data Visualisation": "dimensionality",
    "Big Data Visualization": "dimensionality",
    "Customer Segmentation": "clustering",
    "Targeted Marketing": "clustering",
    "Recommender Systems": "clustering",
    "Recommender System": "clustering",
    "Game AI": "reinforcement",
    "Skill Acquisition": "reinforcement",
    "Learning Tasks": "reinforcement",
    "Robot Navigation": "reinforcement",
    "Real-time decisions": "reinforcement",
    "Real-Time Decisions": "reinforcement",
}


def _section(title, explanation):
    st.title(title)
    st.caption(explanation)


def _show_confusion(y_true, y_pred, labels=None):
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    figure, axis = plt.subplots(figsize=(4.5, 3.5))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    axis.set_xticks(range(len(matrix)))
    axis.set_yticks(range(len(matrix)))
    if labels is not None:
        axis.set_xticklabels(labels)
        axis.set_yticklabels(labels)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    st.pyplot(figure, use_container_width=False)
    plt.close(figure)


def _advertising():
    _section(
        "Advertising Popularity Prediction",
        "A simulated advertising dataset teaches how a linear model estimates sales from channel spend. "
        "It is a learning example, not a real campaign forecast.",
    )
    rng = np.random.default_rng(11)
    spend = rng.uniform([0, 0, 0], [300, 50, 120], size=(240, 3))
    sales = 3.5 + spend @ np.array([0.045, 0.19, 0.002]) + rng.normal(0, 2.5, 240)
    features = pd.DataFrame(spend, columns=["TV", "Radio", "Newspaper"])
    x_train, x_test, y_train, y_test = train_test_split(
        features, sales, test_size=0.25, random_state=11
    )
    model = LinearRegression().fit(x_train, y_train)
    predictions = model.predict(x_test)
    controls = st.columns(3)
    values = [
        controls[0].number_input("TV spend ($ thousands)", 0.0, 300.0, 150.0),
        controls[1].number_input("Radio spend ($ thousands)", 0.0, 50.0, 25.0),
        controls[2].number_input("Newspaper spend ($ thousands)", 0.0, 120.0, 30.0),
    ]
    estimate = float(model.predict(pd.DataFrame([values], columns=features.columns))[0])
    st.metric("Estimated sales (thousands)", f"{estimate:.1f}")
    metrics = st.columns(3)
    metrics[0].metric("Holdout MAE", f"{mean_absolute_error(y_test, predictions):.2f}")
    metrics[1].metric("Holdout R²", f"{r2_score(y_test, predictions):.3f}")
    metrics[2].metric("Intercept", f"{model.intercept_:.2f}")
    st.write("**Learned channel coefficients** (estimated sales per $1,000 of spend)")
    st.dataframe(pd.DataFrame({"Channel": features.columns, "Coefficient": model.coef_}), hide_index=True)
    figure, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for index, axis in enumerate(axes):
        axis.scatter(features.iloc[:, index], sales, alpha=0.45, s=18)
        axis.set_xlabel(f"{features.columns[index]} spend ($000s)")
        axis.set_ylabel("Sales (000s)")
    figure.tight_layout()
    st.pyplot(figure, use_container_width=True)
    plt.close(figure)


def _diagnostics():
    _section(
        "Health Diagnostics — Educational Demo",
        "This synthetic diabetes-risk classifier is for learning only. It is not medical advice, "
        "does not diagnose disease, and must not guide care.",
    )
    x, y = make_classification(
        n_samples=900, n_features=5, n_informative=4, n_redundant=0,
        weights=[0.68, 0.32], class_sep=0.9, random_state=21,
    )
    columns = ["Glucose", "BMI", "Age", "Blood pressure", "Activity"]
    data = pd.DataFrame(x, columns=columns)
    scaler = StandardScaler().fit(data)
    xs = scaler.transform(data)
    x_train, x_test, y_train, y_test = train_test_split(
        xs, y, test_size=0.25, random_state=21, stratify=y
    )
    model = LogisticRegression(max_iter=1000).fit(x_train, y_train)
    inputs = st.columns(3)
    glucose = inputs[0].slider("Glucose (demo scale)", -2.5, 2.5, 0.4, 0.1)
    bmi = inputs[1].slider("BMI (demo scale)", -2.5, 2.5, 0.2, 0.1)
    age = inputs[2].slider("Age (demo scale)", -2.5, 2.5, 0.1, 0.1)
    threshold = st.slider("Flag as higher risk above", 0.1, 0.9, 0.5, 0.05)
    row = np.array([[glucose, bmi, age, 0.0, 0.0]])
    probability = float(model.predict_proba(row)[0, 1])
    st.metric("Illustrative model risk score", f"{probability:.0%}")
    st.warning(
        "Higher-risk demo flag" if probability >= threshold else "Lower-risk demo flag",
        icon="⚠️",
    )
    st.metric("Synthetic holdout accuracy", f"{accuracy_score(y_test, model.predict(x_test)):.1%}")
    _show_confusion(y_test, model.predict(x_test), labels=[0, 1])


def _churn():
    _section(
        "Customer Retention — Churn Prediction",
        "Explore synthetic telecom customers and compare logistic regression with a decision tree.",
    )
    rng = np.random.default_rng(23)
    n = 1200
    tenure = rng.integers(1, 73, n)
    charges = rng.normal(72, 22, n).clip(18, 150)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.20])
    logit = 1.0 - 0.045 * tenure + 0.018 * (charges - 70) + (contract == "Month-to-month") * 0.8 - (contract == "Two year") * 0.9
    probability = 1 / (1 + np.exp(-logit))
    churn = rng.binomial(1, probability)
    encoded_contract = pd.get_dummies(contract, dtype=float)
    data = pd.DataFrame({"Tenure": tenure, "Monthly charges": charges}).join(encoded_contract)
    x_train, x_test, y_train, y_test = train_test_split(
        data, churn, test_size=0.25, random_state=23, stratify=churn
    )
    choice = st.selectbox("Model", ["Logistic Regression", "Decision Tree"])
    model = (
        LogisticRegression(max_iter=1000, class_weight="balanced")
        if choice == "Logistic Regression"
        else DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=23)
    ).fit(x_train, y_train)
    controls = st.columns(3)
    user_tenure = controls[0].slider("Tenure (months)", 1, 72, 18)
    user_charges = controls[1].slider("Monthly charges ($)", 18, 150, 85)
    user_contract = controls[2].selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    row = pd.DataFrame([{
        "Tenure": user_tenure, "Monthly charges": user_charges,
        **{column: float(column == user_contract) for column in encoded_contract.columns},
    }]).reindex(columns=data.columns, fill_value=0)
    churn_probability = float(model.predict_proba(row)[0, 1])
    st.metric("Churn probability", f"{churn_probability:.1%}")
    st.warning("Model predicts churn" if churn_probability >= 0.5 else "Model predicts retention")
    st.metric("Synthetic holdout accuracy", f"{accuracy_score(y_test, model.predict(x_test)):.1%}")
    _show_confusion(y_test, model.predict(x_test), labels=[0, 1])


def _image_classification():
    _section(
        "Image Classification — Handwritten Digits",
        "A logistic-regression model recognizes 8×8 digit images from scikit-learn's built-in digits dataset.",
    )
    digits = load_digits()
    x_train, x_test, y_train, y_test = train_test_split(
        digits.data, digits.target, test_size=0.25, random_state=5, stratify=digits.target
    )
    model = LogisticRegression(max_iter=2000).fit(x_train, y_train)
    sample = st.slider("Test image number", 0, len(x_test) - 1, 0)
    predicted = int(model.predict(x_test[sample:sample + 1])[0])
    display = st.columns([1, 2])
    display[0].image(
        (x_test[sample].reshape(8, 8) / 16.0),
        caption=f"Actual digit: {y_test[sample]}",
        width=180,
        clamp=True,
    )
    display[1].metric("Predicted digit", str(predicted))
    display[1].metric("Test accuracy", f"{accuracy_score(y_test, model.predict(x_test)):.1%}")
    _show_confusion(y_test, model.predict(x_test), labels=list(range(10)))


def _fraud():
    _section(
        "Identity Fraud Detection — Transaction Anomaly Demo",
        "Synthetic transactions are used to teach imbalanced classification. The scores are not suitable "
        "for real fraud decisions.",
    )
    x, y = make_classification(
        n_samples=1600, n_features=5, n_informative=4, n_redundant=0,
        weights=[0.97, 0.03], class_sep=1.2, random_state=31,
    )
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=31, stratify=y
    )
    model = LogisticRegression(class_weight="balanced", max_iter=1000).fit(x_train, y_train)
    controls = st.columns(3)
    amount = controls[0].slider("Amount (standardized)", -3.0, 3.0, 1.0, 0.1)
    distance = controls[1].slider("Location deviation", -3.0, 3.0, 1.1, 0.1)
    velocity = controls[2].slider("Spending velocity", -3.0, 3.0, 0.8, 0.1)
    row = np.array([[amount, distance, velocity, 0.0, 0.0]])
    score = float(model.predict_proba(row)[0, 1])
    cutoff = st.slider("Review transactions above", 0.1, 0.9, 0.5, 0.05)
    st.metric("Illustrative fraud score", f"{score:.1%}")
    st.warning("Flag for manual review" if score >= cutoff else "Not flagged by this demo")
    predictions = model.predict(x_test)
    st.metric("Synthetic holdout accuracy", f"{accuracy_score(y_test, predictions):.1%}")
    _show_confusion(y_test, predictions, labels=[0, 1])
    st.caption(f"Holdout set size: {len(y_test)} transactions.")


def _feature_selection():
    _section(
        "Feature Elicitation / Selection",
        "Adjust the variance and correlation filters, then inspect a model's feature importance on synthetic data.",
    )
    rng = np.random.default_rng(41)
    x, y = make_regression(n_samples=500, n_features=8, n_informative=4, noise=14, random_state=41)
    data = pd.DataFrame(x, columns=[f"Feature {i}" for i in range(1, 9)])
    data["Near-constant"] = rng.normal(0, 0.015, len(data))
    data["Duplicate signal"] = data["Feature 1"] + rng.normal(0, 0.02, len(data))
    variance_cutoff = st.slider("Minimum variance", 0.0, 0.2, 0.01, 0.005)
    correlation_cutoff = st.slider("Drop one of highly correlated pairs above", 0.80, 1.0, 0.98, 0.01)
    variances = data.var()
    retained = list(variances[variances >= variance_cutoff].index)
    corr = data[retained].corr().abs()
    dropped_corr = set()
    for i, column in enumerate(retained):
        if column in dropped_corr:
            continue
        for other in retained[i + 1:]:
            if other not in dropped_corr and corr.loc[column, other] >= correlation_cutoff:
                dropped_corr.add(other)
    retained = [column for column in retained if column not in dropped_corr]
    model = RandomForestClassifier(n_estimators=80, random_state=41)
    labels = pd.qcut(y, q=2, labels=False)
    model.fit(data[retained], labels)
    importance = pd.Series(model.feature_importances_, index=retained).sort_values()
    st.write(f"Kept **{len(retained)} of {data.shape[1]}** features: {', '.join(retained)}")
    figure, axis = plt.subplots(figsize=(8, 4))
    importance.plot.barh(ax=axis, color="#2a9d8f")
    axis.set_title("Tree-based feature importance (synthetic target)")
    figure.tight_layout()
    st.pyplot(figure, use_container_width=True)
    plt.close(figure)
    st.caption(f"Dropped low-variance: {', '.join(variances[variances < variance_cutoff].index) or 'none'}; "
               f"high-correlation: {', '.join(sorted(dropped_corr)) or 'none'}.")


def _structure_discovery():
    _section(
        "Structure Discovery — PCA",
        "Principal Component Analysis compresses correlated measurements into two plotted components.",
    )
    samples = st.slider("Number of synthetic observations", 100, 1200, 500, 100)
    dimensions = st.slider("Original dimensions", 3, 12, 6)
    x, groups = make_blobs(n_samples=samples, n_features=dimensions, centers=4, random_state=51)
    scaled = StandardScaler().fit_transform(x)
    pca = PCA(n_components=2).fit(scaled)
    projected = pca.transform(scaled)
    figure, axis = plt.subplots(figsize=(8, 5))
    points = axis.scatter(projected[:, 0], projected[:, 1], c=groups, cmap="tab10", alpha=0.75)
    axis.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    axis.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    axis.set_title("Synthetic observations projected to two dimensions")
    figure.colorbar(points, ax=axis, label="Generated group")
    figure.tight_layout()
    st.pyplot(figure, use_container_width=True)
    plt.close(figure)
    st.metric("Variance explained by 2D view", f"{pca.explained_variance_ratio_.sum():.1%}")


def _compression():
    _section(
        "Meaningful Compression",
        "Compare PCA low-rank compression with the original synthetic numeric data and measure reconstruction loss.",
    )
    rng = np.random.default_rng(61)
    latent = rng.normal(size=(400, 4))
    mixing = rng.normal(size=(4, 16))
    original = latent @ mixing + rng.normal(scale=0.12, size=(400, 16))
    components = st.slider("Retain PCA components", 1, 16, 4)
    quantization = st.slider("Quantization levels", 2, 32, 8)
    scaler = StandardScaler().fit(original)
    scaled = scaler.transform(original)
    pca = PCA(n_components=components).fit(scaled)
    encoded = pca.transform(scaled)
    reconstructed = scaler.inverse_transform(pca.inverse_transform(encoded))
    low, high = reconstructed.min(), reconstructed.max()
    quantized = np.round((reconstructed - low) / max(high - low, 1e-9) * (quantization - 1))
    quantized = low + quantized / (quantization - 1) * (high - low)
    rmse = float(np.sqrt(np.mean((original - quantized) ** 2)))
    original_values = original.size
    compressed_values = encoded.size + 2 * components
    metrics = st.columns(3)
    metrics[0].metric("Reconstruction RMSE", f"{rmse:.3f}")
    metrics[1].metric("Numeric values retained", f"{compressed_values:,} / {original_values:,}")
    metrics[2].metric("PCA variance retained", f"{pca.explained_variance_ratio_.sum():.1%}")
    figure, axis = plt.subplots(figsize=(7, 3.5))
    axis.plot(original[0], marker="o", label="Original row")
    axis.plot(quantized[0], marker="s", label="Compressed + quantized")
    axis.set_xlabel("Feature index")
    axis.set_ylabel("Value")
    axis.legend()
    figure.tight_layout()
    st.pyplot(figure, use_container_width=True)
    plt.close(figure)


def _big_data():
    _section(
        "Big Data Visualization",
        "A reproducible large synthetic dataset demonstrates bounded sampling so plots stay responsive.",
    )
    row_count = st.select_slider("Generated rows", options=[10_000, 50_000, 100_000, 250_000], value=100_000)
    sample_count = st.slider("Maximum points to draw", 500, 10_000, 4_000, 500)
    rng = np.random.default_rng(71)
    start = pd.Timestamp.now()
    x = rng.normal(size=row_count)
    y = 0.6 * x + rng.normal(size=row_count)
    generated_seconds = (pd.Timestamp.now() - start).total_seconds()
    indices = rng.choice(row_count, min(sample_count, row_count), replace=False)
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.hexbin(x[indices], y[indices], gridsize=35, cmap="viridis", mincnt=1)
    axis.set_xlabel("Feature 1")
    axis.set_ylabel("Feature 2")
    axis.set_title(f"Density view of {len(indices):,} sampled rows")
    figure.tight_layout()
    st.pyplot(figure, use_container_width=True)
    plt.close(figure)
    metrics = st.columns(3)
    metrics[0].metric("Rows generated", f"{row_count:,}")
    metrics[1].metric("Rows plotted", f"{len(indices):,}")
    metrics[2].metric("Generation time", f"{generated_seconds:.3f}s")
    st.caption("Only the selected sample is drawn; full-resolution density aggregation can be enabled as a next step.")


def _customer_segmentation():
    _section(
        "Customer Segmentation",
        "K-Means groups simulated shoppers by annual income and spending score. Cluster numbers are not rankings.",
    )
    rng = np.random.default_rng(81)
    centers = np.array([[25, 25], [30, 75], [60, 45], [80, 80], [80, 20]])
    groups = rng.integers(0, len(centers), 500)
    customers = centers[groups] + rng.normal(0, [9, 10], (len(groups), 2))
    k = st.slider("Number of customer groups (K)", 2, 8, 5)
    model = KMeans(n_clusters=k, random_state=81, n_init=10).fit(customers)
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.scatter(customers[:, 0], customers[:, 1], c=model.labels_, cmap="tab10", alpha=0.65)
    axis.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], marker="X", s=220, c="black", label="Centroids")
    axis.set_xlabel("Annual income (illustrative $000s)")
    axis.set_ylabel("Spending score (0–100)")
    axis.legend()
    figure.tight_layout()
    st.pyplot(figure, use_container_width=True)
    plt.close(figure)
    st.dataframe(pd.DataFrame(model.cluster_centers_, columns=["Income", "Spending score"]), hide_index=True)


def _targeted_marketing():
    _section(
        "Targeted Marketing",
        "Segment simulated shoppers by purchase frequency, average basket, and discount use to explore campaign ideas.",
    )
    rng = np.random.default_rng(91)
    centers = np.array([[2, 25, 0.8], [12, 90, 0.2], [6, 55, 0.7], [18, 140, 0.35]])
    base_groups = rng.integers(0, 4, 600)
    shoppers = centers[base_groups] + rng.normal(0, [2, 18, 0.13], (600, 3))
    shoppers[:, 0] = shoppers[:, 0].clip(0, 30)
    shoppers[:, 1] = shoppers[:, 1].clip(5, 250)
    shoppers[:, 2] = shoppers[:, 2].clip(0, 1)
    count = st.slider("Campaign segments", 2, 7, 4)
    model = KMeans(n_clusters=count, random_state=91, n_init=10).fit(shoppers)
    suggestions = ["Welcome offer", "Loyalty rewards", "Bundle promotion", "Premium preview", "Win-back discount", "Cross-sell", "Seasonal offer"]
    frame = pd.DataFrame(shoppers, columns=["Purchases / month", "Average basket ($)", "Discount share"])
    frame["Segment"] = model.labels_
    summary = frame.groupby("Segment").mean().round(1)
    summary["Suggested campaign"] = [suggestions[int(i) % len(suggestions)] for i in summary.index]
    st.dataframe(summary, use_container_width=True)
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.scatter(shoppers[:, 0], shoppers[:, 1], c=model.labels_, cmap="tab10", alpha=0.6)
    axis.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], c="black", marker="X", s=170)
    axis.set_xlabel("Purchases per month")
    axis.set_ylabel("Average basket ($)")
    figure.tight_layout()
    st.pyplot(figure, use_container_width=True)
    plt.close(figure)


def _recommender():
    _section(
        "Movie Recommender System",
        "Rate a movie to find similar titles with a hybrid of genre similarity and simulated user-rating patterns.",
    )
    movies = pd.DataFrame([
        ("Starlight Voyage", "Sci-Fi Adventure", 4.5), ("Orbit's Edge", "Sci-Fi Adventure", 4.2),
        ("Deep Space", "Sci-Fi Drama", 4.0), ("The Last Detective", "Mystery Thriller", 4.4),
        ("Hidden Clue", "Mystery Thriller", 4.1), ("Night Watch", "Crime Thriller", 3.9),
        ("Summer Letters", "Romance Drama", 4.3), ("Two Seasons", "Romance Drama", 4.0),
        ("The Long Road", "Drama", 4.2), ("Laugh Track", "Comedy", 4.1),
        ("Weekend Plan", "Comedy Romance", 3.8), ("Wild Earth", "Documentary", 4.6),
        ("Tiny Giants", "Documentary", 4.2), ("Dragon School", "Fantasy Adventure", 4.3),
        ("Crown of Ash", "Fantasy Adventure", 4.0), ("Code Breakers", "Sci-Fi Thriller", 4.1),
    ], columns=["Movie", "Genres", "Community rating"])
    genres = sorted({genre for value in movies["Genres"] for genre in value.split()})
    matrix = np.array([[int(g in value.split()) for g in genres] for value in movies["Genres"]], dtype=float)
    movie = st.selectbox("Choose a movie you enjoyed", movies["Movie"].tolist())
    personal_rating = st.slider("Your rating", 1, 5, 5)
    index = int(movies.index[movies["Movie"] == movie][0])
    similarity = matrix @ matrix[index] / np.maximum(
        np.linalg.norm(matrix, axis=1) * np.linalg.norm(matrix[index]), 1e-9
    )
    # Include community ratings while keeping the selected movie out of its own results.
    score = 0.7 * similarity + 0.3 * movies["Community rating"].to_numpy() / 5
    score += (personal_rating - 3) * similarity * 0.04
    score[index] = -1
    indices = np.argsort(score)[::-1][:5]
    results = movies.iloc[indices].copy()
    results.insert(0, "Recommendation score", np.round(score[indices], 3))
    st.subheader(f"Because you liked {movie}")
    st.dataframe(results, hide_index=True, use_container_width=True)
    st.caption("This small offline catalogue is illustrative; recommendations are computed instantly from movie features.")


def _qlearn_grid(size, episodes, obstacles=()):
    start, goal = (0, 0), (size - 1, size - 1)
    actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    q = np.zeros((size, size, 4))
    rng = np.random.default_rng(101)
    rewards = []
    for episode in range(episodes):
        state = start
        total = 0
        for _ in range(size * size * 3):
            action = int(rng.integers(4)) if rng.random() < max(0.05, 0.7 * (1 - episode / episodes)) else int(np.argmax(q[state]))
            next_state = (state[0] + actions[action][0], state[1] + actions[action][1])
            reward = -1
            if not (0 <= next_state[0] < size and 0 <= next_state[1] < size) or next_state in obstacles:
                next_state, reward = state, -5
            elif next_state == goal:
                reward = 20
            q[state][action] += 0.2 * (reward + 0.9 * np.max(q[next_state]) - q[state][action])
            state, total = next_state, total + reward
            if state == goal:
                break
        rewards.append(total)
    return q, rewards, actions


def _grid_demo(title, description, obstacles=()):
    _section(title, description)
    episodes = st.slider("Training episodes", 50, 1500, 500, 50)
    q, rewards, actions = _qlearn_grid(5, episodes, obstacles)
    state, path = (0, 0), [(0, 0)]
    for _ in range(25):
        action = int(np.argmax(q[state]))
        candidate = (state[0] + actions[action][0], state[1] + actions[action][1])
        if not (0 <= candidate[0] < 5 and 0 <= candidate[1] < 5) or candidate in obstacles or candidate in path:
            alternatives = sorted(range(4), key=lambda a: q[state][a], reverse=True)
            candidate = next(((state[0] + actions[a][0], state[1] + actions[a][1]) for a in alternatives
                              if 0 <= state[0] + actions[a][0] < 5 and 0 <= state[1] + actions[a][1] < 5
                              and (state[0] + actions[a][0], state[1] + actions[a][1]) not in obstacles
                              and (state[0] + actions[a][0], state[1] + actions[a][1]) not in path), state)
        if candidate == state:
            break
        state = candidate
        path.append(state)
        if state == (4, 4):
            break
    grid = np.zeros((5, 5))
    for row, column in obstacles:
        grid[row, column] = 1
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].imshow(grid, cmap="Greys", vmin=0, vmax=1)
    axes[0].plot([point[1] for point in path], [point[0] for point in path], "o-", color="#0077b6")
    axes[0].text(0, 0, " S", color="blue", weight="bold")
    axes[0].text(4, 4, " G", color="green", weight="bold")
    axes[0].set_title("Learned path")
    axes[0].set_xticks(range(5)); axes[0].set_yticks(range(5)); axes[0].grid(alpha=0.25)
    axes[1].plot(pd.Series(rewards).rolling(20, min_periods=1).mean())
    axes[1].set_title("Reward (20-episode mean)")
    axes[1].set_xlabel("Episode")
    figure.tight_layout()
    st.pyplot(figure, use_container_width=True)
    plt.close(figure)
    st.write(f"Path length: **{max(0, len(path) - 1)} steps**. The agent receives a goal reward and penalties for obstacles and wandering.")


def _skill_acquisition():
    _section(
        "Skill Acquisition — Sequential Task Learning",
        "A simple agent learns a sequence of five actions: move through each checkpoint in order to finish the task.",
    )
    episodes = st.slider("Practice episodes", 25, 1000, 300, 25)
    rng = np.random.default_rng(111)
    q = np.zeros((6, 2))
    rewards = []
    for episode in range(episodes):
        state = 0
        total = 0
        epsilon = max(0.04, 0.6 * (1 - episode / episodes))
        for _ in range(10):
            action = int(rng.integers(2)) if rng.random() < epsilon else int(np.argmax(q[state]))
            next_state = min(5, state + 1) if action == 1 else max(0, state - 1)
            reward = 10 if next_state == 5 else (-1 if action == 0 else 1)
            q[state, action] += 0.2 * (reward + 0.9 * np.max(q[next_state]) - q[state, action])
            state, total = next_state, total + reward
            if state == 5:
                break
        rewards.append(total)
    st.progress(min(1.0, state / 5), text=f"Latest practice episode: checkpoint {state} of 5")
    figure, axis = plt.subplots(figsize=(8, 3.5))
    axis.plot(pd.Series(rewards).rolling(15, min_periods=1).mean())
    axis.set_xlabel("Practice episode")
    axis.set_ylabel("Smoothed reward")
    axis.set_title("Learning progress")
    figure.tight_layout()
    st.pyplot(figure, use_container_width=True)
    plt.close(figure)
    st.caption("Rewarded actions reinforce the sequence; repeated practice gradually reduces random exploration.")


def _real_time():
    _section(
        "Real-Time Decisions",
        "Simulate a feedback controller that adjusts a decision threshold to meet a target precision on incoming cases.",
    )
    rng = np.random.default_rng(121)
    count = st.slider("Incoming cases", 50, 1000, 300, 50)
    target = st.slider("Target positive rate", 0.05, 0.8, 0.25, 0.05)
    baseline = st.slider("Initial score threshold", 0.1, 0.9, 0.5, 0.05)
    actual = rng.binomial(1, 0.25, count)
    score = np.clip(rng.normal(0.30 + actual * 0.42, 0.2, count), 0, 1)
    threshold = baseline
    rates, thresholds = [], []
    for index in range(count):
        rates.append(float(np.mean(score[:index + 1] >= threshold)))
        thresholds.append(threshold)
        error = rates[-1] - target
        threshold = float(np.clip(threshold + 0.025 * np.sign(error), 0.1, 0.9))
    prediction = score >= threshold
    columns = st.columns(3)
    columns[0].metric("Final threshold", f"{threshold:.2f}")
    columns[1].metric("Cases flagged", f"{prediction.sum()} / {count}")
    columns[2].metric("Simulated precision", f"{(actual[prediction].mean() if prediction.any() else 0):.1%}")
    figure, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    axes[0].plot(rates, label="Observed flag rate")
    axes[0].axhline(target, linestyle="--", label="Target flag rate")
    axes[0].set_ylabel("Flag rate")
    axes[0].legend()
    axes[1].plot(thresholds, color="#e76f51", label="Adaptive threshold")
    axes[1].set_xlabel("Feedback step")
    axes[1].set_ylabel("Score threshold")
    axes[1].legend()
    figure.suptitle("Threshold feedback loop")
    figure.tight_layout()
    st.pyplot(figure, use_container_width=True)
    plt.close(figure)
    st.caption("This simulated policy adjusts from observed feedback; it does not control a live system.")


def render_mini_project(name):
    """Render one topic-specific demo, raising a clear error for unknown topics."""
    if name not in PROJECTS:
        st.error(f"No mini-project is registered for “{name}”. Choose a topic from the diagram.")
        return
    if name == "Advertising Popularity Prediction":
        _advertising()
    elif name == "Diagnostics":
        _diagnostics()
    elif name == "Customer Retention":
        _churn()
    elif name == "Image Classification":
        _image_classification()
    elif name == "Identity Fraud Detection":
        _fraud()
    elif name in ("Feature Elicitation", "Feature Selection"):
        _feature_selection()
    elif name == "Structure Discovery":
        _structure_discovery()
    elif name == "Meaningful Compression":
        _compression()
    elif name in ("Big Data Visualisation", "Big Data Visualization"):
        _big_data()
    elif name == "Customer Segmentation":
        _customer_segmentation()
    elif name == "Targeted Marketing":
        _targeted_marketing()
    elif name in ("Recommender Systems", "Recommender System"):
        _recommender()
    elif name == "Game AI":
        _grid_demo("Game AI — Q-Learning Gridworld", "Train an agent to reach the goal using rewards and penalties.")
    elif name == "Skill Acquisition":
        _skill_acquisition()
    elif name == "Learning Tasks":
        _grid_demo("Learning Tasks — Task Optimization", "Train an agent to optimize a repeatable goal-reaching task.")
    elif name == "Robot Navigation":
        _grid_demo(
            "Robot Navigation",
            "A robot learns a route to the goal while avoiding fixed obstacles.",
            obstacles={(1, 1), (1, 3), (2, 3), (3, 1)},
        )
    elif name in ("Real-time decisions", "Real-Time Decisions"):
        _real_time()


def render_project_launcher(category):
    """Show topic choices when a category node is selected."""
    choices = {
        "Machine Learning": ["Supervised Learning", "Unsupervised Learning", "Reinforcement Learning"],
        "Supervised Learning": ["Regression", "Classification"],
        "Unsupervised Learning": ["Clustering", "Dimensionality Reduction"],
        "Regression": ["Advertising Popularity Prediction"],
        "Classification": ["Diagnostics", "Customer Retention", "Image Classification", "Identity Fraud Detection"],
        "Dimensionality Reduction": ["Feature Elicitation", "Structure Discovery", "Meaningful Compression", "Big Data Visualisation"],
        "Clustering": ["Customer Segmentation", "Targeted Marketing", "Recommender Systems"],
        "Reinforcement Learning": ["Game AI", "Skill Acquisition", "Learning Tasks", "Robot Navigation", "Real-time decisions"],
    }
    projects = choices.get(category)
    if not projects:
        st.info(f"**{category}** is a machine-learning concept. Select a child topic in the diagram to open its interactive demo.")
        return
    st.subheader(f"{category} projects")
    st.caption("Choose a dedicated mini-project to explore its controls, data, and working algorithm.")
    for name in projects:
        if st.button(f"Open {name}", key=f"launch-{category}-{name}", use_container_width=True):
            is_category = name in choices
            destination = "Category" if is_category else "Mini Project"
            st.query_params.clear()
            st.session_state["selected_module"] = destination
            st.query_params["module"] = destination
            st.query_params["node"] = name
            st.rerun()


if __name__ == "__main__":
    st.set_page_config(page_title="ML Mini-projects", page_icon="🧪", layout="wide")
    selected = st.selectbox("Mini-project", list(PROJECTS))
    render_mini_project(selected)
