from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DATA_PATH = Path(__file__).parent / "data" / "population_data.csv"


@st.cache_data
def load_population_data():
    data = pd.read_csv(DATA_PATH)
    required_columns = {"Year", "Population"}
    if not required_columns.issubset(data.columns):
        raise ValueError(
            f"{DATA_PATH.name} must contain Year and Population columns."
        )

    data = data.loc[:, ["Year", "Population"]].copy()
    data["Year"] = pd.to_numeric(data["Year"], errors="raise").astype(int)
    data["Population"] = pd.to_numeric(
        data["Population"], errors="raise"
    ).astype(float)
    if data.empty or data.isnull().any().any():
        raise ValueError(f"{DATA_PATH.name} must contain valid population records.")
    return data.sort_values("Year").reset_index(drop=True)


def render_population_growth_dashboard():
    st.title("Population Growth Prediction")
    st.markdown(
        "Explore historical population data, evaluate a linear regression model, "
        "and generate forecasts for future years."
    )

    data = load_population_data()
    first_year = int(data["Year"].min())
    last_year = int(data["Year"].max())

    st.subheader("Historical data")
    start_year, end_year = st.slider(
        "Select historical years",
        min_value=first_year,
        max_value=last_year,
        value=(first_year, last_year),
        step=1,
    )
    historical = data[data["Year"].between(start_year, end_year)].copy()
    st.dataframe(historical, use_container_width=True, hide_index=True)

    if len(historical) < 2:
        st.warning("Select at least two years to train the regression model.")
        return

    features = historical[["Year"]]
    target = historical["Population"]
    model = LinearRegression().fit(features, target)

    st.subheader("Model evaluation")
    if len(historical) >= 3:
        split_index = max(1, min(len(historical) - 1, int(len(historical) * 0.8)))
        train_features = features.iloc[:split_index]
        test_features = features.iloc[split_index:]
        train_target = target.iloc[:split_index]
        test_target = target.iloc[split_index:]
        evaluation_model = LinearRegression().fit(train_features, train_target)
        test_predictions = evaluation_model.predict(test_features)

        metric_columns = st.columns(4)
        training_predictions = evaluation_model.predict(train_features)
        metric_columns[0].metric(
            "Training R² score",
            f"{r2_score(train_target, training_predictions):.3f}",
        )
        metric_columns[1].metric(
            "Mean absolute error",
            f"{mean_absolute_error(test_target, test_predictions):,.0f}",
        )
        metric_columns[2].metric(
            "Root mean squared error",
            f"{np.sqrt(mean_squared_error(test_target, test_predictions)):,.0f}",
        )
        r_squared = (
            r2_score(test_target, test_predictions)
            if len(test_target) >= 2
            else None
        )
        metric_columns[3].metric(
            "Test R² score",
            f"{r_squared:.3f}" if r_squared is not None else "N/A",
            help="R² is unavailable when the chronological test split has fewer than two records.",
        )
    else:
        st.info("Select at least three years to calculate holdout evaluation metrics.")

    st.subheader("Historical population and regression trend")
    forecast_year = st.number_input(
        "Predict population for year",
        min_value=last_year + 1,
        max_value=2500,
        value=last_year + 1,
        step=1,
    )
    years_to_plot = pd.DataFrame(
        {"Year": np.arange(start_year, int(forecast_year) + 1)}
    )
    predictions = model.predict(years_to_plot)
    predicted_population = float(
        model.predict(pd.DataFrame({"Year": [forecast_year]}))[0]
    )

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.scatter(
        historical["Year"],
        historical["Population"],
        color="#1e88e5",
        label="Historical population",
        zorder=3,
    )
    axis.plot(
        years_to_plot["Year"],
        predictions,
        color="#e65100",
        label="Linear regression",
    )
    axis.scatter(
        [forecast_year],
        [predicted_population],
        color="#8e24aa",
        marker="*",
        s=150,
        label=f"Prediction for {int(forecast_year)}",
        zorder=4,
    )
    axis.set_xlabel("Year")
    axis.set_ylabel("Population")
    axis.set_title("Population growth and linear regression forecast")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    st.pyplot(figure, use_container_width=True)
    plt.close(figure)

    st.metric(
        f"Predicted population in {int(forecast_year)}",
        f"{predicted_population:,.0f}",
    )

    forecast_years = np.arange(last_year + 1, int(forecast_year) + 1)
    forecast_values = model.predict(pd.DataFrame({"Year": forecast_years}))
    forecast_data = pd.DataFrame(
        {
            "Year": forecast_years,
            "PredictedPopulation": np.maximum(0, np.rint(forecast_values)).astype(int),
        }
    )
    st.download_button(
        "Download predictions as CSV",
        data=forecast_data.to_csv(index=False).encode("utf-8"),
        file_name="population_growth_predictions.csv",
        mime="text/csv",
    )
