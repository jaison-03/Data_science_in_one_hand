from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_PATH = Path(__file__).parent / "data" / "Life Expectancy Data.csv"
TARGET_COLUMN = "Life expectancy"
NUMERIC_FEATURES = (
    "Year",
    "Adult Mortality",
    "infant deaths",
    "Alcohol",
    "percentage expenditure",
    "Hepatitis B",
    "Measles",
    "BMI",
    "under-five deaths",
    "Polio",
    "Total expenditure",
    "Diphtheria",
    "HIV/AIDS",
    "GDP",
    "Population",
    "thinness 1-19 years",
    "thinness 5-9 years",
    "Income composition of resources",
    "Schooling",
)


def _normalize_dataset(data):
    data = data.copy()
    data.columns = [str(column).strip() for column in data.columns]
    canonical_columns = {
        column.casefold(): column
        for column in (TARGET_COLUMN, "Status", *NUMERIC_FEATURES)
    }
    data = data.rename(
        columns={
            column: canonical_columns[column.casefold()]
            for column in data.columns
            if column.casefold() in canonical_columns
        }
    )

    if TARGET_COLUMN not in data.columns:
        raise ValueError(
            f"The dataset must contain a '{TARGET_COLUMN}' column."
        )

    feature_columns = [
        column for column in NUMERIC_FEATURES if column in data.columns
    ]
    if not feature_columns:
        raise ValueError(
            "The dataset must contain at least one supported numeric "
            "predictor, such as GDP, Adult Mortality, or Schooling."
        )

    data[TARGET_COLUMN] = pd.to_numeric(data[TARGET_COLUMN], errors="coerce")
    data[TARGET_COLUMN] = data[TARGET_COLUMN].replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=[TARGET_COLUMN]).copy()

    for column in feature_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
        data[column] = data[column].replace([np.inf, -np.inf], np.nan)
    if "Status" in data.columns:
        data["Status"] = (
            data["Status"]
            .replace(r"^\s*$", np.nan, regex=True)
            .fillna("Unknown")
            .astype(str)
        )

    if len(data) < 2:
        raise ValueError(
            "At least two rows with a valid life expectancy value are "
            "required to train a regression model."
        )
    return data.reset_index(drop=True), feature_columns


def _generate_synthetic_data():
    rng = np.random.default_rng(42)
    row_count = 480
    years = rng.integers(2000, 2016, size=row_count)
    status = rng.choice(
        ["Developing", "Developed"], size=row_count, p=[0.78, 0.22]
    )
    developed = (status == "Developed").astype(float)
    schooling = np.clip(
        rng.normal(8.5 + 5.0 * developed, 2.5, row_count), 2, 20
    )
    income_composition = np.clip(
        rng.normal(0.48 + 0.34 * developed, 0.12, row_count), 0.2, 0.98
    )
    adult_mortality = np.clip(
        rng.normal(245 - 125 * developed - 4.2 * schooling, 55, row_count),
        35,
        500,
    )
    hiv_aids = np.clip(
        rng.gamma(shape=1.3, scale=0.8, size=row_count) * (1.4 - 0.75 * developed),
        0,
        18,
    )
    gdp = np.exp(rng.normal(7.5 + 2.4 * developed, 1.3, row_count))
    population = np.exp(rng.normal(15.2, 1.7, row_count))
    bmi = np.clip(rng.normal(22 + 5 * developed, 4, row_count), 12, 45)
    infant_deaths = np.maximum(
        0, np.rint(rng.lognormal(3.6, 1.2, row_count) * (1.3 - developed))
    ).astype(int)
    under_five_deaths = np.maximum(
        infant_deaths,
        np.rint(infant_deaths * rng.uniform(1.1, 1.8, row_count)),
    ).astype(int)
    life_expectancy = np.clip(
        51
        + 15 * income_composition
        + 0.8 * schooling
        - 0.035 * adult_mortality
        - 0.65 * hiv_aids
        + 0.13 * (years - 2000)
        + 0.1 * (bmi - 22)
        + rng.normal(0, 3.2, row_count),
        42,
        88,
    )

    return pd.DataFrame(
        {
            "Country": rng.choice(
                [f"Country {index:02d}" for index in range(1, 25)],
                size=row_count,
            ),
            "Year": years,
            "Status": status,
            TARGET_COLUMN: life_expectancy,
            "Adult Mortality": adult_mortality,
            "infant deaths": infant_deaths,
            "Alcohol": np.clip(
                rng.normal(2 + 7 * developed, 3, row_count), 0, 15
            ),
            "percentage expenditure": np.maximum(
                0, rng.lognormal(3 + 1.5 * developed, 1.2, row_count)
            ),
            "Hepatitis B": np.clip(
                rng.normal(65 + 25 * developed, 18, row_count), 5, 100
            ),
            "Measles": np.maximum(
                0, np.rint(rng.lognormal(2.5, 2.2, row_count))
            ).astype(int),
            "BMI": bmi,
            "under-five deaths": under_five_deaths,
            "Polio": np.clip(
                rng.normal(68 + 27 * developed, 17, row_count), 5, 100
            ),
            "Total expenditure": np.clip(
                rng.normal(4.5 + 2 * developed, 2, row_count), 0, 15
            ),
            "Diphtheria": np.clip(
                rng.normal(70 + 25 * developed, 16, row_count), 5, 100
            ),
            "HIV/AIDS": hiv_aids,
            "GDP": gdp,
            "Population": population,
            "thinness 1-19 years": np.clip(
                rng.normal(6 - 3 * developed, 2.5, row_count), 0, 18
            ),
            "thinness 5-9 years": np.clip(
                rng.normal(6.2 - 3.1 * developed, 2.5, row_count), 0, 18
            ),
            "Income composition of resources": income_composition,
            "Schooling": schooling,
        }
    )


@st.cache_data
def _load_default_dataset():
    if DATA_PATH.is_file():
        data = pd.read_csv(DATA_PATH)
        return data, str(DATA_PATH), False
    return _generate_synthetic_data(), "synthetic sample data", True


def _build_model(numeric_features, categorical_features):
    transformers = [
        (
            "numeric",
            SimpleImputer(strategy="median", keep_empty_features=True),
            numeric_features,
        )
    ]
    if categorical_features:
        categorical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(
            ("categorical", categorical_pipeline, categorical_features)
        )
    preprocessor = ColumnTransformer(transformers)
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )


def render_life_expectancy_dashboard():
    st.title("Life Expectancy Prediction Using Supervised Learning – Regression")
    st.markdown(
        "Explore the factors associated with life expectancy, evaluate a "
        "supervised linear regression model, and try your own predictions."
    )

    uploaded_file = st.file_uploader(
        "Optional: upload a WHO life-expectancy CSV",
        type="csv",
        help=(
            "The CSV should include a 'Life expectancy' target column and "
            "one or more supported numeric predictors."
        ),
    )
    if uploaded_file is not None:
        try:
            raw_data = pd.read_csv(uploaded_file)
            source = uploaded_file.name
            using_synthetic_data = False
        except (OSError, pd.errors.ParserError, UnicodeDecodeError) as error:
            st.error(f"Could not read the uploaded CSV: {error}")
            st.stop()
    else:
        try:
            raw_data, source, using_synthetic_data = _load_default_dataset()
        except (OSError, pd.errors.ParserError, UnicodeDecodeError) as error:
            st.error(f"Could not read the WHO dataset at {DATA_PATH}: {error}")
            st.stop()

    try:
        data, numeric_features = _normalize_dataset(raw_data)
    except ValueError as error:
        st.error(f"Invalid life-expectancy dataset: {error}")
        st.stop()

    if using_synthetic_data:
        st.info(
            f"No WHO CSV was found at `{DATA_PATH}`. Showing reproducible "
            "synthetic sample data; upload a CSV above to use real records."
        )
    else:
        st.caption(f"Dataset source: {source}")

    categorical_features = ["Status"] if "Status" in data.columns else []
    model_features = numeric_features + categorical_features
    features = data[model_features]
    target = data[TARGET_COLUMN]

    st.subheader("Dataset overview")
    metric_columns = st.columns(3)
    metric_columns[0].metric("Records", f"{len(data):,}")
    metric_columns[1].metric("Predictor features", len(model_features))
    metric_columns[2].metric(
        "Average life expectancy", f"{target.mean():.1f} years"
    )
    with st.expander("Preview dataset"):
        st.dataframe(data, use_container_width=True, hide_index=True)

    st.subheader("Regression model evaluation")
    if len(data) >= 3:
        test_size = max(1, int(np.ceil(len(data) * 0.2)))
        if test_size >= len(data):
            test_size = len(data) - 1
        train_features, test_features, train_target, test_target = train_test_split(
            features, target, test_size=test_size, random_state=42
        )
        evaluation_model = _build_model(
            numeric_features, categorical_features
        ).fit(train_features, train_target)
        test_predictions = evaluation_model.predict(test_features)

        scores = st.columns(4)
        scores[0].metric(
            "MAE", f"{mean_absolute_error(test_target, test_predictions):.2f} years"
        )
        scores[1].metric(
            "MSE", f"{mean_squared_error(test_target, test_predictions):.2f}"
        )
        scores[2].metric(
            "RMSE",
            f"{np.sqrt(mean_squared_error(test_target, test_predictions)):.2f} years",
        )
        test_r_squared = (
            r2_score(test_target, test_predictions)
            if len(test_target) >= 2
            else None
        )
        scores[3].metric(
            "R²",
            f"{test_r_squared:.3f}" if test_r_squared is not None else "N/A",
            help="R² needs at least two records in the test split.",
        )
    else:
        st.info(
            "At least three records are needed for a holdout evaluation. "
            "The prediction model will still be trained on all available rows."
        )

    model = _build_model(numeric_features, categorical_features).fit(
        features, target
    )

    st.subheader("Explore the data")
    distribution_column, status_column = st.columns(2)
    with distribution_column:
        figure, axis = plt.subplots(figsize=(8, 4))
        sns.histplot(data=data, x=TARGET_COLUMN, kde=True, ax=axis, color="#1e88e5")
        axis.set_title("Life expectancy distribution")
        axis.set_xlabel("Life expectancy (years)")
        figure.tight_layout()
        st.pyplot(figure, use_container_width=True)
        plt.close(figure)

    with status_column:
        figure, axis = plt.subplots(figsize=(8, 4))
        if "Status" in data.columns:
            sns.boxplot(data=data, x="Status", y=TARGET_COLUMN, ax=axis)
            axis.set_title("Life expectancy by country status")
        else:
            sns.scatterplot(
                data=data,
                x=numeric_features[0],
                y=TARGET_COLUMN,
                ax=axis,
                color="#43a047",
            )
            axis.set_title(f"Life expectancy by {numeric_features[0]}")
        figure.tight_layout()
        st.pyplot(figure, use_container_width=True)
        plt.close(figure)

    with st.expander("Numeric feature correlation heatmap"):
        correlation_columns = numeric_features + [TARGET_COLUMN]
        correlations = data[correlation_columns].corr(numeric_only=True)
        figure, axis = plt.subplots(
            figsize=(max(10, len(correlation_columns) * 0.7), 8)
        )
        sns.heatmap(
            correlations,
            cmap="coolwarm",
            center=0,
            annot=len(correlation_columns) <= 10,
            fmt=".2f",
            ax=axis,
        )
        axis.set_title("Correlation between numeric predictors and life expectancy")
        figure.tight_layout()
        st.pyplot(figure, use_container_width=True)
        plt.close(figure)

    st.subheader("Predict life expectancy")
    st.caption(
        "Adjust socioeconomic and health indicators, then estimate life "
        "expectancy with the model trained on the selected dataset."
    )
    with st.form("life_expectancy_prediction"):
        prediction = {}
        input_columns = st.columns(3)
        for index, feature in enumerate(numeric_features):
            median = data[feature].median()
            default = float(median) if pd.notna(median) else 0.0
            with input_columns[index % len(input_columns)]:
                prediction[feature] = st.number_input(
                    feature,
                    value=default,
                    step=0.1,
                    format="%.3f",
                )

        if categorical_features:
            statuses = sorted(data["Status"].dropna().astype(str).unique())
            if statuses:
                prediction["Status"] = st.selectbox(
                    "Country status", statuses
                )

        submitted = st.form_submit_button("Estimate life expectancy")

    if submitted:
        prediction_data = pd.DataFrame([prediction], columns=model_features)
        predicted_years = float(model.predict(prediction_data)[0])
        st.success(f"Estimated life expectancy: **{predicted_years:.1f} years**")
