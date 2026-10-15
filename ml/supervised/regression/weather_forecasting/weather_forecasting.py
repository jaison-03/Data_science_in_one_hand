from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FEATURE_COLUMNS = (
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_24",
    "rolling_mean_3",
    "rolling_mean_24",
)


@st.cache_data(ttl=3600, show_spinner=False)
def search_cities(city: str) -> list[dict]:
    response = requests.get(
        GEOCODING_URL,
        params={"name": city, "count": 10, "language": "en", "format": "json"},
        timeout=20,
    )
    response.raise_for_status()
    return response.json().get("results", [])


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_weather_forecast(latitude: float, longitude: float, forecast_days: int) -> dict:
    response = requests.get(
        FORECAST_URL,
        params={
            "latitude": latitude,
            "longitude": longitude,
            "current": (
                "temperature_2m,relative_humidity_2m,apparent_temperature,"
                "is_day,precipitation,weather_code,wind_speed_10m"
            ),
            "hourly": "temperature_2m,precipitation_probability,wind_speed_10m",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code",
            "past_days": 2,
            "forecast_days": forecast_days,
            "timezone": "auto",
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_historical_weather(
    latitude: float,
    longitude: float,
    start_date: date,
    end_date: date,
) -> pd.Series:
    response = requests.get(
        ARCHIVE_URL,
        params={
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "hourly": "temperature_2m",
            "timezone": "UTC",
        },
        timeout=60,
    )
    response.raise_for_status()
    hourly = response.json().get("hourly", {})
    if "time" not in hourly or "temperature_2m" not in hourly:
        raise ValueError("The historical weather API returned no hourly temperature data.")

    numeric_temperatures = pd.to_numeric(hourly["temperature_2m"], errors="coerce")
    temperatures = pd.Series(
        numeric_temperatures,
        index=pd.to_datetime(hourly["time"], errors="coerce"),
        name="temperature",
    ).dropna()
    temperatures = temperatures[~temperatures.index.isna()]
    temperatures = temperatures[~temperatures.index.duplicated(keep="last")].sort_index()
    if temperatures.empty:
        raise ValueError("No valid historical temperatures were returned for this location.")
    return temperatures


def make_training_data(temperatures: pd.Series) -> pd.DataFrame:
    """Build one-hour-ahead features using only temperatures available at prediction time."""
    prior = temperatures.shift(1)
    features = pd.DataFrame(index=temperatures.index)
    for lag in (1, 2, 3, 24):
        features[f"lag_{lag}"] = temperatures.shift(lag)
    for window in (3, 24):
        features[f"rolling_mean_{window}"] = prior.rolling(window).mean()
    features["target"] = temperatures
    return features.dropna()


def make_forecast_features(history: list[float]) -> pd.DataFrame:
    if len(history) < 24:
        raise ValueError("At least 24 recent hourly temperatures are needed to forecast.")
    return pd.DataFrame(
        [
            {
                "lag_1": history[-1],
                "lag_2": history[-2],
                "lag_3": history[-3],
                "lag_24": history[-24],
                "rolling_mean_3": np.mean(history[-3:]),
                "rolling_mean_24": np.mean(history[-24:]),
            }
        ],
        columns=FEATURE_COLUMNS,
    )


def forecast_temperatures(
    model: LinearRegression,
    recent_temperatures: pd.Series,
    forecast_index: pd.DatetimeIndex,
) -> pd.Series:
    history = recent_temperatures.astype(float).tolist()
    predictions = []
    for _ in forecast_index:
        prediction = float(model.predict(make_forecast_features(history))[0])
        predictions.append(prediction)
        history.append(prediction)
    return pd.Series(predictions, index=forecast_index, name="ML prediction")


def _weather_location_label(location: dict) -> str:
    parts = [
        location.get("name", "Unknown location"),
        location.get("admin1"),
        location.get("country"),
    ]
    return ", ".join(part for part in parts if part)


def _show_weather_forecast(location: dict, forecast_days: int) -> None:
    latitude = float(location["latitude"])
    longitude = float(location["longitude"])
    end_date = date.today() - timedelta(days=6)
    start_date = end_date - timedelta(days=730)

    try:
        forecast = fetch_weather_forecast(latitude, longitude, forecast_days)
        history = fetch_historical_weather(latitude, longitude, start_date, end_date)
    except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
        st.error(f"Unable to load weather data: {exc}")
        return

    current = forecast.get("current")
    hourly = forecast.get("hourly")
    daily = forecast.get("daily")
    if not current or not hourly or not daily:
        st.error("The weather forecast API did not return current, hourly, and daily data.")
        return

    training_data = make_training_data(history)
    if len(training_data) < 100:
        st.error(
            "There is not enough historical data to train and evaluate the model "
            "for this location."
        )
        return

    split_index = int(len(training_data) * 0.8)
    if split_index < 24 or len(training_data) - split_index < 2:
        st.error("There is not enough historical data for a chronological test split.")
        return

    train_data = training_data.iloc[:split_index]
    test_data = training_data.iloc[split_index:]
    evaluation_model = LinearRegression()
    evaluation_model.fit(train_data[list(FEATURE_COLUMNS)], train_data["target"])
    test_predictions = evaluation_model.predict(test_data[list(FEATURE_COLUMNS)])
    mae = mean_absolute_error(test_data["target"], test_predictions)
    rmse = np.sqrt(mean_squared_error(test_data["target"], test_predictions))
    r_squared = r2_score(test_data["target"], test_predictions)

    model = LinearRegression()
    model.fit(training_data[list(FEATURE_COLUMNS)], training_data["target"])

    hourly_index = pd.to_datetime(hourly.get("time", []), errors="coerce")
    hourly_temperatures = pd.Series(
        pd.to_numeric(hourly.get("temperature_2m", []), errors="coerce"),
        index=hourly_index,
    ).dropna()
    current_time = pd.to_datetime(current.get("time"), errors="coerce")
    if pd.isna(current_time) or hourly_temperatures.empty:
        st.error("The forecast API returned invalid timestamps or hourly temperatures.")
        return
    recent_temperatures = hourly_temperatures.loc[hourly_temperatures.index <= current_time]
    future_api = hourly_temperatures.loc[hourly_temperatures.index > current_time]
    if len(recent_temperatures) < 24 or future_api.empty:
        st.error("The forecast API did not provide enough recent and future hourly data.")
        return

    api_forecast = future_api.iloc[: forecast_days * 24]
    ml_forecast = forecast_temperatures(model, recent_temperatures.iloc[-24:], api_forecast.index)

    st.subheader(_weather_location_label(location))
    current_columns = st.columns(4)
    current_columns[0].metric(
        "Current temperature", f"{current.get('temperature_2m', '—')} °C"
    )
    current_columns[1].metric(
        "Feels like", f"{current.get('apparent_temperature', '—')} °C"
    )
    current_columns[2].metric(
        "Humidity", f"{current.get('relative_humidity_2m', '—')}%"
    )
    current_columns[3].metric(
        "Wind speed", f"{current.get('wind_speed_10m', '—')} km/h"
    )

    st.subheader("Hourly temperature: Open-Meteo vs. supervised ML")
    comparison = go.Figure()
    comparison.add_trace(
        go.Scatter(
            x=api_forecast.index,
            y=api_forecast.values,
            mode="lines",
            name="Open-Meteo forecast",
            line={"color": "#1f77b4", "width": 2},
        )
    )
    comparison.add_trace(
        go.Scatter(
            x=ml_forecast.index,
            y=ml_forecast.values,
            mode="lines",
            name="Linear regression",
            line={"color": "#ff7f0e", "width": 2, "dash": "dash"},
        )
    )
    comparison.update_layout(
        xaxis_title="Local time",
        yaxis_title="Temperature (°C)",
        hovermode="x unified",
        legend_title="Forecast source",
        margin={"t": 25, "b": 20},
    )
    st.plotly_chart(comparison, use_container_width=True)

    st.subheader("Daily forecast from Open-Meteo")
    daily_frame = pd.DataFrame(
        {
            "Date": pd.to_datetime(daily.get("time", [])),
            "High (°C)": daily.get("temperature_2m_max", []),
            "Low (°C)": daily.get("temperature_2m_min", []),
        }
    )
    daily_figure = go.Figure()
    daily_figure.add_trace(
        go.Scatter(
            x=daily_frame["Date"],
            y=daily_frame["High (°C)"],
            mode="lines+markers",
            name="Daily high",
        )
    )
    daily_figure.add_trace(
        go.Scatter(
            x=daily_frame["Date"],
            y=daily_frame["Low (°C)"],
            mode="lines+markers",
            name="Daily low",
        )
    )
    daily_figure.update_layout(
        xaxis_title="Date",
        yaxis_title="Temperature (°C)",
        hovermode="x unified",
        margin={"t": 25, "b": 20},
    )
    st.plotly_chart(daily_figure, use_container_width=True)

    st.subheader("Historical model evaluation")
    metric_columns = st.columns(3)
    metric_columns[0].metric("MAE", f"{mae:.2f} °C")
    metric_columns[1].metric("RMSE", f"{rmse:.2f} °C")
    metric_columns[2].metric("R²", f"{r_squared:.3f}")
    st.caption(
        f"Linear regression trained on {len(train_data):,} hourly observations and "
        f"evaluated on the following {len(test_data):,} observations "
        f"({test_data.index[0]:%Y-%m-%d} to {test_data.index[-1]:%Y-%m-%d}). "
        "The holdout split preserves chronological order; history comes from "
        "Open-Meteo's Historical Weather API."
    )


def render_weather_forecasting_dashboard() -> None:
    st.title("Real-Time Weather Forecasting")
    st.markdown(
        "Compare Open-Meteo's live forecast with a supervised linear-regression "
        "model trained on historical hourly temperatures."
    )

    with st.form("weather_city_search"):
        city = st.text_input("City", placeholder="e.g. Bengaluru")
        search_submitted = st.form_submit_button("Find city")

    if search_submitted:
        if not city.strip():
            st.warning("Enter a city name to search.")
            st.session_state.pop("weather_locations", None)
        else:
            try:
                locations = search_cities(city.strip())
            except requests.RequestException as exc:
                st.error(f"Unable to search for that city: {exc}")
                st.session_state.pop("weather_locations", None)
            else:
                st.session_state["weather_locations"] = locations
                if not locations:
                    st.warning("No matching cities were found. Try another spelling.")

    locations = st.session_state.get("weather_locations", [])
    if not locations:
        return

    labels = [_weather_location_label(location) for location in locations]
    selected_label = st.selectbox("Select a location", labels, key="weather_location")
    selected_location = locations[labels.index(selected_label)]
    forecast_days = st.slider("Forecast horizon (days)", min_value=1, max_value=7, value=3)
    if st.button("Load forecast and train model", type="primary"):
        with st.spinner("Loading Open-Meteo data and training the model..."):
            _show_weather_forecast(selected_location, forecast_days)
