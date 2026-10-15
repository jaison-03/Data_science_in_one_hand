from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from requests.exceptions import RequestException
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


FEATURE_COLUMNS = (
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_5",
    "lag_10",
    "moving_average_5",
    "moving_average_10",
    "moving_average_20",
)


@st.cache_data(ttl=3600, show_spinner=False)
def download_market_data(ticker: str, start_date: date, end_date: date) -> pd.DataFrame:
    """Download adjusted daily prices; Yahoo Finance's end date is exclusive."""
    data = yf.download(
        ticker,
        start=start_date,
        end=end_date + timedelta(days=1),
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if data.empty:
        raise ValueError(
            f"No market data was returned for {ticker}. Check the ticker and date range."
        )

    try:
        close = data["Close"]
    except KeyError as exc:
        raise ValueError(
            f"No closing-price column was returned for {ticker}."
        ) from exc
    if isinstance(close, pd.DataFrame):
        if ticker in close.columns:
            close = close[ticker]
        else:
            close = close.iloc[:, 0]

    prices = pd.to_numeric(close, errors="coerce").dropna().sort_index()
    prices.index = pd.to_datetime(prices.index)
    if prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)
    prices = prices[~prices.index.duplicated(keep="last")]
    if prices.empty:
        raise ValueError(f"No valid closing prices were returned for {ticker}.")
    return prices.rename("Close").to_frame()


def make_training_data(prices: pd.Series) -> pd.DataFrame:
    """Build next-close features using only prices available before each date."""
    prior_prices = prices.shift(1)
    features = pd.DataFrame(index=prices.index)
    for lag in (1, 2, 3, 5, 10):
        features[f"lag_{lag}"] = prices.shift(lag)
    for window in (5, 10, 20):
        features[f"moving_average_{window}"] = prior_prices.rolling(window).mean()
    features["target"] = prices
    return features.dropna()


def make_forecast_features(history: list[float]) -> pd.DataFrame:
    """Create one forecast row from observed prices and prior predictions."""
    return pd.DataFrame(
        [
            {
                "lag_1": history[-1],
                "lag_2": history[-2],
                "lag_3": history[-3],
                "lag_5": history[-5],
                "lag_10": history[-10],
                "moving_average_5": np.mean(history[-5:]),
                "moving_average_10": np.mean(history[-10:]),
                "moving_average_20": np.mean(history[-20:]),
            }
        ],
        columns=FEATURE_COLUMNS,
    )


def forecast_future_prices(
    model: LinearRegression,
    prices: pd.Series,
    forecast_days: int,
) -> pd.Series:
    """Forecast iteratively on weekdays following the latest observed price."""
    history = prices.astype(float).tolist()
    dates = pd.bdate_range(
        start=prices.index[-1] + pd.Timedelta(days=1), periods=forecast_days
    )
    predictions = []
    for _ in dates:
        next_price = float(model.predict(make_forecast_features(history))[0])
        predictions.append(next_price)
        history.append(next_price)
    return pd.Series(predictions, index=dates, name="Forecast")


def _show_market_forecast(
    ticker: str,
    start_date: date,
    end_date: date,
    test_fraction: float,
    forecast_days: int,
) -> None:
    try:
        data = download_market_data(ticker, start_date, end_date)
    except (
        RequestException,
        ConnectionError,
        TimeoutError,
        ValueError,
        RuntimeError,
    ) as exc:
        st.error(f"Unable to load market data: {exc}")
        return

    prices = data["Close"]
    supervised_data = make_training_data(prices)
    if len(supervised_data) < 40:
        st.error(
            "This date range does not contain enough daily prices to build the "
            "20-day moving-average features. Choose a longer date range."
        )
        return

    test_count = max(2, int(np.ceil(len(supervised_data) * test_fraction)))
    split_index = len(supervised_data) - test_count
    if split_index < 20:
        st.error("Choose a longer date range or a smaller test-set percentage.")
        return

    train_data = supervised_data.iloc[:split_index]
    test_data = supervised_data.iloc[split_index:]
    evaluation_model = LinearRegression()
    evaluation_model.fit(train_data[list(FEATURE_COLUMNS)], train_data["target"])
    test_predictions = evaluation_model.predict(test_data[list(FEATURE_COLUMNS)])
    baseline_predictions = test_data["lag_1"].to_numpy()

    metric_rows = []
    for name, predictions in (
        ("Linear regression", test_predictions),
        ("Previous close baseline", baseline_predictions),
    ):
        metric_rows.append(
            {
                "Model": name,
                "MAE": mean_absolute_error(test_data["target"], predictions),
                "RMSE": np.sqrt(mean_squared_error(test_data["target"], predictions)),
                "R²": r2_score(test_data["target"], predictions),
            }
        )
    metrics = pd.DataFrame(metric_rows).set_index("Model")

    forecast_model = LinearRegression()
    forecast_model.fit(
        supervised_data[list(FEATURE_COLUMNS)], supervised_data["target"]
    )
    forecast = forecast_future_prices(forecast_model, prices, forecast_days)
    latest_price = float(prices.iloc[-1])
    next_price = float(forecast.iloc[0])
    first_test = test_data.index[0].strftime("%b %d, %Y")
    last_test = test_data.index[-1].strftime("%b %d, %Y")

    st.subheader(f"{ticker} · adjusted daily close")
    metric_columns = st.columns(4)
    metric_columns[0].metric("Latest close", f"${latest_price:,.2f}")
    metric_columns[1].metric(
        f"Next weekday estimate ({forecast.index[0]:%b %d})",
        f"${next_price:,.2f}",
        f"{next_price - latest_price:+,.2f}",
    )
    metric_columns[2].metric("Test window", f"{len(test_data)} trading sessions")
    metric_columns[3].metric("Training window", f"{len(train_data)} sessions")

    st.markdown(
        f"**Out-of-sample evaluation:** {first_test} to {last_test}. "
        "The model is compared with a persistence baseline that predicts "
        "tomorrow's close as today's close."
    )
    st.dataframe(
        metrics.style.format({"MAE": "${:,.2f}", "RMSE": "${:,.2f}", "R²": "{:.4f}"}),
        use_container_width=True,
    )

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=prices.index,
            y=prices,
            name="Historical adjusted close",
            mode="lines",
            line={"color": "#4c78a8"},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=test_data.index,
            y=test_predictions,
            name="Test regression predictions",
            mode="lines",
            line={"color": "#f58518", "dash": "dot"},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=test_data.index,
            y=baseline_predictions,
            name="Test previous-close baseline",
            mode="lines",
            line={"color": "#54a24b", "dash": "dash"},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=forecast.index,
            y=forecast,
            name="Iterative forecast",
            mode="lines+markers",
            line={"color": "#b279a2", "dash": "dash"},
        )
    )
    figure.update_layout(
        title="Historical prices, held-out predictions, and forecast",
        xaxis_title="Date",
        yaxis_title="Adjusted close (USD)",
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
        margin={"t": 90},
    )
    st.plotly_chart(figure, use_container_width=True)

    with st.expander("View recent market data and future estimates"):
        st.dataframe(data.tail(10), use_container_width=True)
        st.dataframe(
            forecast.rename_axis("Date").to_frame(),
            use_container_width=True,
        )
        st.download_button(
            "Download forecast CSV",
            data=forecast.rename_axis("Date").to_frame().to_csv().encode("utf-8"),
            file_name=f"{ticker.lower()}_market_forecast.csv",
            mime="text/csv",
        )

    st.caption(
        "Future dates use weekdays and do not exclude exchange-specific market "
        "holidays. This educational model is not financial advice; forecasts "
        "are uncertain and should not be used as the sole basis for investment decisions."
    )


def render_market_forecasting_dashboard() -> None:
    st.title("Market Forecasting Using Supervised Machine Learning – Regression")
    st.markdown(
        "Explore historical market prices, evaluate a linear regression model "
        "against a previous-close baseline, and generate an iterative weekday forecast."
    )
    st.info(
        "Market predictions are inherently uncertain. This interactive example "
        "is for educational purposes only and is not financial advice."
    )

    with st.form("market_forecasting_controls"):
        ticker_options = ["AAPL", "AMZN", "GOOGL", "MSFT", "NVDA", "SPY", "TSLA", "Custom"]
        selected_ticker = st.selectbox("Market ticker", ticker_options, index=0)
        ticker = (
            st.text_input("Enter a ticker symbol", value="AAPL")
            if selected_ticker == "Custom"
            else selected_ticker
        )
        col_start, col_end = st.columns(2)
        today = date.today()
        with col_start:
            start_date = st.date_input(
                "Start date",
                value=today - timedelta(days=365 * 5),
                max_value=today,
            )
        with col_end:
            end_date = st.date_input("End date", value=today, max_value=today)
        col_test, col_forecast = st.columns(2)
        with col_test:
            test_fraction = st.slider(
                "Chronological test-set share",
                min_value=0.1,
                max_value=0.4,
                value=0.2,
                step=0.05,
            )
        with col_forecast:
            forecast_days = st.slider(
                "Weekday forecast horizon", min_value=1, max_value=30, value=10
            )
        submitted = st.form_submit_button("Download data and run forecast")

    if submitted:
        st.session_state["market_forecast_submitted"] = True
    if not st.session_state.get("market_forecast_submitted", False):
        st.caption(
            "Select a ticker and date range, then run the forecast. Historical "
            "data is downloaded from Yahoo Finance when requested."
        )
        return
    if not ticker.strip():
        st.error("Enter a ticker symbol.")
        return
    if start_date >= end_date:
        st.error("The start date must be earlier than the end date.")
        return

    with st.spinner(f"Downloading {ticker.strip().upper()} market data and fitting the model..."):
        _show_market_forecast(
            ticker.strip().upper(),
            start_date,
            end_date,
            test_fraction,
            forecast_days,
        )
