from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor


TICKER = "SPY"
START_DATE = "2010-01-01"
END_DATE = "2025-01-01"
MONTHS_TO_PROJECT = 120
ROLLING_WINDOW = 6
RANDOM_SEED = 42


@dataclass
class MonthlySavingsRoi:
    month: pd.Timestamp
    roi: float
    rolling_average_12m: float


@dataclass
class SavingsProjection:
    monthly_roi: list[MonthlySavingsRoi] = field(default_factory=list)

    def add(self, month: pd.Timestamp, roi: float) -> None:
        recent_roi = [item.roi for item in self.monthly_roi[-11:]]
        recent_roi.append(roi)
        rolling_average_12m = float(np.mean(recent_roi))
        self.monthly_roi.append(
            MonthlySavingsRoi(
                month=month,
                roi=roi,
                rolling_average_12m=rolling_average_12m,
            )
        )

    def __str__(self) -> str:
        lines = ["Projected monthly ROI"]
        for item in self.monthly_roi:
            lines.append(
                f"{item.month.strftime('%Y-%m')}: ROI {item.roi:.2%}, "
                f"12M Avg {item.rolling_average_12m:.2%}"
            )
        return "\n".join(lines)


def plot_projection_roi(projection: SavingsProjection) -> None:
    months = [item.month for item in projection.monthly_roi]
    roi_values = [item.roi for item in projection.monthly_roi]
    rolling_average_values = [item.rolling_average_12m for item in projection.monthly_roi]

    plt.figure(figsize=(12, 6))
    plt.plot(months, roi_values, marker="o", linewidth=1.5, label="Monthly ROI")
    plt.plot(months, rolling_average_values, linewidth=2.0, label="12M Rolling Average")
    plt.title("Projected Monthly ROI Over Time")
    plt.xlabel("Month")
    plt.ylabel("ROI")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def load_price_history(
    ticker: str = TICKER,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
) -> pd.Series:
    data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)
    if data.empty:
        raise ValueError(f"No price history returned for {ticker}.")

    close_data = data["Close"]
    if isinstance(close_data, pd.DataFrame):
        close_data = close_data.iloc[:, 0]

    monthly_close = close_data.resample("ME").last().dropna()
    if monthly_close.empty:
        raise ValueError(f"No monthly close data available for {ticker}.")
    return monthly_close


def build_feature_frame(monthly_close: pd.Series, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    frame = pd.DataFrame({"Close": monthly_close})
    frame["Lag1"] = frame["Close"].pct_change()
    frame["MA_Ratio"] = frame["Close"] / frame["Close"].rolling(window=window).mean()
    frame["Vol"] = frame["Lag1"].rolling(window=window).std()
    frame["Target"] = frame["Close"].shift(-1) / frame["Close"] - 1
    return frame.dropna()


def train_model(feature_frame: pd.DataFrame) -> tuple[RandomForestRegressor, float]:
    feature_columns = ["Lag1", "MA_Ratio", "Vol"]
    split = int(len(feature_frame) * 0.8)
    if split <= 0 or split >= len(feature_frame):
        raise ValueError("Not enough data to train and test the model.")

    x_train = feature_frame.iloc[:split][feature_columns]
    y_train = feature_frame.iloc[:split]["Target"]
    x_test = feature_frame.iloc[split:][feature_columns]

    model = RandomForestRegressor(
        n_estimators=100,
        min_samples_split=10,
        random_state=RANDOM_SEED,
    )
    model.fit(x_train, y_train)

    noise_scale = float(x_test["Vol"].mean()) if not x_test.empty else float(feature_frame["Vol"].mean())
    return model, noise_scale


def get_next_month_roi(
    current_features: list[float],
    model: RandomForestRegressor,
    noise_scale: float,
    rng: np.random.Generator,
) -> float:
    feature_frame = pd.DataFrame(
        [current_features],
        columns=["Lag1", "MA_Ratio", "Vol"],
    )
    predicted_roi = float(model.predict(feature_frame)[0])
    noise = float(rng.normal(0, noise_scale))
    return predicted_roi + noise


def generate_projection(
    monthly_close: pd.Series,
    model: RandomForestRegressor,
    noise_scale: float,
    months_to_project: int = MONTHS_TO_PROJECT,
    window: int = ROLLING_WINDOW,
) -> SavingsProjection:
    projection = SavingsProjection()
    simulated_close = monthly_close.copy()
    rng = np.random.default_rng(RANDOM_SEED)

    for _ in range(months_to_project):
        last_returns = simulated_close.pct_change().dropna()
        lag1 = float(last_returns.iloc[-1])
        ma_ratio = float(simulated_close.iloc[-1] / simulated_close.rolling(window=window).mean().iloc[-1])
        vol = float(last_returns.rolling(window=window).std().iloc[-1])

        next_roi = get_next_month_roi([lag1, ma_ratio, vol], model, noise_scale, rng)
        next_month = simulated_close.index[-1] + pd.offsets.MonthEnd(1)
        next_close = simulated_close.iloc[-1] * (1 + next_roi)

        simulated_close.loc[next_month] = next_close
        projection.add(next_month, next_roi)

    return projection


def main() -> None:
    monthly_close = load_price_history()
    feature_frame = build_feature_frame(monthly_close)
    model, noise_scale = train_model(feature_frame)
    projection = generate_projection(monthly_close, model, noise_scale)
    print(projection)
    plot_projection_roi(projection)


if __name__ == "__main__":
    main()
