from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf


TICKER = "SPY"
START_DATE = "2000-01-01"
END_DATE = "2025-01-01"
MONTHS_TO_PROJECT = 120
ROLLING_AVERAGE_WINDOW = 12


@dataclass
class MonthlyRoiPoint:
    month: pd.Timestamp
    roi: float
    rolling_average_12m: float


@dataclass
class Roi:
    monthly_roi: list[MonthlyRoiPoint] = field(default_factory=list)
    ticker: str = TICKER
    monthly_close: pd.Series | None = None
    return_frame: pd.DataFrame | None = None
    monthly_mean_return: float = 0.0
    monthly_volatility: float = 0.0
    historical_returns: np.ndarray | None = None

    def add(self, month: pd.Timestamp, roi: float) -> None:
        recent_roi = [item.roi for item in self.monthly_roi[-(ROLLING_AVERAGE_WINDOW - 1):]]
        recent_roi.append(roi)
        rolling_average_12m = float(np.mean(recent_roi))
        self.monthly_roi.append(
            MonthlyRoiPoint(
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
        lines.append(f"Ending growth of $1.00: ${self.ending_value():.2f}")
        return "\n".join(lines)

    def ending_value(self, starting_value: float = 1.0) -> float:
        value = starting_value
        for item in self.monthly_roi:
            value *= 1 + item.roi
        return value

    def train(self, ticker: str = TICKER) -> pd.DataFrame:
        self.ticker = ticker
        self.monthly_close = load_price_history(ticker=ticker)
        self.return_frame = build_return_frame(self.monthly_close)
        (
            self.monthly_mean_return,
            self.monthly_volatility,
            self.historical_returns,
        ) = calibrate_growth_model(self.return_frame)
        return self.return_frame

    def project(
        self,
        ticker: str = TICKER,
        months_to_project: int = MONTHS_TO_PROJECT,
        seed: int | None = None,
    ) -> "Roi":
        if self.return_frame is None or self.monthly_close is None or self.historical_returns is None:
            self.train(ticker=ticker)

        generated = generate_projection(
            monthly_close=self.monthly_close,
            monthly_mean_return=self.monthly_mean_return,
            monthly_volatility=self.monthly_volatility,
            historical_returns=self.historical_returns,
            months_to_project=months_to_project,
            seed=seed,
            ticker=self.ticker,
            return_frame=self.return_frame,
        )
        self.monthly_roi = generated.monthly_roi
        return self


def _plot_projection_axis(axis_left: plt.Axes, roi: Roi, title: str) -> None:
    months = [item.month for item in roi.monthly_roi]
    roi_values = [item.roi for item in roi.monthly_roi]
    rolling_average_values = [item.rolling_average_12m for item in roi.monthly_roi]
    cumulative_values = np.cumprod([1 + item.roi for item in roi.monthly_roi])

    axis_left.plot(months, roi_values, marker="o", linewidth=1.2, label="Monthly ROI")
    axis_left.plot(months, rolling_average_values, linewidth=2.0, label="12M Rolling Average")
    axis_left.set_xlabel("Month")
    axis_left.set_ylabel("ROI")
    axis_left.grid(True, alpha=0.3)

    axis_right = axis_left.twinx()
    axis_right.plot(months, cumulative_values, color="black", linestyle="--", linewidth=1.8, label="Growth of $1")
    axis_right.set_ylabel("Growth of $1")

    axis_left.set_title(title)
    lines_left, labels_left = axis_left.get_legend_handles_labels()
    lines_right, labels_right = axis_right.get_legend_handles_labels()
    axis_left.legend(lines_left + lines_right, labels_left + labels_right, loc="upper left")


def _plot_projection_with_history_axis(
    axis_left: plt.Axes,
    return_frame: pd.DataFrame,
    roi: Roi,
    title: str,
) -> None:
    historical_roi = return_frame["Monthly_Return"].copy()
    historical_rolling = historical_roi.rolling(ROLLING_AVERAGE_WINDOW, min_periods=1).mean()
    projected_months = [item.month for item in roi.monthly_roi]
    projected_roi = pd.Series([item.roi for item in roi.monthly_roi], index=projected_months)
    projected_rolling = pd.Series(
        [item.rolling_average_12m for item in roi.monthly_roi],
        index=projected_months,
    )

    full_roi = pd.concat([historical_roi, projected_roi])
    full_rolling = pd.concat([historical_rolling, projected_rolling])
    full_growth = (1 + full_roi).cumprod()

    axis_left.plot(full_roi.index, full_roi.values, linewidth=1.0, label="Monthly ROI")
    axis_left.plot(full_rolling.index, full_rolling.values, linewidth=2.0, label="12M Rolling Average")
    axis_left.axvline(projected_months[0], color="gray", linestyle=":", linewidth=1.5, label="Projection Start")
    axis_left.set_xlabel("Month")
    axis_left.set_ylabel("ROI")
    axis_left.grid(True, alpha=0.3)

    axis_right = axis_left.twinx()
    axis_right.plot(
        full_growth.index,
        full_growth.values,
        color="black",
        linestyle="--",
        linewidth=1.8,
        label="Growth of $1",
    )
    axis_right.set_ylabel("Growth of $1")

    axis_left.set_title(title)
    lines_left, labels_left = axis_left.get_legend_handles_labels()
    lines_right, labels_right = axis_right.get_legend_handles_labels()
    axis_left.legend(lines_left + lines_right, labels_left + labels_right, loc="upper left")


def plot_projection_roi(roi: Roi, show: bool = True) -> None:
    figure, axis = plt.subplots(figsize=(12, 6))
    _plot_projection_axis(axis, roi, "Projected Monthly ROI Over Time")
    plt.tight_layout()
    if show:
        plt.show()


def plot_projection_with_history(
    return_frame: pd.DataFrame,
    roi: Roi,
    show: bool = True,
) -> None:
    figure, axis = plt.subplots(figsize=(14, 7))
    _plot_projection_with_history_axis(axis, return_frame, roi, "Historical and Projected Monthly ROI Over Time")
    plt.tight_layout()
    if show:
        plt.show()


def plot_projection_views(
    return_frame: pd.DataFrame,
    roi: Roi,
    show: bool = True,
) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(14, 12))
    _plot_projection_axis(axes[0], roi, "Projected Monthly ROI Over Time")
    _plot_projection_with_history_axis(
        axes[1],
        return_frame,
        roi,
        "Historical and Projected Monthly ROI Over Time",
    )
    plt.tight_layout()
    if show:
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


def build_return_frame(monthly_close: pd.Series) -> pd.DataFrame:
    frame = pd.DataFrame({"Close": monthly_close})
    frame["Monthly_Return"] = frame["Close"].pct_change()
    return frame.dropna()


def calibrate_growth_model(return_frame: pd.DataFrame) -> tuple[float, float, np.ndarray]:
    monthly_mean_return = float(return_frame["Monthly_Return"].mean())
    monthly_volatility = float(return_frame["Monthly_Return"].std())
    historical_returns = return_frame["Monthly_Return"].to_numpy()
    return monthly_mean_return, monthly_volatility, historical_returns


def get_next_month_roi(
    monthly_mean_return: float,
    monthly_volatility: float,
    historical_returns: np.ndarray,
    rng: np.random.Generator,
) -> float:
    sampled_return = float(rng.choice(historical_returns))
    mean_reversion = 0.15 * (monthly_mean_return - sampled_return)
    shock = float(rng.normal(0, monthly_volatility * 0.15))
    return sampled_return + mean_reversion + shock


def generate_projection(
    monthly_close: pd.Series,
    monthly_mean_return: float,
    monthly_volatility: float,
    historical_returns: np.ndarray,
    months_to_project: int = MONTHS_TO_PROJECT,
    seed: int | None = None,
    ticker: str = TICKER,
    return_frame: pd.DataFrame | None = None,
) -> Roi:
    roi = Roi(
        ticker=ticker,
        monthly_close=monthly_close.copy(),
        return_frame=return_frame,
        monthly_mean_return=monthly_mean_return,
        monthly_volatility=monthly_volatility,
        historical_returns=historical_returns,
    )
    simulated_close = monthly_close.copy()
    rng = np.random.default_rng(seed)

    for _ in range(months_to_project):
        next_roi = get_next_month_roi(
            monthly_mean_return,
            monthly_volatility,
            historical_returns,
            rng,
        )
        next_month = simulated_close.index[-1] + pd.offsets.MonthEnd(1)
        next_close = simulated_close.iloc[-1] * (1 + next_roi)

        simulated_close.loc[next_month] = next_close
        roi.add(next_month, next_roi)

    return roi


def prep_projection(
    ticker: str = TICKER,
    months_to_project: int = MONTHS_TO_PROJECT,
    seed: int | None = None,
) -> tuple[Roi, float, float, pd.DataFrame]:
    roi = Roi()
    return_frame = roi.train(ticker=ticker)
    roi.project(ticker=ticker, months_to_project=months_to_project, seed=seed)
    return roi, roi.monthly_mean_return, roi.monthly_volatility, return_frame
