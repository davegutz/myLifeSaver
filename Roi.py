from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

from utils import build_life_horizon_dates, reindex_life_horizon_values, required_life_horizon_months


TICKER = "SPY"
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
    current_date: pd.Timestamp | None = None
    history_years: int | None = None
    start_clock: str | pd.Timestamp | None = None
    man_dob: str | pd.Timestamp | None = None
    woman_dob: str | pd.Timestamp | None = None
    man_age_at_death: float | None = None
    woman_age_at_death: float | None = None
    monthly_close: pd.Series | None = None
    return_frame: pd.DataFrame | None = None
    monthly_mean_return: float = 0.0
    monthly_volatility: float = 0.0
    historical_returns: np.ndarray | None = None
    life_horizon_roi: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    life_horizon_dates: np.ndarray = field(default_factory=lambda: np.array([], dtype="datetime64[ns]"))
    life_horizon_roi_cum: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))

    def __str__(self) -> str:
        lines = ["Projected monthly ROI"]
        for item in self.monthly_roi:
            lines.append(
                f"{item.month.strftime('%Y-%m')}: ROI {item.roi:.2%}, "
                f"12M Avg {item.rolling_average_12m:.2%}"
            )
        lines.append(f"Ending growth of $1.00: ${self.ending_value():.2f}")
        return "\n".join(lines)

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

    def build_life_horizon_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        horizon_dates = build_life_horizon_dates(
            start_clock=self.start_clock,
            man_dob=self.man_dob,
            woman_dob=self.woman_dob,
            man_age_at_death=self.man_age_at_death,
            woman_age_at_death=self.woman_age_at_death,
        )
        historical_roi = pd.Series(dtype=float)
        if self.return_frame is not None:
            historical_roi = self.return_frame["Monthly_Return"].copy()
        return reindex_life_horizon_values(
            horizon_dates=horizon_dates,
            projected_dates=list(historical_roi.index) + [item.month for item in self.monthly_roi],
            projected_values=list(historical_roi.to_numpy(dtype=float)) + [item.roi for item in self.monthly_roi],
            series_label="ROI",
        )

    @staticmethod
    def build_return_frame(monthly_close: pd.Series) -> pd.DataFrame:
        frame = pd.DataFrame({"Close": monthly_close})
        frame["Monthly_Return"] = frame["Close"].pct_change()
        return frame.dropna()

    def calibrate_growth_model(self) -> tuple[float, float, np.ndarray]:
        if self.return_frame is None:
            raise ValueError("return_frame must be loaded before calibrating the growth model.")
        monthly_mean_return = float(self.return_frame["Monthly_Return"].mean())
        monthly_volatility = float(self.return_frame["Monthly_Return"].std())
        historical_returns = self.return_frame["Monthly_Return"].to_numpy()
        return monthly_mean_return, monthly_volatility, historical_returns

    def ending_value(self, starting_value: float = 1.0) -> float:
        value = starting_value
        for item in self.monthly_roi:
            value *= 1 + item.roi
        return value

    def generate_projection(self, months_to_project: int, seed: int | None = None) -> "Roi":
        if self.monthly_close is None or self.historical_returns is None or self.return_frame is None:
            raise ValueError("ROI history must be trained before generating a projection.")

        roi = Roi(
            ticker=self.ticker,
            current_date=self.current_date,
            history_years=self.history_years,
            start_clock=self.start_clock,
            man_dob=self.man_dob,
            woman_dob=self.woman_dob,
            man_age_at_death=self.man_age_at_death,
            woman_age_at_death=self.woman_age_at_death,
            monthly_close=self.monthly_close.copy(),
            return_frame=self.return_frame,
            monthly_mean_return=self.monthly_mean_return,
            monthly_volatility=self.monthly_volatility,
            historical_returns=self.historical_returns,
        )
        simulated_close = self.monthly_close.copy()
        rng = np.random.default_rng(seed)

        for _ in range(months_to_project):
            next_roi = self.get_next_month_roi(rng)
            next_month = simulated_close.index[-1] + pd.offsets.MonthEnd(1)
            next_close = simulated_close.iloc[-1] * (1 + next_roi)
            simulated_close.loc[next_month] = next_close
            roi.add(next_month, next_roi)

        return roi

    def get_next_month_roi(self, rng: np.random.Generator) -> float:
        if self.historical_returns is None:
            raise ValueError("historical_returns must be available before sampling ROI.")
        sampled_return = float(rng.choice(self.historical_returns))
        mean_reversion = 0.15 * (self.monthly_mean_return - sampled_return)
        shock = float(rng.normal(0, self.monthly_volatility * 0.15))
        return sampled_return + mean_reversion + shock

    def load_price_history(self, ticker: str = TICKER) -> pd.Series:
        if self.history_years is None:
            raise ValueError("history_years must be supplied when loading ROI price history.")
        end_date_to_use = pd.Timestamp.today().normalize() if self.current_date is None else pd.Timestamp(self.current_date)
        start_date_to_use = end_date_to_use - pd.DateOffset(years=self.history_years)
        data = yf.download(
            ticker,
            start=start_date_to_use,
            end=end_date_to_use,
            auto_adjust=True,
            progress=False,
        )
        if data.empty:
            raise ValueError(f"No price history returned for {ticker}.")

        close_data: pd.Series = data["Close"].squeeze("columns")
        if not isinstance(close_data, pd.Series):
            raise ValueError(f"Unexpected close-data shape returned for {ticker}.")

        monthly_close = close_data.resample("ME").last().dropna()
        if monthly_close.empty:
            raise ValueError(f"No monthly close data available for {ticker}.")
        return monthly_close

    @classmethod
    def prep_projection(
        cls,
        ticker: str = TICKER,
        current_date: str | pd.Timestamp | None = None,
        history_years: int | None = None,
        seed: int | None = None,
        start_clock: str | pd.Timestamp | None = None,
        man_dob: str | pd.Timestamp | None = None,
        woman_dob: str | pd.Timestamp | None = None,
        man_age_at_death: float | None = None,
        woman_age_at_death: float | None = None,
    ) -> "Roi":
        if (
            current_date is None
            or history_years is None
            or start_clock is None
            or man_dob is None
            or woman_dob is None
            or man_age_at_death is None
            or woman_age_at_death is None
        ):
            raise ValueError("Roi current date, history_years, and life-horizon parameters must be supplied by the caller.")

        roi = cls(
            ticker=ticker,
            current_date=pd.Timestamp(current_date),
            history_years=history_years,
            start_clock=start_clock,
            man_dob=man_dob,
            woman_dob=woman_dob,
            man_age_at_death=man_age_at_death,
            woman_age_at_death=woman_age_at_death,
        )
        roi.train(ticker=ticker)
        roi.project(ticker=ticker, seed=seed)
        return roi

    def project(
        self,
        ticker: str = TICKER,
        seed: int | None = None,
    ) -> "Roi":
        if self.return_frame is None or self.monthly_close is None or self.historical_returns is None:
            self.train(ticker=ticker)
        months_to_project = self.required_projection_months(self.return_frame.index[-1])
        generated = self.generate_projection(months_to_project=months_to_project, seed=seed)
        self.monthly_roi = generated.monthly_roi
        self.life_horizon_roi, self.life_horizon_dates = self.build_life_horizon_arrays()
        self.life_horizon_roi_cum = np.cumprod(1 + self.life_horizon_roi)
        return self

    def required_projection_months(self, last_historical_month: pd.Timestamp) -> int:
        if (
            self.start_clock is None
            or self.man_dob is None
            or self.woman_dob is None
            or self.man_age_at_death is None
            or self.woman_age_at_death is None
        ):
            raise ValueError("Roi life-horizon parameters must be set at instantiation.")

        first_projected_month = last_historical_month + pd.offsets.MonthEnd(1)
        return required_life_horizon_months(
            first_projection_month=first_projected_month,
            start_clock=self.start_clock,
            man_dob=self.man_dob,
            woman_dob=self.woman_dob,
            man_age_at_death=self.man_age_at_death,
            woman_age_at_death=self.woman_age_at_death,
        )

    def train(self, ticker: str = TICKER) -> pd.DataFrame:
        self.ticker = ticker
        if self.current_date is None or self.history_years is None:
            raise ValueError("current_date and history_years must be set at Roi instantiation before training.")
        self.monthly_close = self.load_price_history(ticker=ticker)
        self.return_frame = self.build_return_frame(self.monthly_close)
        (
            self.monthly_mean_return,
            self.monthly_volatility,
            self.historical_returns,
        ) = self.calibrate_growth_model()
        return self.return_frame


def plot_projection_axis(axis_left: plt.Axes, roi: Roi, title: str) -> None:
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


def plot_projection_with_history_axis(
    axis_left: plt.Axes,
    return_frame: pd.DataFrame,
    roi: Roi,
    title: str,
) -> None:
    historical_roi = return_frame["Monthly_Return"].copy()
    historical_rolling = historical_roi.rolling(ROLLING_AVERAGE_WINDOW, min_periods=1).mean()
    projected_months = pd.DatetimeIndex([item.month for item in roi.monthly_roi])
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
    if len(projected_months) > 0:
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


def plot_projection_with_history(
    return_frame: pd.DataFrame,
    roi: Roi,
    show: bool = True,
) -> None:
    figure, axis = plt.subplots(figsize=(14, 7))
    plot_projection_with_history_axis(axis, return_frame, roi, "Historical and Projected Monthly ROI Over Time")
    plt.tight_layout()
    if show:
        plt.show()


def plot_projection_views(
    return_frame: pd.DataFrame,
    roi: Roi,
    show: bool = True,
) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(14, 12))
    plot_projection_axis(axes[0], roi, "Projected Monthly ROI Over Time")
    plot_projection_with_history_axis(
        axes[1],
        return_frame,
        roi,
        "Historical and Projected Monthly ROI Over Time",
    )
    plt.tight_layout()
    if show:
        plt.show()
