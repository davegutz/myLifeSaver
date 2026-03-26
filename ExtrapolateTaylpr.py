import argparse
from dataclasses import dataclass, field
from urllib.error import URLError
from urllib.request import urlopen

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf


TICKER = "SPY"
START_DATE = "2000-01-01"
END_DATE = "2025-01-01"
MONTHS_TO_PROJECT = 120
ROLLING_AVERAGE_WINDOW = 12
CPI_SERIES_ID = "CPIAUCSL"
CPI_HISTORY_YEARS = 25
GP_LENGTH_SCALE = 18.0
GP_SIGNAL_VARIANCE = 0.000025
GP_NOISE_VARIANCE = 0.000004
START_CLOCK = "2026-07-01",
MAN_DOB = "1957-07-26",
WOMAN_DOB = "1956-04-11",
MAN_AGE_TO_AL = 71,
WOMAN_AGE_TO_AL = 71,
MAN_AGE_AT_DEATH = 77,
WOMAN_AGE_AT_DEATH = 77,
PILE_AT_START = 5700000,

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


@dataclass
class Taylor_life:
    def __init__(
        self,
        projection: Roi,
        start_clock: str = START_CLOCK,
        man_dob: str = MAN_DOB,
        woman_dob: str = WOMAN_DOB,
        man_age_to_al: float = MAN_AGE_TO_AL,
        man_age_at_death: float = MAN_AGE_AT_DEATH,
        woman_age_to_al: float = WOMAN_AGE_TO_AL,
        woman_age_at_death: float = WOMAN_AGE_AT_DEATH,
        pile_at_start: float = PILE_AT_START,
        esc: float = 1,
        exp: float = 1000,
    ) -> None:
        # cc_exp_2: float = 3750,
        # cc_exp_1: float = 3150,
        # al_exp: float = 9200,
        # lc_exp_2: float = 9600,
        # lc_exp_1: float = 8100,
        self.projection = projection
        self.start_clock = start_clock
        self.man_dob = man_dob
        self.woman_dob = woman_dob
        self.man_age_to_al = man_age_to_al
        self.man_age_at_death = man_age_at_death
        self.woman_age_to_al = woman_age_to_al
        self.woman_age_at_death = woman_age_at_death
        self.pile_at_start = pile_at_start
        self.esc_al = esc_al
        self.esc_cc = esc_cc
        self.non_Taylor_exp = non_Taylor_exp
        self.exp_2 = exp_2
        self.exp_1 = exp_1

    def age(self, date: str | pd.Timestamp, birth_date: str | pd.Timestamp) -> float:
        date_ts = pd.Timestamp(date)
        birth_ts = pd.Timestamp(birth_date)
        return float((date_ts - birth_ts).days / 365.2425)

    def deceased(self, date: str | pd.Timestamp) -> bool:
        date_ts = pd.Timestamp(date)
        man_deceased = self.age(date_ts, self.man_dob) >= self.man_age_at_death
        woman_deceased = self.age(date_ts, self.woman_dob) >= self.woman_age_at_death
        return man_deceased and woman_deceased

    def al(self, date: str | pd.Timestamp) -> bool:
        date_ts = pd.Timestamp(date)
        man_in_al = self.man_age_to_al <= self.age(date_ts, self.man_dob) < self.man_age_at_death
        woman_in_al = self.woman_age_to_al <= self.age(date_ts, self.woman_dob) < self.woman_age_at_death
        return man_in_al or woman_in_al

    def cc_lc(self, date: str | pd.Timestamp) -> int:
        date_ts = pd.Timestamp(date)
        man_alive = self.age(date_ts, self.man_dob) < self.man_age_at_death
        woman_alive = self.age(date_ts, self.woman_dob) < self.woman_age_at_death
        return int(man_alive) + int(woman_alive)

    def de_escalate(
        self,
        date: str | pd.Timestamp,
        projection: Roi | None = None,
    ) -> float:
        date_ts = pd.Timestamp(date)
        start_ts = pd.Timestamp(self.start_clock)
        projection_to_use = projection if projection is not None else self.projection

        if date_ts <= start_ts:
            return 1.0

        full_growth_since_start = 1.0
        for item in projection_to_use.monthly_roi:
            if start_ts < item.month <= date_ts:
                full_growth_since_start *= 1 + item.roi

        de_escalate = 1 / full_growth_since_start
        return de_escalate


@dataclass
class MonthlyInflationPoint:
    month: pd.Timestamp
    inflation: float
    rolling_average_12m: float
    lower_bound: float
    upper_bound: float


@dataclass
class GaussianProcessRegression:
    length_scale: float = GP_LENGTH_SCALE
    signal_variance: float = GP_SIGNAL_VARIANCE
    noise_variance: float = GP_NOISE_VARIANCE
    mean_value: float = 0.0
    x_train: np.ndarray | None = None
    y_train: np.ndarray | None = None
    alpha: np.ndarray | None = None
    cholesky_factor: np.ndarray | None = None

    def _kernel(self, x_left: np.ndarray, x_right: np.ndarray) -> np.ndarray:
        squared_distance = (x_left[:, None] - x_right[None, :]) ** 2
        return self.signal_variance * np.exp(-0.5 * squared_distance / (self.length_scale ** 2))

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        self.x_train = np.asarray(x_train, dtype=float)
        self.y_train = np.asarray(y_train, dtype=float)
        centered_y_train = self.y_train - self.mean_value
        covariance = self._kernel(self.x_train, self.x_train)
        covariance += np.eye(len(self.x_train)) * self.noise_variance
        self.cholesky_factor = np.linalg.cholesky(covariance + np.eye(len(self.x_train)) * 1e-12)
        intermediate = np.linalg.solve(self.cholesky_factor, centered_y_train)
        self.alpha = np.linalg.solve(self.cholesky_factor.T, intermediate)

    def predict(self, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.x_train is None or self.alpha is None or self.cholesky_factor is None:
            raise ValueError("GaussianProcessRegression must be fit before calling predict.")

        x_test = np.asarray(x_test, dtype=float)
        cross_covariance = self._kernel(x_test, self.x_train)
        mean = self.mean_value + cross_covariance @ self.alpha
        solved = np.linalg.solve(self.cholesky_factor, cross_covariance.T)
        prior_variance = np.diag(self._kernel(x_test, x_test))
        variance = np.maximum(prior_variance - np.sum(solved ** 2, axis=0), 1e-12)
        return mean, np.sqrt(variance)


@dataclass
class Inflation:
    monthly_inflation: list[MonthlyInflationPoint] = field(default_factory=list)
    current_date: pd.Timestamp | None = None
    gp_model: GaussianProcessRegression | None = None
    inflation_frame: pd.DataFrame | None = None
    monthly_mean_inflation: float = 0.0
    monthly_inflation_volatility: float = 0.0
    historical_inflation: np.ndarray | None = None

    def add(
        self,
        month: pd.Timestamp,
        inflation: float,
        lower_bound: float,
        upper_bound: float,
    ) -> None:
        recent_inflation = [
            item.inflation for item in self.monthly_inflation[-(ROLLING_AVERAGE_WINDOW - 1):]
        ]
        recent_inflation.append(inflation)
        rolling_average_12m = float(np.mean(recent_inflation))
        self.monthly_inflation.append(
            MonthlyInflationPoint(
                month=month,
                inflation=inflation,
                rolling_average_12m=rolling_average_12m,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
            )
        )

    def train(self, current_date: str | pd.Timestamp) -> pd.DataFrame:
        self.current_date = pd.Timestamp(current_date)
        monthly_cpi = load_cpi_history(end_date=self.current_date)
        self.inflation_frame = build_inflation_frame(monthly_cpi)
        self.gp_model = calibrate_inflation_model(self.inflation_frame)
        (
            self.monthly_mean_inflation,
            self.monthly_inflation_volatility,
            self.historical_inflation,
        ) = calibrate_inflation_statistics(self.inflation_frame)
        return self.inflation_frame

    def project(
        self,
        current_date: str | pd.Timestamp,
        months_to_project: int = MONTHS_TO_PROJECT,
    ) -> "Inflation":
        inflation_frame = self.train(current_date)
        generated = generate_inflation_projection(
            inflation_frame=inflation_frame,
            gp_model=self.gp_model,
            current_date=current_date,
            months_to_project=months_to_project,
        )
        self.monthly_inflation = generated.monthly_inflation
        return self

    def __str__(self) -> str:
        lines = ["Projected monthly inflation"]
        for item in self.monthly_inflation:
            lines.append(
                f"{item.month.strftime('%Y-%m')}: Inflation {item.inflation:.2%}, "
                f"12M Avg {item.rolling_average_12m:.2%}, "
                f"95% CI [{item.lower_bound:.2%}, {item.upper_bound:.2%}]"
            )
        lines.append(f"Implied annualized inflation: {self.annualized_inflation:.2%}")
        return "\n".join(lines)

    @property
    def annualized_inflation(self) -> float:
        if not self.monthly_inflation:
            return 0.0
        monthly_mean = float(np.mean([item.inflation for item in self.monthly_inflation]))
        return float((1 + monthly_mean) ** 12 - 1)

    def de_escalate(
        self,
        date: str | pd.Timestamp,
        start_clock: str | pd.Timestamp = START_CLOCK,
    ) -> float:
        date_ts = pd.Timestamp(date)
        start_ts = pd.Timestamp(start_clock)

        if date_ts <= start_ts:
            return 1.0

        full_inflation_since_start = 1.0
        for item in self.monthly_inflation:
            if start_ts < item.month <= date_ts:
                full_inflation_since_start *= 1 + item.inflation

        return 1 / full_inflation_since_start


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


def _plot_inflation_axis(axis_left: plt.Axes, inflation: Inflation, title: str) -> None:
    months = [item.month for item in inflation.monthly_inflation]
    inflation_values = [item.inflation for item in inflation.monthly_inflation]
    rolling_average_values = [item.rolling_average_12m for item in inflation.monthly_inflation]
    lower_bounds = [item.lower_bound for item in inflation.monthly_inflation]
    upper_bounds = [item.upper_bound for item in inflation.monthly_inflation]
    growth_of_one_dollar = np.cumprod([1 + item.inflation for item in inflation.monthly_inflation])

    axis_left.plot(months, inflation_values, marker="o", linewidth=1.2, label="Monthly Inflation")
    axis_left.plot(months, rolling_average_values, linewidth=2.0, label="12M Rolling Average")
    axis_left.fill_between(months, lower_bounds, upper_bounds, alpha=0.2, label="95% GP Interval")
    axis_left.set_xlabel("Month")
    axis_left.set_ylabel("Inflation")
    axis_left.grid(True, alpha=0.3)

    axis_right = axis_left.twinx()
    axis_right.plot(
        months,
        growth_of_one_dollar,
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


def _plot_inflation_with_history_axis(
    axis_left: plt.Axes,
    inflation_frame: pd.DataFrame,
    inflation: Inflation,
    title: str,
) -> None:
    historical_inflation = inflation_frame["Monthly_Inflation"].copy()
    historical_rolling = historical_inflation.rolling(ROLLING_AVERAGE_WINDOW, min_periods=1).mean()
    projected_months = [item.month for item in inflation.monthly_inflation]
    projected_inflation = pd.Series(
        [item.inflation for item in inflation.monthly_inflation],
        index=projected_months,
    )
    projected_rolling = pd.Series(
        [item.rolling_average_12m for item in inflation.monthly_inflation],
        index=projected_months,
    )
    projected_lower = pd.Series(
        [item.lower_bound for item in inflation.monthly_inflation],
        index=projected_months,
    )
    projected_upper = pd.Series(
        [item.upper_bound for item in inflation.monthly_inflation],
        index=projected_months,
    )

    full_inflation = pd.concat([historical_inflation, projected_inflation])
    full_rolling = pd.concat([historical_rolling, projected_rolling])
    growth_of_one_dollar = (1 + full_inflation).cumprod()

    axis_left.plot(full_inflation.index, full_inflation.values, linewidth=1.0, label="Monthly Inflation")
    axis_left.plot(full_rolling.index, full_rolling.values, linewidth=2.0, label="12M Rolling Average")
    axis_left.fill_between(
        projected_months,
        projected_lower.values,
        projected_upper.values,
        alpha=0.2,
        label="95% GP Interval",
    )
    axis_left.axvline(projected_months[0], color="gray", linestyle=":", linewidth=1.5, label="Projection Start")
    axis_left.set_xlabel("Month")
    axis_left.set_ylabel("Inflation")
    axis_left.grid(True, alpha=0.3)

    axis_right = axis_left.twinx()
    axis_right.plot(
        growth_of_one_dollar.index,
        growth_of_one_dollar.values,
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


def plot_inflation_views(
    inflation_frame: pd.DataFrame,
    inflation: Inflation,
    show: bool = True,
) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(14, 12))
    _plot_inflation_axis(axes[0], inflation, "Projected Monthly Inflation Over Time")
    _plot_inflation_with_history_axis(
        axes[1],
        inflation_frame,
        inflation,
        "Historical and Projected Monthly Inflation Over Time",
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


def load_cpi_history(
    series_id: str = CPI_SERIES_ID,
    end_date: str | pd.Timestamp | None = None,
    years: int = CPI_HISTORY_YEARS,
) -> pd.Series:
    end_timestamp = pd.Timestamp.today().normalize() if end_date is None else pd.Timestamp(end_date)
    start_timestamp = end_timestamp - pd.DateOffset(years=years)
    fred_url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"

    try:
        with urlopen(fred_url) as response:
            cpi_frame = pd.read_csv(response)
    except (URLError, OSError) as exc:
        raise ValueError(f"Unable to download CPI history for {series_id}.") from exc

    cpi_frame.columns = [str(column).strip().lstrip("\ufeff") for column in cpi_frame.columns]
    date_column = next(
        (column for column in cpi_frame.columns if column.upper() == "DATE"),
        cpi_frame.columns[0] if len(cpi_frame.columns) > 0 else None,
    )
    value_column = next(
        (column for column in cpi_frame.columns if column.strip() == series_id),
        cpi_frame.columns[1] if len(cpi_frame.columns) > 1 else None,
    )
    if date_column is None or value_column is None:
        raise ValueError(f"Unexpected CPI CSV schema for {series_id}: {list(cpi_frame.columns)}")

    cpi_frame = cpi_frame.rename(columns={date_column: "DATE", value_column: series_id})
    cpi_frame["DATE"] = pd.to_datetime(cpi_frame["DATE"])
    cpi_frame[series_id] = pd.to_numeric(cpi_frame[series_id], errors="coerce")
    cpi_frame = cpi_frame.dropna(subset=[series_id])
    cpi_frame = cpi_frame.loc[
        (cpi_frame["DATE"] >= start_timestamp) & (cpi_frame["DATE"] <= end_timestamp),
        ["DATE", series_id],
    ]
    if cpi_frame.empty:
        raise ValueError(f"No CPI history returned for {series_id}.")

    monthly_cpi = cpi_frame.set_index("DATE")[series_id].resample("ME").last().dropna()
    if monthly_cpi.empty:
        raise ValueError(f"No monthly CPI data available for {series_id}.")
    return monthly_cpi


def build_return_frame(monthly_close: pd.Series) -> pd.DataFrame:
    frame = pd.DataFrame({"Close": monthly_close})
    frame["Monthly_Return"] = frame["Close"].pct_change()
    return frame.dropna()


def build_inflation_frame(monthly_cpi: pd.Series) -> pd.DataFrame:
    frame = pd.DataFrame({"CPI": monthly_cpi})
    frame["Monthly_Inflation"] = frame["CPI"].pct_change()
    return frame.dropna()


def calibrate_growth_model(return_frame: pd.DataFrame) -> tuple[float, float, np.ndarray]:
    monthly_mean_return = float(return_frame["Monthly_Return"].mean())
    monthly_volatility = float(return_frame["Monthly_Return"].std())
    historical_returns = return_frame["Monthly_Return"].to_numpy()
    return monthly_mean_return, monthly_volatility, historical_returns


def calibrate_inflation_model(inflation_frame: pd.DataFrame) -> GaussianProcessRegression:
    recent_mean = float(
        inflation_frame["Monthly_Inflation"]
        .tail(ROLLING_AVERAGE_WINDOW)
        .mean()
    )
    gp_model = GaussianProcessRegression(mean_value=recent_mean)
    x_train = np.arange(len(inflation_frame), dtype=float)
    y_train = inflation_frame["Monthly_Inflation"].to_numpy(dtype=float)
    gp_model.fit(x_train, y_train)
    return gp_model


def calibrate_inflation_statistics(inflation_frame: pd.DataFrame) -> tuple[float, float, np.ndarray]:
    monthly_mean_inflation = float(inflation_frame["Monthly_Inflation"].mean())
    monthly_inflation_volatility = float(inflation_frame["Monthly_Inflation"].std())
    historical_inflation = inflation_frame["Monthly_Inflation"].to_numpy(dtype=float)
    return monthly_mean_inflation, monthly_inflation_volatility, historical_inflation


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


def get_next_month_inflation(
    monthly_mean_inflation: float,
    monthly_inflation_volatility: float,
    historical_inflation: np.ndarray,
    rng: np.random.Generator,
) -> float:
    sampled_inflation = float(rng.choice(historical_inflation))
    mean_reversion = 0.15 * (monthly_mean_inflation - sampled_inflation)
    shock = float(rng.normal(0, monthly_inflation_volatility * 0.15))
    return sampled_inflation + mean_reversion + shock


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


def generate_inflation_projection(
    inflation_frame: pd.DataFrame,
    gp_model: GaussianProcessRegression | None,
    current_date: str | pd.Timestamp,
    months_to_project: int = MONTHS_TO_PROJECT,
    monthly_mean_inflation: float = 0.0,
    monthly_inflation_volatility: float = 0.0,
    historical_inflation: np.ndarray | None = None,
    seed: int | None = None,
) -> Inflation:
    if gp_model is None:
        raise ValueError("A fit GaussianProcessRegression model is required for inflation projection.")
    if historical_inflation is None:
        raise ValueError("Historical inflation data is required for inflation projection.")

    current_timestamp = pd.Timestamp(current_date)
    last_historical_month = inflation_frame.index[-1]
    first_projection_month = max(
        last_historical_month + pd.offsets.MonthEnd(1),
        current_timestamp + pd.offsets.MonthEnd(0),
    )

    projection_months = pd.date_range(
        start=first_projection_month,
        periods=months_to_project,
        freq="ME",
    )
    rng = np.random.default_rng(seed)
    inflation = Inflation(
        current_date=current_timestamp,
        gp_model=gp_model,
        inflation_frame=inflation_frame,
        monthly_mean_inflation=monthly_mean_inflation,
        monthly_inflation_volatility=monthly_inflation_volatility,
        historical_inflation=historical_inflation,
    )
    x_test = np.arange(len(inflation_frame), len(inflation_frame) + months_to_project, dtype=float)
    _, predicted_std = gp_model.predict(x_test)

    for month, std_value in zip(projection_months, predicted_std):
        next_inflation = get_next_month_inflation(
            monthly_mean_inflation,
            monthly_inflation_volatility,
            historical_inflation,
            rng,
        )
        lower_bound = float(next_inflation - 1.96 * std_value)
        upper_bound = float(next_inflation + 1.96 * std_value)
        inflation.add(
            month=month,
            inflation=float(next_inflation),
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )
    return inflation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monte Carlo monthly ROI projection anchored to historical long-run growth."
    )
    parser.add_argument("--ticker", default=TICKER, help="Ticker symbol to download, default: SPY")
    parser.add_argument("--months", type=int, default=MONTHS_TO_PROJECT, help="Months to project")
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed for reproducible runs")
    return parser.parse_args()


def prep_projection(
    ticker: str = TICKER,
    months_to_project: int = MONTHS_TO_PROJECT,
    seed: int | None = None,
) -> tuple[Roi, float, float, pd.DataFrame]:
    roi = Roi()
    return_frame = roi.train(ticker=ticker)
    roi.project(ticker=ticker, months_to_project=months_to_project, seed=seed)
    return roi, roi.monthly_mean_return, roi.monthly_volatility, return_frame


def prep_inflation(
    current_date: str | pd.Timestamp,
    months_to_project: int = MONTHS_TO_PROJECT,
    seed: int | None = None,
) -> tuple[Inflation, float, pd.DataFrame]:
    inflation_model = Inflation()
    inflation_frame = inflation_model.train(current_date=current_date)
    inflation = generate_inflation_projection(
        inflation_frame=inflation_frame,
        gp_model=inflation_model.gp_model,
        current_date=current_date,
        months_to_project=months_to_project,
        monthly_mean_inflation=inflation_model.monthly_mean_inflation,
        monthly_inflation_volatility=inflation_model.monthly_inflation_volatility,
        historical_inflation=inflation_model.historical_inflation,
        seed=seed,
    )
    annualized_inflation = inflation.annualized_inflation
    if inflation.inflation_frame is None:
        raise ValueError("Inflation history was not loaded during preparation.")
    return inflation, annualized_inflation, inflation.inflation_frame


def main() -> None:
    args = parse_args()
    roi, monthly_mean_return, monthly_volatility, return_frame = prep_projection(
        ticker=args.ticker,
        months_to_project=args.months,
        seed=args.seed,
    )
    current_date = pd.Timestamp.today().normalize()
    inflation, annualized_inflation, inflation_frame = prep_inflation(
        current_date=current_date,
        months_to_project=args.months,
        seed=args.seed,
    )

    annualized_mean = (1 + monthly_mean_return) ** 12 - 1
    annualized_mean_cpi = annualized_inflation
    print(
        f"Ticker: {args.ticker}\n"
        f"Historical monthly mean return: {monthly_mean_return:.2%}\n"
        f"Implied annualized return: {annualized_mean:.2%}\n"
        f"Monthly volatility: {monthly_volatility:.2%}\n"
        f"Seed: {args.seed}\n"
        f"CPI current date: {current_date.date()}\n"
        f"Implied annualized CPI inflation: {annualized_mean_cpi:.2%}"
    )
    print(roi)
    print(inflation)
    plot_projection_views(return_frame, roi, show=False)
    plot_inflation_views(inflation_frame, inflation, show=False)
    plt.show()


if __name__ == "__main__":
    main()
