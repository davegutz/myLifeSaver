from dataclasses import dataclass, field
from urllib.error import URLError
from urllib.request import urlopen

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROLLING_AVERAGE_WINDOW = 12
CPI_SERIES_ID = "CPIAUCSL"
GP_LENGTH_SCALE = 18.0
GP_SIGNAL_VARIANCE = 0.000025
GP_NOISE_VARIANCE = 0.000004


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
    history_years: int | None = None
    start_clock: str | pd.Timestamp | None = None
    man_dob: str | pd.Timestamp | None = None
    woman_dob: str | pd.Timestamp | None = None
    man_age_at_death: float | None = None
    woman_age_at_death: float | None = None
    gp_model: GaussianProcessRegression | None = None
    inflation_frame: pd.DataFrame | None = None
    monthly_mean_inflation: float = 0.0
    monthly_inflation_volatility: float = 0.0
    historical_inflation: np.ndarray | None = None
    life_horizon_inflation: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    life_horizon_dates: np.ndarray = field(default_factory=lambda: np.array([], dtype="datetime64[ns]"))

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
        if self.history_years is None:
            raise ValueError("history_years must be set at Inflation instantiation before training.")
        monthly_cpi = load_cpi_history(end_date=self.current_date, years=self.history_years)
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
        seed: int | None = None,
    ) -> "Inflation":
        if self.inflation_frame is None or self.gp_model is None or self.historical_inflation is None:
            self.train(current_date=current_date)
        if (
            self.start_clock is None
            or self.man_dob is None
            or self.woman_dob is None
            or self.man_age_at_death is None
            or self.woman_age_at_death is None
        ):
            raise ValueError("Inflation life-horizon parameters must be set at instantiation.")

        first_projection_month = max(
            self.inflation_frame.index[-1] + pd.offsets.MonthEnd(1),
            pd.Timestamp(current_date) + pd.offsets.MonthEnd(0),
        )
        months_to_project = _required_projection_months(
            first_projection_month=first_projection_month,
            start_clock=self.start_clock,
            man_dob=self.man_dob,
            woman_dob=self.woman_dob,
            man_age_at_death=self.man_age_at_death,
            woman_age_at_death=self.woman_age_at_death,
        )

        generated = generate_inflation_projection(
            inflation_frame=self.inflation_frame,
            gp_model=self.gp_model,
            current_date=current_date,
            months_to_project=months_to_project,
            monthly_mean_inflation=self.monthly_mean_inflation,
            monthly_inflation_volatility=self.monthly_inflation_volatility,
            historical_inflation=self.historical_inflation,
            seed=seed,
            start_clock=self.start_clock,
            man_dob=self.man_dob,
            woman_dob=self.woman_dob,
            man_age_at_death=self.man_age_at_death,
            woman_age_at_death=self.woman_age_at_death,
        )
        self.monthly_inflation = generated.monthly_inflation
        self.life_horizon_inflation, self.life_horizon_dates = _build_life_horizon_arrays(self)
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
        start_clock: str | pd.Timestamp | None = None,
    ) -> float:
        date_ts = pd.Timestamp(date)
        start_clock_to_use = self.start_clock if start_clock is None else start_clock
        if start_clock_to_use is None:
            raise ValueError("start_clock must be provided either at Inflation instantiation or method call.")
        start_ts = pd.Timestamp(start_clock_to_use)

        if date_ts <= start_ts:
            return 1.0

        full_inflation_since_start = 1.0
        for item in self.monthly_inflation:
            if start_ts < item.month <= date_ts:
                full_inflation_since_start *= 1 + item.inflation

        return 1 / full_inflation_since_start


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


def load_cpi_history(
    series_id: str = CPI_SERIES_ID,
    end_date: str | pd.Timestamp | None = None,
    years: int | None = None,
) -> pd.Series:
    if years is None:
        raise ValueError("years must be supplied when loading CPI history.")
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


def build_inflation_frame(monthly_cpi: pd.Series) -> pd.DataFrame:
    frame = pd.DataFrame({"CPI": monthly_cpi})
    frame["Monthly_Inflation"] = frame["CPI"].pct_change()
    return frame.dropna()


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


def generate_inflation_projection(
    inflation_frame: pd.DataFrame,
    gp_model: GaussianProcessRegression | None,
    current_date: str | pd.Timestamp,
    months_to_project: int,
    monthly_mean_inflation: float = 0.0,
    monthly_inflation_volatility: float = 0.0,
    historical_inflation: np.ndarray | None = None,
    seed: int | None = None,
    start_clock: str | pd.Timestamp | None = None,
    man_dob: str | pd.Timestamp | None = None,
    woman_dob: str | pd.Timestamp | None = None,
    man_age_at_death: float | None = None,
    woman_age_at_death: float | None = None,
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
        start_clock=start_clock,
        man_dob=man_dob,
        woman_dob=woman_dob,
        man_age_at_death=man_age_at_death,
        woman_age_at_death=woman_age_at_death,
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


def _date_at_age(birth_date: str | pd.Timestamp, age_years: float) -> pd.Timestamp:
    birth_ts = pd.Timestamp(birth_date)
    whole_years = int(age_years)
    remaining_months = int(round((age_years - whole_years) * 12))
    return birth_ts + pd.DateOffset(years=whole_years, months=remaining_months)


def _build_life_horizon_arrays(
    inflation: Inflation,
) -> tuple[np.ndarray, np.ndarray]:
    if (
        inflation.start_clock is None
        or inflation.man_dob is None
        or inflation.woman_dob is None
        or inflation.man_age_at_death is None
        or inflation.woman_age_at_death is None
    ):
        raise ValueError("Inflation life-horizon parameters must be set at instantiation.")

    start_month = pd.Timestamp(inflation.start_clock) + pd.offsets.MonthEnd(0)
    end_date = max(
        _date_at_age(inflation.man_dob, inflation.man_age_at_death),
        _date_at_age(inflation.woman_dob, inflation.woman_age_at_death),
    )
    end_month = end_date + pd.offsets.MonthEnd(0)
    horizon_dates = pd.date_range(start=start_month, end=end_month, freq="ME")
    projected_inflation = pd.Series(
        [item.inflation for item in inflation.monthly_inflation],
        index=pd.DatetimeIndex([item.month for item in inflation.monthly_inflation]),
    )
    horizon_inflation = projected_inflation.reindex(horizon_dates)
    if horizon_inflation.isna().any():
        missing_dates = horizon_inflation.index[horizon_inflation.isna()]
        raise ValueError(
            "Inflation projection does not cover the required life horizon: "
            f"{missing_dates[0].date()} to {missing_dates[-1].date()}"
        )
    return horizon_inflation.to_numpy(dtype=float), horizon_dates.to_numpy()


def _required_projection_months(
    first_projection_month: pd.Timestamp,
    start_clock: str | pd.Timestamp,
    man_dob: str | pd.Timestamp,
    woman_dob: str | pd.Timestamp,
    man_age_at_death: float,
    woman_age_at_death: float,
) -> int:
    start_month = pd.Timestamp(start_clock) + pd.offsets.MonthEnd(0)
    end_date = max(
        _date_at_age(man_dob, man_age_at_death),
        _date_at_age(woman_dob, woman_age_at_death),
    )
    end_month = end_date + pd.offsets.MonthEnd(0)
    effective_start = min(first_projection_month, start_month)
    required_months = len(pd.date_range(start=effective_start, end=end_month, freq="ME"))
    return max(required_months, 0)


def prep_inflation(
    current_date: str | pd.Timestamp,
    history_years: int | None = None,
    seed: int | None = None,
    start_clock: str | pd.Timestamp | None = None,
    man_dob: str | pd.Timestamp | None = None,
    woman_dob: str | pd.Timestamp | None = None,
    man_age_at_death: float | None = None,
    woman_age_at_death: float | None = None,
) -> tuple[Inflation, float, pd.DataFrame]:
    if (
        history_years is None
        or
        start_clock is None
        or man_dob is None
        or woman_dob is None
        or man_age_at_death is None
        or woman_age_at_death is None
    ):
        raise ValueError("Inflation history_years and life-horizon parameters must be supplied by the caller.")

    inflation_model = Inflation(
        history_years=history_years,
        start_clock=start_clock,
        man_dob=man_dob,
        woman_dob=woman_dob,
        man_age_at_death=man_age_at_death,
        woman_age_at_death=woman_age_at_death,
    )
    inflation_frame = inflation_model.train(current_date=current_date)
    inflation_model.project(current_date=current_date, seed=seed)
    inflation = inflation_model
    annualized_inflation = inflation.annualized_inflation
    if inflation.inflation_frame is None:
        raise ValueError("Inflation history was not loaded during preparation.")
    return inflation, annualized_inflation, inflation.inflation_frame
