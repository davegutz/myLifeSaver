from dataclasses import dataclass, field
from io import StringIO
from urllib.error import URLError
from urllib.request import urlopen

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import build_life_horizon_dates, reindex_life_horizon_values, required_life_horizon_months

ROLLING_AVERAGE_WINDOW = 12
CPI_SERIES_ID = "CPIAUCSL"
GP_LENGTH_SCALE = 18.0
GP_SIGNAL_VARIANCE = 0.000025
GP_NOISE_VARIANCE = 0.000004
DEFAULT_MEAN_REVERSION = 0.15


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
    al_cum_running_avg_yrs: int | float | None = None
    start_clock: str | pd.Timestamp | None = None
    man_dob: str | pd.Timestamp | None = None
    woman_dob: str | pd.Timestamp | None = None
    man_age_at_death: float | None = None
    woman_age_at_death: float | None = None
    gp_model: GaussianProcessRegression | None = None
    inflation_frame: pd.DataFrame | None = None
    monthly_mean_inflation: float = 0.0
    monthly_inflation_volatility: float = 0.0
    mean_reversion_strength: float = DEFAULT_MEAN_REVERSION
    mean_shift: float = 0.0
    vol_multiplier: float = 1.0
    historical_inflation: np.ndarray | None = None
    life_horizon_inflation: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    life_horizon_dates: np.ndarray = field(default_factory=lambda: np.array([], dtype="datetime64[ns]"))
    life_horizon_inflation_cum: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    life_horizon_cpi_running_avg: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    result: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))

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

    def build_life_horizon_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        horizon_dates = build_life_horizon_dates(
            start_clock=self.start_clock,
            man_dob=self.man_dob,
            woman_dob=self.woman_dob,
            man_age_at_death=self.man_age_at_death,
            woman_age_at_death=self.woman_age_at_death,
        )
        historical_inflation = pd.Series(dtype=float)
        if self.inflation_frame is not None:
            historical_inflation = self.inflation_frame["Monthly_Inflation"].copy()
        return reindex_life_horizon_values(
            horizon_dates=horizon_dates,
            projected_dates=list(historical_inflation.index) + [item.month for item in self.monthly_inflation],
            projected_values=list(historical_inflation.to_numpy(dtype=float)) + [item.inflation for item in self.monthly_inflation],
            series_label="Inflation",
        )

    @staticmethod
    def build_inflation_frame(monthly_cpi: pd.Series) -> pd.DataFrame:
        frame = pd.DataFrame({"CPI": monthly_cpi})
        frame["Monthly_Inflation"] = frame["CPI"].pct_change()
        return frame.dropna()

    @staticmethod
    def build_right_axis(
        axis_left: plt.Axes,
        growth_dates: pd.Index | pd.DatetimeIndex | list[pd.Timestamp],
        growth_values: pd.Series | np.ndarray,
        running_avg_cpi_growth: pd.Series,
        cpi_running_average: pd.Series,
    ) -> plt.Axes:
        axis_right = axis_left.twinx()
        axis_right.plot(
            growth_dates,
            growth_values,
            color="black",
            linestyle="--",
            linewidth=1.8,
            label="Growth of $1",
        )
        if not cpi_running_average.isna().all():
            axis_right.plot(
                running_avg_cpi_growth.index,
                running_avg_cpi_growth.values,
                color="dimgray",
                linestyle="-.",
                linewidth=1.8,
                label="Running Avg CPI Growth",
            )
        axis_right.set_ylabel("Growth of $1")
        return axis_right

    def build_life_horizon_cpi_running_avg(self) -> np.ndarray:
        if self.al_cum_running_avg_yrs is None:
            raise ValueError("Inflation running-average years must be set at instantiation.")
        if self.life_horizon_inflation.size == 0:
            return np.array([], dtype=float)

        window_months = max(1, int(round(float(self.al_cum_running_avg_yrs) * 12)))
        running_average = pd.Series(self.life_horizon_inflation).rolling(window_months, min_periods=1).mean()
        return running_average.to_numpy(dtype=float)

    def build_life_horizon_inflation_cum(self) -> np.ndarray:
        if self.life_horizon_inflation.size == 0:
            return np.array([], dtype=float)
        return np.cumprod(1 + self.life_horizon_inflation)

    def build_life_horizon_result(self) -> np.ndarray:
        if self.life_horizon_dates.size == 0:
            return np.array([], dtype=float)
        return np.array(
            [self.de_escalate(pd.Timestamp(date)) for date in self.life_horizon_dates],
            dtype=float,
        )

    def calibrate_inflation_model(self) -> GaussianProcessRegression:
        if self.inflation_frame is None:
            raise ValueError("inflation_frame must be loaded before calibrating the inflation model.")
        recent_mean = float(
            self.inflation_frame["Monthly_Inflation"]
            .tail(ROLLING_AVERAGE_WINDOW)
            .mean()
        )
        gp_model = GaussianProcessRegression(mean_value=recent_mean)
        x_train = np.arange(len(self.inflation_frame), dtype=float)
        y_train = self.inflation_frame["Monthly_Inflation"].to_numpy(dtype=float)
        gp_model.fit(x_train, y_train)
        return gp_model

    def calibrate_inflation_statistics(self) -> tuple[float, float, np.ndarray]:
        if self.inflation_frame is None:
            raise ValueError("inflation_frame must be loaded before calibrating inflation statistics.")
        monthly_mean_inflation = float(self.inflation_frame["Monthly_Inflation"].mean())
        monthly_inflation_volatility = float(self.inflation_frame["Monthly_Inflation"].std())
        historical_inflation = self.inflation_frame["Monthly_Inflation"].to_numpy(dtype=float)
        return monthly_mean_inflation, monthly_inflation_volatility, historical_inflation

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
        if self.life_horizon_dates.size == 0 or self.life_horizon_inflation.size == 0:
            raise ValueError("life_horizon_inflation must be built before calling de_escalate.")

        horizon_dates = pd.DatetimeIndex(self.life_horizon_dates)
        matching_positions = np.where(horizon_dates == date_ts)[0]
        if matching_positions.size == 0:
            raise ValueError(f"No inflation value available for {date_ts.date()}.")

        return 1 / float(self.life_horizon_inflation[int(matching_positions[0])])

    def load_cpi_history(self) -> pd.Series:
        if self.history_years is None:
            raise ValueError("history_years must be set at Inflation instantiation before loading CPI history.")
        end_timestamp = pd.Timestamp.today().normalize() if self.current_date is None else pd.Timestamp(self.current_date)
        start_timestamp = end_timestamp - pd.DateOffset(years=self.history_years)
        fred_url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={CPI_SERIES_ID}"

        try:
            with urlopen(fred_url) as response:
                csv_text = response.read().decode("utf-8-sig")
                cpi_frame = pd.read_csv(StringIO(csv_text))
        except (URLError, OSError) as exc:
            raise ValueError(f"Unable to download CPI history for {CPI_SERIES_ID}.") from exc

        cpi_frame.columns = [str(column).strip().lstrip("\ufeff") for column in cpi_frame.columns]
        date_column = next(
            (column for column in cpi_frame.columns if column.upper() == "DATE"),
            cpi_frame.columns[0] if len(cpi_frame.columns) > 0 else None,
        )
        value_column = next(
            (column for column in cpi_frame.columns if column.strip() == CPI_SERIES_ID),
            cpi_frame.columns[1] if len(cpi_frame.columns) > 1 else None,
        )
        if date_column is None or value_column is None:
            raise ValueError(f"Unexpected CPI CSV schema for {CPI_SERIES_ID}: {list(cpi_frame.columns)}")

        cpi_frame = cpi_frame.rename(columns={date_column: "DATE", value_column: CPI_SERIES_ID})
        cpi_frame["DATE"] = pd.to_datetime(cpi_frame["DATE"])
        cpi_frame[CPI_SERIES_ID] = pd.to_numeric(cpi_frame[CPI_SERIES_ID], errors="coerce")
        cpi_frame = cpi_frame.dropna(subset=[CPI_SERIES_ID])
        cpi_frame = cpi_frame.loc[
            (cpi_frame["DATE"] >= start_timestamp) & (cpi_frame["DATE"] <= end_timestamp),
            ["DATE", CPI_SERIES_ID],
        ]
        if cpi_frame.empty:
            raise ValueError(f"No CPI history returned for {CPI_SERIES_ID}.")

        monthly_cpi = cpi_frame.set_index("DATE")[CPI_SERIES_ID].resample("ME").last().dropna()
        if monthly_cpi.empty:
            raise ValueError(f"No monthly CPI data available for {CPI_SERIES_ID}.")
        return monthly_cpi

    def get_next_month_inflation(self, rng: np.random.Generator) -> float:
        if self.historical_inflation is None:
            raise ValueError("historical_inflation must be available before sampling inflation.")
        sampled_inflation = float(rng.choice(self.historical_inflation))
        target_mean = self.monthly_mean_inflation + self.mean_shift
        adjusted_volatility = self.monthly_inflation_volatility * self.vol_multiplier
        mean_reversion = self.mean_reversion_strength * (target_mean - sampled_inflation)
        shock = float(rng.normal(0, adjusted_volatility * 0.15))
        return sampled_inflation + mean_reversion + shock

    def generate_inflation_projection(
        self,
        current_date: str | pd.Timestamp,
        months_to_project: int,
        seed: int | None = None,
    ) -> "Inflation":
        if self.gp_model is None:
            raise ValueError("A fit GaussianProcessRegression model is required for inflation projection.")
        if self.historical_inflation is None:
            raise ValueError("Historical inflation data is required for inflation projection.")
        if self.inflation_frame is None:
            raise ValueError("inflation_frame must be loaded before projection.")

        current_timestamp = pd.Timestamp(current_date)
        last_historical_month = self.inflation_frame.index[-1]
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
            history_years=self.history_years,
            al_cum_running_avg_yrs=self.al_cum_running_avg_yrs,
            start_clock=self.start_clock,
            man_dob=self.man_dob,
            woman_dob=self.woman_dob,
            man_age_at_death=self.man_age_at_death,
            woman_age_at_death=self.woman_age_at_death,
            gp_model=self.gp_model,
            inflation_frame=self.inflation_frame,
            monthly_mean_inflation=self.monthly_mean_inflation,
            monthly_inflation_volatility=self.monthly_inflation_volatility,
            mean_reversion_strength=self.mean_reversion_strength,
            mean_shift=self.mean_shift,
            vol_multiplier=self.vol_multiplier,
            historical_inflation=self.historical_inflation,
        )
        x_test = np.arange(len(self.inflation_frame), len(self.inflation_frame) + months_to_project, dtype=float)
        _, predicted_std = self.gp_model.predict(x_test)

        for month, std_value in zip(projection_months, predicted_std):
            next_inflation = self.get_next_month_inflation(rng)
            lower_bound = float(next_inflation - 1.96 * std_value)
            upper_bound = float(next_inflation + 1.96 * std_value)
            inflation.add(
                month=month,
                inflation=float(next_inflation),
                lower_bound=lower_bound,
                upper_bound=upper_bound,
            )
        return inflation

    @classmethod
    def prep_inflation(
        cls,
        current_date: str | pd.Timestamp,
        history_years: int | None = None,
        al_cum_running_avg_yrs: int | float | None = None,
        seed: int | None = None,
        start_clock: str | pd.Timestamp | None = None,
        man_dob: str | pd.Timestamp | None = None,
        woman_dob: str | pd.Timestamp | None = None,
        man_age_at_death: float | None = None,
        woman_age_at_death: float | None = None,
    ) -> tuple["Inflation", float, pd.DataFrame]:
        inflation_model = cls(
            history_years=history_years,
            al_cum_running_avg_yrs=al_cum_running_avg_yrs,
            start_clock=start_clock,
            man_dob=man_dob,
            woman_dob=woman_dob,
            man_age_at_death=man_age_at_death,
            woman_age_at_death=woman_age_at_death,
        )
        inflation_frame = inflation_model.train(current_date=current_date)
        inflation_model.project(current_date=current_date, seed=seed)
        annualized_inflation = inflation_model.annualized_inflation
        if inflation_model.inflation_frame is None:
            raise ValueError("Inflation history was not loaded during preparation.")
        return inflation_model, annualized_inflation, inflation_frame

    def project(
        self,
        current_date: str | pd.Timestamp,
        seed: int | None = None,
    ) -> "Inflation":
        if self.inflation_frame is None or self.gp_model is None or self.historical_inflation is None:
            self.train(current_date=current_date)

        first_projection_month = max(
            self.inflation_frame.index[-1] + pd.offsets.MonthEnd(1),
            pd.Timestamp(current_date) + pd.offsets.MonthEnd(0),
        )
        months_to_project = self.required_projection_months(first_projection_month)

        generated = self.generate_inflation_projection(
            current_date=current_date,
            months_to_project=months_to_project,
            seed=seed,
        )
        self.monthly_inflation = generated.monthly_inflation
        self.life_horizon_inflation, self.life_horizon_dates = self.build_life_horizon_arrays()
        self.life_horizon_inflation_cum = self.build_life_horizon_inflation_cum()
        self.life_horizon_cpi_running_avg = self.build_life_horizon_cpi_running_avg()
        self.result = self.build_life_horizon_result()
        self.validate_life_horizon_result()
        return self

    def required_projection_months(self, first_projection_month: pd.Timestamp) -> int:
        return required_life_horizon_months(
            first_projection_month=first_projection_month,
            start_clock=self.start_clock,
            man_dob=self.man_dob,
            woman_dob=self.woman_dob,
            man_age_at_death=self.man_age_at_death,
            woman_age_at_death=self.woman_age_at_death,
        )

    def train(self, current_date: str | pd.Timestamp) -> pd.DataFrame:
        self.current_date = pd.Timestamp(current_date)
        monthly_cpi = self.load_cpi_history()
        self.inflation_frame = self.build_inflation_frame(monthly_cpi)
        self.gp_model = self.calibrate_inflation_model()
        (
            self.monthly_mean_inflation,
            self.monthly_inflation_volatility,
            self.historical_inflation,
        ) = self.calibrate_inflation_statistics()
        return self.inflation_frame

    def validate_life_horizon_result(self) -> None:
        if self.life_horizon_inflation.size == 0 or self.result.size == 0:
            return

        normalized_growth = self.result * self.life_horizon_inflation
        expected_level = np.full_like(normalized_growth, normalized_growth[0], dtype=float)
        if not np.allclose(normalized_growth, expected_level, rtol=1e-9, atol=1e-9):
            raise ValueError(
                "Inflation result scale factor failed validation against life_horizon_inflation."
            )


def plot_inflation_axis(axis_left: plt.Axes, inflation: Inflation, title: str) -> None:
    months = [item.month for item in inflation.monthly_inflation]
    inflation_values = [item.inflation for item in inflation.monthly_inflation]
    rolling_average_values = [item.rolling_average_12m for item in inflation.monthly_inflation]
    cpi_running_average = pd.Series(
        inflation.life_horizon_cpi_running_avg,
        index=pd.DatetimeIndex(inflation.life_horizon_dates),
    ).reindex(months)
    lower_bounds = [item.lower_bound for item in inflation.monthly_inflation]
    upper_bounds = [item.upper_bound for item in inflation.monthly_inflation]
    growth_of_one_dollar = np.cumprod([1 + item.inflation for item in inflation.monthly_inflation])
    running_avg_cpi_growth = (1 + cpi_running_average.fillna(0.0)).cumprod()

    axis_left.plot(months, inflation_values, marker="o", linewidth=1.2, label="Monthly Inflation")
    axis_left.plot(months, rolling_average_values, linewidth=2.0, label="12M Rolling Average")
    if not cpi_running_average.isna().all():
        axis_left.plot(
            cpi_running_average.index,
            cpi_running_average.values,
            linewidth=2.0,
            linestyle="-.",
            label="CPI Running Average",
        )
    axis_left.fill_between(months, lower_bounds, upper_bounds, alpha=0.2, label="95% GP Interval")
    axis_left.set_xlabel("Month")
    axis_left.set_ylabel("Inflation")
    axis_left.grid(True, alpha=0.3)

    axis_right = Inflation.build_right_axis(
        axis_left=axis_left,
        growth_dates=months,
        growth_values=growth_of_one_dollar,
        running_avg_cpi_growth=running_avg_cpi_growth,
        cpi_running_average=cpi_running_average,
    )

    axis_left.set_title(title)
    lines_left, labels_left = axis_left.get_legend_handles_labels()
    lines_right, labels_right = axis_right.get_legend_handles_labels()
    axis_left.legend(lines_left + lines_right, labels_left + labels_right, loc="upper left")


def plot_inflation_with_history_axis(
    axis_left: plt.Axes,
    inflation_frame: pd.DataFrame,
    inflation: Inflation,
    title: str,
) -> None:
    historical_inflation = inflation_frame["Monthly_Inflation"].copy()
    historical_rolling = historical_inflation.rolling(ROLLING_AVERAGE_WINDOW, min_periods=1).mean()
    projected_months = pd.DatetimeIndex([item.month for item in inflation.monthly_inflation])
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
    projected_cpi_running_average = pd.Series(
        inflation.life_horizon_cpi_running_avg,
        index=pd.DatetimeIndex(inflation.life_horizon_dates),
    ).reindex(projected_months)

    full_inflation = pd.concat([historical_inflation, projected_inflation])
    full_rolling = pd.concat([historical_rolling, projected_rolling])
    growth_of_one_dollar = (1 + full_inflation).cumprod()
    running_avg_cpi_growth = (1 + projected_cpi_running_average.fillna(0.0)).cumprod()

    axis_left.plot(full_inflation.index, full_inflation.values, linewidth=1.0, label="Monthly Inflation")
    axis_left.plot(full_rolling.index, full_rolling.values, linewidth=2.0, label="12M Rolling Average")
    if not projected_cpi_running_average.isna().all():
        axis_left.plot(
            projected_cpi_running_average.index,
            projected_cpi_running_average.values,
            linewidth=2.0,
            linestyle="-.",
            label="CPI Running Average",
        )
    axis_left.fill_between(
        projected_months,
        projected_lower.values,
        projected_upper.values,
        alpha=0.2,
        label="95% GP Interval",
    )
    if len(projected_months) > 0:
        axis_left.axvline(projected_months[0], color="gray", linestyle=":", linewidth=1.5, label="Projection Start")
    axis_left.set_xlabel("Month")
    axis_left.set_ylabel("Inflation")
    axis_left.grid(True, alpha=0.3)

    axis_right = Inflation.build_right_axis(
        axis_left=axis_left,
        growth_dates=growth_of_one_dollar.index,
        growth_values=growth_of_one_dollar.values,
        running_avg_cpi_growth=running_avg_cpi_growth,
        cpi_running_average=projected_cpi_running_average,
    )

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
    plot_inflation_axis(axes[0], inflation, "Projected Monthly Inflation Over Time")
    plot_inflation_with_history_axis(
        axes[1],
        inflation_frame,
        inflation,
        "Historical and Projected Monthly Inflation Over Time",
    )
    plt.tight_layout()
    if show:
        plt.show()
