import pandas as pd
import numpy as np


def date_at_age(birth_date: str | pd.Timestamp, age_years: float) -> pd.Timestamp:
    birth_ts = pd.Timestamp(birth_date)
    whole_years = int(age_years)
    remaining_months = int(round((age_years - whole_years) * 12))
    return birth_ts + pd.DateOffset(years=whole_years, months=remaining_months)


def build_life_horizon_dates(
    start_clock: str | pd.Timestamp,
    man_dob: str | pd.Timestamp,
    woman_dob: str | pd.Timestamp,
    man_age_at_death: float,
    woman_age_at_death: float,
) -> pd.DatetimeIndex:
    start_month = pd.Timestamp(start_clock) + pd.offsets.MonthEnd(0)
    end_date = max(
        date_at_age(man_dob, man_age_at_death),
        date_at_age(woman_dob, woman_age_at_death),
    )
    end_month = end_date + pd.offsets.MonthEnd(0)
    return pd.date_range(start=start_month, end=end_month, freq="ME")


def required_life_horizon_months(
    first_projection_month: pd.Timestamp,
    start_clock: str | pd.Timestamp,
    man_dob: str | pd.Timestamp,
    woman_dob: str | pd.Timestamp,
    man_age_at_death: float,
    woman_age_at_death: float,
) -> int:
    horizon_dates = build_life_horizon_dates(
        start_clock=start_clock,
        man_dob=man_dob,
        woman_dob=woman_dob,
        man_age_at_death=man_age_at_death,
        woman_age_at_death=woman_age_at_death,
    )
    effective_start = min(pd.Timestamp(first_projection_month), horizon_dates[0])
    required_months = len(pd.date_range(start=effective_start, end=horizon_dates[-1], freq="ME"))
    return max(required_months, 0)


def reindex_life_horizon_values(
    horizon_dates: pd.DatetimeIndex,
    projected_dates: list[pd.Timestamp],
    projected_values: list[float],
    series_label: str,
) -> tuple[np.ndarray, np.ndarray]:
    projected_series = pd.Series(
        projected_values,
        index=pd.DatetimeIndex(projected_dates),
    )
    projected_series = projected_series.sort_index()
    projected_series = projected_series[~projected_series.index.duplicated(keep="last")]
    horizon_values = projected_series.reindex(horizon_dates)
    if horizon_values.isna().any():
        missing_dates = horizon_values.index[horizon_values.isna()]
        raise ValueError(
            f"{series_label} projection does not cover the required life horizon: "
            f"{missing_dates[0].date()} to {missing_dates[-1].date()}"
        )
    return horizon_values.to_numpy(dtype=float), horizon_dates.to_numpy()
