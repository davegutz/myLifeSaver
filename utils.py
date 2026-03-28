from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from Taylor import LhsScenario, ScenarioRunContext, TaylorLife, TaylorLifeResult


def age(date: str | pd.Timestamp, birth_date: str | pd.Timestamp) -> float:
    date_ts = pd.Timestamp(date)
    birth_ts = pd.Timestamp(birth_date)
    return float((date_ts - birth_ts).days / 365.2425)


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


def date_at_age(birth_date: str | pd.Timestamp, age_years: float) -> pd.Timestamp:
    birth_ts = pd.Timestamp(birth_date)
    whole_years = int(age_years)
    remaining_months = int(round((age_years - whole_years) * 12))
    return birth_ts + pd.DateOffset(years=whole_years, months=remaining_months)


def plot_taylor_life_exp_non_taylor(this_life: "TaylorLife", show: bool = True) -> None:
    if not this_life.exp_non_taylor_history:
        this_life.calc_result()

    dates = pd.DatetimeIndex(this_life.dates)
    num_il = np.asarray(this_life.num_il_1, dtype=float) + np.asarray(this_life.num_il_2, dtype=float)
    num_al = np.asarray(this_life.num_al_1, dtype=float) + np.asarray(this_life.num_al_2, dtype=float)
    num_non_taylor = (
        np.asarray(this_life.num_non_taylor_1, dtype=float) + np.asarray(this_life.num_non_taylor_2, dtype=float)
    )

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(14, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    axis_top, axis_bottom = axes
    axis_top.plot(pd.DatetimeIndex(this_life.dates), this_life.exp_non_taylor_history, linewidth=2.0, label="exp_non_taylor")
    axis_top.plot(dates, this_life.exp_norm_non_taylor, linewidth=2.0, linestyle="--", label="exp_norm_non_taylor")
    axis_top.plot(dates, this_life.exp_al_cc_history, linewidth=2.0, label="exp_al_cc")
    axis_top.plot(dates, this_life.exp_norm_al_cc, linewidth=2.0, linestyle="--", label="exp_norm_al_cc")
    axis_top.plot(dates, this_life.exp_cc_history, linewidth=2.0, label="exp_cc")
    axis_top.plot(dates, this_life.exp_norm_cc, linewidth=2.0, linestyle="--", label="exp_norm_cc")
    axis_top.plot(dates, this_life.exp_lc_history, linewidth=2.0, label="exp_lc")
    axis_top.plot(dates, this_life.exp_norm_lc, linewidth=2.0, linestyle="--", label="exp_norm_lc")
    axis_top.plot(dates, this_life.exp_total_cc_history, linewidth=4.0, label="exp_total_cc")
    axis_top.plot(dates, this_life.exp_norm_total_cc, linewidth=4.0, linestyle="--", label="exp_norm_total_cc")
    axis_top.plot(dates, this_life.exp_total_lc_history, linewidth=4.0, label="exp_total_lc")
    axis_top.plot(dates, this_life.exp_norm_total_lc, linewidth=4.0, linestyle="--", label="exp_norm_total_lc")
    axis_top.plot(dates, this_life.worth_lc_history, linewidth=2.0, label="worth_lc")
    axis_top.plot(dates, this_life.worth_norm_lc_history, linewidth=2.0, linestyle="--", label="worth_norm_lc")
    axis_top.plot(dates, this_life.worth_cc_history, linewidth=2.0, label="worth_cc")
    axis_top.plot(dates, this_life.worth_norm_cc_history, linewidth=2.0, linestyle="--", label="worth_norm_cc")
    axis_top.plot(dates, this_life.earn_lc_history, linewidth=2.0, label="earn_lc")
    axis_top.plot(dates, this_life.earn_norm_lc_history, linewidth=2.0, linestyle="--", label="earn_norm_lc")
    axis_top.plot(dates, this_life.earn_cc_history, linewidth=2.0, label="earn_cc")
    axis_top.plot(dates, this_life.earn_norm_cc_history, linewidth=2.0, linestyle="--", label="earn_norm_cc")
    axis_top.set_ylabel("Expense")
    axis_top.set_title("Taylor Life Expenses Over Time")
    axis_top.grid(True, alpha=0.3)
    axis_top.legend(loc="upper left")

    axis_bottom.plot(dates, num_il, linewidth=2.0, label="num_il")
    axis_bottom.plot(dates, num_al, linewidth=2.0, label="num_al")
    axis_bottom.plot(dates, num_non_taylor, linewidth=2.0, label="num_non_taylor")
    axis_bottom.set_xlabel("Date")
    axis_bottom.set_ylabel("Count")
    axis_bottom.grid(True, alpha=0.3)
    axis_bottom.legend(loc="upper left")
    plt.tight_layout()
    if show:
        plt.show()


def evaluate_lhs_scenario(
    scenario: LhsScenario,
    context: ScenarioRunContext | None = None,
) -> tuple[TaylorLife, TaylorLifeResult]:
    from Taylor import TaylorLife, TaylorLifeResult

    model = TaylorLife.from_lhs_scenario(scenario=scenario, context=context)
    worth_lc, worth_norm_lc, worth_cc, worth_norm_cc = model.calc_result()
    return model, TaylorLifeResult(
        worth_lc=worth_lc,
        worth_norm_lc=worth_norm_lc,
        worth_cc=worth_cc,
        worth_norm_cc=worth_norm_cc,
    )


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

