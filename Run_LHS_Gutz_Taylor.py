"""
Run_LHS_Gutz_Taylor.py

Latin Hypercube Sampling (LHS) Monte Carlo analysis centered around the Gutz case inputs
from Run_one_Taylor.py local_run_overrides. Includes edge cases, replay cases, and all
plotting features.

Output: lhs_gutz_taylor_results.csv

The centerpoint scenario and ranges are defined below; modify them to explore different
regions of the scenario space.
"""


# User inputs
#  To force the probability both man and woman go to AL instead of dying right away
force_al = False
DEFAULT_LHS_POINTS = 1


import argparse
from dataclasses import asdict
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import cast
from Center_LHS_Gutz_Taylor import (
    CENTERPOINT_CONSTANT_MONTHLY_CPI,
    CENTERPOINT_CONSTANT_MONTHLY_ROI,
    CENTERPOINT_INFLATION_SEED,
    CENTERPOINT_MAN_ASSISTED_YRS,
    CENTERPOINT_MAN_GOES_TO_AL,
    CENTERPOINT_MAN_GOES_TO_AL_SEED,
    CENTERPOINT_MAN_INDEPENDENT_YRS,
    CENTERPOINT_ROI_SEED,
    CENTERPOINT_USE_CONSTANT_RATES,
    CENTERPOINT_WOMAN_ASSISTED_YRS,
    CENTERPOINT_WOMAN_GOES_TO_AL,
    CENTERPOINT_WOMAN_GOES_TO_AL_SEED,
    CENTERPOINT_WOMAN_INDEPENDENT_YRS,
)
from default_case import (
    AL_ESC_RUNNING_AVG_YRS,
    DEFAULT_SEED,
    HISTORY_YEARS,
    MAN_DOB,
    P_MAN_AL,
    P_WOMAN_AL,
    PILE_AT_START,
    START_CLOCK,
    WOMAN_DOB,
    apy_percent_to_monthly_fraction,
)
from edges import build_replay_case_scenarios_gutz, format_apy_suffix
from lhs_plotting import plot_lhs_figure1, plot_lhs_figure2_worth_subplots
from Roi import TICKER
from Taylor import LhsScenario, LhsScenarioSummary, ScenarioRunContext, TaylorLife, TaylorLifeResult
from utils import age, evaluate_lhs_scenario



# ============================================================================
# IMPORTANT: ROI AND INFLATION RATES
# ============================================================================
# CONSTANT_MONTHLY_ROI and CONSTANT_MONTHLY_CPI below determine ROI/inflation
# for ALL edge cases. Currently set to None, meaning stochastic/historical model.
# 
# Current Edge Case ROI/CPI Configuration:
#   - ROI:       None (uses stochastic model from historical data)
#   - Inflation: None (uses stochastic model from historical data)
#
# To use FIXED rates for edge cases, set:
#   - CONSTANT_MONTHLY_ROI = 0.10 / 12  # 10% annual = 0.833% monthly
#   - CONSTANT_MONTHLY_CPI = 0.05 / 12   # 5% annual = 0.417% monthly
# ============================================================================

# ============================================================================
# LHS VARIATION RANGES (±% around centerpoint for life parameters)
# ============================================================================
# For life parameters, use ±50% range around centerpoint
LIFE_PARAM_VARIATION = 0.50
MAN_INDEPENDENT_YRS_RANGE = (
    CENTERPOINT_MAN_INDEPENDENT_YRS * (1.0 - LIFE_PARAM_VARIATION),
    CENTERPOINT_MAN_INDEPENDENT_YRS * (1.0 + LIFE_PARAM_VARIATION),
)
WOMAN_INDEPENDENT_YRS_RANGE = (
    CENTERPOINT_WOMAN_INDEPENDENT_YRS * (1.0 - LIFE_PARAM_VARIATION),
    CENTERPOINT_WOMAN_INDEPENDENT_YRS * (1.0 + LIFE_PARAM_VARIATION),
)
MAN_ASSISTED_YRS_RANGE = (
    CENTERPOINT_MAN_ASSISTED_YRS * (1.0 - LIFE_PARAM_VARIATION),
    CENTERPOINT_MAN_ASSISTED_YRS * (1.0 + LIFE_PARAM_VARIATION),
)
WOMAN_ASSISTED_YRS_RANGE = (
    CENTERPOINT_WOMAN_ASSISTED_YRS * (1.0 - LIFE_PARAM_VARIATION),
    CENTERPOINT_WOMAN_ASSISTED_YRS * (1.0 + LIFE_PARAM_VARIATION),
)

# Seed and model parameter ranges (same as Run_LHS_Taylor)
SEED_RANGE = (0, 1000000)
ROI_MEAN_SHIFT_RANGE = (-0.01, 0.01)
ROI_VOL_MULTIPLIER_RANGE = (0.5, 1.5)
ROI_MEAN_REVERSION_RANGE = (0.0, 0.5)
INFLATION_MEAN_SHIFT_RANGE = (-0.005, 0.005)
INFLATION_VOL_MULTIPLIER_RANGE = (0.5, 1.5)
INFLATION_MEAN_REVERSION_RANGE = (0.0, 0.5)

# Set to True to force all stochastic LHS scenarios to go to AL regardless of seed.
# None (or False) uses the seed-based Bernoulli draw with P_MAN_AL / P_WOMAN_AL.
if force_al:
    LHS_MAN_GOES_TO_AL: bool | None = True
    LHS_WOMAN_GOES_TO_AL: bool | None = True
else:
    LHS_MAN_GOES_TO_AL: bool | None = None
    LHS_WOMAN_GOES_TO_AL: bool | None = None

PLOT_EDGE_CASES_IN_LHS_PLOT = True
# Edge points are explicit (roi_apy, cpi_apy) pairs instead of a Cartesian grid.
EDGE_CASE_ROI_CPI_APY_PAIRS = [
    (0.0, 0.0),
    (0.0, 12.0),
    (2.0, 6.0),
    (5.0, 4.0),
    (6.0, 6.0),
]
# Keep these for subplot layout/CLI compatibility; generation uses PAIRS above.
EDGE_CASE_ROI_APY_PERCENTS = sorted({pair[0] for pair in EDGE_CASE_ROI_CPI_APY_PAIRS})
EDGE_CASE_CPI_APY_PERCENTS = sorted({pair[1] for pair in EDGE_CASE_ROI_CPI_APY_PAIRS})
PLOT_MAIN_TITLE = "Taylor Community Lifecare / Continuing Care Decision,  2026 for Katherine and David Gutz"


CSV_COLUMNS = [
    "run_id",
    "yrs_il_single",
    "yrs_il_double",
    "yrs_sum_al",
    "total_living_yrs",
    "elapsed_time_yrs",
    "earning_potential",      # (PILE_AT_START - entrance_fee_lc) * (roi_cum[-1] / cpi_cum[-1]) * elapsed_time_yrs
    "earning_potential_cc",   # (PILE_AT_START - entrance_fee_cc) * (roi_cum[-1] / cpi_cum[-1]) * elapsed_time_yrs
    "added_lc_worth_norm",
    "man_independent_yrs",
    "woman_independent_yrs",
    "man_assisted_yrs",
    "woman_assisted_yrs",
    "roi_seed",
    "inflation_seed",
    "apy_roi",
    "apy_cpi",
    "roi_one_dollar_at_end",
    "cpi_one_dollar_at_end",
    "norm_one_dollar_at_end",
    "roi_mean_shift",
    "roi_vol_multiplier",
    "roi_mean_reversion",
    "inflation_mean_shift",
    "inflation_vol_multiplier",
    "inflation_mean_reversion",
    "man_goes_to_al_seed",
    "woman_goes_to_al_seed",
    "man_goes_to_al",
    "woman_goes_to_al",
    "man_age_to_al",
    "woman_age_to_al",
    "man_age_at_death",
    "woman_age_at_death",
    "exp_norm_al_cc",
    "exp_norm_cc",
    "exp_norm_lc",
    "exp_norm_non_taylor",
    "exp_norm_total_cc",
    "exp_norm_total_lc",
    "entrance_fee_cc",
    "entrance_fee_lc",
    "earn_norm_cc",
    "earn_norm_lc",
    "cum_mo_earn_lc_norm",
    "cum_mo_earn_cc_norm",
    "cum_mo_earn_ss_man_norm",
    "cum_mo_earn_ss_woman_norm",
    "cum_mo_earn_pen_man_norm",
    "cum_mo_earn_pen_woman_norm",
    "cum_mo_exp_lc_norm",
    "cum_mo_exp_cc_norm",
    "cum_mo_exp_al_cc_norm",
    "cum_mo_exp_non_taylor_norm",
    "cum_mo_exp_total_lc_norm",
    "cum_mo_exp_total_cc_norm",
    "start_pile",
    "final_worth_norm_cc",
    "final_worth_norm_lc",
    "worth_norm_lc",
    "worth_norm_cc",
    "man_age_at_start",
    "woman_age_at_start",
    # Context constants from this run
    "ticker",
    "current_date",
    "history_years",
    "al_cum_running_avg_yrs",
    "start_clock",
    "man_dob",
    "woman_dob",
    "constant_monthly_roi",
    "constant_monthly_cpi",
]

SCREEN_MIN_COL_WIDTH = 14


def format_screen_number(value: int | float, width: int) -> str:
    if isinstance(value, (np.integer, int)):
        text = f"{int(value):.3g}"
    else:
        numeric = float(value)
        if math.isnan(numeric):
            return " " * width
        text = f"{numeric:.3g}"
    if "e" in text or "E" in text:
        mantissa, exponent = text.lower().split("e", maxsplit=1)
        exponent_text = f"e{int(exponent):+03d}"
    else:
        mantissa = text
        exponent_text = ""
    if "." not in mantissa:
        mantissa = f"{mantissa}."
    left, right = mantissa.split(".", maxsplit=1)
    decimal_target = max(2, width - max(len(exponent_text), 4) - 1)
    left_padding = max(decimal_target - len(left), 0)
    aligned = f"{' ' * left_padding}{left}.{right}{exponent_text}"
    return aligned.rjust(width)


def format_screen_cell(value: object, width: int) -> str:
    if isinstance(value, (int, float, np.integer, np.floating)):
        return format_screen_number(value=value, width=width)
    return str(value).rjust(width)


def print_screen_row(row: dict[str, object], columns: list[str], widths: dict[str, int]) -> None:
    print(" ".join(format_screen_cell(row[column], widths[column]) for column in columns))


def add_lifecare_reference_line(axis: plt.Axes) -> None:
    """Add a bold y=0 reference line and a label just above it."""
    axis.axhline(0.0, color="black", linewidth=3.0, alpha=0.95, zorder=0)
    x_min, x_max = axis.get_xlim()
    y_min, y_max = axis.get_ylim()
    x_text = x_min + 0.02 * (x_max - x_min)
    y_text = 0.0 + 0.02 * (y_max - y_min)
    axis.text(
        x_text,
        y_text,
        "Lifecare better",
        color="black",
        fontsize=10,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def sample_lhs_points(num_points: int, dimensions: int, seed: int) -> np.ndarray:
    if num_points <= 0:
        raise ValueError("num_points must be positive for LHS sampling.")
    rng = np.random.default_rng(seed)
    lhs = np.zeros((num_points, dimensions), dtype=float)
    for dim in range(dimensions):
        cut_points = (np.arange(num_points, dtype=float) + rng.random(num_points)) / num_points
        lhs[:, dim] = cut_points[rng.permutation(num_points)]
    return lhs


def scale_lhs_column(values: np.ndarray, bounds: tuple[float, float]) -> np.ndarray:
    low, high = bounds
    return low + values * (high - low)


def build_lhs_scenarios(num_points: int, seed: int) -> list[LhsScenario]:
    sampled = sample_lhs_points(num_points, dimensions=14, seed=seed)
    scenarios: list[LhsScenario] = []
    for idx in range(num_points):
        man_goes_to_al_seed = int(round(scale_lhs_column(sampled[:, 12], SEED_RANGE)[idx]))
        woman_goes_to_al_seed = int(round(scale_lhs_column(sampled[:, 13], SEED_RANGE)[idx]))
        scenario = cast(
            LhsScenario,
            LhsScenario(
                man_independent_yrs=float(scale_lhs_column(sampled[:, 0], MAN_INDEPENDENT_YRS_RANGE)[idx]),
                woman_independent_yrs=float(scale_lhs_column(sampled[:, 1], WOMAN_INDEPENDENT_YRS_RANGE)[idx]),
                man_assisted_yrs=float(scale_lhs_column(sampled[:, 2], MAN_ASSISTED_YRS_RANGE)[idx]),
                woman_assisted_yrs=float(scale_lhs_column(sampled[:, 3], WOMAN_ASSISTED_YRS_RANGE)[idx]),
                roi_seed=int(round(scale_lhs_column(sampled[:, 4], SEED_RANGE)[idx])),
                inflation_seed=int(round(scale_lhs_column(sampled[:, 5], SEED_RANGE)[idx])),
                roi_mean_shift=float(scale_lhs_column(sampled[:, 6], ROI_MEAN_SHIFT_RANGE)[idx]),
                roi_vol_multiplier=float(scale_lhs_column(sampled[:, 7], ROI_VOL_MULTIPLIER_RANGE)[idx]),
                roi_mean_reversion=float(scale_lhs_column(sampled[:, 8], ROI_MEAN_REVERSION_RANGE)[idx]),
                inflation_mean_shift=float(scale_lhs_column(sampled[:, 9], INFLATION_MEAN_SHIFT_RANGE)[idx]),
                inflation_vol_multiplier=float(scale_lhs_column(sampled[:, 10], INFLATION_VOL_MULTIPLIER_RANGE)[idx]),
                inflation_mean_reversion=float(scale_lhs_column(sampled[:, 11], INFLATION_MEAN_REVERSION_RANGE)[idx]),
                man_goes_to_al_seed=man_goes_to_al_seed,
                woman_goes_to_al_seed=woman_goes_to_al_seed,
                man_goes_to_al=LHS_MAN_GOES_TO_AL if LHS_MAN_GOES_TO_AL is not None else bool(np.random.default_rng(man_goes_to_al_seed).binomial(1, P_MAN_AL)),
                woman_goes_to_al=LHS_WOMAN_GOES_TO_AL if LHS_WOMAN_GOES_TO_AL is not None else bool(np.random.default_rng(woman_goes_to_al_seed).binomial(1, P_WOMAN_AL)),
            ),
        )
        scenarios.append(scenario)
    return scenarios


def build_centerpoint_scenario() -> LhsScenario:
    """Build the explicit centerpoint scenario for the Gutz LHS run."""
    return LhsScenario(
        man_independent_yrs=CENTERPOINT_MAN_INDEPENDENT_YRS,
        woman_independent_yrs=CENTERPOINT_WOMAN_INDEPENDENT_YRS,
        man_assisted_yrs=CENTERPOINT_MAN_ASSISTED_YRS,
        woman_assisted_yrs=CENTERPOINT_WOMAN_ASSISTED_YRS,
        roi_seed=CENTERPOINT_ROI_SEED,
        inflation_seed=CENTERPOINT_INFLATION_SEED,
        man_goes_to_al_seed=CENTERPOINT_MAN_GOES_TO_AL_SEED,
        woman_goes_to_al_seed=CENTERPOINT_WOMAN_GOES_TO_AL_SEED,
        man_goes_to_al=CENTERPOINT_MAN_GOES_TO_AL,
        woman_goes_to_al=CENTERPOINT_WOMAN_GOES_TO_AL,
        roi_mean_shift=(ROI_MEAN_SHIFT_RANGE[0] + ROI_MEAN_SHIFT_RANGE[1]) / 2.0,
        roi_vol_multiplier=(ROI_VOL_MULTIPLIER_RANGE[0] + ROI_VOL_MULTIPLIER_RANGE[1]) / 2.0,
        roi_mean_reversion=(ROI_MEAN_REVERSION_RANGE[0] + ROI_MEAN_REVERSION_RANGE[1]) / 2.0,
        inflation_mean_shift=(INFLATION_MEAN_SHIFT_RANGE[0] + INFLATION_MEAN_SHIFT_RANGE[1]) / 2.0,
        inflation_vol_multiplier=(INFLATION_VOL_MULTIPLIER_RANGE[0] + INFLATION_VOL_MULTIPLIER_RANGE[1]) / 2.0,
        inflation_mean_reversion=(INFLATION_MEAN_REVERSION_RANGE[0] + INFLATION_MEAN_REVERSION_RANGE[1]) / 2.0,
    )


def last_value(values: list[float]) -> float:
    return float(values[-1]) if values else 0.0



def monthly_rate_to_apy(monthly_rate: float) -> float:
    return (1.0 + monthly_rate) ** 12 - 1.0


def realized_monthly_rate(path, fallback: float) -> float:
    if len(path) == 0:
        return fallback
    growth = (1.0 + path).prod()
    if growth <= 0.0:
        return fallback
    months = float(len(path))
    return float(growth ** (1.0 / months) - 1.0)


def effective_apy_from_cumulative(cumulative_path: np.ndarray, monthly_fallback: float) -> float:
    if cumulative_path.size > 0:
        final_growth = float(cumulative_path[-1])
        if final_growth > 0.0:
            months = float(cumulative_path.size)
            return (final_growth ** (12.0 / months) - 1.0) * 100.0
    return monthly_rate_to_apy(monthly_fallback) * 100.0


def format_constant_monthly_output(value: float | None) -> float | str:
    return "stochastic" if value is None else value


def normalize_centerpoint_constant_monthly(value: float | None) -> float | None:
    """
    Accept either:
      - monthly fraction (e.g., 0.008 for 0.8%/mo), or
      - APY percent (e.g., 10.0 for 10% APY)
    for centerpoint constant ROI/CPI inputs.
    """
    if value is None:
        return None
    numeric = float(value)
    if abs(numeric) > 1.0:
        return apy_percent_to_monthly_fraction(numeric)
    return numeric


def summarize_lhs_run(
    run_id: int | str,
    scenario: LhsScenario,
    model: TaylorLife,
    result: TaylorLifeResult,
    context: ScenarioRunContext,
) -> LhsScenarioSummary:
    roi_effective_apy = effective_apy_from_cumulative(model.roi.life_horizon_roi_cum, model.roi.monthly_mean_return)
    cpi_effective_apy = effective_apy_from_cumulative(
        model.cpi.life_horizon_inflation_cum,
        model.cpi.monthly_mean_inflation,
    )
    return LhsScenarioSummary(
        run_id=run_id,
        man_independent_yrs=scenario.man_independent_yrs,
        woman_independent_yrs=scenario.woman_independent_yrs,
        man_assisted_yrs=scenario.man_assisted_yrs,
        woman_assisted_yrs=scenario.woman_assisted_yrs,
        roi_seed=scenario.roi_seed,
        inflation_seed=scenario.inflation_seed,
        apy_roi=roi_effective_apy,
        apy_cpi=cpi_effective_apy,
        roi_one_dollar_at_end=float(model.roi.life_horizon_roi_cum[-1]),
        cpi_one_dollar_at_end=float(model.cpi.life_horizon_inflation_cum[-1]),
        norm_one_dollar_at_end=float(model.cpi.life_horizon_inflation_cum[-1]),
        roi_mean_shift=scenario.roi_mean_shift,
        roi_vol_multiplier=scenario.roi_vol_multiplier,
        roi_mean_reversion=scenario.roi_mean_reversion,
        inflation_mean_shift=scenario.inflation_mean_shift,
        inflation_vol_multiplier=scenario.inflation_vol_multiplier,
        inflation_mean_reversion=scenario.inflation_mean_reversion,
        exp_norm_al_cc=last_value(model.exp_norm_al_cc),
        exp_norm_cc=last_value(model.exp_norm_cc),
        exp_norm_lc=last_value(model.exp_norm_lc),
        exp_norm_non_taylor=last_value(model.exp_norm_non_taylor),
        exp_norm_total_cc=last_value(model.exp_norm_total_cc),
        exp_norm_total_lc=last_value(model.exp_norm_total_lc),
        entrance_fee_cc=model.entrance_fee_cc,
        entrance_fee_lc=model.entrance_fee_lc,
        earn_norm_cc=last_value(model.earn_norm_cc_history),
        earn_norm_lc=last_value(model.earn_norm_lc_history),
        cum_mo_earn_lc_norm=last_value(model.cum_mo_earn_lc_norm),
        cum_mo_earn_cc_norm=last_value(model.cum_mo_earn_cc_norm),
        cum_mo_earn_ss_man_norm=last_value(model.cum_mo_earn_ss_man_norm),
        cum_mo_earn_ss_woman_norm=last_value(model.cum_mo_earn_ss_woman_norm),
        cum_mo_earn_pen_man_norm=last_value(model.cum_mo_earn_pen_man_norm),
        cum_mo_earn_pen_woman_norm=last_value(model.cum_mo_earn_pen_woman_norm),
        cum_mo_exp_lc_norm=last_value(model.cum_mo_exp_lc_norm),
        cum_mo_exp_cc_norm=last_value(model.cum_mo_exp_cc_norm),
        cum_mo_exp_al_cc_norm=last_value(model.cum_mo_exp_al_cc_norm),
        cum_mo_exp_non_taylor_norm=last_value(model.cum_mo_exp_non_taylor_norm),
        cum_mo_exp_total_lc_norm=last_value(model.cum_mo_exp_total_lc_norm),
        cum_mo_exp_total_cc_norm=last_value(model.cum_mo_exp_total_cc_norm),
        start_pile=float(PILE_AT_START),
        final_worth_norm_cc=float(PILE_AT_START + last_value(model.cum_mo_earn_cc_norm) - last_value(model.cum_mo_exp_total_cc_norm)),
        final_worth_norm_lc=float(PILE_AT_START + last_value(model.cum_mo_earn_lc_norm) - last_value(model.cum_mo_exp_total_lc_norm)),
        worth_norm_lc=result.worth_norm_lc,
        worth_norm_cc=result.worth_norm_cc,
        added_lc_worth_norm=result.worth_norm_lc - result.worth_norm_cc,
        yrs_il_double=min(scenario.man_independent_yrs, scenario.woman_independent_yrs),
        yrs_il_single=abs(scenario.man_independent_yrs - scenario.woman_independent_yrs),
        yrs_sum_al=scenario.man_assisted_yrs + scenario.woman_assisted_yrs,
        total_living_yrs=scenario.man_independent_yrs + scenario.woman_independent_yrs + scenario.man_assisted_yrs + scenario.woman_assisted_yrs,
        elapsed_time_yrs=float((pd.Timestamp(model.dates[-1]) - model.start_clock).days / 365.2425),
        earning_potential=float((PILE_AT_START - model.entrance_fee_lc) * (model.roi.life_horizon_roi_cum[-1] / model.cpi.life_horizon_inflation_cum[-1]) * float((pd.Timestamp(model.dates[-1]) - model.start_clock).days / 365.2425)),
        earning_potential_cc=float((PILE_AT_START - model.entrance_fee_cc) * (model.roi.life_horizon_roi_cum[-1] / model.cpi.life_horizon_inflation_cum[-1]) * float((pd.Timestamp(model.dates[-1]) - model.start_clock).days / 365.2425)),
        man_goes_to_al_seed=scenario.man_goes_to_al_seed,
        woman_goes_to_al_seed=scenario.woman_goes_to_al_seed,
        man_goes_to_al=model.man_goes_to_al,
        woman_goes_to_al=model.woman_goes_to_al,
        man_age_to_al=model.man_age_to_al if model.man_goes_to_al else '',
        woman_age_to_al=model.woman_age_to_al if model.woman_goes_to_al else '',
        man_age_at_death=model.man_age_at_death,
        woman_age_at_death=model.woman_age_at_death,
        man_age_at_start=age(context.start_clock, context.man_dob),
        woman_age_at_start=age(context.start_clock, context.woman_dob),
        # Context constants
        ticker=context.ticker,
        current_date=str(context.current_date),
        history_years=context.history_years,
        al_cum_running_avg_yrs=context.al_cum_running_avg_yrs,
        start_clock=context.start_clock,
        man_dob=context.man_dob,
        woman_dob=context.woman_dob,
        constant_monthly_roi=format_constant_monthly_output(context.constant_monthly_roi),
        constant_monthly_cpi=format_constant_monthly_output(context.constant_monthly_cpi),
    )


def run_lhs_driver(num_points: int, context: ScenarioRunContext, output_path: Path, seed: int) -> pd.DataFrame:
    if context.constant_monthly_roi is not None or context.constant_monthly_cpi is not None:
        print(
            "Using fixed monthly ROI/CPI from default_case.py; "
            "apy_roi and apy_cpi reflect effective APY from final growth of $1 "
            "under those configured constants."
        )
    scenarios = build_lhs_scenarios(num_points=num_points, seed=seed)
    rows = []
    column_widths = {column: max(len(column), SCREEN_MIN_COL_WIDTH) for column in CSV_COLUMNS}
    print(" ".join(column.rjust(column_widths[column]) for column in CSV_COLUMNS))
    
    # Process random LHS scenarios
    for run_id, scenario in enumerate(scenarios, start=1):
        model, result = evaluate_lhs_scenario(scenario=scenario, context=context)
        row = asdict(summarize_lhs_run(run_id=run_id, scenario=scenario, model=model, result=result, context=context))
        ordered_row = {column: row[column] for column in CSV_COLUMNS}
        print_screen_row(row=ordered_row, columns=CSV_COLUMNS, widths=column_widths)
        rows.append(ordered_row)
    
    # Process explicit fixed edge points from centerpoint scenario (no stochastic rates).
    for roi_apy, cpi_apy in EDGE_CASE_ROI_CPI_APY_PAIRS:
        fixed_monthly_roi = apy_percent_to_monthly_fraction(roi_apy)
        fixed_monthly_cpi = apy_percent_to_monthly_fraction(cpi_apy)
        edge_context = ScenarioRunContext(
            ticker=context.ticker,
            current_date=context.current_date,
            history_years=context.history_years,
            al_cum_running_avg_yrs=context.al_cum_running_avg_yrs,
            start_clock=context.start_clock,
            man_dob=context.man_dob,
            woman_dob=context.woman_dob,
            constant_monthly_roi=fixed_monthly_roi,
            constant_monthly_cpi=fixed_monthly_cpi,
        )
        edge_scenario = build_centerpoint_scenario()
        case_name = f"EC_CENTERPOINT_{format_apy_suffix(roi_apy)}_{format_apy_suffix(cpi_apy)}"
        model, result = evaluate_lhs_scenario(scenario=edge_scenario, context=edge_context)
        row = asdict(
            summarize_lhs_run(
                run_id=case_name,
                scenario=edge_scenario,
                model=model,
                result=result,
                context=edge_context,
            )
        )
        ordered_row = {column: row[column] for column in CSV_COLUMNS}
        print_screen_row(row=ordered_row, columns=CSV_COLUMNS, widths=column_widths)
        rows.append(ordered_row)

    # Process Gutz replay cases once, outside the fixed edge-point loop.
    replay_cases = build_replay_case_scenarios_gutz()
    for case_name, scenario, replay_constant_roi, replay_constant_cpi in replay_cases:
        replay_context = ScenarioRunContext(
            ticker=context.ticker,
            current_date=context.current_date,
            history_years=context.history_years,
            al_cum_running_avg_yrs=context.al_cum_running_avg_yrs,
            start_clock=context.start_clock,
            man_dob=context.man_dob,
            woman_dob=context.woman_dob,
            constant_monthly_roi=normalize_centerpoint_constant_monthly(replay_constant_roi),
            constant_monthly_cpi=normalize_centerpoint_constant_monthly(replay_constant_cpi),
        )
        model, result = evaluate_lhs_scenario(scenario=scenario, context=replay_context)
        row = asdict(
            summarize_lhs_run(
                run_id=case_name,
                scenario=scenario,
                model=model,
                result=result,
                context=replay_context,
            )
        )
        ordered_row = {column: row[column] for column in CSV_COLUMNS}
        print_screen_row(row=ordered_row, columns=CSV_COLUMNS, widths=column_widths)
        rows.append(ordered_row)

    # Process the explicit centerpoint scenario once and append as CENTERPOINT.
    centerpoint_context = context
    if CENTERPOINT_USE_CONSTANT_RATES:
        centerpoint_context = ScenarioRunContext(
            ticker=context.ticker,
            current_date=context.current_date,
            history_years=context.history_years,
            al_cum_running_avg_yrs=context.al_cum_running_avg_yrs,
            start_clock=context.start_clock,
            man_dob=context.man_dob,
            woman_dob=context.woman_dob,
            constant_monthly_roi=normalize_centerpoint_constant_monthly(CENTERPOINT_CONSTANT_MONTHLY_ROI),
            constant_monthly_cpi=normalize_centerpoint_constant_monthly(CENTERPOINT_CONSTANT_MONTHLY_CPI),
        )
    centerpoint_scenario = build_centerpoint_scenario()
    model, result = evaluate_lhs_scenario(scenario=centerpoint_scenario, context=centerpoint_context)
    centerpoint_row = asdict(
        summarize_lhs_run(
            run_id="CENTERPOINT",
            scenario=centerpoint_scenario,
            model=model,
            result=result,
            context=centerpoint_context,
        )
    )
    ordered_centerpoint_row = {column: centerpoint_row[column] for column in CSV_COLUMNS}
    print_screen_row(row=ordered_centerpoint_row, columns=CSV_COLUMNS, widths=column_widths)
    rows.append(ordered_centerpoint_row)

    frame = pd.DataFrame(rows, columns=CSV_COLUMNS)
    frame.to_csv(output_path, index=False)
    return frame


def plot_edge_case_subplots(
    results: pd.DataFrame,
    roi_apy_percents: list[float],
    cpi_apy_percents: list[float],
    shared_y_scale: bool = True,
    show: bool = True,
) -> None:
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    # Only use generated edge-case rows for Figures 2/3 (exclude replay string run_ids).
    edge_results = results[
        results["run_id"].apply(lambda v: isinstance(v, str) and v.startswith("EC_"))
    ]
    if edge_results.empty:
        return

    n_roi = len(roi_apy_percents)
    n_cpi = len(cpi_apy_percents)
    if n_roi == 0 or n_cpi == 0:
        return
    figure, axes = plt.subplots(
        n_roi,
        n_cpi,
        figsize=(6 * n_cpi, 5 * n_roi),
        squeeze=False,
        constrained_layout=True,
    )
    suffix = "(Shared Y-Scale)" if shared_y_scale else "(Free Axis Scale)"
    figure.suptitle(
        f"{PLOT_MAIN_TITLE}\nEdge Cases: Added Worth (normalized) {suffix}",
        fontsize=14,
    )

    cmap = LinearSegmentedColormap.from_list("bright_rg", ["#ff0000", "#00ff00"])
    independent_all = edge_results["man_independent_yrs"] + edge_results["woman_independent_yrs"]
    norm = Normalize(vmin=float(independent_all.min()), vmax=float(independent_all.max()))

    y_min = 0.0
    y_max = 0.0
    if shared_y_scale:
        # Use the full results range so Figure 2's y-scale matches Figure 1.
        y_all = results["added_lc_worth_norm"].to_numpy(dtype=float)
        y_min = float(np.nanmin(y_all))
        y_max = float(np.nanmax(y_all))
        if math.isclose(y_min, y_max):
            pad = abs(y_min) * 0.01 + 1e-6
            y_min -= pad
            y_max += pad

    mappable = None
    for row_idx, roi_apy in enumerate(roi_apy_percents):
        for col_idx, cpi_apy in enumerate(cpi_apy_percents):
            ax = axes[row_idx, col_idx]
            ec_suffix = f"_{format_apy_suffix(roi_apy)}_{format_apy_suffix(cpi_apy)}"
            combo_rows = edge_results[
                edge_results["run_id"].apply(lambda v, s=ec_suffix: isinstance(v, str) and v.endswith(s))
            ]
            if not combo_rows.empty:
                assisted_total = combo_rows["man_assisted_yrs"] + combo_rows["woman_assisted_yrs"]
                independent_total = combo_rows["man_independent_yrs"] + combo_rows["woman_independent_yrs"]
                mappable = ax.scatter(
                    assisted_total,
                    combo_rows["added_lc_worth_norm"],
                    c=independent_total,
                    cmap=cmap,
                    norm=norm,
                    marker="o",
                    alpha=0.85,
                    s=50,
                )
            ax.set_xlabel("Sum of Assisted Living Years: yrs_sum_al (Years)")
            ax.set_ylabel("Added Worth (normalized to 2026 dollars)")
            ax.set_title(f"ROI={roi_apy:.3g}%  CPI={cpi_apy:.3g}%")
            if shared_y_scale:
                ax.set_ylim(y_min, y_max)
            ax.grid(True, alpha=0.3)
            add_lifecare_reference_line(ax)

    if mappable is not None:
        figure.colorbar(mappable, ax=axes.ravel().tolist(), label="combined independent years")

    if show:
        plt.show()


def plot_lhs_summary(
    results: pd.DataFrame,
    include_edge_cases: bool = True,
    roi_apy_percents: list[float] | None = None,
    cpi_apy_percents: list[float] | None = None,
    show: bool = True,
) -> None:
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    centerpoint_rows = results[results["run_id"].apply(lambda v: str(v) == "CENTERPOINT")]
    lhs_rows = results[results["run_id"].apply(lambda v: not isinstance(v, str))]
    edge_rows = results[
        results["run_id"].apply(lambda v: isinstance(v, str) and str(v) != "CENTERPOINT")
    ]

    # Three subplots (3×1): each uses a different x-axis variable.
    x_configs = [
        ("yrs_sum_al",    "Sum of Assisted Living Years: yrs_sum_al (Years)",       "Added Worth (normalized) vs Sum of AL Years"),
        ("yrs_il_single", "yrs_il_single (years exactly one in IL)", "Added Worth (normalized) vs Years Single in IL"),
        ("yrs_il_double", "yrs_il_double (years both in IL)",        "Added Worth (normalized) vs Years Both in IL"),
    ]

    figure, axes = plt.subplots(3, 1, figsize=(12, 18), constrained_layout=True)
    figure.suptitle(
        f"{PLOT_MAIN_TITLE}\nAdded Worth (normalized) vs Life Structure Parameters",
        fontsize=14,
    )

    cmap = LinearSegmentedColormap.from_list("bright_rg", ["#ff0000", "#00ff00"])
    norm = None
    mappable = None
    if include_edge_cases and not edge_rows.empty:
        independent_total_all = edge_rows["man_independent_yrs"] + edge_rows["woman_independent_yrs"]
        norm = Normalize(vmin=float(independent_total_all.min()), vmax=float(independent_total_all.max()))

    for ax, (x_col, x_label, title) in zip(axes, x_configs):
        if not lhs_rows.empty:
            lhs_x = lhs_rows[x_col].to_numpy(dtype=float)
            ax.scatter(lhs_x, lhs_rows["added_lc_worth_norm"], alpha=0.25, color="lightgray", marker="o", label="LHS")

        if include_edge_cases and not edge_rows.empty and norm is not None:
            independent_total = edge_rows["man_independent_yrs"] + edge_rows["woman_independent_yrs"]
            x_vals = edge_rows[x_col].to_numpy(dtype=float)
            mappable = ax.scatter(
                x_vals,
                edge_rows["added_lc_worth_norm"],
                c=independent_total,
                cmap=cmap,
                norm=norm,
                marker="o",
                alpha=0.9,
                s=55,
                label="edge cases",
            )

        if not centerpoint_rows.empty:
            cx = centerpoint_rows[x_col].to_numpy(dtype=float)
            cy = centerpoint_rows["added_lc_worth_norm"].to_numpy(dtype=float)
            ax.scatter(
                cx,
                cy,
                color="blue",
                marker="*",
                s=300,
                edgecolors="black",
                linewidths=0.8,
                zorder=5,
                label="CENTERPOINT",
            )

        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles=handles, loc="best", fontsize=9)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Added Worth (normalized to 2026 dollars)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        add_lifecare_reference_line(ax)

    if mappable is not None:
        figure.colorbar(mappable, ax=axes.tolist(), label="combined independent years")

    if show:
        plt.show()


def _format_gutz_figure1_annotation(row: pd.Series) -> str:
    life_params = (
        f"{float(row['man_independent_yrs']):.2f}/"
        f"{float(row['woman_independent_yrs']):.2f}/"
        f"{float(row['man_assisted_yrs']):.2f}/"
        f"{float(row['woman_assisted_yrs']):.2f}"
    )
    apy_params = f"{float(row['apy_roi']):.2f}%/{float(row['apy_cpi']):.2f}%"
    worth_norm_lc = f"worth_norm_lc=${float(row['worth_norm_lc']):,.0f}"
    worth_norm_cc = f"worth_norm_cc=${float(row['worth_norm_cc']):,.0f}"
    return f"{life_params}\n{apy_params}\n{worth_norm_lc}\n{worth_norm_cc}"


def _lc_norm_total(df: pd.DataFrame) -> pd.Series:
    """Return normalized total LC expenses, always >= entrance_fee_lc.
    Works with both new CSVs (exp_norm_total_lc column) and old CSVs (fallback)."""
    if "exp_norm_total_lc" in df.columns:
        return df["exp_norm_total_lc"]
    fee = df["entrance_fee_lc"] if "entrance_fee_lc" in df.columns else pd.Series(0.0, index=df.index)
    return df["exp_norm_lc"] + fee


def plot_worth_vs_earn(results: pd.DataFrame, show: bool = True) -> plt.Figure:
    lhs = results[pd.to_numeric(results["run_id"], errors="coerce").notna()]
    cp = results[results["run_id"].apply(lambda v: str(v) == "CENTERPOINT")]

    figure, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(28, 6), constrained_layout=True)
    figure.suptitle("Worth vs Earnings (normalized)", fontsize=14)

    # Subplot 1: worth_norm vs earn_norm
    ax1.scatter(lhs["earn_norm_lc"], lhs["worth_norm_lc"],
                marker="o", s=18, alpha=0.6, label="LC")
    ax1.scatter(lhs["earn_norm_cc"], lhs["worth_norm_cc"],
                marker="x", s=18, alpha=0.6, label="CC")

    # Subplot 2: earn_norm vs total_living_yrs
    ax2.scatter(lhs["total_living_yrs"], lhs["earn_norm_lc"],
                marker="o", s=18, alpha=0.6, label="LC")
    ax2.scatter(lhs["total_living_yrs"], lhs["earn_norm_cc"],
                marker="x", s=18, alpha=0.6, label="CC")

    # Subplot 3: worth_norm vs earning_potential (entrance-fee-adjusted principal)
    ep_lc_col = "earning_potential"
    ep_cc_col = "earning_potential_cc" if "earning_potential_cc" in lhs.columns else "earning_potential"
    ax3.scatter(lhs[ep_lc_col], lhs["worth_norm_lc"],
                marker="o", s=18, alpha=0.6, label="LC")
    ax3.scatter(lhs[ep_cc_col], lhs["worth_norm_cc"],
                marker="x", s=18, alpha=0.6, label="CC")

    # Subplot 4: normalized total expenses (entrance fee in present-value dollars) vs total_living_yrs
    ax4.scatter(lhs["total_living_yrs"], _lc_norm_total(lhs),
                marker="o", s=18, alpha=0.6, label="LC total")
    ax4.scatter(lhs["total_living_yrs"], lhs["exp_norm_total_cc"],
                marker="x", s=18, alpha=0.6, label="CC total")

    if not cp.empty:
        row = cp.iloc[0]
        earn_lc  = float(row["earn_norm_lc"])
        worth_lc = float(row["worth_norm_lc"])
        earn_cc  = float(row["earn_norm_cc"])
        worth_cc = float(row["worth_norm_cc"])
        cp_total_living = float(row["total_living_yrs"])
        cp_ep_lc = float(row["earning_potential"])
        cp_ep_cc = float(row["earning_potential_cc"]) if "earning_potential_cc" in results.columns else cp_ep_lc
        cp_exp_lc = float(_lc_norm_total(cp).iloc[0])
        cp_exp_cc = float(row["exp_norm_total_cc"])

        ax1.scatter([earn_lc], [worth_lc], marker="*", s=260, alpha=0.4,
                    color="green", edgecolors="green", zorder=5, label="CP LC")
        ax1.scatter([earn_cc], [worth_cc], marker="*", s=260, alpha=0.4,
                    color="red", edgecolors="red", zorder=5, label="CP CC")

        ax2.scatter([cp_total_living], [earn_lc], marker="*", s=260, alpha=0.4,
                    color="green", edgecolors="green", zorder=5, label="CP LC")
        ax2.scatter([cp_total_living], [earn_cc], marker="*", s=260, alpha=0.4,
                    color="red", edgecolors="red", zorder=5, label="CP CC")

        ax3.scatter([cp_ep_lc], [worth_lc], marker="*", s=260, alpha=0.4,
                    color="green", edgecolors="green", zorder=5, label="CP LC")
        ax3.scatter([cp_ep_cc], [worth_cc], marker="*", s=260, alpha=0.4,
                    color="red", edgecolors="red", zorder=5, label="CP CC")

        ax4.scatter([cp_total_living], [cp_exp_lc], marker="*", s=260, alpha=0.4,
                    color="green", edgecolors="green", zorder=5, label="CP LC")
        ax4.scatter([cp_total_living], [cp_exp_cc], marker="*", s=260, alpha=0.4,
                    color="red", edgecolors="red", zorder=5, label="CP CC")

    ax1.set_xlabel("Normalized Earnings")
    ax1.set_ylabel("Normalized Worth")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("total_living_yrs (years)")
    ax2.set_ylabel("Normalized Earnings")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3.set_xlabel("earning_potential")
    ax3.set_ylabel("Normalized Worth")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    ax4.set_xlabel("total_living_yrs (years)")
    ax4.set_ylabel("Normalized Expenses")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    if show:
        plt.show()
    return figure


def plot_demographic_stats(results: pd.DataFrame, show: bool = True) -> plt.Figure:
    MAN_COLOR = "lightblue"
    WOMAN_COLOR = "fuchsia"

    results = results[pd.to_numeric(results["run_id"], errors="coerce").notna()]
    man_al = results[results["man_goes_to_al"] == True]
    woman_al = results[results["woman_goes_to_al"] == True]
    man_age_to_al = pd.to_numeric(man_al["man_age_to_al"], errors="coerce").dropna()
    woman_age_to_al = pd.to_numeric(woman_al["woman_age_to_al"], errors="coerce").dropna()

    figure, axes = plt.subplot_mosaic(
        [["death_cdf", "al_cdf"], ["man_al", "woman_al"], ["pdf", "pdf"], ["age_start", "age_start"]],
        figsize=(13, 18), constrained_layout=True,
    )
    figure.suptitle("Demographic Stats", fontsize=14)

    def _plot_cdf(ax, data, color, label):
        sorted_data = np.sort(data)
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        ax.plot(sorted_data, cdf, color=color, linewidth=2, label=label)
        ax.fill_between(sorted_data, cdf, alpha=0.2, color=color)

    # age at death CDF, both genders
    _plot_cdf(axes["death_cdf"], results["man_age_at_death"].values, MAN_COLOR, "Man")
    _plot_cdf(axes["death_cdf"], results["woman_age_at_death"].values, WOMAN_COLOR, "Woman")
    axes["death_cdf"].set_title("Age at death")
    axes["death_cdf"].set_xlabel("Age")
    axes["death_cdf"].set_ylabel("Cumulative probability")
    axes["death_cdf"].set_ylim(0, 1)
    axes["death_cdf"].legend()

    # age to AL CDF, both genders
    if len(man_age_to_al):
        _plot_cdf(axes["al_cdf"], man_age_to_al.values, MAN_COLOR, f"Man (n={len(man_age_to_al)})")
    if len(woman_age_to_al):
        _plot_cdf(axes["al_cdf"], woman_age_to_al.values, WOMAN_COLOR, f"Woman (n={len(woman_age_to_al)})")
    axes["al_cdf"].set_title(f"Age to AL  (of {len(results)} runs)")
    axes["al_cdf"].set_xlabel("Age")
    axes["al_cdf"].set_ylabel("Cumulative probability")
    axes["al_cdf"].set_ylim(0, 1)
    axes["al_cdf"].legend()

    # man time in AL dot plot
    axes["man_al"].scatter(
        range(len(man_al)), man_al["man_assisted_yrs"],
        marker="x", color=MAN_COLOR, linewidths=1.2, s=40,
    )
    axes["man_al"].set_title("Man time in AL")
    axes["man_al"].set_xlabel("Run index")
    axes["man_al"].set_ylabel("Years")

    # woman time in AL dot plot
    axes["woman_al"].scatter(
        range(len(woman_al)), woman_al["woman_assisted_yrs"],
        marker="x", color=WOMAN_COLOR, linewidths=1.2, s=40,
    )
    axes["woman_al"].set_title("Woman time in AL")
    axes["woman_al"].set_xlabel("Run index")
    axes["woman_al"].set_ylabel("Years")

    # age at death PDF, both genders (full-width)
    axes["pdf"].hist(results["man_age_at_death"], bins=20, density=True,
                     color=MAN_COLOR, edgecolor="green", alpha=0.6, label="Man")
    axes["pdf"].hist(results["woman_age_at_death"], bins=20, density=True,
                     color=WOMAN_COLOR, edgecolor="deeppink", alpha=0.6, label="Woman")
    axes["pdf"].set_title("Age at Death — Probability Density")
    axes["pdf"].set_xlabel("Age")
    axes["pdf"].set_ylabel("Density")
    axes["pdf"].legend()

    # age at start dot plot, both genders (full-width)
    axes["age_start"].scatter(
        range(len(results)), results["man_age_at_start"],
        marker="x", color=MAN_COLOR, linewidths=1.2, s=40, label="Man",
    )
    axes["age_start"].scatter(
        range(len(results)), results["woman_age_at_start"],
        marker="x", color=WOMAN_COLOR, linewidths=1.2, s=40, label="Woman",
    )
    axes["age_start"].set_title("Age at Start")
    axes["age_start"].set_xlabel("Run index")
    axes["age_start"].set_ylabel("Age")
    axes["age_start"].legend()

    if show:
        plt.show()
    return figure


def plot_gutz_lhs_figure1(results: pd.DataFrame, show: bool = True) -> tuple[plt.Figure, plt.Axes]:
    figure, axis, _ = plot_lhs_figure1(
        results,
        main_title=PLOT_MAIN_TITLE,
        subtitle=(
            "Added Worth (normalized) vs yrs_sum_al (Gutz Centerpoint LHS)\n"
            "Params: man_IL/woman_IL/man_AL/woman_AL/roi_apy/cpi_apy"
        ),
        add_reference_line=add_lifecare_reference_line,
        annotation_formatter=_format_gutz_figure1_annotation,
        subtitle_y=1.08,
        color_mode="worth_override",
        annotate_centerpoint=True,
        show=show,
    )
    return figure, axis


def plot_gutz_lhs_worth_subplots(results: pd.DataFrame, show: bool = True) -> tuple[plt.Figure, np.ndarray]:
    return plot_lhs_figure2_worth_subplots(
        results,
        main_title=PLOT_MAIN_TITLE,
        add_reference_line=None,
        annotation_formatter=_format_gutz_figure1_annotation,
        show=show,
    )


def _select_nearest_ep_lhs(results: pd.DataFrame, n: int = 100) -> pd.DataFrame:
    """Return the n LHS rows nearest to the centerpoint earning_potential (symmetric
    above/below) plus the centerpoint row itself.  If one side has fewer than n//2
    points, the count is clamped symmetrically (e.g. 45 below and 45 above)."""
    centerpoint_rows = results[results["run_id"].apply(lambda v: str(v) == "CENTERPOINT")]
    lhs_rows = results[results["run_id"].apply(lambda v: not isinstance(v, str))].copy()

    if centerpoint_rows.empty or lhs_rows.empty:
        return results

    cp_ep = float(centerpoint_rows.iloc[0]["earning_potential"])
    ep = lhs_rows["earning_potential"].astype(float)

    below = lhs_rows[ep < cp_ep].sort_values("earning_potential", ascending=False)
    above = lhs_rows[ep >= cp_ep].sort_values("earning_potential", ascending=True)

    n_sym = min(n // 2, len(below), len(above))
    selected = pd.concat([below.head(n_sym), above.head(n_sym), centerpoint_rows])
    return selected.reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LHS Monte Carlo anchored to Gutz case centerpoint from Run_one_Taylor.py"
    )
    parser.add_argument("--ticker", default=TICKER, help="Ticker symbol to download, default: SPY")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"RNG seed, default: {DEFAULT_SEED}")
    parser.add_argument(
        "--current-date",
        default="2026-03-29",
        help=f"Historical data cutoff date in YYYY-MM-DD, default: 2026-03-29",
    )
    parser.add_argument(
        "--lhs-points",
        type=int,
        default=DEFAULT_LHS_POINTS,
        help=f"Run a Latin hypercube sample with this many points. Default: {DEFAULT_LHS_POINTS}",
    )
    parser.add_argument(
        "--lhs-output",
        default="lhs_gutz_taylor_results.csv",
        help="CSV output path for LHS runs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    current_date = pd.Timestamp(args.current_date).normalize()
    context = ScenarioRunContext(
        ticker=args.ticker,
        current_date=current_date,
        history_years=HISTORY_YEARS,
        al_cum_running_avg_yrs=AL_ESC_RUNNING_AVG_YRS,
        start_clock=START_CLOCK,
        man_dob=MAN_DOB,
        woman_dob=WOMAN_DOB,
        # Keep sampled LHS rows stochastic by default; CENTERPOINT row constants
        # are controlled separately via CENTERPOINT_USE_CONSTANT_RATES.
        constant_monthly_roi=None,
        constant_monthly_cpi=None,
    )
    if args.lhs_points > 0:
        output_path = Path(args.lhs_output)
        results = run_lhs_driver(
            num_points=args.lhs_points,
            context=context,
            output_path=output_path,
            seed=args.seed,
        )
        print(
            f"LHS runs completed: {len(results)}\n"
            f"Output CSV: {output_path}\n"
            f"Worth LC range: {results['worth_norm_lc'].min():,.0f} to {results['worth_norm_lc'].max():,.0f}\n"
            f"Worth CC range: {results['worth_norm_cc'].min():,.0f} to {results['worth_norm_cc'].max():,.0f}"
        )
        
        plot_gutz_lhs_figure1(results, show=False)
        plot_gutz_lhs_worth_subplots(results, show=False)
        plot_worth_vs_earn(results, show=False)

        # Figures 4-6: figures 1-3 filtered to the 100 LHS points
        # with earning_potential nearest to the centerpoint (symmetric above/below).
        nearest_results = _select_nearest_ep_lhs(results, n=100)
        plot_gutz_lhs_figure1(nearest_results, show=False)
        plot_gutz_lhs_worth_subplots(nearest_results, show=False)
        plot_worth_vs_earn(nearest_results, show=False)

        plot_demographic_stats(results, show=False)

        plot_lhs_summary(
            results,
            include_edge_cases=PLOT_EDGE_CASES_IN_LHS_PLOT,
            roi_apy_percents=EDGE_CASE_ROI_APY_PERCENTS,
            cpi_apy_percents=EDGE_CASE_CPI_APY_PERCENTS,
            show=False,
        )
        plot_edge_case_subplots(
            results,
            EDGE_CASE_ROI_APY_PERCENTS,
            EDGE_CASE_CPI_APY_PERCENTS,
            shared_y_scale=True,
            show=False,
        )
        plot_edge_case_subplots(
            results,
            EDGE_CASE_ROI_APY_PERCENTS,
            EDGE_CASE_CPI_APY_PERCENTS,
            shared_y_scale=False,
            show=False,
        )

        plt.show()
        return


if __name__ == "__main__":
    main()

