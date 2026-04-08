"""
Run_LHS_Gutz_Taylor.py

Latin Hypercube Sampling (LHS) Monte Carlo analysis centered around the Gutz case inputs
from Run_one_Taylor.py local_run_overrides. Includes edge cases, replay cases, and all
plotting features.

Output: lhs_gutz+5_taylor_results.csv

The centerpoint scenario and ranges are defined below; modify them to explore different
regions of the scenario space.
"""


import argparse
from Run_LHS_Taylor import (
    add_lifecare_reference_line,
    print_screen_row,
    summarize_lhs_run,
    _select_nearest_ep_lhs,
    CSV_COLUMNS,
    SCREEN_MIN_COL_WIDTH,
    build_lhs_scenarios,
    plot_edge_case_subplots,
    plot_lhs_summary,
    plot_worth_vs_earn,
    plot_demographic_stats,
)
from dataclasses import asdict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from Center_LHS_Gutz_Taylor import (
    CENTERPOINT_CONSTANT_MONTHLY_CPI,
    CENTERPOINT_CONSTANT_MONTHLY_ROI,
    CENTERPOINT_INFLATION_SEED,
    CENTERPOINT_MAN_ASSISTED_YRS,
    CENTERPOINT_MAN_GOES_TO_AL,
    CENTERPOINT_MAN_GOES_TO_AL_SEED,
    CENTERPOINT_ROI_SEED,
    CENTERPOINT_USE_CONSTANT_RATES,
    CENTERPOINT_WOMAN_ASSISTED_YRS,
    CENTERPOINT_WOMAN_GOES_TO_AL,
    CENTERPOINT_WOMAN_GOES_TO_AL_SEED,
)
from default_case import (
    AL_ESC_RUNNING_AVG_YRS,
    DEFAULT_SEED,
    DEFAULT_CURRENT_DATE,
    HISTORY_YEARS,
    P_MAN_AL,
    P_WOMAN_AL,
    START_CLOCK,
    apy_percent_to_monthly_fraction,
)
from edges import build_replay_case_scenarios_gutz, format_apy_suffix
from lhs_plotting import plot_lhs_figure1, plot_lhs_figure2_worth_subplots
from Taylor import LhsScenario, ScenarioRunContext
from utils import evaluate_lhs_scenario
from Roi import TICKER

# User inputs
#  To force the probability both man and woman go to AL instead of dying right away
force_al = False
plotting = True
LIFE_PARAM_VARIATION = 0.5 # For life parameters, use ±50% range around centerpoint (0.5)
DEFAULT_LHS_POINTS = 1000
ROI_MEAN_SHIFT_RANGE = (-0.003, 0.003)
ROI_VOL_MULTIPLIER_RANGE = (0.8, 1.2)
ROI_MEAN_REVERSION_RANGE = (0.1, 0.3)
MAN_DOB = "1952-07-26"
WOMAN_DOB = "1951-04-11"
# MAN_INDEPENDENT_YRS_RANGE = (74.0 - age(START_CLOCK, MAN_DOB), 90.0 - age(START_CLOCK, MAN_DOB))
# WOMAN_INDEPENDENT_YRS_RANGE = (75.0 - age(START_CLOCK, WOMAN_DOB), 90.0 - age(START_CLOCK, WOMAN_DOB))

CENTERPOINT_MAN_INDEPENDENT_YRS = 5.
CENTERPOINT_WOMAN_INDEPENDENT_YRS = 4.4

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

# Seed and model parameter ranges (same as Run_LHS_Taylor with scalar)
SEED_RANGE = (0, 1000000)
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


def normalize_centerpoint_constant_monthly(value: float | None) -> float | None:
    """Convert APY percent to monthly fraction. Always treats input as APY percent."""
    if value is None:
        return None
    return apy_percent_to_monthly_fraction(float(value))


def run_lhs_driver(num_points: int, context: ScenarioRunContext, output_path: Path, seed: int) -> pd.DataFrame:
    if context.constant_monthly_roi is not None or context.constant_monthly_cpi is not None:
        print(
            "Using fixed monthly ROI/CPI from default_case.py; "
            "apy_roi and apy_cpi reflect effective APY from final growth of $1 "
            "under those configured constants."
        )
    scenarios = build_lhs_scenarios(
        num_points=num_points,
        seed=seed,
        man_independent_yrs_range=MAN_INDEPENDENT_YRS_RANGE,
        woman_independent_yrs_range=WOMAN_INDEPENDENT_YRS_RANGE,
        man_assisted_yrs_range=MAN_ASSISTED_YRS_RANGE,
        woman_assisted_yrs_range=WOMAN_ASSISTED_YRS_RANGE,
        roi_mean_shift_range=ROI_MEAN_SHIFT_RANGE,
        roi_vol_multiplier_range=ROI_VOL_MULTIPLIER_RANGE,
        roi_mean_reversion_range=ROI_MEAN_REVERSION_RANGE,
        inflation_mean_shift_range=INFLATION_MEAN_SHIFT_RANGE,
        inflation_vol_multiplier_range=INFLATION_VOL_MULTIPLIER_RANGE,
        inflation_mean_reversion_range=INFLATION_MEAN_REVERSION_RANGE,
        p_man_al=P_MAN_AL,
        p_woman_al=P_WOMAN_AL,
        lhs_man_goes_to_al=LHS_MAN_GOES_TO_AL,
        lhs_woman_goes_to_al=LHS_WOMAN_GOES_TO_AL,
    )
    rows = []
    column_widths = {column: max(len(column), SCREEN_MIN_COL_WIDTH) for column in CSV_COLUMNS}
    print(" ".join(column.rjust(column_widths[column]) for column in CSV_COLUMNS))
    
    # Process random LHS scenarios
    for run_id, scenario in enumerate(scenarios, start=1):
        model, result = evaluate_lhs_scenario(scenario=scenario, context=context)
        row = asdict(summarize_lhs_run(run_id=run_id, scenario=scenario, model=model, context=context))
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
        row = asdict(summarize_lhs_run(run_id=case_name, scenario=edge_scenario, model=model, context=edge_context))
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
        row = asdict(summarize_lhs_run(run_id=case_name, scenario=scenario, model=model, context=replay_context))
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
    centerpoint_row = asdict(summarize_lhs_run(run_id="CENTERPOINT", scenario=centerpoint_scenario, model=model,
                                               context=centerpoint_context))
    ordered_centerpoint_row = {column: centerpoint_row[column] for column in CSV_COLUMNS}
    print_screen_row(row=ordered_centerpoint_row, columns=CSV_COLUMNS, widths=column_widths)
    rows.append(ordered_centerpoint_row)

    frame = pd.DataFrame(rows, columns=CSV_COLUMNS)
    frame.to_csv(output_path, index=False)
    return frame


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


def parse_args(
    description: str = "Monte Carlo monthly ROI projection anchored to historical long-run growth.",
    default_ticker: str = TICKER,
    default_seed: int = DEFAULT_SEED,
    default_current_date: str = DEFAULT_CURRENT_DATE,
    default_lhs_points: int = DEFAULT_LHS_POINTS,
    default_lhs_output: str = "lhs_gutz+5_taylor_results.csv",
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--ticker", default=default_ticker, help=f"Ticker symbol to download, default: {default_ticker}")
    parser.add_argument("--seed", type=int, default=default_seed, help=f"RNG seed, default: {default_seed}")
    parser.add_argument(
        "--current-date",
        default=default_current_date,
        help=f"Historical data cutoff date in YYYY-MM-DD, default: {default_current_date}",
    )
    parser.add_argument(
        "--lhs-points",
        type=int,
        default=default_lhs_points,
        help=f"Run a Latin hypercube sample with this many points. Default: {default_lhs_points}",
    )
    parser.add_argument(
        "--lhs-output",
        default=default_lhs_output,
        help=f"CSV output path for LHS runs. Default: {default_lhs_output}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args(
        description="LHS Monte Carlo anchored to Gutz case centerpoint from Run_one_Taylor.py",
        default_current_date="2026-03-29",
        default_lhs_output="lhs_gutz_taylor_results.csv",
    )
    current_date = pd.Timestamp(args.current_date)
    assert isinstance(current_date, pd.Timestamp), f"Invalid current_date: {args.current_date}"
    current_date = current_date.normalize()
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

        if plotting:
            plot_gutz_lhs_figure1(results, show=False)
            plot_gutz_lhs_worth_subplots(results, show=False)
            plot_worth_vs_earn(results, show=False, main_title="Gutz Worth vs Earnings (normalized)")

            # Figures 4-6: figures 1-3 filtered to the 100 LHS points
            # with earning_potential nearest to the centerpoint (symmetric above/below).
            nearest_results = _select_nearest_ep_lhs(results, n=100)
            plot_gutz_lhs_figure1(nearest_results, show=False)
            plot_gutz_lhs_worth_subplots(nearest_results, show=False)
            plot_worth_vs_earn(nearest_results, show=False, main_title="Gutz Worth vs Earnings (normalized) - Nearest EP")

            plot_demographic_stats(results, show=False, main_title="Gutz Demographic Stats")

            plot_lhs_summary(
                results,
                include_edge_cases=PLOT_EDGE_CASES_IN_LHS_PLOT,
                show=False,
                main_title=PLOT_MAIN_TITLE,
            )
            plot_edge_case_subplots(
                results,
                EDGE_CASE_ROI_APY_PERCENTS,
                EDGE_CASE_CPI_APY_PERCENTS,
                shared_y_scale=True,
                show=False,
                main_title=PLOT_MAIN_TITLE,
            )
            plot_edge_case_subplots(
                results,
                EDGE_CASE_ROI_APY_PERCENTS,
                EDGE_CASE_CPI_APY_PERCENTS,
                shared_y_scale=False,
                show=False,
                main_title=PLOT_MAIN_TITLE,
            )

            plt.show()
        return


if __name__ == "__main__":
    main()

