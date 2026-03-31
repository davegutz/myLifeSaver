"""
Run_LHS_Gutz_Taylor.py

Latin Hypercube Sampling (LHS) Monte Carlo analysis centered around the Gutz case inputs
from Run_one_Taylor.py local_run_overrides. Includes edge cases, replay cases, and all
plotting features.

Output: lhs_gutz_taylor_results.csv

The centerpoint scenario and ranges are defined below; modify them to explore different
regions of the scenario space.
"""

import argparse
from dataclasses import asdict
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import cast
from default_case import (
    AL_ESC_RUNNING_AVG_YRS,
    CONSTANT_MONTHLY_CPI,
    CONSTANT_MONTHLY_ROI,
    DEFAULT_CURRENT_DATE,
    DEFAULT_SEED,
    HISTORY_YEARS,
    MAN_DOB,
    START_CLOCK,
    WOMAN_DOB,
    apy_percent_to_monthly_fraction,
)
from edges import build_edge_case_scenarios, build_replay_case_scenarios, format_apy_suffix, build_custom_edge_cases_gutz, get_edge_cases_gutz, CUSTOM_EDGE_CASES_GUTZ
from Inflation import plot_inflation_views
from Roi import TICKER, plot_projection_views
from Taylor import LhsScenario, LhsScenarioSummary, ScenarioRunContext, TaylorLife, TaylorLifeResult
from utils import age, evaluate_lhs_scenario, plot_taylor_life_exp_non_taylor

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
# CENTERPOINT SCENARIO (from Run_one_Taylor.py local_run_overrides)
# ============================================================================
CENTERPOINT_MAN_INDEPENDENT_YRS = 10.0
CENTERPOINT_WOMAN_INDEPENDENT_YRS = 15.5
CENTERPOINT_MAN_ASSISTED_YRS = 2.35
CENTERPOINT_WOMAN_ASSISTED_YRS = 5.5
CENTERPOINT_ROI_SEED = 740264
CENTERPOINT_INFLATION_SEED = 898910

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

DEFAULT_LHS_POINTS = 1000
PLOT_EDGE_CASES_IN_LHS_PLOT = True
EDGE_CASE_ROI_APY_PERCENTS = [6.0, 12.0]
EDGE_CASE_CPI_APY_PERCENTS = [0.0, 2.0, 6.0]
PLOT_MAIN_TITLE = "Taylor Community Lifecare / Continuing Care Decision,  2026 for Katherine and David Gutz"


CSV_COLUMNS = [
    "run_id",
    "yrs_il_single",
    "yrs_il_double",
    "yrs_sum_al",
    "added_lc_worth_norm",
    "man_independent_yrs",
    "woman_independent_yrs",
    "man_assisted_yrs",
    "woman_assisted_yrs",
    "roi_seed",
    "inflation_seed",
    "apy_roi",
    "apy_cpi",
    "roi_mean_shift",
    "roi_vol_multiplier",
    "roi_mean_reversion",
    "inflation_mean_shift",
    "inflation_vol_multiplier",
    "inflation_mean_reversion",
    "exp_al_cc",
    "exp_norm_al_cc",
    "exp_cc",
    "exp_norm_cc",
    "exp_lc",
    "exp_norm_lc",
    "exp_non_taylor",
    "exp_norm_non_taylor",
    "exp_total_cc",
    "exp_norm_total_cc",
    "earn_cc",
    "earn_norm_cc",
    "earn_lc",
    "earn_norm_lc",
    "worth_lc",
    "worth_norm_lc",
    "worth_cc",
    "worth_norm_cc",
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
    sampled = sample_lhs_points(num_points, dimensions=12, seed=seed)
    scenarios: list[LhsScenario] = []
    for idx in range(num_points):
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
            ),
        )
        scenarios.append(scenario)
    return scenarios


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
        roi_mean_shift=scenario.roi_mean_shift,
        roi_vol_multiplier=scenario.roi_vol_multiplier,
        roi_mean_reversion=scenario.roi_mean_reversion,
        inflation_mean_shift=scenario.inflation_mean_shift,
        inflation_vol_multiplier=scenario.inflation_vol_multiplier,
        inflation_mean_reversion=scenario.inflation_mean_reversion,
        exp_al_cc=last_value(model.exp_al_cc_history),
        exp_norm_al_cc=last_value(model.exp_norm_al_cc),
        exp_cc=last_value(model.exp_cc_history),
        exp_norm_cc=last_value(model.exp_norm_cc),
        exp_lc=last_value(model.exp_lc_history),
        exp_norm_lc=last_value(model.exp_norm_lc),
        exp_non_taylor=last_value(model.exp_non_taylor_history),
        exp_norm_non_taylor=last_value(model.exp_norm_non_taylor),
        exp_total_cc=last_value(model.exp_total_cc_history),
        exp_norm_total_cc=last_value(model.exp_norm_total_cc),
        earn_cc=last_value(model.earn_cc_history),
        earn_norm_cc=last_value(model.earn_norm_cc_history),
        earn_lc=last_value(model.earn_lc_history),
        earn_norm_lc=last_value(model.earn_norm_lc_history),
        worth_lc=result.worth_lc,
        worth_norm_lc=result.worth_norm_lc,
        worth_cc=result.worth_cc,
        worth_norm_cc=result.worth_norm_cc,
        added_lc_worth_norm=result.worth_norm_lc - result.worth_norm_cc,
        yrs_il_double=min(scenario.man_independent_yrs, scenario.woman_independent_yrs),
        yrs_il_single=abs(scenario.man_independent_yrs - scenario.woman_independent_yrs),
        yrs_sum_al=scenario.man_assisted_yrs + scenario.woman_assisted_yrs,
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
    
    # Process edge cases for every ROI  CPI combination (both fixed)
    for roi_apy in EDGE_CASE_ROI_APY_PERCENTS:
        for cpi_apy in EDGE_CASE_CPI_APY_PERCENTS:
            fixed_monthly_roi = apy_percent_to_monthly_fraction(roi_apy)
            fixed_monthly_cpi = apy_percent_to_monthly_fraction(cpi_apy)
            both_context = ScenarioRunContext(
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
            edge_cases = get_edge_cases_gutz(
                roi_apy=roi_apy,
                cpi_apy=cpi_apy,
                centerpoint_man_independent_yrs=CENTERPOINT_MAN_INDEPENDENT_YRS,
                centerpoint_woman_independent_yrs=CENTERPOINT_WOMAN_INDEPENDENT_YRS,
                centerpoint_man_assisted_yrs=CENTERPOINT_MAN_ASSISTED_YRS,
                centerpoint_woman_assisted_yrs=CENTERPOINT_WOMAN_ASSISTED_YRS,
                centerpoint_roi_seed=CENTERPOINT_ROI_SEED,
                centerpoint_inflation_seed=CENTERPOINT_INFLATION_SEED,
            )
            for case_name, scenario in edge_cases:
                model, result = evaluate_lhs_scenario(scenario=scenario, context=both_context)
                row = asdict(summarize_lhs_run(run_id=case_name, scenario=scenario, model=model, result=result, context=both_context))
                ordered_row = {column: row[column] for column in CSV_COLUMNS}
                print_screen_row(row=ordered_row, columns=CSV_COLUMNS, widths=column_widths)
                rows.append(ordered_row)

    # Process replay cases once, outside the ROI  CPI edge-case matrix loop.
    replay_cases = build_replay_case_scenarios()
    for case_name, scenario in replay_cases:
        model, result = evaluate_lhs_scenario(scenario=scenario, context=context)
        row = asdict(
            summarize_lhs_run(
                run_id=case_name,
                scenario=scenario,
                model=model,
                result=result,
                context=context,
            )
        )
        ordered_row = {column: row[column] for column in CSV_COLUMNS}
        print_screen_row(row=ordered_row, columns=CSV_COLUMNS, widths=column_widths)
        rows.append(ordered_row)

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

    lhs_rows = results[results["run_id"].apply(lambda v: not isinstance(v, str))]
    edge_rows = results[results["run_id"].apply(lambda v: isinstance(v, str))]

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
        constant_monthly_roi=CONSTANT_MONTHLY_ROI,
        constant_monthly_cpi=CONSTANT_MONTHLY_CPI,
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
            f"Worth LC range: {results['worth_lc'].min():,.0f} to {results['worth_lc'].max():,.0f}\n"
            f"Worth CC range: {results['worth_cc'].min():,.0f} to {results['worth_cc'].max():,.0f}"
        )
        
        # Figure 1 – centerpoint only
        lhs_only = results[results["run_id"].apply(lambda v: isinstance(v, int))]
        fig1, ax1 = plt.subplots(figsize=(12, 7))
        if not lhs_only.empty:
            x_vals = lhs_only["yrs_sum_al"].to_numpy(dtype=float)
            y_vals = lhs_only["added_lc_worth_norm"].to_numpy(dtype=float)
            positive_mask = y_vals > 0.0
            non_positive_mask = ~positive_mask
            if np.any(non_positive_mask):
                ax1.scatter(
                    x_vals[non_positive_mask],
                    y_vals[non_positive_mask],
                    alpha=0.8,
                    color="red",
                    marker="x",
                    s=18,
                    label="stochastic LHS (<= 0)",
                )
            if np.any(positive_mask):
                ax1.scatter(
                    x_vals[positive_mask],
                    y_vals[positive_mask],
                    alpha=0.8,
                    color="black",
                    marker="x",
                    s=18,
                    label="stochastic LHS (> 0)",
                )
            ax1.legend(loc="best", fontsize=9)
        ax1.set_xlabel("Sum of Assisted Living Years: yrs_sum_al (Years)")
        ax1.set_ylabel("Added Worth (normalized to 2026 dollars)")
        ax1.set_title(PLOT_MAIN_TITLE, fontweight="bold", pad=20)
        ax1.text(
            0.5,
            1.01,
            "Added Worth (normalized) vs yrs_sum_al (Gutz Centerpoint LHS)",
            transform=ax1.transAxes,
            ha="center",
            va="bottom",
        )
        ax1.grid(True, alpha=0.3)
        add_lifecare_reference_line(ax1)

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

