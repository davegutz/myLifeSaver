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
from edges import build_edge_case_scenarios, build_replay_case_scenarios, format_apy_suffix
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

# Fixed and default varied parameters are imported from default_case.py.
MAN_INDEPENDENT_YRS_RANGE = (69.0 - age(START_CLOCK, MAN_DOB), 90.0 - age(START_CLOCK, MAN_DOB))
WOMAN_INDEPENDENT_YRS_RANGE = (70.0 - age(START_CLOCK, WOMAN_DOB), 90.0 - age(START_CLOCK, WOMAN_DOB))
MAN_ASSISTED_YRS_RANGE = (0., 10.0)
WOMAN_ASSISTED_YRS_RANGE = (0., 10.0)
SEED_RANGE = (0, 1000000)
ROI_MEAN_SHIFT_RANGE = (-0.01, 0.01)
ROI_VOL_MULTIPLIER_RANGE = (0.5, 1.5)
ROI_MEAN_REVERSION_RANGE = (0.0, 0.5)
INFLATION_MEAN_SHIFT_RANGE = (-0.005, 0.005)
INFLATION_VOL_MULTIPLIER_RANGE = (0.5, 1.5)
INFLATION_MEAN_REVERSION_RANGE = (0.0, 0.5)
DEFAULT_LHS_POINTS = 200
PLOT_EDGE_CASES_IN_LHS_PLOT = True
EDGE_CASE_ROI_APY_PERCENTS = [6.0, 12.0]  # Fixed ROI rates (APY %) for edge case matrix
EDGE_CASE_CPI_APY_PERCENTS = [2.0, 6.0]  # Fixed CPI rates (APY %) for edge case matrix
# EDGE_CASE_ROI_APY_PERCENTS = [10.0]  # Fixed ROI rates (APY %) for edge case matrix
# EDGE_CASE_CPI_APY_PERCENTS = [5.0]  # Fixed CPI rates (APY %) for edge case matrix

CSV_COLUMNS = [
    "run_id",
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


def realized_monthly_rate(path: np.ndarray, fallback: float) -> float:
    if path.size == 0:
        return fallback
    growth = np.cumprod(1.0 + np.asarray(path, dtype=float))
    if growth.size == 0 or growth[-1] <= 0.0:
        return fallback
    months = float(path.size)
    return float(growth[-1] ** (1.0 / months) - 1.0)


def effective_apy_from_cumulative(cumulative_path: np.ndarray, monthly_fallback: float) -> float:
    if cumulative_path.size > 0:
        final_growth = float(cumulative_path[-1])
        if final_growth > 0.0:
            months = float(cumulative_path.size)
            return (final_growth ** (12.0 / months) - 1.0) * 100.0
    return monthly_rate_to_apy(monthly_fallback) * 100.0


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
        # Context constants
        ticker=context.ticker,
        current_date=str(context.current_date),
        history_years=context.history_years,
        al_cum_running_avg_yrs=context.al_cum_running_avg_yrs,
        start_clock=context.start_clock,
        man_dob=context.man_dob,
        woman_dob=context.woman_dob,
        constant_monthly_roi=context.constant_monthly_roi,
        constant_monthly_cpi=context.constant_monthly_cpi,
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
    
    # Process edge cases for every ROI × CPI combination (both fixed)
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
            both_edge_cases = build_edge_case_scenarios(roi_apy_percent=roi_apy, cpi_apy_percent=cpi_apy)
            for case_name, scenario in both_edge_cases:
                model, result = evaluate_lhs_scenario(scenario=scenario, context=both_context)
                row = asdict(summarize_lhs_run(run_id=case_name, scenario=scenario, model=model, result=result, context=both_context))
                ordered_row = {column: row[column] for column in CSV_COLUMNS}
                print_screen_row(row=ordered_row, columns=CSV_COLUMNS, widths=column_widths)
                rows.append(ordered_row)

    # Process replay cases once, outside the ROI × CPI edge-case matrix loop.
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
    from matplotlib.lines import Line2D

    edge_results = results[results["run_id"].apply(lambda v: isinstance(v, str))]
    if edge_results.empty:
        return

    n_roi = len(roi_apy_percents)
    n_cpi = len(cpi_apy_percents)
    figure, axes = plt.subplots(
        n_roi,
        n_cpi,
        figsize=(6 * n_cpi, 5 * n_roi),
        squeeze=False,
        constrained_layout=True,
    )
    suffix = "(Shared Y-Scale)" if shared_y_scale else "(Free Axis Scale)"
    figure.suptitle(f"Edge Cases: Normalized Worth vs Combined Assisted Years {suffix}", fontsize=14)

    # Color encodes combined independent years (red=lower, green=higher).
    cmap = LinearSegmentedColormap.from_list("bright_rg", ["#ff0000", "#00ff00"])
    independent_all = edge_results["man_independent_yrs"] + edge_results["woman_independent_yrs"]
    norm = Normalize(vmin=float(independent_all.min()), vmax=float(independent_all.max()))

    y_min = 0.0
    y_max = 0.0
    if shared_y_scale:
        # Force a shared y-range across all panels for direct visual comparison.
        y_all = np.concatenate([
            edge_results["worth_norm_lc"].to_numpy(dtype=float),
            edge_results["worth_norm_cc"].to_numpy(dtype=float),
        ])
        y_min = float(np.nanmin(y_all))
        y_max = float(np.nanmax(y_all))
        if math.isclose(y_min, y_max):
            pad = abs(y_min) * 0.01 + 1e-6
            y_min -= pad
            y_max += pad

    style_handles = [
        Line2D([0], [0], marker="o", color="gray", linestyle="None", markersize=7, label="lc"),
        Line2D([0], [0], marker="^", color="gray", linestyle="None", markersize=7, label="cc"),
    ]

    mappable = None
    for row_idx, roi_apy in enumerate(roi_apy_percents):
        for col_idx, cpi_apy in enumerate(cpi_apy_percents):
            ax = axes[row_idx][col_idx]
            suffix = f"_{format_apy_suffix(roi_apy)}_{format_apy_suffix(cpi_apy)}"
            combo_rows = edge_results[
                edge_results["run_id"].apply(lambda v, s=suffix: isinstance(v, str) and v.endswith(s))
            ]
            if not combo_rows.empty:
                assisted_total = combo_rows["man_assisted_yrs"] + combo_rows["woman_assisted_yrs"]
                independent_total = combo_rows["man_independent_yrs"] + combo_rows["woman_independent_yrs"]
                mappable = ax.scatter(
                    assisted_total,
                    combo_rows["worth_norm_lc"],
                    c=independent_total,
                    cmap=cmap,
                    norm=norm,
                    marker="o",
                    alpha=0.85,
                    s=50,
                )
                ax.scatter(
                    assisted_total,
                    combo_rows["worth_norm_cc"],
                    c=independent_total,
                    cmap=cmap,
                    norm=norm,
                    marker="^",
                    alpha=0.55,
                    s=50,
                )
            ax.set_xlabel("man_assisted_yrs + woman_assisted_yrs")
            ax.set_ylabel("Worth (normalized)")
            ax.set_title(f"ROI={roi_apy:.3g}%  CPI={cpi_apy:.3g}%")
            if shared_y_scale:
                ax.set_ylim(y_min, y_max)
            ax.grid(True, alpha=0.3)
            ax.legend(handles=style_handles, loc="best", fontsize=7)

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
    from matplotlib.lines import Line2D

    lhs_rows = results[results["run_id"].apply(lambda v: not isinstance(v, str))]
    edge_rows = results[results["run_id"].apply(lambda v: isinstance(v, str))]

    figure, axis = plt.subplots(figsize=(12, 7))

    if not lhs_rows.empty:
        lhs_x = lhs_rows["man_assisted_yrs"] + lhs_rows["woman_assisted_yrs"]
        axis.scatter(lhs_x, lhs_rows["worth_norm_lc"], alpha=0.25, color="lightgray", marker="o", label="LHS lc")
        axis.scatter(lhs_x, lhs_rows["worth_norm_cc"], alpha=0.25, color="gray", marker="^", label="LHS cc")

    mappable = None
    if include_edge_cases and not edge_rows.empty:
        cmap = LinearSegmentedColormap.from_list("bright_rg", ["#ff0000", "#00ff00"])
        independent_total = edge_rows["man_independent_yrs"] + edge_rows["woman_independent_yrs"]
        norm = Normalize(vmin=float(independent_total.min()), vmax=float(independent_total.max()))
        x_vals = edge_rows["man_assisted_yrs"] + edge_rows["woman_assisted_yrs"]
        mappable = axis.scatter(
            x_vals,
            edge_rows["worth_norm_lc"],
            c=independent_total,
            cmap=cmap,
            norm=norm,
            marker="o",
            alpha=0.9,
            s=55,
            label="edge lc",
        )
        axis.scatter(
            x_vals,
            edge_rows["worth_norm_cc"],
            c=independent_total,
            cmap=cmap,
            norm=norm,
            marker="^",
            alpha=0.55,
            s=55,
            label="edge cc",
        )

    handles, _ = axis.get_legend_handles_labels()
    style_handles = [
        Line2D([0], [0], marker="o", color="gray", linestyle="None", markersize=8, label="o = lc"),
        Line2D([0], [0], marker="^", color="gray", linestyle="None", markersize=8, label="^ = cc"),
    ]
    axis.legend(handles=handles + style_handles, loc="best", fontsize=9)

    if mappable is not None:
        figure.colorbar(mappable, ax=axis, label="combined independent years")

    axis.set_xlabel("man_assisted_yrs + woman_assisted_yrs")
    axis.set_ylabel("Worth (normalized)")
    axis.set_title("Normalized Worth vs Combined Assisted Years")
    axis.grid(True, alpha=0.3)
    plt.tight_layout()
    if show:
        plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monte Carlo monthly ROI projection anchored to historical long-run growth."
    )
    parser.add_argument("--ticker", default=TICKER, help="Ticker symbol to download, default: SPY")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"RNG seed, default: {DEFAULT_SEED}")
    parser.add_argument(
        "--current-date",
        default=DEFAULT_CURRENT_DATE,
        help=f"Historical data cutoff date in YYYY-MM-DD, default: {DEFAULT_CURRENT_DATE}",
    )
    parser.add_argument(
        "--lhs-points",
        type=int,
        default=DEFAULT_LHS_POINTS,
        help=f"Run a Latin hypercube sample with this many points. Default: {DEFAULT_LHS_POINTS}",
    )
    parser.add_argument(
        "--lhs-output",
        default="lhs_taylor_results.csv",
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

    scenario = LhsScenario(
        roi_seed=args.seed,
        inflation_seed=args.seed,
    )
    this_life, result = evaluate_lhs_scenario(scenario=scenario, context=context)
    roi = this_life.roi
    cpi = this_life.cpi
    if cpi.inflation_frame is None:
        raise ValueError("Inflation history was not loaded during projection.")
    inflation_frame = cpi.inflation_frame
    worth_lc = result.worth_lc
    worth_cc = result.worth_cc

    effective_monthly_roi = realized_monthly_rate(roi.life_horizon_roi, roi.monthly_mean_return)
    effective_monthly_cpi = realized_monthly_rate(cpi.life_horizon_inflation, cpi.monthly_mean_inflation)
    annualized_mean = monthly_rate_to_apy(effective_monthly_roi)
    annualized_mean_cpi = monthly_rate_to_apy(effective_monthly_cpi)
    print(
        f"Ticker: {args.ticker}\n"
        f"Effective APY return: {annualized_mean:.2%}\n"
        f"Monthly volatility: {roi.monthly_volatility:.2%}\n"
        f"ROI seed: {scenario.roi_seed}\n"
        f"Inflation seed: {scenario.inflation_seed}\n"
        f"CPI current date: {current_date.date()}\n"
        f"Effective annualized CPI inflation: {annualized_mean_cpi:.2%}\n"
        f"Cumulative inflation growth of $1 since {START_CLOCK}: ${cpi.life_horizon_inflation_cum[-1]:.4f}"
    )
    # print(roi)
    # print(cpi)
    total_expenses_cc = this_life.exp_cc_history[-1] if this_life.exp_cc_history else 0.0
    total_expenses_lc = this_life.exp_lc_history[-1] if this_life.exp_lc_history else 0.0
    total_al_expenses_cc = this_life.exp_al_cc_history[-1] if this_life.exp_al_cc_history else 0.0
    total_al_expenses_lc = this_life.exp_al_lc_history[-1] if this_life.exp_al_lc_history else 0.0
    total_non_taylor_cc = this_life.exp_non_taylor_history[-1] if this_life.exp_non_taylor_history else 0.0
    total_non_taylor_lc = total_non_taylor_cc
    grand_total_cc = this_life.exp_total_cc_history[-1] if this_life.exp_total_cc_history else 0.0
    grand_total_lc = this_life.exp_total_lc_history[-1] if this_life.exp_total_lc_history else 0.0
    total_returns_cc = this_life.earn_cc_history[-1] if this_life.earn_cc_history else 0.0
    total_returns_lc = this_life.earn_lc_history[-1] if this_life.earn_lc_history else 0.0
    worth_cc = result.worth_cc
    worth_lc = result.worth_lc
    header_rows = [
        ("apy roi %", annualized_mean * 100.0, annualized_mean * 100.0),
        ("apy cpi %", annualized_mean_cpi * 100.0, annualized_mean_cpi * 100.0),
        ("man independent yrs", this_life.man_independent_yrs, this_life.man_independent_yrs),
        ("man assisted yrs", this_life.man_assisted_yrs, this_life.man_assisted_yrs),
        ("man age to al", this_life.man_age_to_al, this_life.man_age_to_al),
        ("man age at death", this_life.man_age_at_death, this_life.man_age_at_death),
        ("woman independent yrs", this_life.woman_independent_yrs, this_life.woman_independent_yrs),
        ("woman assisted yrs", this_life.woman_assisted_yrs, this_life.woman_assisted_yrs),
        ("woman age to al", this_life.woman_age_to_al, this_life.woman_age_to_al),
        ("woman age at death", this_life.woman_age_at_death, this_life.woman_age_at_death),
    ]
    table_rows = [
        ("total expenses", total_expenses_cc, total_expenses_lc),
        ("total al expenses", total_al_expenses_cc, total_al_expenses_lc),
        ("total non-taylor expenses", total_non_taylor_cc, total_non_taylor_lc),
        ("grand total expenses", grand_total_cc, grand_total_lc),
        ("total returns", total_returns_cc, total_returns_lc),
        ("final worth", worth_cc, worth_lc),
    ]
    print(f"{'item':<28}{'cc':>15}{'lc':>15}")
    print(f"{'-' * 28}{'-' * 15}{'-' * 15}")
    for item, cc_value, lc_value in header_rows:
        print(f"{item:<28}{cc_value:>15.1f}{lc_value:>15.1f}")
    print(f"{'-' * 28}{'-' * 15}{'-' * 15}")
    for item, cc_value, lc_value in table_rows:
        print(f"{item:<28}{cc_value:>15,.0f}{lc_value:>15,.0f}")
    if roi.return_frame is None:
        raise ValueError("ROI history was not loaded during projection.")
    plot_projection_views(roi.return_frame, roi, show=False)
    plot_inflation_views(inflation_frame, cpi, show=False)
    plot_taylor_life_exp_non_taylor(this_life, show=False)
    plt.show()


if __name__ == "__main__":
    main()

