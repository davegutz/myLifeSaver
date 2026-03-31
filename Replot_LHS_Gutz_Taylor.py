"""
Replot_LHS_Gutz_Taylor.py

Regenerates all Run_LHS_Gutz_Taylor figures from an existing
lhs_gutz_taylor_results.csv without re-running any simulations.

Usage:
    python Replot_LHS_Gutz_Taylor.py
    python Replot_LHS_Gutz_Taylor.py --lhs-csv path/to/lhs_gutz_taylor_results.csv
    python Replot_LHS_Gutz_Taylor.py --roi-apys 6 12 --cpi-apys 0 2 6
    python Replot_LHS_Gutz_Taylor.py --no-edge-cases
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Run_LHS_Gutz_Taylor import (
    EDGE_CASE_CPI_APY_PERCENTS,
    EDGE_CASE_ROI_APY_PERCENTS,
    PLOT_MAIN_TITLE,
    add_lifecare_reference_line,
    plot_edge_case_subplots,
    plot_lhs_summary,
)

DEFAULT_LHS_CSV = "lhs_gutz_taylor_results.csv"


def _fix_run_id_types(results: pd.DataFrame) -> pd.DataFrame:
    """
    When pandas reads the LHS CSV, the run_id column is object dtype and every
    value is a str.  The plot functions distinguish LHS rows (int run_id) from
    edge-case rows (str run_id) via isinstance checks, so restore the original
    types: pure-numeric strings become int, everything else stays str.
    """
    def coerce(v: object) -> int | str:
        s = str(v).strip()
        return int(s) if s.lstrip("-").isdigit() else s

    results = results.copy()
    results["run_id"] = results["run_id"].apply(coerce)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate all Run_LHS_Gutz_Taylor plots from an existing CSV."
    )
    parser.add_argument(
        "--lhs-csv",
        default=DEFAULT_LHS_CSV,
        help=f"Path to the LHS Gutz results CSV produced by Run_LHS_Gutz_Taylor.py. "
             f"Default: {DEFAULT_LHS_CSV}",
    )
    parser.add_argument(
        "--roi-apys",
        nargs="+",
        type=float,
        default=EDGE_CASE_ROI_APY_PERCENTS,
        metavar="PCT",
        help=(
            "ROI APY %% values that identify edge-case subplots. "
            f"Default (from Run_LHS_Gutz_Taylor): {EDGE_CASE_ROI_APY_PERCENTS}"
        ),
    )
    parser.add_argument(
        "--cpi-apys",
        nargs="+",
        type=float,
        default=EDGE_CASE_CPI_APY_PERCENTS,
        metavar="PCT",
        help=(
            "CPI APY %% values that identify edge-case subplots. "
            f"Default (from Run_LHS_Gutz_Taylor): {EDGE_CASE_CPI_APY_PERCENTS}"
        ),
    )
    parser.add_argument(
        "--no-edge-cases",
        action="store_true",
        help="Suppress edge-case overlays on the summary plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    include_edge: bool = not args.no_edge_cases
    roi_apys: list[float] = args.roi_apys
    cpi_apys: list[float] = args.cpi_apys

    print(f"Loading '{args.lhs_csv}' ...")
    results = pd.read_csv(args.lhs_csv)
    results = _fix_run_id_types(results)
    n_lhs = results["run_id"].apply(lambda v: isinstance(v, int)).sum()
    n_edge = results["run_id"].apply(lambda v: isinstance(v, str)).sum()
    print(f"  {len(results)} rows loaded  ({n_lhs} stochastic LHS, {n_edge} edge/replay).")
    print(f"  ROI APY grid:  {roi_apys}")
    print(f"  CPI APY grid:  {cpi_apys}")
    print("Generating plots ...")

    # Figure 1 – added_lc_worth_norm vs yrs_sum_al (stochastic rows only, no edge cases)
    lhs_only = results[results["run_id"].apply(lambda v: isinstance(v, int))]
    centerpoint_rows = results[results["run_id"].apply(lambda v: str(v) == "CENTERPOINT")]
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
    if not centerpoint_rows.empty:
        ax1.scatter(
            centerpoint_rows["yrs_sum_al"].to_numpy(dtype=float),
            centerpoint_rows["added_lc_worth_norm"].to_numpy(dtype=float),
            color="blue",
            marker="*",
            s=360,
            edgecolors="black",
            linewidths=0.9,
            zorder=6,
            label="CENTERPOINT",
        )
    if not lhs_only.empty or not centerpoint_rows.empty:
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

    # Figure 2 – added worth (normalized) vs life structure params (3×1 subplots)
    plot_lhs_summary(
        results,
        include_edge_cases=include_edge,
        roi_apy_percents=roi_apys,
        cpi_apy_percents=cpi_apys,
        show=False,
    )

    # Figure 3 – added worth (normalized) edge-case subplots, shared y-scale
    plot_edge_case_subplots(
        results,
        roi_apys,
        cpi_apys,
        shared_y_scale=True,
        show=False,
    )

    # Figure 4 – added worth (normalized) edge-case subplots, free y-scale
    plot_edge_case_subplots(
        results,
        roi_apys,
        cpi_apys,
        shared_y_scale=False,
        show=False,
    )

    plt.show()


if __name__ == "__main__":
    main()

