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

import matplotlib.pyplot as plt
import pandas as pd
from Replot_LHS_Taylor import _fix_run_id_types, parse_args
from Run_LHS_Gutz_Taylor import (
    EDGE_CASE_CPI_APY_PERCENTS,
    EDGE_CASE_ROI_APY_PERCENTS,
    _select_nearest_ep_lhs,
    plot_demographic_stats,
    plot_edge_case_subplots,
    plot_gutz_lhs_figure1,
    plot_gutz_lhs_worth_subplots,
    plot_lhs_summary,
    plot_worth_vs_earn,
)

DEFAULT_LHS_CSV = "lhs_gutz_taylor_results.csv"


def main() -> None:
    args = parse_args(
        description="Regenerate all Run_LHS_Gutz_Taylor plots from an existing CSV.",
        default_lhs_csv=DEFAULT_LHS_CSV,
        default_roi_apys=EDGE_CASE_ROI_APY_PERCENTS,
        default_cpi_apys=EDGE_CASE_CPI_APY_PERCENTS,
    )
    include_edge: bool = not args.no_edge_cases
    roi_apys: list[float] = args.roi_apys
    cpi_apys: list[float] = args.cpi_apys

    print(f"Loading '{args.lhs_csv}' ...")
    results: pd.DataFrame = pd.read_csv(args.lhs_csv)  # type: ignore[assignment]
    results = _fix_run_id_types(results)
    n_lhs = results["run_id"].apply(lambda v: isinstance(v, int)).sum()
    n_edge = results["run_id"].apply(lambda v: isinstance(v, str)).sum()
    print(f"  {len(results)} rows loaded  ({n_lhs} stochastic LHS, {n_edge} edge/replay).")
    print(f"  ROI APY grid:  {roi_apys}")
    print(f"  CPI APY grid:  {cpi_apys}")
    print("Generating plots ...")

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

    # Figure 7 – added worth (normalized) vs life structure params (3×1 subplots)
    plot_lhs_summary(
        results,
        include_edge_cases=include_edge,
        show=False,
    )

    # Figure 8 – added worth (normalized) edge-case subplots, shared y-scale
    plot_edge_case_subplots(
        results,
        roi_apys,
        cpi_apys,
        shared_y_scale=True,
        show=False,
    )

    # Figure 9 – added worth (normalized) edge-case subplots, free y-scale
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

