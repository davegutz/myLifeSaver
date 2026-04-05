import argparse
from dataclasses import asdict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Inflation import plot_inflation_views
from Roi import TICKER, plot_projection_views
from Taylor import LhsScenario, ScenarioRunContext
from Run_one_Taylor import configurate_local_run, configurate_base_run, run_one, merge_run_config
from default_case import (
    AL_ESC_RUNNING_AVG_YRS,
    CONSTANT_MONTHLY_CPI,
    CONSTANT_MONTHLY_ROI,
    DEFAULT_CURRENT_DATE,
    DEFAULT_SEED,
    HISTORY_YEARS,
    INFLATION_MEAN_REVERSION,
    INFLATION_MEAN_SHIFT,
    INFLATION_VOL_MULTIPLIER,
    MAN_ASSISTED_YRS,
    MAN_DOB,
    MAN_INDEPENDENT_YRS,
    P_MAN_AL,
    P_WOMAN_AL,
    PILE_AT_START,
    ROI_MEAN_REVERSION,
    ROI_MEAN_SHIFT,
    ROI_VOL_MULTIPLIER,
    START_CLOCK,
    WOMAN_ASSISTED_YRS,
    WOMAN_DOB,
    WOMAN_INDEPENDENT_YRS,
    apy_percent_to_monthly_fraction,
    load_default_case,
)
from utils import evaluate_lhs_scenario, plot_taylor_life_exp_non_taylor


RUN_ONE_CASE_NAME: str | None = None  # e.g. "RUN_ONE_PRESENT" or "DEFAULT"
# RUN_ONE_CASE_NAME: str | None = 'RUN_ONE_PRESENT'  # e.g. "RUN_ONE_PRESENT" or "DEFAULT"


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    active_case_name = RUN_ONE_CASE_NAME
    base_run_config = configurate_base_run(args)

    case_run_config = None
    if active_case_name is not None:
        case_scenario_kwargs, case_context_kwargs = load_default_case(active_case_name)
        case_run_config = {
            "scenario": case_scenario_kwargs,
            "context": case_context_kwargs,
        }

    ins = [
        [0, 8, 4, 10, 8.8],
        [1, 8, 4, 10, 8.8],
        [2, 8, 4, 10, 8.8],
        [4, 8, 4, 10, 8.8],
        [8, 8, 4, 10, 8.8],
        [16, 8, 4, 10, 8.8],
    ]

    cc_worth_norm = []
    lc_worth_norm = []
    added_lc_benefit =[]
    print(
        f"yrs_al_total,"
        f"roi_fixed,"
        f"cpi_fixed,"
        f"yrs_il_man,"
        f"yrs_il_woman,"
        f"yrs_al_total_cc_norm,"
        f"start_pile,"
        f"entrance_fee_cc,"
        f"cum_mo_earn_total_cc_norm,"
        f"cum_mo_exp_total_cc_norm,"
        f"final_worth_cc_norm,"
        f"entrance_fee_lc,"
        f"cum_mo_earn_total_lc_norm,"
        f"cum_mo_exp_total_lc_norm,"
        f"final_worth_lc_norm,"
    )
    for [yrs_al_total, roi_fixed, cpi_fixed, yrs_il_man, yrs_il_woman] in ins:
        local_run_overrides = configurate_local_run(yrs_al_total=0., roi_fixed=4., cpi_fixed=4., yrs_il_man=10., yrs_il_woman=8.8)
        run_config = merge_run_config(base_run_config, case_run_config, local_run_overrides)
        r = run_one(run_config=run_config, active_case_name=active_case_name, plot=False, printing=False)
        print(
            f" {yrs_al_total},"
            f" {roi_fixed},"
            f" {cpi_fixed},"
            f" {yrs_il_man},"
            f" {yrs_il_woman},"
            f" {r['start_pile']:.0f},"
            f" {r['entrance_fee_cc']:.0f},"
            f" {r['cum_mo_earn_total_cc_norm']:.0f},"
            f" {r['cum_mo_exp_total_cc_norm']:.0f},"
            f" {r['final_worth_cc_norm']:.0f},"
            f" {r['entrance_fee_lc']:.0f},"
            f" {r['cum_mo_earn_total_lc_norm']:.0f},"
            f" {r['cum_mo_exp_total_lc_norm']:.0f},"
            f" {r['final_worth_lc_norm']:.0f},"
        )
        cc_worth_norm.append(0)

if __name__ == "__main__":
    main()
