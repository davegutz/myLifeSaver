"""
Replay_LHS_Case.py

Replays a single stochastic case from a previous Run_LHS_Taylor run.
Provide the integer run_id (case number printed in the LHS table) and
the path to the LHS CSV.  Every scenario parameter is read directly
from the CSV row so the run is bit-for-bit identical to the original.

Usage:
    python Replay_LHS_Case.py 42
    python Replay_LHS_Case.py 42 --lhs-csv path/to/lhs_taylor_results.csv
    python Replay_LHS_Case.py 42 --ticker SPY --current-date 2025-01-01
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Inflation import plot_inflation_views
from Roi import TICKER, plot_projection_views
from Taylor import LhsScenario, ScenarioRunContext
from default_case import (
    AL_ESC_RUNNING_AVG_YRS,
    CONSTANT_MONTHLY_CPI,
    CONSTANT_MONTHLY_ROI,
    DEFAULT_CURRENT_DATE,
    HISTORY_YEARS,
    MAN_DOB,
    START_CLOCK,
    WOMAN_DOB,
)
from utils import evaluate_lhs_scenario, plot_taylor_life_exp_non_taylor

# Default path written by Run_LHS_Taylor.py
DEFAULT_LHS_CSV = "lhs_taylor_results.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay a single stochastic case from a previous Run_LHS_Taylor run."
    )
    parser.add_argument(
        "run_id",
        type=int,
        nargs="?",
        default=None,
        help="Integer run_id (case number) from the LHS results CSV. "
             "If omitted, available IDs are listed and you will be prompted.",
    )
    parser.add_argument(
        "--lhs-csv",
        default=DEFAULT_LHS_CSV,
        help=f"Path to the LHS results CSV produced by Run_LHS_Taylor.py. Default: {DEFAULT_LHS_CSV}",
    )
    parser.add_argument(
        "--ticker",
        default=TICKER,
        help=f"Ticker symbol for ROI history download. Default: {TICKER}",
    )
    parser.add_argument(
        "--current-date",
        default=DEFAULT_CURRENT_DATE,
        help=f"Historical data cutoff date YYYY-MM-DD. Default: {DEFAULT_CURRENT_DATE}",
    )
    return parser.parse_args()


def list_stochastic_run_ids(csv_path: str) -> list[int]:
    """Return the sorted list of integer run_ids from the LHS CSV."""
    df = pd.read_csv(csv_path)
    df["run_id"] = df["run_id"].astype(str)
    stochastic = df[df["run_id"].apply(lambda v: str(v).isdigit())]
    return sorted(stochastic["run_id"].apply(int).tolist())


def prompt_for_run_id(csv_path: str) -> int:
    """Print available stochastic run_ids and ask the user to choose one."""
    try:
        available = list_stochastic_run_ids(csv_path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"LHS CSV not found: '{csv_path}'. "
            f"Run Run_LHS_Taylor.py first or supply --lhs-csv."
        )
    print(f"Available stochastic run_ids in '{csv_path}': {available}")
    while True:
        raw = input("Enter run_id: ").strip()
        if raw.isdigit() and int(raw) in available:
            return int(raw)
        print(f"  '{raw}' is not a valid run_id. Please choose from {available}.")


def load_scenario_from_csv(csv_path: str, run_id: int) -> LhsScenario:
    """Read a stochastic LHS row from the CSV and reconstruct its LhsScenario."""
    # Force run_id column to string so mixed int/str CSV column is handled uniformly.
    df = pd.read_csv(csv_path)
    df["run_id"] = df["run_id"].astype(str)

    # Stochastic rows have purely numeric run_ids; edge-case rows start with letters.
    stochastic = df[df["run_id"].apply(lambda v: str(v).isdigit())]
    row = stochastic[stochastic["run_id"].apply(lambda v: int(v) == run_id)]

    if row.empty:
        available = sorted(stochastic["run_id"].apply(int).tolist())
        raise ValueError(
            f"run_id={run_id} not found in '{csv_path}'.\n"
            f"Available stochastic run_ids: {available}"
        )

    r = row.iloc[0]
    return LhsScenario(
        man_independent_yrs=float(r["man_independent_yrs"]),
        woman_independent_yrs=float(r["woman_independent_yrs"]),
        man_assisted_yrs=float(r["man_assisted_yrs"]),
        woman_assisted_yrs=float(r["woman_assisted_yrs"]),
        roi_seed=int(float(r["roi_seed"])),
        inflation_seed=int(float(r["inflation_seed"])),
        roi_mean_shift=float(r["roi_mean_shift"]),
        roi_vol_multiplier=float(r["roi_vol_multiplier"]),
        roi_mean_reversion=float(r["roi_mean_reversion"]),
        inflation_mean_shift=float(r["inflation_mean_shift"]),
        inflation_vol_multiplier=float(r["inflation_vol_multiplier"]),
        inflation_mean_reversion=float(r["inflation_mean_reversion"]),
    )


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


def main() -> None:
    args = parse_args()

    run_id: int = args.run_id if args.run_id is not None else prompt_for_run_id(args.lhs_csv)

    print(f"Loading run_id={run_id} from '{args.lhs_csv}' ...")
    scenario = load_scenario_from_csv(args.lhs_csv, run_id)
    print(f"Scenario parameters loaded — running replay.")

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

    this_life, result = evaluate_lhs_scenario(scenario=scenario, context=context)
    roi = this_life.roi
    cpi = this_life.cpi

    if cpi.inflation_frame is None:
        raise ValueError("Inflation history was not loaded during projection.")
    inflation_frame = cpi.inflation_frame

    effective_monthly_roi = realized_monthly_rate(roi.life_horizon_roi, roi.monthly_mean_return)
    effective_monthly_cpi = realized_monthly_rate(cpi.life_horizon_inflation, cpi.monthly_mean_inflation)
    annualized_mean = monthly_rate_to_apy(effective_monthly_roi)
    annualized_mean_cpi = monthly_rate_to_apy(effective_monthly_cpi)

    print(
        f"\n=== Replay of LHS run_id={run_id} from '{args.lhs_csv}' ===\n"
        f"Ticker:                     {args.ticker}\n"
        f"Effective APY return:       {annualized_mean:.2%}\n"
        f"Monthly volatility:         {roi.monthly_volatility:.2%}\n"
        f"ROI seed:                   {scenario.roi_seed}\n"
        f"Inflation seed:             {scenario.inflation_seed}\n"
        f"CPI current date:           {current_date.date()}\n"
        f"Effective annualized CPI:   {annualized_mean_cpi:.2%}\n"
        f"Cum. inflation of $1 since {START_CLOCK}: ${cpi.life_horizon_inflation_cum[-1]:.4f}"
    )

    total_expenses_cc    = this_life.exp_cc_history[-1]         if this_life.exp_cc_history         else 0.0
    total_expenses_lc    = this_life.exp_lc_history[-1]         if this_life.exp_lc_history         else 0.0
    total_al_expenses_cc = this_life.exp_al_cc_history[-1]      if this_life.exp_al_cc_history      else 0.0
    total_al_expenses_lc = this_life.exp_al_cc_history[-1]      if this_life.exp_al_cc_history      else 0.0
    total_non_taylor_cc  = this_life.exp_non_taylor_history[-1] if this_life.exp_non_taylor_history else 0.0
    total_non_taylor_lc  = total_non_taylor_cc
    grand_total_cc       = this_life.exp_total_cc_history[-1]   if this_life.exp_total_cc_history   else 0.0
    grand_total_lc       = this_life.exp_total_lc_history[-1]   if this_life.exp_total_lc_history   else 0.0
    total_returns_cc     = this_life.earn_cc_history[-1]         if this_life.earn_cc_history         else 0.0
    total_returns_lc     = this_life.earn_lc_history[-1]         if this_life.earn_lc_history         else 0.0
    worth_cc = result.worth_cc
    worth_lc = result.worth_lc

    header_rows = [
        ("apy roi %",            annualized_mean * 100.0,      annualized_mean * 100.0),
        ("apy cpi %",            annualized_mean_cpi * 100.0,  annualized_mean_cpi * 100.0),
        ("man independent yrs",  this_life.man_independent_yrs,  this_life.man_independent_yrs),
        ("man assisted yrs",     this_life.man_assisted_yrs,     this_life.man_assisted_yrs),
        ("man age to al",        this_life.man_age_to_al,        this_life.man_age_to_al),
        ("man age at death",     this_life.man_age_at_death,     this_life.man_age_at_death),
        ("woman independent yrs",this_life.woman_independent_yrs,this_life.woman_independent_yrs),
        ("woman assisted yrs",   this_life.woman_assisted_yrs,   this_life.woman_assisted_yrs),
        ("woman age to al",      this_life.woman_age_to_al,      this_life.woman_age_to_al),
        ("woman age at death",   this_life.woman_age_at_death,   this_life.woman_age_at_death),
    ]
    table_rows = [
        ("total expenses",           total_expenses_cc,    total_expenses_lc),
        ("total al expenses",        total_al_expenses_cc, total_al_expenses_lc),
        ("total non-taylor expenses",total_non_taylor_cc,  total_non_taylor_lc),
        ("grand total expenses",     grand_total_cc,       grand_total_lc),
        ("total returns",            total_returns_cc,     total_returns_lc),
        ("final worth",              worth_cc,             worth_lc),
    ]

    print(f"\n{'item':<28}{'cc':>15}{'lc':>15}")
    print(f"{'-' * 28}{'-' * 15}{'-' * 15}")
    for item, cc_value, lc_value in header_rows:
        print(f"{item:<28}{cc_value:>15.1f}{lc_value:>15.1f}")
    print(f"{'-' * 28}{'-' * 15}{'-' * 15}")
    for item, cc_value, lc_value in table_rows:
        print(f"{item:<28}{cc_value:>15,.0f}{lc_value:>15,.0f}")

    # Write monthly CSV alongside the run
    out_csv = f"replay_lhs_case_monthly_{run_id}.csv"
    
    # Compute monthly APY values from the actual stochastic paths.
    # Each month's rate is annualized: (1 + monthly_rate)^12 - 1
    monthly_apy_roi = (1.0 + np.asarray(this_life.roi.life_horizon_roi)) ** 12 - 1.0
    monthly_apy_cpi = (1.0 + np.asarray(this_life.cpi.life_horizon_inflation)) ** 12 - 1.0
    
    df_monthly = pd.DataFrame({
        "date":              pd.to_datetime(this_life.dates),
        "apy_roi":           monthly_apy_roi * 100.0,
        "apy_cpi":           monthly_apy_cpi * 100.0,
        "earn_lc":           this_life.earn_lc_history,
        "earn_cc":           this_life.earn_cc_history,
        "earn_norm_lc":      this_life.earn_norm_lc_history,
        "earn_norm_cc":      this_life.earn_norm_cc_history,
        "exp_total_lc":      this_life.exp_total_lc_history,
        "exp_total_cc":      this_life.exp_total_cc_history,
        "exp_norm_total_lc": this_life.exp_norm_total_lc,
        "exp_norm_total_cc": this_life.exp_norm_total_cc,
        "worth_lc":          this_life.worth_lc_history,
        "worth_cc":          this_life.worth_cc_history,
        "worth_norm_lc":     this_life.worth_norm_lc_history,
        "worth_norm_cc":     this_life.worth_norm_cc_history,
        "exp_al_cc":         this_life.exp_al_cc_history,
        "exp_cc":            this_life.exp_cc_history,
        "exp_lc":            this_life.exp_lc_history,
        "exp_non_taylor":    this_life.exp_non_taylor_history,
    })
    df_monthly.to_csv(out_csv, index=False)
    print(f"\nMonthly detail written to '{out_csv}'.")

    if roi.return_frame is None:
        raise ValueError("ROI history was not loaded during projection.")
    plot_projection_views(roi.return_frame, roi, show=False)
    plot_inflation_views(inflation_frame, cpi, show=False)
    plot_taylor_life_exp_non_taylor(this_life, show=False)
    plt.show()


if __name__ == "__main__":
    main()


