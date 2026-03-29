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
import ast
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import re
from Inflation import plot_inflation_views
from Roi import plot_projection_views
from Taylor import LhsScenario, ScenarioRunContext
from utils import evaluate_lhs_scenario, plot_taylor_life_exp_non_taylor

# Default path written by Run_LHS_Taylor.py
DEFAULT_LHS_CSV = "lhs_taylor_results.csv"
REPLAY_CASE_FILE = Path(__file__).with_name("replay_case.py")
REPLAY_FIELD_ORDER = [
    "man_independent_yrs",
    "woman_independent_yrs",
    "man_assisted_yrs",
    "woman_assisted_yrs",
    "roi_seed",
    "inflation_seed",
    "roi_mean_shift",
    "roi_vol_multiplier",
    "roi_mean_reversion",
    "inflation_mean_shift",
    "inflation_vol_multiplier",
    "inflation_mean_reversion",
]


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
        default=None,
        help="Ticker symbol for ROI history download. If omitted, uses value from CSV.",
    )
    parser.add_argument(
        "--current-date",
        default=None,
        help="Historical data cutoff date YYYY-MM-DD. If omitted, uses value from CSV.",
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


def load_scenario_from_csv(csv_path: str, run_id: int) -> tuple[LhsScenario, dict]:
    """
    Read a stochastic LHS row from the CSV and reconstruct its LhsScenario and context.
    Returns (scenario, context_dict) where context_dict has all context fields.
    """
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
    scenario = LhsScenario(
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
    
    # Extract context fields from CSV
    context_dict = {
        "ticker": str(r["ticker"]),
        "current_date": str(r["current_date"]),
        "history_years": int(float(r["history_years"])),
        "al_cum_running_avg_yrs": float(r["al_cum_running_avg_yrs"]),
        "start_clock": str(r["start_clock"]),
        "man_dob": str(r["man_dob"]),
        "woman_dob": str(r["woman_dob"]),
        "constant_monthly_roi": float(r["constant_monthly_roi"]) if pd.notna(r["constant_monthly_roi"]) else None,
        "constant_monthly_cpi": float(r["constant_monthly_cpi"]) if pd.notna(r["constant_monthly_cpi"]) else None,
    }
    
    return scenario, context_dict


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


def _format_replay_case_block(case_name: str, scenario: LhsScenario) -> str:
    lines = [f'    "{case_name}": {{']
    for key in REPLAY_FIELD_ORDER:
        value = getattr(scenario, key)
        if isinstance(value, int):
            value_text = str(value)
        else:
            value_text = repr(float(value))
        lines.append(f'        "{key}": {value_text},')
    lines.append("    },")
    return "\n".join(lines)


def upsert_replay_case_definition(run_id: int, scenario: LhsScenario) -> Path:
    """Insert REPLAY_CASES['REPLAY_<run_id>'] at the bottom of replay_case.py."""
    case_name = f"REPLAY_{run_id}"
    file_path = REPLAY_CASE_FILE
    if not file_path.exists():
        raise FileNotFoundError(f"Expected replay case file not found: '{file_path}'")

    content = file_path.read_text(encoding="utf-8")
    block = _format_replay_case_block(case_name, scenario)

    entry_pattern = re.compile(
        rf'(^\s*"{re.escape(case_name)}"\s*:\s*\{{\n(?:.*\n)*?^\s*\}},\n?)',
        flags=re.MULTILINE,
    )
    # Remove existing entry (if any), then always append the refreshed block at the end.
    content_wo_entry = entry_pattern.sub("", content)
    marker = "REPLAY_CASES: dict[str, dict[str, float | int]] = {"
    marker_index = content_wo_entry.find(marker)
    if marker_index == -1:
        raise ValueError("Could not find REPLAY_CASES dictionary in replay_case.py")
    close_index = content_wo_entry.find("\n}", marker_index)
    if close_index == -1:
        raise ValueError("Could not find REPLAY_CASES closing brace in replay_case.py")
    insertion = ("\n" if content_wo_entry[close_index - 1] != "\n" else "") + block + "\n"
    updated = content_wo_entry[:close_index] + insertion + content_wo_entry[close_index:]

    # Validate generated Python before writing.
    ast.parse(updated)
    file_path.write_text(updated, encoding="utf-8")
    return file_path


def main() -> None:
    args = parse_args()

    run_id: int = args.run_id if args.run_id is not None else prompt_for_run_id(args.lhs_csv)

    print(f"Loading run_id={run_id} from '{args.lhs_csv}' ...")
    scenario, context_dict = load_scenario_from_csv(args.lhs_csv, run_id)
    print(f"Scenario and context parameters loaded — running replay.")

    # Override context fields with command-line args if provided
    if args.ticker is not None:
        context_dict["ticker"] = args.ticker
    if args.current_date is not None:
        context_dict["current_date"] = args.current_date

    current_date = pd.Timestamp(context_dict["current_date"]).normalize()
    context = ScenarioRunContext(
        ticker=context_dict["ticker"],
        current_date=current_date,
        history_years=context_dict["history_years"],
        al_cum_running_avg_yrs=context_dict["al_cum_running_avg_yrs"],
        start_clock=context_dict["start_clock"],
        man_dob=context_dict["man_dob"],
        woman_dob=context_dict["woman_dob"],
        constant_monthly_roi=context_dict["constant_monthly_roi"],
        constant_monthly_cpi=context_dict["constant_monthly_cpi"],
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
        f"Ticker:                     {context.ticker}\n"
        f"Effective APY return:       {annualized_mean:.2%}\n"
        f"Monthly volatility:         {roi.monthly_volatility:.2%}\n"
        f"ROI seed:                   {scenario.roi_seed}\n"
        f"Inflation seed:             {scenario.inflation_seed}\n"
        f"CPI current date:           {current_date.date()}\n"
        f"Effective annualized CPI:   {annualized_mean_cpi:.2%}\n"
        f"Cum. inflation of $1 since {context.start_clock}: ${cpi.life_horizon_inflation_cum[-1]:.4f}"
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
        ("final worth_norm",         result.worth_norm_cc, result.worth_norm_lc),
    ]

    print(f"\n{'item':<28}{'cc':>15}{'lc':>15}")
    print(f"{'-' * 28}{'-' * 15}{'-' * 15}")
    for item, cc_value, lc_value in header_rows:
        print(f"{item:<28}{cc_value:>15.1f}{lc_value:>15.1f}")
    print(f"{'-' * 28}{'-' * 15}{'-' * 15}")
    for item, cc_value, lc_value in table_rows:
        print(f"{item:<28}{cc_value:>15,.0f}{lc_value:>15,.0f}")
    replay_case_path = upsert_replay_case_definition(run_id=run_id, scenario=scenario)
    print(f"Replay edge-case definition updated in '{replay_case_path.name}' as REPLAY_CASES['REPLAY_{run_id}'].")

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


