"""
Replay_LHS_Gutz_Case.py

Replays a single stochastic case from a previous Run_LHS_Gutz_Taylor run.
Provide the integer run_id (case number printed in the LHS table) and
the path to the LHS CSV.  Every scenario parameter is read directly
from the CSV row so the run is bit-for-bit identical to the original.

Usage:
    python Replay_LHS_Gutz_Case.py 42
    python Replay_LHS_Gutz_Case.py 42 --lhs-csv path/to/lhs_gutz_taylor_results.csv
    python Replay_LHS_Gutz_Case.py 42 --ticker SPY --current-date 2026-03-29
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from Inflation import plot_inflation_views
from Roi import plot_projection_views
from default_case import PILE_AT_START
from Taylor import ScenarioRunContext
from utils import evaluate_lhs_scenario, plot_taylor_life_exp_non_taylor
from Replay_LHS_Case import (
    prompt_for_run_id,
    load_scenario_from_csv,
    monthly_rate_to_apy,
    realized_monthly_rate,
    upsert_replay_case_definition,
)

# Default path written by Run_LHS_Gutz_Taylor.py
DEFAULT_LHS_CSV = "lhs_gutz_taylor_results.csv"
REPLAY_CASE_FILE = Path(__file__).with_name("replay_gutz_case.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay a single stochastic case from a previous Run_LHS_Gutz_Taylor run."
    )
    parser.add_argument(
        "run_id",
        type=int,
        nargs="?",
        default=None,
        help="Integer run_id (case number) from the LHS Gutz results CSV. "
             "If omitted, available IDs are listed and you will be prompted.",
    )
    parser.add_argument(
        "--lhs-csv",
        default=DEFAULT_LHS_CSV,
        help=f"Path to the LHS Gutz results CSV produced by Run_LHS_Gutz_Taylor.py. "
             f"Default: {DEFAULT_LHS_CSV}",
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

def main() -> None:
    args = parse_args()

    run_id: int = args.run_id if args.run_id is not None else prompt_for_run_id(args.lhs_csv)

    print(f"Loading run_id={run_id} from '{args.lhs_csv}' ...")
    scenario, context_dict = load_scenario_from_csv(args.lhs_csv, run_id)
    print(f"Scenario and context parameters loaded — running Gutz replay.")

    if args.ticker is not None:
        context_dict["ticker"] = args.ticker
    if args.current_date is not None:
        context_dict["current_date"] = args.current_date

    current_date = pd.Timestamp(context_dict["current_date"])
    assert isinstance(current_date, pd.Timestamp), f"Invalid current_date: {args.current_date}"
    current_date = current_date.normalize()
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
        f"\n=== Gutz Replay of LHS run_id={run_id} from '{args.lhs_csv}' ===\n"
        f"Ticker:                     {context.ticker}\n"
        f"Effective APY return:       {annualized_mean:.2%}\n"
        f"Monthly volatility:         {roi.monthly_volatility:.2%}\n"
        f"ROI seed:                   {scenario.roi_seed}\n"
        f"Inflation seed:             {scenario.inflation_seed}\n"
        f"Man goes-to-AL seed:        {scenario.man_goes_to_al_seed}  -> {this_life.man_goes_to_al}\n"
        f"Woman goes-to-AL seed:      {scenario.woman_goes_to_al_seed}  -> {this_life.woman_goes_to_al}\n"
        f"CPI current date:           {current_date.date()}\n"
        f"Effective annualized CPI:   {annualized_mean_cpi:.2%}\n"
        f"Cum. inflation of $1 since {context.start_clock}: ${cpi.life_horizon_inflation_cum[-1]:.4f}"
    )

    header_rows = [
        ("apy roi %",            annualized_mean * 100.0,      annualized_mean * 100.0),
        ("apy cpi %",            annualized_mean_cpi * 100.0,  annualized_mean_cpi * 100.0),
        ("roi_one_dollar_at_end", roi.life_horizon_roi_cum[-1], roi.life_horizon_roi_cum[-1]),
        ("cpi_one_dollar_at_end", cpi.life_horizon_inflation_cum[-1], cpi.life_horizon_inflation_cum[-1]),
        ("norm_one_dollar_at_end", cpi.life_horizon_inflation_cum[-1], cpi.life_horizon_inflation_cum[-1]),
        ("man independent yrs",  this_life.man_independent_yrs,  this_life.man_independent_yrs),
        ("man assisted yrs",     this_life.man_assisted_yrs,     this_life.man_assisted_yrs),
        ("man age to al",        this_life.man_age_to_al,        this_life.man_age_to_al),
        ("man age at death",     this_life.man_age_at_death,     this_life.man_age_at_death),
        ("woman independent yrs",this_life.woman_independent_yrs,this_life.woman_independent_yrs),
        ("woman assisted yrs",   this_life.woman_assisted_yrs,   this_life.woman_assisted_yrs),
        ("woman age to al",      this_life.woman_age_to_al,      this_life.woman_age_to_al),
        ("woman age at death",   this_life.woman_age_at_death,   this_life.woman_age_at_death),
        ("yrs il double",        min(this_life.man_independent_yrs, this_life.woman_independent_yrs),
                                 min(this_life.man_independent_yrs, this_life.woman_independent_yrs)),
        ("CC_2",                 this_life.initial_cc_2 * 12.0, 0.0),
        ("LC_2",                 0.0, this_life.initial_lc_2 * 12.0),
        ("yrs il single",        abs(this_life.man_independent_yrs - this_life.woman_independent_yrs),
                                 abs(this_life.man_independent_yrs - this_life.woman_independent_yrs)),
        ("CC_1_1",               this_life.initial_cc_1_1 * 12.0, 0.0),
        ("CC_1_2",               this_life.initial_cc_1_2 * 12.0, 0.0),
        ("LC_1_1",               0.0, this_life.initial_lc_1_1 * 12.0),
        ("LC_1_2",               0.0, this_life.initial_lc_1_2 * 12.0),
        ("yrs al double",        min(this_life.man_assisted_yrs, this_life.woman_assisted_yrs),
                                 min(this_life.man_assisted_yrs, this_life.woman_assisted_yrs)),
        ("AL_CC_2",              this_life.initial_al_cc_2 * 12.0, 0.0),
        ("yrs al single",        abs(this_life.man_assisted_yrs - this_life.woman_assisted_yrs),
                                 abs(this_life.man_assisted_yrs - this_life.woman_assisted_yrs)),
        ("AL_CC_1_1",            this_life.initial_al_cc_1_1 * 12.0, 0.0),
        ("AL_CC_1_2",            this_life.initial_al_cc_1_2 * 12.0, 0.0),
        ("exp_entrance_fee_cc",  this_life.entrance_fee_cc, 0.0),
        ("exp_entrance_fee_lc",  0.0, this_life.entrance_fee_lc),
        ("SS_MAN mo",            this_life.ss_man, this_life.ss_man),
        ("SS_WOMAN mo",          this_life.ss_woman, this_life.ss_woman),
        ("PEN_MAN mo",           this_life.pen_man, this_life.pen_man),
        ("PEN_WOMAN mo",         this_life.pen_woman, this_life.pen_woman),
        ("yrs sum al",           this_life.man_assisted_yrs + this_life.woman_assisted_yrs,
                                 this_life.man_assisted_yrs + this_life.woman_assisted_yrs),
    ]

    def _last(lst: list) -> float:
        return float(lst[-1]) if lst else 0.0

    table_rows = [
        ("start_pile",             PILE_AT_START, PILE_AT_START),
        ("",                       None, None),
        ("cum_mo_earn_ss_norm",  _last(this_life.cum_mo_earn_ss_norm),  _last(this_life.cum_mo_earn_ss_norm)),
        ("cum_mo_earn_pen_norm", _last(this_life.cum_mo_earn_pen_norm), _last(this_life.cum_mo_earn_pen_norm)),
        ("cum_mo_earn_inv_norm", _last(this_life.cum_mo_earn_inv_cc_norm), _last(this_life.cum_mo_earn_inv_lc_norm)),
        ("---",                    None, None),
        ("cum_mo_earn_norm",       _last(this_life.cum_mo_earn_cc_norm), _last(this_life.cum_mo_earn_lc_norm)),
        ("",                       None, None),
        ("exp_entrance_fee",       this_life.entrance_fee_cc, this_life.entrance_fee_lc),
        ("cum_mo_exp_norm",        _last(this_life.cum_mo_exp_cc_norm), _last(this_life.cum_mo_exp_lc_norm)),
        ("cum_mo_exp_al_norm",     _last(this_life.cum_mo_exp_al_cc_norm), 0.0),
        ("cum_mo_exp_non_taylor_norm", _last(this_life.cum_mo_exp_non_taylor_norm), _last(this_life.cum_mo_exp_non_taylor_norm)),
        ("---",                    None, None),
        ("cum_mo_exp_total_norm",  _last(this_life.cum_mo_exp_total_cc_norm), _last(this_life.cum_mo_exp_total_lc_norm)),
        ("",                       None, None),
        ("final worth norm",
         PILE_AT_START + _last(this_life.cum_mo_earn_cc_norm) - _last(this_life.cum_mo_exp_total_cc_norm),
         PILE_AT_START + _last(this_life.cum_mo_earn_lc_norm) - _last(this_life.cum_mo_exp_total_lc_norm)),
    ]

    print(f"\n{'item':<28}{'cc':>15}{'lc':>15}")
    print(f"{'-' * 28}{'-' * 15}{'-' * 15}")
    for item, cc_value, lc_value in header_rows:
        print(f"{item:<28}{cc_value:>15.1f}{lc_value:>15.1f}")
    print(f"{'-' * 28}{'-' * 15}{'-' * 15}")
    for item, cc_value, lc_value in table_rows:
        if cc_value is None:
            if item == "---":
                print(f"{'-' * 28}{'-' * 15}{'-' * 15}")
            else:
                print()
        else:
            print(f"{item:<28}{cc_value:>15,.0f}{lc_value:>15,.0f}")

    worth_norm_cc = PILE_AT_START + _last(this_life.cum_mo_earn_cc_norm) - _last(this_life.cum_mo_exp_total_cc_norm)
    worth_norm_lc = PILE_AT_START + _last(this_life.cum_mo_earn_lc_norm) - _last(this_life.cum_mo_exp_total_lc_norm)
    added_lc_worth_norm = worth_norm_lc - worth_norm_cc
    print(f"\nadded worth (norm lc - norm cc): {added_lc_worth_norm:>15,.0f}")

    replay_case_path = upsert_replay_case_definition(
        run_id=run_id,
        scenario=scenario,
        file_path=REPLAY_CASE_FILE,
        dict_name="REPLAY_CASES_GUTZ",
        case_prefix="REPLAY_GUTZ",
    )
    print(
        f"Replay edge-case definition updated in '{replay_case_path.name}' "
        f"as REPLAY_CASES_GUTZ['REPLAY_GUTZ_{run_id}']."
    )
    print(
        f"\nTo include this case in the next Run_LHS_Gutz_Taylor run, add "
        f"'REPLAY_GUTZ_{run_id}' to build_replay_case_scenarios_gutz() in edges.py "
        f"(it was auto-written to replay_gutz_case.py under that key)."
    )

    # Write monthly CSV
    out_csv = f"replay_lhs_gutz_case_monthly_{run_id}.csv"

    monthly_apy_roi = (1.0 + np.asarray(this_life.roi.life_horizon_roi)) ** 12 - 1.0
    monthly_apy_cpi = (1.0 + np.asarray(this_life.cpi.life_horizon_inflation)) ** 12 - 1.0

    df_monthly = pd.DataFrame({
        "date":               pd.to_datetime(this_life.dates),
        "apy_roi":            monthly_apy_roi * 100.0,
        "apy_cpi":            monthly_apy_cpi * 100.0,
        "num_al":             (np.asarray(this_life.num_al_1_1) + np.asarray(this_life.num_al_1_2) + np.asarray(this_life.num_al_2)).tolist(),
        "earn_norm_lc":               this_life.earn_norm_lc_history,
        "earn_norm_cc":               this_life.earn_norm_cc_history,
        "mo_earn_lc_norm":            this_life.mo_earn_lc_norm,
        "cum_mo_earn_lc_norm":        this_life.cum_mo_earn_lc_norm,
        "mo_earn_cc_norm":            this_life.mo_earn_cc_norm,
        "cum_mo_earn_cc_norm":        this_life.cum_mo_earn_cc_norm,
        "mo_earn_inv_lc_norm":        this_life.mo_earn_inv_lc_norm,
        "cum_mo_earn_inv_lc_norm":    this_life.cum_mo_earn_inv_lc_norm,
        "mo_earn_inv_cc_norm":        this_life.mo_earn_inv_cc_norm,
        "cum_mo_earn_inv_cc_norm":    this_life.cum_mo_earn_inv_cc_norm,
        "mo_earn_ss_man_norm":        this_life.mo_earn_ss_man_norm,
        "cum_mo_earn_ss_man_norm":    this_life.cum_mo_earn_ss_man_norm,
        "mo_earn_ss_woman_norm":      this_life.mo_earn_ss_woman_norm,
        "cum_mo_earn_ss_woman_norm":  this_life.cum_mo_earn_ss_woman_norm,
        "mo_earn_ss_norm":            this_life.mo_earn_ss_norm,
        "cum_mo_earn_ss_norm":        this_life.cum_mo_earn_ss_norm,
        "mo_earn_pen_man_norm":       this_life.mo_earn_pen_man_norm,
        "cum_mo_earn_pen_man_norm":   this_life.cum_mo_earn_pen_man_norm,
        "mo_earn_pen_woman_norm":     this_life.mo_earn_pen_woman_norm,
        "cum_mo_earn_pen_woman_norm": this_life.cum_mo_earn_pen_woman_norm,
        "mo_earn_pen_norm":           this_life.mo_earn_pen_norm,
        "cum_mo_earn_pen_norm":       this_life.cum_mo_earn_pen_norm,
        "exp_norm_total_lc":  this_life.exp_norm_total_lc,
        "exp_norm_total_cc":  this_life.exp_norm_total_cc,
        "worth_norm_lc":      this_life.worth_norm_lc_history,
        "worth_norm_cc":      this_life.worth_norm_cc_history,
        "added_lc_worth_norm":[added_lc_worth_norm] * len(this_life.dates),
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
