import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from Inflation import plot_inflation_views
from Roi import TICKER, plot_projection_views
from Taylor import LhsScenario, ScenarioRunContext, TaylorLife, evaluate_lhs_scenario
from default_case import (
    AL_ESC_RUNNING_AVG_YRS,
    DEFAULT_CURRENT_DATE,
    DEFAULT_SEED,
    HISTORY_YEARS,
    MAN_DOB,
    START_CLOCK,
    WOMAN_DOB,
)



def plot_taylor_life_exp_non_taylor(this_life: TaylorLife, show: bool = True) -> None:
    if not this_life.exp_non_taylor_history:
        this_life.calc_result()

    dates = pd.DatetimeIndex(this_life.dates)
    num_il = np.asarray(this_life.num_il_1, dtype=float) + np.asarray(this_life.num_il_2, dtype=float)
    num_al = np.asarray(this_life.num_al_1, dtype=float) + np.asarray(this_life.num_al_2, dtype=float)
    num_non_taylor = (
        np.asarray(this_life.num_non_taylor_1, dtype=float) + np.asarray(this_life.num_non_taylor_2, dtype=float)
    )

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(14, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    axis_top, axis_bottom = axes
    axis_top.plot(
        pd.DatetimeIndex(this_life.dates),
        this_life.exp_non_taylor_history,
        linewidth=2.0,
        label="exp_non_taylor",
    )
    axis_top.plot(
        dates,
        this_life.exp_norm_non_taylor,
        linewidth=2.0,
        linestyle="--",
        label="exp_norm_non_taylor",
    )
    axis_top.plot(
        dates,
        this_life.exp_al_cc_history,
        linewidth=2.0,
        label="exp_al_cc",
    )
    axis_top.plot(
        dates,
        this_life.exp_norm_al_cc,
        linewidth=2.0,
        linestyle="--",
        label="exp_norm_al_cc",
    )
    axis_top.plot(
        dates,
        this_life.exp_cc_history,
        linewidth=2.0,
        label="exp_cc",
    )
    axis_top.plot(
        dates,
        this_life.exp_norm_cc,
        linewidth=2.0,
        linestyle="--",
        label="exp_norm_cc",
    )
    axis_top.plot(
        dates,
        this_life.exp_lc_history,
        linewidth=2.0,
        label="exp_lc",
    )
    axis_top.plot(
        dates,
        this_life.exp_norm_lc,
        linewidth=2.0,
        linestyle="--",
        label="exp_norm_lc",
    )
    axis_top.plot(
        dates,
        this_life.exp_total_cc_history,
        linewidth=4.0,
        label="exp_total_cc",
    )
    axis_top.plot(
        dates,
        this_life.exp_norm_total_cc,
        linewidth=4.0,
        linestyle="--",
        label="exp_norm_total_cc",
    )
    axis_top.plot(
        dates,
        this_life.exp_total_lc_history,
        linewidth=4.0,
        label="exp_total_lc",
    )
    axis_top.plot(
        dates,
        this_life.exp_norm_total_lc,
        linewidth=4.0,
        linestyle="--",
        label="exp_norm_total_lc",
    )
    axis_top.plot(
        dates,
        this_life.worth_lc_history,
        linewidth=2.0,
        label="worth_lc",
    )
    axis_top.plot(
        dates,
        this_life.worth_norm_lc_history,
        linewidth=2.0,
        linestyle="--",
        label="worth_norm_lc",
    )
    axis_top.plot(
        dates,
        this_life.worth_cc_history,
        linewidth=2.0,
        label="worth_cc",
    )
    axis_top.plot(
        dates,
        this_life.worth_norm_cc_history,
        linewidth=2.0,
        linestyle="--",
        label="worth_norm_cc",
    )
    axis_top.plot(
        dates,
        this_life.earn_lc_history,
        linewidth=2.0,
        label="earn_lc",
    )
    axis_top.plot(
        dates,
        this_life.earn_norm_lc_history,
        linewidth=2.0,
        linestyle="--",
        label="earn_norm_lc",
    )
    axis_top.plot(
        dates,
        this_life.earn_cc_history,
        linewidth=2.0,
        label="earn_cc",
    )
    axis_top.plot(
        dates,
        this_life.earn_norm_cc_history,
        linewidth=2.0,
        linestyle="--",
        label="earn_norm_cc",
    )
    axis_top.set_ylabel("Expense")
    axis_top.set_title("Taylor Life Expenses Over Time")
    axis_top.grid(True, alpha=0.3)
    axis_top.legend(loc="upper left")

    axis_bottom.plot(dates, num_il, linewidth=2.0, label="num_il")
    axis_bottom.plot(dates, num_al, linewidth=2.0, label="num_al")
    axis_bottom.plot(dates, num_non_taylor, linewidth=2.0, label="num_non_taylor")
    axis_bottom.set_xlabel("Date")
    axis_bottom.set_ylabel("Count")
    axis_bottom.grid(True, alpha=0.3)
    axis_bottom.legend(loc="upper left")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    current_date = pd.Timestamp(args.current_date).normalize()
    scenario = LhsScenario(
        roi_seed=args.seed,
        inflation_seed=args.seed,
    )
    context = ScenarioRunContext(
        ticker=args.ticker,
        current_date=current_date,
        history_years=HISTORY_YEARS,
        al_cum_running_avg_yrs=AL_ESC_RUNNING_AVG_YRS,
        start_clock=START_CLOCK,
        man_dob=MAN_DOB,
        woman_dob=WOMAN_DOB,
    )
    this_life, result = evaluate_lhs_scenario(scenario=scenario, context=context)
    roi = this_life.roi
    cpi = this_life.cpi
    annualized_inflation = cpi.annualized_inflation
    if cpi.inflation_frame is None:
        raise ValueError("Inflation history was not loaded during projection.")
    inflation_frame = cpi.inflation_frame
    worth_lc = result.worth_lc
    worth_cc = result.worth_cc

    annualized_mean = (1 + roi.monthly_mean_return) ** 12 - 1
    annualized_mean_cpi = annualized_inflation
    print(
        f"Ticker: {args.ticker}\n"
        f"Historical monthly mean return: {roi.monthly_mean_return:.2%}\n"
        f"Implied annualized return: {annualized_mean:.2%}\n"
        f"Monthly volatility: {roi.monthly_volatility:.2%}\n"
        f"ROI seed: {scenario.roi_seed}\n"
        f"Inflation seed: {scenario.inflation_seed}\n"
        f"CPI current date: {current_date.date()}\n"
        f"Implied annualized CPI inflation: {annualized_mean_cpi:.2%}\n"
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
    header_rows = [
        ("man age to al", this_life.man_age_to_al, this_life.man_age_to_al),
        ("man age at death", this_life.man_age_at_death, this_life.man_age_at_death),
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
    
    # Write monthly results to CSV
    df = pd.DataFrame({
        'date': pd.to_datetime(this_life.dates),
        'earn_lc': this_life.earn_lc_history,
        'earn_cc': this_life.earn_cc_history,
        'earn_norm_lc': this_life.earn_norm_lc_history,
        'earn_norm_cc': this_life.earn_norm_cc_history,
        'exp_total_lc': this_life.exp_total_lc_history,
        'exp_total_cc': this_life.exp_total_cc_history,
        'exp_norm_total_lc': this_life.exp_norm_total_lc,
        'exp_norm_total_cc': this_life.exp_norm_total_cc,
        'worth_lc': this_life.worth_lc_history,
        'worth_cc': this_life.worth_cc_history,
        'worth_norm_lc': this_life.worth_norm_lc_history,
        'worth_norm_cc': this_life.worth_norm_cc_history,
        'exp_al_cc': this_life.exp_al_cc_history,
        'exp_cc': this_life.exp_cc_history,
        'exp_lc': this_life.exp_lc_history,
        'exp_non_taylor': this_life.exp_non_taylor_history,
    })
    df.to_csv('taylor_life_monthly.csv', index=False)
    
    if roi.return_frame is None:
        raise ValueError("ROI history was not loaded during projection.")
    plot_projection_views(roi.return_frame, roi, show=False)
    plot_inflation_views(inflation_frame, cpi, show=False)
    plot_taylor_life_exp_non_taylor(this_life, show=False)
    plt.show()


if __name__ == "__main__":
    main()
