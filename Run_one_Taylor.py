import argparse
from dataclasses import asdict
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


def merge_run_config(*configs: dict[str, dict[str, object]] | None) -> dict[str, dict[str, object]]:
    merged: dict[str, dict[str, object]] = {"scenario": {}, "context": {}}
    for config in configs:
        if config is None:
            continue
        for section in ("scenario", "context"):
            merged[section].update(config.get(section, {}))
    return merged


def normalize_run_one_inputs(run_config: dict[str, dict[str, object]]) -> tuple[LhsScenario, ScenarioRunContext]:
    scenario_kwargs = dict(run_config.get("scenario", {}))
    context_kwargs = dict(run_config.get("context", {}))

    def normalize_constant_rate(key: str) -> None:
        raw = context_kwargs.get(key)
        if raw is None:
            return
        value = float(raw)
        if abs(value) > 1.0:
            monthly = apy_percent_to_monthly_fraction(value)
            print(f"Interpreting {key}={value} as APY percent; using monthly fraction {monthly:.8f}")
            context_kwargs[key] = monthly
        else:
            context_kwargs[key] = value

    normalize_constant_rate("constant_monthly_roi")
    normalize_constant_rate("constant_monthly_cpi")
    if "current_date" in context_kwargs and context_kwargs["current_date"] is not None:
        context_kwargs["current_date"] = pd.Timestamp(context_kwargs["current_date"]).normalize()
    man_goes_to_al_seed = int(scenario_kwargs.pop("man_goes_to_al_seed", DEFAULT_SEED))
    woman_goes_to_al_seed = int(scenario_kwargs.pop("woman_goes_to_al_seed", DEFAULT_SEED))
    if "man_goes_to_al" not in scenario_kwargs:
        scenario_kwargs["man_goes_to_al"] = bool(
            np.random.default_rng(man_goes_to_al_seed).binomial(1, P_MAN_AL)
        )
    if "woman_goes_to_al" not in scenario_kwargs:
        scenario_kwargs["woman_goes_to_al"] = bool(
            np.random.default_rng(woman_goes_to_al_seed).binomial(1, P_WOMAN_AL)
        )
    return LhsScenario(**scenario_kwargs), ScenarioRunContext(**context_kwargs)


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


def run_one(run_config: dict[str, dict[str, object]], active_case_name: str | None = None) -> None:
    scenario, context = normalize_run_one_inputs(run_config)
    current_date = pd.Timestamp(context.current_date).normalize()
    if active_case_name is not None:
        print(f"Using default case '{active_case_name}' from default_case.py")
    print("Resolved run_one inputs:")
    print(f"  scenario: {asdict(scenario)}")
    print(f"  context:  {asdict(context)}")
    this_life, result = evaluate_lhs_scenario(scenario=scenario, context=context)
    roi = this_life.roi
    cpi = this_life.cpi
    annualized_inflation = cpi.annualized_inflation
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
        f"Ticker: {context.ticker}\n"
        f"Effective APY return: {annualized_mean:.2%}\n"
        f"Monthly volatility: {roi.monthly_volatility:.2%}\n"
        f"ROI seed: {scenario.roi_seed}\n"
        f"Inflation seed: {scenario.inflation_seed}\n"
        f"Man goes to AL: {this_life.man_goes_to_al}\n"
        f"Woman goes to AL: {this_life.woman_goes_to_al}\n"
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
        ("yrs il double", min(this_life.man_independent_yrs, this_life.woman_independent_yrs),
                          min(this_life.man_independent_yrs, this_life.woman_independent_yrs)),
        ("CC_2", this_life.initial_cc_2 * 12.0, 0.0),
        ("LC_2", 0.0, this_life.initial_lc_2 * 12.0),
        ("yrs il single", abs(this_life.man_independent_yrs - this_life.woman_independent_yrs),
                          abs(this_life.man_independent_yrs - this_life.woman_independent_yrs)),
        ("CC_1", this_life.initial_cc_1 * 12.0, 0.0),
        ("LC_1", 0.0, this_life.initial_lc_1 * 12.0),
        ("yrs al double", min(this_life.man_assisted_yrs, this_life.woman_assisted_yrs),
                          min(this_life.man_assisted_yrs, this_life.woman_assisted_yrs)),
        ("AL_CC_2", this_life.initial_al_cc_2 * 12.0, 0.0),
        ("yrs al single", abs(this_life.man_assisted_yrs - this_life.woman_assisted_yrs),
                          abs(this_life.man_assisted_yrs - this_life.woman_assisted_yrs)),
        ("AL_CC_1", this_life.initial_al_cc_1 * 12.0, 0.0),
        ("yrs sum al", this_life.man_assisted_yrs + this_life.woman_assisted_yrs,
                       this_life.man_assisted_yrs + this_life.woman_assisted_yrs),
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
    
    added_lc_worth_norm = result.worth_norm_lc - result.worth_norm_cc
    print(f"\nadded worth (norm lc - norm cc): {added_lc_worth_norm:>15,.0f}")

    # Write monthly results to CSV
    df = pd.DataFrame({
        'date': pd.to_datetime(this_life.dates),
        'apy_roi': [annualized_mean * 100.0] * len(this_life.dates),
        'apy_cpi': [annualized_mean_cpi * 100.0] * len(this_life.dates),
        'num_il_2': this_life.num_il_2,
        'num_il_1': this_life.num_il_1,
        'num_al_2': this_life.num_al_2,
        'num_al_1': this_life.num_al_1,
        'num_il': (np.asarray(this_life.num_il_1) + np.asarray(this_life.num_il_2)).tolist(),
        'num_non_taylor_2': this_life.num_non_taylor_2,
        'num_non_taylor_1': this_life.num_non_taylor_1,
        'num_non_taylor': (np.asarray(this_life.num_non_taylor_1) + np.asarray(this_life.num_non_taylor_2)).tolist(),
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
         'added_lc_worth_norm': [result.worth_norm_lc - result.worth_norm_cc] * len(this_life.dates),
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


def main() -> None:
    args = parse_args()
    active_case_name = RUN_ONE_CASE_NAME
    base_run_config = {
        "scenario": {
            "man_independent_yrs": MAN_INDEPENDENT_YRS,
            "woman_independent_yrs": WOMAN_INDEPENDENT_YRS,
            "man_assisted_yrs": MAN_ASSISTED_YRS,
            "woman_assisted_yrs": WOMAN_ASSISTED_YRS,
            "roi_seed": args.seed,
            "inflation_seed": args.seed,
            "roi_mean_shift": ROI_MEAN_SHIFT,
            "roi_vol_multiplier": ROI_VOL_MULTIPLIER,
            "roi_mean_reversion": ROI_MEAN_REVERSION,
            "inflation_mean_shift": INFLATION_MEAN_SHIFT,
            "inflation_vol_multiplier": INFLATION_VOL_MULTIPLIER,
            "inflation_mean_reversion": INFLATION_MEAN_REVERSION,
        },
        "context": {
            "ticker": args.ticker,
            "current_date": args.current_date,
            "history_years": HISTORY_YEARS,
            "al_cum_running_avg_yrs": AL_ESC_RUNNING_AVG_YRS,
            "start_clock": START_CLOCK,
            "man_dob": MAN_DOB,
            "woman_dob": WOMAN_DOB,
            "constant_monthly_roi": CONSTANT_MONTHLY_ROI,
            "constant_monthly_cpi": CONSTANT_MONTHLY_CPI,
        },
    }
    case_run_config = None
    if active_case_name is not None:
        case_scenario_kwargs, case_context_kwargs = load_default_case(active_case_name)
        case_run_config = {
            "scenario": case_scenario_kwargs,
            "context": case_context_kwargs,
        }

    # Hand-edit these local overrides as your normal workflow.
    # Precedence is: base defaults -> named default case -> local overrides.
    local_run_overrides = {
        "scenario": {
            "man_independent_yrs": 10.,  # google age men enter al - current age
            "woman_independent_yrs": 15.5,  # google age women enter al - current age
            "man_assisted_yrs": 2.35,  # google men in al; assume no mc (conservative for yes on lc decision)
            "woman_assisted_yrs": 5.5,  # google women in al; assume no mc (conservative for yes on lc decision)
            "roi_seed": 740264,
            "inflation_seed": 898910,
            # "man_goes_to_al": True,  # Uncomment this to force True
            # "woman_goes_to_al": True,  # Uncomment this to force True
            "roi_mean_shift": 0.0080464851559136,
            "inflation_mean_shift": -0.00459579542225717,
        },
        "context": {
            "ticker": "SPY",
            "current_date": "2026-03-29",
            # "constant_monthly_roi": None,  # was 10. — None → stochastic
            # "constant_monthly_cpi": None,  # was  5. — None → stochas
            # "constant_monthly_roi": 4.,  # was 10. — None → stochastic
            # "constant_monthly_cpi": 5.,  # was  5. — None → stochas
            "constant_monthly_roi": 0.,  # was 10. — None → stochastic
            "constant_monthly_cpi": 0.,  # was  5. — None → stochas
        },
    }

    run_config = merge_run_config(base_run_config, case_run_config, local_run_overrides)
    run_one(run_config=run_config, active_case_name=active_case_name)


if __name__ == "__main__":
    main()
