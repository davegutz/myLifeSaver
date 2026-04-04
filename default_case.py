import pandas as pd


def age_at_date_for_dob(date: str, birth_date: str) -> float:
	return float((pd.Timestamp(date) - pd.Timestamp(birth_date)).days / 365.2425)


def apy_percent_to_monthly_fraction(apy_percent: float) -> float:
	return (1.0 + apy_percent / 100.0) ** (1.0 / 12.0) - 1.0


# Fixed parameters
HISTORY_YEARS = 25
AL_ESC_RUNNING_AVG_YRS = 2
START_CLOCK = "2026-07-01"
DEFAULT_CURRENT_DATE = "2026-03-27"
MAN_DOB = "1957-07-26"
WOMAN_DOB = "1956-04-11"
PILE_AT_START = 6300000.0
NON_TAYLOR_2 = 9612.0 / 2.0
NON_TAYLOR_1 = 5492.0
ENTRANCE_FEE_CC = 481000.
ENTRANCE_FEE_LC = 900000.
AL_CC_1 = 9200.0
AL_CC_2 = AL_CC_1 * 2.0
CC_1 = 3150.0
CC_2 = 3750.0 / 2.0
LC_1 = 8100.0
LC_2 = 9600.0 / 2.0
# CONSTANT_ROI_APY_PERCENT = 10.0
# CONSTANT_CPI_APY_PERCENT = 5.0
# CONSTANT_MONTHLY_ROI: float | None = apy_percent_to_monthly_fraction(CONSTANT_ROI_APY_PERCENT)
# CONSTANT_MONTHLY_CPI: float | None = apy_percent_to_monthly_fraction(CONSTANT_CPI_APY_PERCENT)
CONSTANT_MONTHLY_ROI: float | None = None  # Fraction per month
CONSTANT_MONTHLY_CPI: float | None = None  # Fraction per month
AL_INFLATION_FACTOR = 2.0  # AL escalates at 2x inflation
LC_INFLATION_FACTOR = 1.5  # LC escalates at 1.5x inflation
P_MAN_AL = 0.7
P_WOMAN_AL = 0.7
# P_MAN_AL = 1.  # disable the Pal logic
# P_WOMAN_AL = 1.  # disable the Pal logic

# To be varied
MAN_INDEPENDENT_YRS = 10.
WOMAN_INDEPENDENT_YRS = MAN_INDEPENDENT_YRS
MAN_ASSISTED_YRS = 5.0
WOMAN_ASSISTED_YRS = 5.0
DEFAULT_SEED = 0
ROI_MEAN_SHIFT = 0.0
ROI_VOL_MULTIPLIER = 1.0
ROI_MEAN_REVERSION = 0.15
INFLATION_MEAN_SHIFT = 0.0
INFLATION_VOL_MULTIPLIER = 1.0
INFLATION_MEAN_REVERSION = 0.15


DEFAULT_CASES: dict[str, dict[str, dict[str, object]]] = {
	"RUN_ONE_PRESENT": {
		"scenario": {
			"man_independent_yrs": MAN_INDEPENDENT_YRS,
			"woman_independent_yrs": WOMAN_INDEPENDENT_YRS,
			"man_assisted_yrs": MAN_ASSISTED_YRS,
			"woman_assisted_yrs": WOMAN_ASSISTED_YRS,
			"roi_mean_shift": ROI_MEAN_SHIFT,
			"roi_vol_multiplier": ROI_VOL_MULTIPLIER,
			"roi_mean_reversion": ROI_MEAN_REVERSION,
			"inflation_mean_shift": INFLATION_MEAN_SHIFT,
			"inflation_vol_multiplier": INFLATION_VOL_MULTIPLIER,
			"inflation_mean_reversion": INFLATION_MEAN_REVERSION,
		},
		"context": {
			"history_years": HISTORY_YEARS,
			"al_cum_running_avg_yrs": AL_ESC_RUNNING_AVG_YRS,
			"start_clock": START_CLOCK,
			"man_dob": MAN_DOB,
			"woman_dob": WOMAN_DOB,
			"constant_monthly_roi": CONSTANT_MONTHLY_ROI,
			"constant_monthly_cpi": CONSTANT_MONTHLY_CPI,
		},
	},
	"DEFAULT": {
		"scenario": {
			"man_independent_yrs": MAN_INDEPENDENT_YRS,
			"woman_independent_yrs": WOMAN_INDEPENDENT_YRS,
			"man_assisted_yrs": MAN_ASSISTED_YRS,
			"woman_assisted_yrs": WOMAN_ASSISTED_YRS,
			"roi_mean_shift": ROI_MEAN_SHIFT,
			"roi_vol_multiplier": ROI_VOL_MULTIPLIER,
			"roi_mean_reversion": ROI_MEAN_REVERSION,
			"inflation_mean_shift": INFLATION_MEAN_SHIFT,
			"inflation_vol_multiplier": INFLATION_VOL_MULTIPLIER,
			"inflation_mean_reversion": INFLATION_MEAN_REVERSION,
		},
		"context": {
			"history_years": HISTORY_YEARS,
			"al_cum_running_avg_yrs": AL_ESC_RUNNING_AVG_YRS,
			"start_clock": START_CLOCK,
			"man_dob": MAN_DOB,
			"woman_dob": WOMAN_DOB,
			"constant_monthly_roi": CONSTANT_MONTHLY_ROI,
			"constant_monthly_cpi": CONSTANT_MONTHLY_CPI,
		},
	},
}


def default_case_names() -> list[str]:
	return sorted(DEFAULT_CASES)


def load_default_case(case_name: str) -> tuple[dict[str, object], dict[str, object]]:
	try:
		case = DEFAULT_CASES[case_name]
	except KeyError as exc:
		available = ", ".join(default_case_names())
		raise ValueError(f"Unknown default case '{case_name}'. Available cases: {available}") from exc
	scenario = dict(case["scenario"])
	if "man_independence_yrs" in scenario and "man_independent_yrs" not in scenario:
		scenario["man_independent_yrs"] = scenario.pop("man_independence_yrs")
	if "woman_independence_yrs" in scenario and "woman_independent_yrs" not in scenario:
		scenario["woman_independent_yrs"] = scenario.pop("woman_independence_yrs")
	if "man_linger" in scenario and "man_assisted_yrs" not in scenario:
		scenario["man_assisted_yrs"] = scenario.pop("man_linger")
	if "woman_linger" in scenario and "woman_assisted_yrs" not in scenario:
		scenario["woman_assisted_yrs"] = scenario.pop("woman_linger")
	return scenario, dict(case["context"])


