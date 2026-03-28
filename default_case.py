import pandas as pd


def _age(date: str, birth_date: str) -> float:
	return float((pd.Timestamp(date) - pd.Timestamp(birth_date)).days / 365.2425)


# Fixed parameters
HISTORY_YEARS = 25
AL_ESC_RUNNING_AVG_YRS = 2
START_CLOCK = "2026-07-01"
DEFAULT_CURRENT_DATE = "2026-03-27"
MAN_DOB = "1957-07-26"
WOMAN_DOB = "1956-04-11"
PILE_AT_START = 5700000.0
NON_TAYLOR_2 = 9612.0 / 2.0
NON_TAYLOR_1 = 5492.0
AL_CC_1 = 9200.0
AL_CC_2 = AL_CC_1 * 2.0
CC_1 = 3150.0
CC_2 = 3750.0 / 2.0
LC_1 = 8100.0
LC_2 = 9600.0 / 2.0
CONSTANT_MONTHLY_ROI: float | None = 10.0 / 100.0 / 12.0  # Fraction per month
CONSTANT_MONTHLY_CPI: float | None = 5.0 / 100.0 / 12.0  # Fraction per month
# CONSTANT_MONTHLY_ROI: float | None = None  # Fraction per month
# CONSTANT_MONTHLY_CPI: float | None = None  # Fraction per month
AL_AND_LC_INFLATION_FACTOR = 2.0  # LTC escalates at 2x inflation

# To be varied
MAN_INDEPENDENCE_YRS = 79.0 - _age(START_CLOCK, MAN_DOB)
WOMAN_INDEPENDENCE_YRS = 80.29 - _age(START_CLOCK, WOMAN_DOB)
MAN_LINGER = 5.0
WOMAN_LINGER = 5.0
DEFAULT_SEED = 0
ROI_MEAN_SHIFT = 0.0
ROI_VOL_MULTIPLIER = 1.0
ROI_MEAN_REVERSION = 0.15
INFLATION_MEAN_SHIFT = 0.0
INFLATION_VOL_MULTIPLIER = 1.0
INFLATION_MEAN_REVERSION = 0.15

