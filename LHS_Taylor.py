import argparse
from dataclasses import asdict, dataclass
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from default_case import (
    AL_AND_LC_INFLATION_FACTOR,
    AL_CC_1,
    AL_CC_2,
    AL_ESC_RUNNING_AVG_YRS,
    CC_1,
    CC_2,
    CONSTANT_MONTHLY_CPI,
    CONSTANT_MONTHLY_ROI,
    DEFAULT_CURRENT_DATE,
    DEFAULT_SEED,
    HISTORY_YEARS,
    INFLATION_MEAN_REVERSION,
    INFLATION_MEAN_SHIFT,
    INFLATION_VOL_MULTIPLIER,
    LC_1,
    LC_2,
    MAN_AGE_TO_AL,
    MAN_DOB,
    MAN_LINGER,
    NON_TAYLOR_1,
    NON_TAYLOR_2,
    PILE_AT_START,
    ROI_MEAN_REVERSION,
    ROI_MEAN_SHIFT,
    ROI_VOL_MULTIPLIER,
    START_CLOCK,
    WOMAN_AGE_TO_AL,
    WOMAN_DOB,
    WOMAN_LINGER,
)
from Inflation import Inflation, MonthlyInflationPoint, plot_inflation_views
from Roi import MonthlyRoiPoint, TICKER, Roi, plot_projection_views
from utils import age

# ============================================================================
# IMPORTANT: ROI AND INFLATION RATES
# ============================================================================
# CONSTANT_MONTHLY_ROI and CONSTANT_MONTHLY_CPI below determine ROI/inflation
# for ALL edge cases. Currently set to None, meaning stochastic/historical model.
# 
# Current Edge Case ROI/CPI Configuration:
#   - ROI:       None (uses stochastic model from historical data)
#   - Inflation: None (uses stochastic model from historical data)
#
# To use FIXED rates for edge cases, set:
#   - CONSTANT_MONTHLY_ROI = 0.10 / 12  # 10% annual = 0.833% monthly
#   - CONSTANT_MONTHLY_CPI = 0.05 / 12   # 5% annual = 0.417% monthly
# ============================================================================

# Fixed and default varied parameters are imported from default_case.py.
MAN_AGE_TO_AL_RANGE = (69.0, 90.0)
WOMAN_AGE_TO_AL_RANGE = (70.0, 90.0)
MAN_LINGER_RANGE = (0., 10.0)
WOMAN_LINGER_RANGE = (0., 10.0)
SEED_RANGE = (0, 1000000)
ROI_MEAN_SHIFT_RANGE = (-0.01, 0.01)
ROI_VOL_MULTIPLIER_RANGE = (0.5, 1.5)
ROI_MEAN_REVERSION_RANGE = (0.0, 0.5)
INFLATION_MEAN_SHIFT_RANGE = (-0.005, 0.005)
INFLATION_VOL_MULTIPLIER_RANGE = (0.5, 1.5)
INFLATION_MEAN_REVERSION_RANGE = (0.0, 0.5)
DEFAULT_LHS_POINTS = 100


@dataclass(frozen=True)
class LhsScenario:
    man_age_to_al: float = MAN_AGE_TO_AL
    woman_age_to_al: float = WOMAN_AGE_TO_AL
    man_linger: float = MAN_LINGER
    woman_linger: float = WOMAN_LINGER
    roi_seed: int = DEFAULT_SEED
    inflation_seed: int = DEFAULT_SEED
    roi_mean_shift: float = ROI_MEAN_SHIFT
    roi_vol_multiplier: float = ROI_VOL_MULTIPLIER
    roi_mean_reversion: float = ROI_MEAN_REVERSION
    inflation_mean_shift: float = INFLATION_MEAN_SHIFT
    inflation_vol_multiplier: float = INFLATION_VOL_MULTIPLIER
    inflation_mean_reversion: float = INFLATION_MEAN_REVERSION


@dataclass(frozen=True)
class ScenarioRunContext:
    ticker: str = TICKER
    current_date: pd.Timestamp | str = DEFAULT_CURRENT_DATE
    history_years: int = HISTORY_YEARS
    al_cum_running_avg_yrs: int | float = AL_ESC_RUNNING_AVG_YRS
    start_clock: str = START_CLOCK
    man_dob: str = MAN_DOB
    woman_dob: str = WOMAN_DOB


@dataclass(frozen=True)
class TaylorLifeResult:
    worth_lc: int
    worth_norm_lc: int
    worth_cc: int
    worth_norm_cc: int


@dataclass(frozen=True)
class LhsScenarioSummary:
    run_id: int | str
    man_age_to_al: float
    woman_age_to_al: float
    man_linger: float
    woman_linger: float
    roi_seed: int
    inflation_seed: int
    roi_mean_shift: float
    roi_vol_multiplier: float
    roi_mean_reversion: float
    inflation_mean_shift: float
    inflation_vol_multiplier: float
    inflation_mean_reversion: float
    exp_al_cc: float
    exp_norm_al_cc: float
    exp_cc: float
    exp_norm_cc: float
    exp_lc: float
    exp_norm_lc: float
    exp_non_taylor: float
    exp_norm_non_taylor: float
    exp_total_cc: float
    exp_norm_total_cc: float
    earn_cc: float
    earn_norm_cc: float
    earn_lc: float
    earn_norm_lc: float
    worth_lc: int
    worth_norm_lc: int
    worth_cc: int
    worth_norm_cc: int


CSV_COLUMNS = [
    "run_id",
    "man_age_to_al",
    "woman_age_to_al",
    "man_linger",
    "woman_linger",
    "roi_seed",
    "inflation_seed",
    "roi_mean_shift",
    "roi_vol_multiplier",
    "roi_mean_reversion",
    "inflation_mean_shift",
    "inflation_vol_multiplier",
    "inflation_mean_reversion",
    "exp_al_cc",
    "exp_norm_al_cc",
    "exp_cc",
    "exp_norm_cc",
    "exp_lc",
    "exp_norm_lc",
    "exp_non_taylor",
    "exp_norm_non_taylor",
    "exp_total_cc",
    "exp_norm_total_cc",
    "earn_cc",
    "earn_norm_cc",
    "earn_lc",
    "earn_norm_lc",
    "worth_lc",
    "worth_norm_lc",
    "worth_cc",
    "worth_norm_cc",
]

SCREEN_MIN_COL_WIDTH = 14


def format_screen_number(value: int | float, width: int) -> str:
    if isinstance(value, (np.integer, int)):
        text = f"{int(value):.3g}"
    else:
        numeric = float(value)
        if math.isnan(numeric):
            return " " * width
        text = f"{numeric:.3g}"
    if "e" in text or "E" in text:
        mantissa, exponent = text.lower().split("e", maxsplit=1)
        exponent_text = f"e{int(exponent):+03d}"
    else:
        mantissa = text
        exponent_text = ""
    if "." not in mantissa:
        mantissa = f"{mantissa}."
    left, right = mantissa.split(".", maxsplit=1)
    decimal_target = max(2, width - max(len(exponent_text), 4) - 1)
    left_padding = max(decimal_target - len(left), 0)
    aligned = f"{' ' * left_padding}{left}.{right}{exponent_text}"
    return aligned.rjust(width)


def format_screen_cell(value: object, width: int) -> str:
    if isinstance(value, (int, float, np.integer, np.floating)):
        return format_screen_number(value=value, width=width)
    return str(value).rjust(width)


def print_screen_row(row: dict[str, object], columns: list[str], widths: dict[str, int]) -> None:
    print(" ".join(format_screen_cell(row[column], widths[column]) for column in columns))

class TaylorLife:
    def __init__(
        self,
        roi: Roi,
        cpi: Inflation,
        man_dob: str = MAN_DOB,
        woman_dob: str = WOMAN_DOB,
        man_age_to_al: float = MAN_AGE_TO_AL,
        man_linger: float = MAN_LINGER,
        woman_age_to_al: float = WOMAN_AGE_TO_AL,
        woman_linger: float = WOMAN_LINGER,
        worth_at_start: float = PILE_AT_START,
        al_cc_2: float = AL_CC_2,
        al_cc_1: float = AL_CC_1,
        cc_2: float = CC_2,
        cc_1: float = CC_1,
        lc_2: float = LC_2,
        lc_1: float = LC_1,
        non_taylor_2: float = NON_TAYLOR_2,
        non_taylor_1: float = NON_TAYLOR_1,
    ) -> None:
        self.roi = roi
        self.cpi = cpi
        self.start_clock = roi.start_clock if roi.start_clock is not None else cpi.start_clock
        self.dates = roi.life_horizon_dates
        self.man_dob = man_dob
        self.woman_dob = woman_dob
        self.man_age_to_al = man_age_to_al
        self.man_linger = man_linger
        self.man_age_at_death = self.man_age_to_al + self.man_linger
        self.woman_age_to_al = woman_age_to_al
        self.woman_linger = woman_linger
        self.woman_age_at_death = self.woman_age_to_al + self.woman_linger
        self.worth_at_start = worth_at_start
        self.initial_al_cc_2 = al_cc_2
        self.initial_al_cc_1 = al_cc_1
        self.initial_cc_2 = cc_2
        self.initial_cc_1 = cc_1
        self.initial_lc_2 = lc_2
        self.initial_lc_1 = lc_1
        self.initial_non_taylor_2 = non_taylor_2
        self.initial_non_taylor_1 = non_taylor_1
        self.worth_lc = self.worth_at_start
        self.worth_cc = self.worth_at_start
        self.worth_norm_lc = self.worth_lc
        self.worth_norm_cc = self.worth_cc
        self.earn_lc = 0.0
        self.earn_cc = 0.0
        self.earn_norm_lc = 0.0
        self.earn_norm_cc = 0.0
        self.earn_lc_history: list[float] = []
        self.earn_cc_history: list[float] = []
        self.earn_norm_lc_history: list[float] = []
        self.earn_norm_cc_history: list[float] = []
        self.worth_lc_history: list[float] = []
        self.worth_cc_history: list[float] = []
        self.worth_norm_lc_history: list[float] = []
        self.worth_norm_cc_history: list[float] = []
        self.exp_al_lc = 0.0
        self.num_il_2: list[float] = []
        self.num_il_1: list[float] = []
        self.num_al_2: list[float] = []
        self.num_al_1: list[float] = []
        self.num_il: list[float] = []
        self.num_al: list[float] = []
        self.num_non_taylor : list[float] = []
        self.al_cc_2 = al_cc_2
        self.al_cc_1 = al_cc_1
        self.mo_al_cc_del_2 = 0.
        self.mo_al_cc_del_1 = 0.
        self.mo_al_cc = 0.
        self.exp_al_cc = 0.
        self.exp_al_cc_history: list[float] = []
        self.exp_norm_al_cc: list[float] = []
        self.exp_al_lc_history: list[float] = []
        self.exp_norm_al_lc: list[float] = []
        self.cc_2 = cc_2
        self.cc_1 = cc_1
        self.mo_cc_del_2 = 0.
        self.mo_cc_del_1 = 0.
        self.mo_cc = 0.
        self.exp_cc = 0.
        self.exp_cc_history: list[float] = []
        self.exp_norm_cc: list[float] = []
        self.exp_total_cc = 0.0
        self.exp_total_cc_history: list[float] = []
        self.exp_norm_total_cc: list[float] = []
        self.lc_2 = lc_2
        self.lc_1 = lc_1
        self.mo_lc_del_2 = 0.
        self.mo_lc_del_1 = 0.
        self.mo_lc = 0.
        self.exp_lc = 0.
        self.exp_lc_history: list[float] = []
        self.exp_norm_lc: list[float] = []
        self.exp_total_lc = 0.0
        self.exp_total_lc_history: list[float] = []
        self.exp_norm_total_lc: list[float] = []
        self.num_non_taylor_2: list[float] = []
        self.num_non_taylor_1: list[float] = []
        self.num_non_taylor_2: list[float] = []
        self.num_non_taylor_1: list[float] = []
        self.non_taylor_2 = non_taylor_2
        self.non_taylor_1 = non_taylor_1
        self.mo_non_taylor_del_2 = 0.
        self.mo_non_taylor_del_1 = 0.
        self.mo_non_taylor = 0.
        self.exp_non_taylor = 0.
        self.exp_non_taylor_history: list[float] = []
        self.exp_norm_non_taylor: list[float] = []
        self.exp_norm_taylor: list[float] = []

    @classmethod
    def from_lhs_scenario(
        cls,
        scenario: LhsScenario,
        context: ScenarioRunContext | None = None,
    ) -> "TaylorLife":
        run_context = context if context is not None else ScenarioRunContext()
        current_date = pd.Timestamp(run_context.current_date).normalize()
        man_age_at_death = scenario.man_age_to_al + scenario.man_linger
        woman_age_at_death = scenario.woman_age_to_al + scenario.woman_linger

        roi = Roi(
            ticker=run_context.ticker,
            current_date=current_date,
            history_years=run_context.history_years,
            start_clock=run_context.start_clock,
            man_dob=run_context.man_dob,
            woman_dob=run_context.woman_dob,
            man_age_at_death=man_age_at_death,
            woman_age_at_death=woman_age_at_death,
            mean_reversion_strength=scenario.roi_mean_reversion,
            mean_shift=scenario.roi_mean_shift,
            vol_multiplier=scenario.roi_vol_multiplier,
        )
        roi.train(ticker=run_context.ticker)
        roi.project(ticker=run_context.ticker, seed=scenario.roi_seed)
        cls.apply_constant_roi(roi)

        cpi = Inflation(
            history_years=run_context.history_years,
            al_cum_running_avg_yrs=run_context.al_cum_running_avg_yrs,
            start_clock=run_context.start_clock,
            man_dob=run_context.man_dob,
            woman_dob=run_context.woman_dob,
            man_age_at_death=man_age_at_death,
            woman_age_at_death=woman_age_at_death,
            mean_reversion_strength=scenario.inflation_mean_reversion,
            mean_shift=scenario.inflation_mean_shift,
            vol_multiplier=scenario.inflation_vol_multiplier,
        )
        cpi.train(current_date=current_date)
        cpi.project(current_date=current_date, seed=scenario.inflation_seed)
        cls.apply_constant_cpi(cpi)

        return cls(
            roi=roi,
            cpi=cpi,
            man_dob=run_context.man_dob,
            woman_dob=run_context.woman_dob,
            man_age_to_al=scenario.man_age_to_al,
            man_linger=scenario.man_linger,
            woman_age_to_al=scenario.woman_age_to_al,
            woman_linger=scenario.woman_linger,
        )

    @staticmethod
    def apply_constant_roi(roi: Roi) -> None:
        if CONSTANT_MONTHLY_ROI is None:
            return
        roi.life_horizon_roi = np.full(len(roi.life_horizon_dates), CONSTANT_MONTHLY_ROI, dtype=float)
        roi.life_horizon_roi_cum = np.cumprod(1.0 + roi.life_horizon_roi)
        roi.monthly_roi = [
            MonthlyRoiPoint(
                month=pd.Timestamp(date),
                roi=CONSTANT_MONTHLY_ROI,
                rolling_average_12m=CONSTANT_MONTHLY_ROI,
            )
            for date in pd.DatetimeIndex(roi.life_horizon_dates)
        ]

    @staticmethod
    def apply_constant_cpi(cpi: Inflation) -> None:
        if CONSTANT_MONTHLY_CPI is None:
            return
        cpi.life_horizon_inflation = np.full(len(cpi.life_horizon_dates), CONSTANT_MONTHLY_CPI, dtype=float)
        cpi.life_horizon_inflation_cum = np.cumprod(1.0 + cpi.life_horizon_inflation)
        cpi.life_horizon_cpi_running_avg = np.full(len(cpi.life_horizon_dates), CONSTANT_MONTHLY_CPI, dtype=float)
        cpi.monthly_inflation = [
            MonthlyInflationPoint(
                month=pd.Timestamp(date),
                inflation=CONSTANT_MONTHLY_CPI,
                rolling_average_12m=CONSTANT_MONTHLY_CPI,
                lower_bound=CONSTANT_MONTHLY_CPI,
                upper_bound=CONSTANT_MONTHLY_CPI,
            )
            for date in pd.DatetimeIndex(cpi.life_horizon_dates)
        ]

    def calc_result(self):
        self.worth_lc = self.worth_at_start
        self.worth_cc = self.worth_at_start
        self.worth_norm_lc = self.worth_lc
        self.worth_norm_cc = self.worth_cc
        self.earn_lc = 0.0
        self.earn_cc = 0.0
        self.earn_norm_lc = 0.0
        self.earn_norm_cc = 0.0
        self.earn_lc_history = []
        self.earn_cc_history = []
        self.earn_norm_lc_history = []
        self.earn_norm_cc_history = []
        self.worth_lc_history = []
        self.worth_cc_history = []
        self.worth_norm_lc_history = []
        self.worth_norm_cc_history = []
        self.count_all()
        self.exp_al_cc_history = self.build_expense_history(
            self.initial_al_cc_2,
            self.initial_al_cc_1,
            self.num_al_2,
            self.num_al_1,
            inflation_factor=AL_AND_LC_INFLATION_FACTOR,
        )
        self.exp_cc_history = self.build_expense_history(
            self.initial_cc_2,
            self.initial_cc_1,
            self.num_il_2,
            self.num_il_1,
        )
        self.exp_lc_history = self.build_expense_history(
            self.initial_lc_2,
            self.initial_lc_1,
            self.num_il_2,
            self.num_il_1,
            inflation_factor=AL_AND_LC_INFLATION_FACTOR,
        )
        self.exp_non_taylor_history = self.build_expense_history(
            self.initial_non_taylor_2,
            self.initial_non_taylor_1,
            self.num_non_taylor_2,
            self.num_non_taylor_1,
        )
        inflation_cum = self.cpi.life_horizon_inflation_cum
        al_active = np.array(self.num_al_1, dtype=float) + np.array(self.num_al_2, dtype=float) > 0.0
        il_active = np.array(self.num_il_1, dtype=float) + np.array(self.num_il_2, dtype=float) > 0.0
        non_taylor_active = (
            np.array(self.num_non_taylor_1, dtype=float) + np.array(self.num_non_taylor_2, dtype=float) > 0.0
        )
        self.exp_al_lc_history = np.zeros(len(inflation_cum), dtype=float).tolist()
        self.exp_norm_al_lc = np.zeros(len(inflation_cum), dtype=float).tolist()
        self.exp_total_cc_history = (
            np.asarray(self.exp_cc_history, dtype=float)
            + np.asarray(self.exp_non_taylor_history, dtype=float)
            + np.asarray(self.exp_al_cc_history, dtype=float)
        ).tolist()
        self.exp_total_lc_history = (
            np.asarray(self.exp_lc_history, dtype=float)
            + np.asarray(self.exp_non_taylor_history, dtype=float)
            + np.asarray(self.exp_al_lc_history, dtype=float)
        ).tolist()
        worth_lc_history = self.build_worth_history(self.worth_at_start, self.exp_total_lc_history)
        worth_cc_history = self.build_worth_history(self.worth_at_start, self.exp_total_cc_history)
        earn_lc_history = self.build_earn_history(self.worth_at_start, worth_lc_history)
        earn_cc_history = self.build_earn_history(self.worth_at_start, worth_cc_history)
        self.earn_lc_history = earn_lc_history.tolist()
        self.earn_cc_history = earn_cc_history.tolist()
        self.earn_norm_lc_history = (earn_lc_history / inflation_cum).tolist()
        self.earn_norm_cc_history = (earn_cc_history / inflation_cum).tolist()
        self.worth_lc_history = worth_lc_history.tolist()
        self.worth_cc_history = worth_cc_history.tolist()
        self.worth_norm_lc_history = (worth_lc_history / inflation_cum).tolist()
        self.worth_norm_cc_history = (worth_cc_history / inflation_cum).tolist()

        if len(self.exp_al_cc_history) > 0:
            self.exp_al_cc = self.exp_al_cc_history[-1]
            self.exp_cc = self.exp_cc_history[-1]
            self.exp_total_cc = self.exp_total_cc_history[-1]
            self.exp_lc = self.exp_lc_history[-1]
            self.exp_total_lc = self.exp_total_lc_history[-1]
            self.exp_non_taylor = self.exp_non_taylor_history[-1]
            self.earn_lc = float(earn_lc_history[-1])
            self.earn_cc = float(earn_cc_history[-1])
            self.earn_norm_lc = float(self.earn_norm_lc_history[-1])
            self.earn_norm_cc = float(self.earn_norm_cc_history[-1])
            self.worth_lc = float(worth_lc_history[-1])
            self.worth_cc = float(worth_cc_history[-1])
            self.worth_norm_lc = float(self.worth_norm_lc_history[-1])
            self.worth_norm_cc = float(self.worth_norm_cc_history[-1])

        self.exp_norm_al_cc = self.normalize_history(self.exp_al_cc_history, al_active, inflation_cum)
        self.exp_norm_cc = self.normalize_history(self.exp_cc_history, il_active, inflation_cum)
        self.exp_norm_lc = self.normalize_history(self.exp_lc_history, il_active, inflation_cum)
        self.exp_norm_non_taylor = self.normalize_history(
            self.exp_non_taylor_history,
            non_taylor_active,
            inflation_cum,
        )
        self.exp_norm_total_cc = self.normalize_history(self.exp_total_cc_history, il_active | al_active, inflation_cum)
        self.exp_norm_total_lc = self.normalize_history(self.exp_total_lc_history, il_active | non_taylor_active, inflation_cum)

        result = int(self.worth_lc), int(self.worth_norm_lc), int(self.worth_cc), int(self.worth_norm_cc)

        return result

    def count_all(self) -> None:
        self.num_il_2 = []
        self.num_il_1 = []
        self.num_al_2 = []
        self.num_al_1 = []
        self.num_non_taylor_2 = []
        self.num_non_taylor_1 = []

        for date in self.roi.life_horizon_dates:
            man_age = age(date, self.man_dob)
            woman_age = age(date, self.woman_dob)
            man_in_al = self.man_age_to_al <= man_age < self.man_age_at_death
            woman_in_al = self.woman_age_to_al <= woman_age < self.woman_age_at_death
            man_pre_al = man_age < self.man_age_to_al
            woman_pre_al = woman_age < self.woman_age_to_al

            # Summarize
            num_il_2 = 2.0 * float(man_pre_al and woman_pre_al)
            num_il_1 = float(man_pre_al != woman_pre_al)
            num_al_2 = 2.0 * float(man_in_al and woman_in_al)
            num_al_1 = float(man_in_al != woman_in_al)
            num_non_taylor_2 = num_il_2
            num_non_taylor_1 = num_il_1
            self.num_il_2.append(num_il_2)
            self.num_il_1.append(num_il_1)
            self.num_al_2.append(num_al_2)
            self.num_al_1.append(num_al_1)
            self.num_non_taylor_2.append(num_non_taylor_2)
            self.num_non_taylor_1.append(num_non_taylor_1)
            self.num_il = self.num_il_1 + self.num_il_2
            self.num_al = self.num_al_1 + self.num_al_2
            self.num_non_taylor = self.num_non_taylor_1 + self.num_non_taylor_2

    @staticmethod
    def normalize_history(
        expense_history: list[float],
        active_mask: np.ndarray,
        inflation_cum: np.ndarray,
    ) -> list[float]:
        if len(expense_history) == 0:
            return []
        raw = np.divide(
            np.asarray(expense_history, dtype=float),
            np.asarray(inflation_cum, dtype=float),
            out=np.zeros(len(expense_history), dtype=float),
            where=np.asarray(inflation_cum, dtype=float) != 0.0,
        )
        candidate = np.where(active_mask, raw, np.nan)
        return pd.Series(candidate).ffill().fillna(0.0).to_list()

    def build_expense_history(
        self,
        initial_cost_2: float,
        initial_cost_1: float,
        num_2: list[float],
        num_1: list[float],
        inflation_factor: float = 1.0,
    ) -> list[float]:
        inflation_growth = np.cumprod(
            1.0 + np.asarray(self.cpi.life_horizon_inflation, dtype=float) * inflation_factor
        )
        cost_2_path = initial_cost_2 * inflation_growth
        cost_1_path = initial_cost_1 * inflation_growth
        monthly_expense = cost_2_path * np.asarray(num_2, dtype=float) + cost_1_path * np.asarray(num_1, dtype=float)
        return np.cumsum(monthly_expense).tolist()

    def build_worth_history(
        self,
        worth_at_start: float,
        expense_history: list[float],
    ) -> np.ndarray:
        if len(expense_history) == 0:
            return np.array([], dtype=float)
        growth = np.cumprod(1.0 + np.asarray(self.roi.life_horizon_roi, dtype=float))
        expense = np.asarray(expense_history, dtype=float)
        worth = growth * worth_at_start - expense

        return worth

    def build_earn_history(
        self,
        worth_at_start: float,
        worth_history: np.ndarray,
    ) -> np.ndarray:
        if worth_history.size == 0:
            return np.array([], dtype=float)
        prior_worth = np.concatenate(([worth_at_start], worth_history[:-1]))
        monthly_earn = prior_worth * np.asarray(self.roi.life_horizon_roi, dtype=float)
        return np.cumsum(monthly_earn)

    def deceased(self, date: str | pd.Timestamp) -> bool:
        date_ts = pd.Timestamp(date)
        man_deceased = age(date_ts, self.man_dob) >= self.man_age_at_death
        woman_deceased = age(date_ts, self.woman_dob) >= self.woman_age_at_death
        return man_deceased and woman_deceased

    def deflate(
        self,
        date: str | pd.Timestamp,
        projection: Inflation | None = None,
    ) -> float:
        date_ts = pd.Timestamp(date)
        if self.start_clock is None:
            raise ValueError("start_clock must be set on roi or cpi before calling deflate.")
        start_ts = pd.Timestamp(self.start_clock)
        projection_to_use = projection if projection is not None else self.cpi

        if date_ts <= start_ts:
            return 1.0

        full_inflation_since_start = 1.0
        for item in projection_to_use.monthly_inflation:
            if start_ts < item.month <= date_ts:
                full_inflation_since_start *= 1 + item.inflation

        return 1 / full_inflation_since_start


def evaluate_lhs_scenario(
    scenario: LhsScenario,
    context: ScenarioRunContext | None = None,
) -> tuple[TaylorLife, TaylorLifeResult]:
    model = TaylorLife.from_lhs_scenario(scenario=scenario, context=context)
    worth_lc, worth_norm_lc, worth_cc, worth_norm_cc = model.calc_result()
    return model, TaylorLifeResult(
        worth_lc=worth_lc,
        worth_norm_lc=worth_norm_lc,
        worth_cc=worth_cc,
        worth_norm_cc=worth_norm_cc,
    )


def sample_lhs_points(num_points: int, dimensions: int, seed: int) -> np.ndarray:
    if num_points <= 0:
        raise ValueError("num_points must be positive for LHS sampling.")
    rng = np.random.default_rng(seed)
    lhs = np.zeros((num_points, dimensions), dtype=float)
    for dim in range(dimensions):
        cut_points = (np.arange(num_points, dtype=float) + rng.random(num_points)) / num_points
        lhs[:, dim] = cut_points[rng.permutation(num_points)]
    return lhs


def scale_lhs_column(values: np.ndarray, bounds: tuple[float, float]) -> np.ndarray:
    low, high = bounds
    return low + values * (high - low)


def build_lhs_scenarios(num_points: int, seed: int) -> list[LhsScenario]:
    sampled = sample_lhs_points(num_points, dimensions=12, seed=seed)
    return [
        LhsScenario(
            man_age_to_al=float(scale_lhs_column(sampled[:, 0], MAN_AGE_TO_AL_RANGE)[idx]),
            woman_age_to_al=float(scale_lhs_column(sampled[:, 1], WOMAN_AGE_TO_AL_RANGE)[idx]),
            man_linger=float(scale_lhs_column(sampled[:, 2], MAN_LINGER_RANGE)[idx]),
            woman_linger=float(scale_lhs_column(sampled[:, 3], WOMAN_LINGER_RANGE)[idx]),
            roi_seed=int(round(scale_lhs_column(sampled[:, 4], SEED_RANGE)[idx])),
            inflation_seed=int(round(scale_lhs_column(sampled[:, 5], SEED_RANGE)[idx])),
            roi_mean_shift=float(scale_lhs_column(sampled[:, 6], ROI_MEAN_SHIFT_RANGE)[idx]),
            roi_vol_multiplier=float(scale_lhs_column(sampled[:, 7], ROI_VOL_MULTIPLIER_RANGE)[idx]),
            roi_mean_reversion=float(scale_lhs_column(sampled[:, 8], ROI_MEAN_REVERSION_RANGE)[idx]),
            inflation_mean_shift=float(scale_lhs_column(sampled[:, 9], INFLATION_MEAN_SHIFT_RANGE)[idx]),
            inflation_vol_multiplier=float(scale_lhs_column(sampled[:, 10], INFLATION_VOL_MULTIPLIER_RANGE)[idx]),
            inflation_mean_reversion=float(scale_lhs_column(sampled[:, 11], INFLATION_MEAN_REVERSION_RANGE)[idx]),
        )
        for idx in range(num_points)
    ]


def build_edge_case_scenarios() -> list[tuple[str, LhsScenario]]:
    """
    Define explicit edge case scenarios for testing.
    Returns a list of tuples (case_name, scenario).
    
    NOTE: The ROI and CPI behavior for each edge case depends on global CONSTANT_MONTHLY_ROI
    and CONSTANT_MONTHLY_CPI settings:
    - If CONSTANT_MONTHLY_ROI is None: Uses stochastic/historical ROI model
    - If CONSTANT_MONTHLY_ROI is set: Uses that fixed monthly rate for all edge cases
    - Same applies for CONSTANT_MONTHLY_CPI (currently None = historical inflation model)
    
    Current settings in this file:
    - CONSTANT_MONTHLY_ROI = None (uses stochastic model with roi_mean_shift, roi_vol_multiplier, etc.)
    - CONSTANT_MONTHLY_CPI = None (uses stochastic model with inflation_mean_shift, inflation_vol_multiplier, etc.)
    
    To use fixed rates like in ExtrapolateTaylor.py, set at top of file:
    - CONSTANT_MONTHLY_ROI = 0.10 / 12  # 10% annual ROI
    - CONSTANT_MONTHLY_CPI = 0.05 / 12   # 5% annual inflation
    """
    return [
        (
            "EC_0_0",
            LhsScenario(
                man_age_to_al=69.0+0,  # now
                woman_age_to_al=70.29+0,  # now
                man_linger=0.0,  # longest linger
                woman_linger=0.0,  # longest linger
                roi_seed=DEFAULT_SEED,
                inflation_seed=DEFAULT_SEED,
                roi_mean_shift=0.01,  # highest ROI boost
                roi_vol_multiplier=0.5,  # lowest volatility
                roi_mean_reversion=0.5,  # highest mean reversion
                inflation_mean_shift=-0.005,  # lowest inflation boost
                inflation_vol_multiplier=0.5,  # lowest volatility
                inflation_mean_reversion=0.5,  # highest mean reversion
            )
        ),
        (
            "EC_0_5",
            LhsScenario(
                man_age_to_al=69.0+0,  # now
                woman_age_to_al=70.29+0,  # now
                man_linger=5.0,  # longest linger
                woman_linger=5.0,  # longest linger
                roi_seed=DEFAULT_SEED,
                inflation_seed=DEFAULT_SEED,
                roi_mean_shift=0.01,  # highest ROI boost
                roi_vol_multiplier=0.5,  # lowest volatility
                roi_mean_reversion=0.5,  # highest mean reversion
                inflation_mean_shift=-0.005,  # lowest inflation boost
                inflation_vol_multiplier=0.5,  # lowest volatility
                inflation_mean_reversion=0.5,  # highest mean reversion
            )
        ),
        (
            "EC_0_10",
            LhsScenario(
                man_age_to_al=69.0+0,  # now
                woman_age_to_al=70.29+0,  # now
                man_linger=10.0,  # longest linger
                woman_linger=10.0,  # longest linger
                roi_seed=DEFAULT_SEED,
                inflation_seed=DEFAULT_SEED,
                roi_mean_shift=0.01,  # highest ROI boost
                roi_vol_multiplier=0.5,  # lowest volatility
                roi_mean_reversion=0.5,  # highest mean reversion
                inflation_mean_shift=-0.005,  # lowest inflation boost
                inflation_vol_multiplier=0.5,  # lowest volatility
                inflation_mean_reversion=0.5,  # highest mean reversion
            )
        ),
        (
            "EC_0_15",
            LhsScenario(
                man_age_to_al=69.0+0,  # now
                woman_age_to_al=70.29+0,  # now
                man_linger=15.0,  # longest linger
                woman_linger=15.0,  # longest linger
                roi_seed=DEFAULT_SEED,
                inflation_seed=DEFAULT_SEED,
                roi_mean_shift=0.01,  # highest ROI boost
                roi_vol_multiplier=0.5,  # lowest volatility
                roi_mean_reversion=0.5,  # highest mean reversion
                inflation_mean_shift=-0.005,  # lowest inflation boost
                inflation_vol_multiplier=0.5,  # lowest volatility
                inflation_mean_reversion=0.5,  # highest mean reversion
            )
        ),
        (
            "EC_5_0",
            LhsScenario(
                man_age_to_al=69.0+5,  # 5
                woman_age_to_al=70.29+5,  # 5
                man_linger=0.0,  # longest linger
                woman_linger=0.0,  # longest linger
                roi_seed=DEFAULT_SEED,
                inflation_seed=DEFAULT_SEED,
                roi_mean_shift=0.01,  # highest ROI boost
                roi_vol_multiplier=0.5,  # lowest volatility
                roi_mean_reversion=0.5,  # highest mean reversion
                inflation_mean_shift=-0.005,  # lowest inflation boost
                inflation_vol_multiplier=0.5,  # lowest volatility
                inflation_mean_reversion=0.5,  # highest mean reversion
            )
        ),
        (
            "EC_5_5",
            LhsScenario(
                man_age_to_al=69.0+5,  # 5
                woman_age_to_al=70.29+5,  # 5
                man_linger=5.0,  # longest linger
                woman_linger=5.0,  # longest linger
                roi_seed=DEFAULT_SEED,
                inflation_seed=DEFAULT_SEED,
                roi_mean_shift=0.01,  # highest ROI boost
                roi_vol_multiplier=0.5,  # lowest volatility
                roi_mean_reversion=0.5,  # highest mean reversion
                inflation_mean_shift=-0.005,  # lowest inflation boost
                inflation_vol_multiplier=0.5,  # lowest volatility
                inflation_mean_reversion=0.5,  # highest mean reversion
            )
        ),
        (
            "EC_5_10",
            LhsScenario(
                man_age_to_al=69.0+5,  # 5
                woman_age_to_al=70.29+5,  # 5
                man_linger=10.0,  # longest linger
                woman_linger=10.0,  # longest linger
                roi_seed=DEFAULT_SEED,
                inflation_seed=DEFAULT_SEED,
                roi_mean_shift=0.01,  # highest ROI boost
                roi_vol_multiplier=0.5,  # lowest volatility
                roi_mean_reversion=0.5,  # highest mean reversion
                inflation_mean_shift=-0.005,  # lowest inflation boost
                inflation_vol_multiplier=0.5,  # lowest volatility
                inflation_mean_reversion=0.5,  # highest mean reversion
            )
        ),
        (
            "EC_5_15",
            LhsScenario(
                man_age_to_al=69.0+5,  # 5
                woman_age_to_al=70.29+5,  # 5
                man_linger=15.0,  # longest linger
                woman_linger=15.0,  # longest linger
                roi_seed=DEFAULT_SEED,
                inflation_seed=DEFAULT_SEED,
                roi_mean_shift=0.01,  # highest ROI boost
                roi_vol_multiplier=0.5,  # lowest volatility
                roi_mean_reversion=0.5,  # highest mean reversion
                inflation_mean_shift=-0.005,  # lowest inflation boost
                inflation_vol_multiplier=0.5,  # lowest volatility
                inflation_mean_reversion=0.5,  # highest mean reversion
            )
        ),
        (
            "EC_10_0",
            LhsScenario(
                man_age_to_al=69.0+10,  # 10
                woman_age_to_al=70.29+10,  # 10
                man_linger=0.0,  # longest linger
                woman_linger=0.0,  # longest linger
                roi_seed=DEFAULT_SEED,
                inflation_seed=DEFAULT_SEED,
                roi_mean_shift=0.01,  # highest ROI boost
                roi_vol_multiplier=0.5,  # lowest volatility
                roi_mean_reversion=0.5,  # highest mean reversion
                inflation_mean_shift=-0.005,  # lowest inflation boost
                inflation_vol_multiplier=0.5,  # lowest volatility
                inflation_mean_reversion=0.5,  # highest mean reversion
            )
        ),
        (
            "EC_10_5",
            LhsScenario(
                man_age_to_al=69.0+10,  # 10
                woman_age_to_al=70.29+10,  # 10
                man_linger=5.0,  # longest linger
                woman_linger=5.0,  # longest linger
                roi_seed=DEFAULT_SEED,
                inflation_seed=DEFAULT_SEED,
                roi_mean_shift=0.01,  # highest ROI boost
                roi_vol_multiplier=0.5,  # lowest volatility
                roi_mean_reversion=0.5,  # highest mean reversion
                inflation_mean_shift=-0.005,  # lowest inflation boost
                inflation_vol_multiplier=0.5,  # lowest volatility
                inflation_mean_reversion=0.5,  # highest mean reversion
            )
        ),
        (
            "EC_10_10",
            LhsScenario(
                man_age_to_al=69.0+10,  # 10
                woman_age_to_al=70.29+10,  # 10
                man_linger=10.0,  # longest linger
                woman_linger=10.0,  # longest linger
                roi_seed=DEFAULT_SEED,
                inflation_seed=DEFAULT_SEED,
                roi_mean_shift=0.01,  # highest ROI boost
                roi_vol_multiplier=0.5,  # lowest volatility
                roi_mean_reversion=0.5,  # highest mean reversion
                inflation_mean_shift=-0.005,  # lowest inflation boost
                inflation_vol_multiplier=0.5,  # lowest volatility
                inflation_mean_reversion=0.5,  # highest mean reversion
            )
        ),
        (
            "EC_10_15",
            LhsScenario(
                man_age_to_al=69.0+10,  # 10
                woman_age_to_al=70.29+10,  # 10
                man_linger=15.0,  # longest linger
                woman_linger=15.0,  # longest linger
                roi_seed=DEFAULT_SEED,
                inflation_seed=DEFAULT_SEED,
                roi_mean_shift=0.01,  # highest ROI boost
                roi_vol_multiplier=0.5,  # lowest volatility
                roi_mean_reversion=0.5,  # highest mean reversion
                inflation_mean_shift=-0.005,  # lowest inflation boost
                inflation_vol_multiplier=0.5,  # lowest volatility
                inflation_mean_reversion=0.5,  # highest mean reversion
            )
        ),
        (
            "EC_15_0",
            LhsScenario(
                man_age_to_al=69.0+15,  # 15
                woman_age_to_al=70.29+15,  # 15
                man_linger=0.0,  # longest linger
                woman_linger=0.0,  # longest linger
                roi_seed=DEFAULT_SEED,
                inflation_seed=DEFAULT_SEED,
                roi_mean_shift=0.01,  # highest ROI boost
                roi_vol_multiplier=0.5,  # lowest volatility
                roi_mean_reversion=0.5,  # highest mean reversion
                inflation_mean_shift=-0.005,  # lowest inflation boost
                inflation_vol_multiplier=0.5,  # lowest volatility
                inflation_mean_reversion=0.5,  # highest mean reversion
            )
        ),
        (
            "EC_15_5",
            LhsScenario(
                man_age_to_al=69.0+15,  # 15
                woman_age_to_al=70.29+15,  # 15
                man_linger=5.0,  # longest linger
                woman_linger=5.0,  # longest linger
                roi_seed=DEFAULT_SEED,
                inflation_seed=DEFAULT_SEED,
                roi_mean_shift=0.01,  # highest ROI boost
                roi_vol_multiplier=0.5,  # lowest volatility
                roi_mean_reversion=0.5,  # highest mean reversion
                inflation_mean_shift=-0.005,  # lowest inflation boost
                inflation_vol_multiplier=0.5,  # lowest volatility
                inflation_mean_reversion=0.5,  # highest mean reversion
            )
        ),
        (
            "EC_15_10",
            LhsScenario(
                man_age_to_al=69.0+15,  # 15
                woman_age_to_al=70.29+15,  # 15
                man_linger=10.0,  # longest linger
                woman_linger=10.0,  # longest linger
                roi_seed=DEFAULT_SEED,
                inflation_seed=DEFAULT_SEED,
                roi_mean_shift=0.01,  # highest ROI boost
                roi_vol_multiplier=0.5,  # lowest volatility
                roi_mean_reversion=0.5,  # highest mean reversion
                inflation_mean_shift=-0.005,  # lowest inflation boost
                inflation_vol_multiplier=0.5,  # lowest volatility
                inflation_mean_reversion=0.5,  # highest mean reversion
            )
        ),
        (
            "EC_15_15",
            LhsScenario(
                man_age_to_al=69.0+15,  # 15
                woman_age_to_al=70.29+15,  # 15
                man_linger=15.0,  # longest linger
                woman_linger=15.0,  # longest linger
                roi_seed=DEFAULT_SEED,
                inflation_seed=DEFAULT_SEED,
                roi_mean_shift=0.01,  # highest ROI boost
                roi_vol_multiplier=0.5,  # lowest volatility
                roi_mean_reversion=0.5,  # highest mean reversion
                inflation_mean_shift=-0.005,  # lowest inflation boost
                inflation_vol_multiplier=0.5,  # lowest volatility
                inflation_mean_reversion=0.5,  # highest mean reversion
            )
        ),
        (
            "EC_85_1",
            LhsScenario(
                man_age_to_al=85.0,
                woman_age_to_al=85.0,
                man_linger=1.0,  # very short life after AL
                woman_linger=1.0,
                roi_seed=DEFAULT_SEED,
                inflation_seed=DEFAULT_SEED,
                roi_mean_shift=0.005,
                roi_vol_multiplier=1.2,
                roi_mean_reversion=0.15,
                inflation_mean_shift=0.002,
                inflation_vol_multiplier=1.1,
                inflation_mean_reversion=0.15,
            )
        ),
    ]


def last_value(values: list[float]) -> float:
    return float(values[-1]) if values else 0.0


def summarize_lhs_run(run_id: int | str, scenario: LhsScenario, model: TaylorLife, result: TaylorLifeResult) -> LhsScenarioSummary:
    return LhsScenarioSummary(
        run_id=run_id,
        man_age_to_al=scenario.man_age_to_al,
        woman_age_to_al=scenario.woman_age_to_al,
        man_linger=scenario.man_linger,
        woman_linger=scenario.woman_linger,
        roi_seed=scenario.roi_seed,
        inflation_seed=scenario.inflation_seed,
        roi_mean_shift=scenario.roi_mean_shift,
        roi_vol_multiplier=scenario.roi_vol_multiplier,
        roi_mean_reversion=scenario.roi_mean_reversion,
        inflation_mean_shift=scenario.inflation_mean_shift,
        inflation_vol_multiplier=scenario.inflation_vol_multiplier,
        inflation_mean_reversion=scenario.inflation_mean_reversion,
        exp_al_cc=last_value(model.exp_al_cc_history),
        exp_norm_al_cc=last_value(model.exp_norm_al_cc),
        exp_cc=last_value(model.exp_cc_history),
        exp_norm_cc=last_value(model.exp_norm_cc),
        exp_lc=last_value(model.exp_lc_history),
        exp_norm_lc=last_value(model.exp_norm_lc),
        exp_non_taylor=last_value(model.exp_non_taylor_history),
        exp_norm_non_taylor=last_value(model.exp_norm_non_taylor),
        exp_total_cc=last_value(model.exp_total_cc_history),
        exp_norm_total_cc=last_value(model.exp_norm_total_cc),
        earn_cc=last_value(model.earn_cc_history),
        earn_norm_cc=last_value(model.earn_norm_cc_history),
        earn_lc=last_value(model.earn_lc_history),
        earn_norm_lc=last_value(model.earn_norm_lc_history),
        worth_lc=result.worth_lc,
        worth_norm_lc=result.worth_norm_lc,
        worth_cc=result.worth_cc,
        worth_norm_cc=result.worth_norm_cc,
    )


def run_lhs_driver(num_points: int, context: ScenarioRunContext, output_path: Path, seed: int) -> pd.DataFrame:
    global CONSTANT_MONTHLY_ROI, CONSTANT_MONTHLY_CPI
    scenarios = build_lhs_scenarios(num_points=num_points, seed=seed)
    edge_cases = build_edge_case_scenarios()
    rows = []
    column_widths = {column: max(len(column), SCREEN_MIN_COL_WIDTH) for column in CSV_COLUMNS}
    print(" ".join(column.rjust(column_widths[column]) for column in CSV_COLUMNS))
    
    # Save original constant values
    original_roi = CONSTANT_MONTHLY_ROI
    original_cpi = CONSTANT_MONTHLY_CPI
    
    # Process random LHS scenarios
    for run_id, scenario in enumerate(scenarios, start=1):
        model, result = evaluate_lhs_scenario(scenario=scenario, context=context)
        row = asdict(summarize_lhs_run(run_id=run_id, scenario=scenario, model=model, result=result))
        ordered_row = {column: row[column] for column in CSV_COLUMNS}
        print_screen_row(row=ordered_row, columns=CSV_COLUMNS, widths=column_widths)
        rows.append(ordered_row)
    
    # Process edge case scenarios
    for case_name, scenario in edge_cases:
        model, result = evaluate_lhs_scenario(scenario=scenario, context=context)
        row = asdict(summarize_lhs_run(run_id=case_name, scenario=scenario, model=model, result=result))
        ordered_row = {column: row[column] for column in CSV_COLUMNS}
        print_screen_row(row=ordered_row, columns=CSV_COLUMNS, widths=column_widths)
        rows.append(ordered_row)
    
    # Restore original constant values
    CONSTANT_MONTHLY_ROI = original_roi
    CONSTANT_MONTHLY_CPI = original_cpi
    
    frame = pd.DataFrame(rows, columns=CSV_COLUMNS)
    frame.to_csv(output_path, index=False)
    return frame


def plot_lhs_summary(results: pd.DataFrame, show: bool = True) -> None:
    linger_total = results["man_linger"] + results["woman_linger"]
    figure, axis = plt.subplots(figsize=(12, 7))
    axis.scatter(linger_total, results["worth_norm_lc"], alpha=0.7, label="worth_norm_lc")
    axis.scatter(linger_total, results["worth_norm_cc"], alpha=0.7, label="worth_norm_cc")
    axis.set_xlabel("man_linger + woman_linger")
    axis.set_ylabel("Worth (normalized)")
    axis.set_title("Normalized Worth vs Combined Linger")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best")
    plt.tight_layout()
    if show:
        plt.show()


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
    parser.add_argument(
        "--lhs-points",
        type=int,
        default=DEFAULT_LHS_POINTS,
        help=f"Run a Latin hypercube sample with this many points. Default: {DEFAULT_LHS_POINTS}",
    )
    parser.add_argument(
        "--lhs-output",
        default="lhs_taylor_results.csv",
        help="CSV output path for LHS runs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    current_date = pd.Timestamp(args.current_date).normalize()
    context = ScenarioRunContext(
        ticker=args.ticker,
        current_date=current_date,
        history_years=HISTORY_YEARS,
        al_cum_running_avg_yrs=AL_ESC_RUNNING_AVG_YRS,
        start_clock=START_CLOCK,
        man_dob=MAN_DOB,
        woman_dob=WOMAN_DOB,
    )
    if args.lhs_points > 0:
        output_path = Path(args.lhs_output)
        results = run_lhs_driver(
            num_points=args.lhs_points,
            context=context,
            output_path=output_path,
            seed=args.seed,
        )
        print(
            f"LHS runs completed: {len(results)}\n"
            f"Output CSV: {output_path}\n"
            f"Worth LC range: {results['worth_lc'].min():,.0f} to {results['worth_lc'].max():,.0f}\n"
            f"Worth CC range: {results['worth_cc'].min():,.0f} to {results['worth_cc'].max():,.0f}"
        )
        plot_lhs_summary(results)
        return

    scenario = LhsScenario(
        roi_seed=args.seed,
        inflation_seed=args.seed,
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
    worth_cc = result.worth_cc
    worth_lc = result.worth_lc
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
    if roi.return_frame is None:
        raise ValueError("ROI history was not loaded during projection.")
    plot_projection_views(roi.return_frame, roi, show=False)
    plot_inflation_views(inflation_frame, cpi, show=False)
    plot_taylor_life_exp_non_taylor(this_life, show=False)
    plt.show()


if __name__ == "__main__":
    main()
