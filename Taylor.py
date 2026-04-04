from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from default_case import (
    AL_INFLATION_FACTOR,
    AL_CC_1,
    AL_CC_2,
    AL_ESC_RUNNING_AVG_YRS,
    CC_1,
    CC_2,
    CONSTANT_MONTHLY_CPI,
    CONSTANT_MONTHLY_ROI,
    DEFAULT_CURRENT_DATE,
    DEFAULT_SEED,
    ENTRANCE_FEE_CC,
    ENTRANCE_FEE_LC,
    SS_MAN,
    SS_WOMAN,
    PEN_MAN,
    PEN_WOMAN,
    HISTORY_YEARS,
    INFLATION_MEAN_REVERSION,
    INFLATION_MEAN_SHIFT,
    INFLATION_VOL_MULTIPLIER,
    LC_1,
    LC_2,
    LC_INFLATION_FACTOR,
    MAN_ASSISTED_YRS,
    MAN_DOB,
    MAN_INDEPENDENT_YRS,
    NON_TAYLOR_1,
    NON_TAYLOR_2,
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
)
from Inflation import Inflation, MonthlyInflationPoint
from Roi import MonthlyRoiPoint, Roi, TICKER
from utils import age, date_after_years


@dataclass(frozen=True)
class LhsScenario:
    man_independent_yrs: float = MAN_INDEPENDENT_YRS
    woman_independent_yrs: float = WOMAN_INDEPENDENT_YRS
    man_assisted_yrs: float = MAN_ASSISTED_YRS
    woman_assisted_yrs: float = WOMAN_ASSISTED_YRS
    roi_seed: int = DEFAULT_SEED
    inflation_seed: int = DEFAULT_SEED
    roi_mean_shift: float = ROI_MEAN_SHIFT
    roi_vol_multiplier: float = ROI_VOL_MULTIPLIER
    roi_mean_reversion: float = ROI_MEAN_REVERSION
    inflation_mean_shift: float = INFLATION_MEAN_SHIFT
    inflation_vol_multiplier: float = INFLATION_VOL_MULTIPLIER
    inflation_mean_reversion: float = INFLATION_MEAN_REVERSION
    man_goes_to_al_seed: int = DEFAULT_SEED
    woman_goes_to_al_seed: int = DEFAULT_SEED
    man_goes_to_al: bool = field(default_factory=lambda: bool(np.random.binomial(1, P_MAN_AL)))
    woman_goes_to_al: bool = field(default_factory=lambda: bool(np.random.binomial(1, P_WOMAN_AL)))


@dataclass(frozen=True)
class LhsScenarioSummary:
    run_id: int | str
    man_independent_yrs: float
    woman_independent_yrs: float
    man_assisted_yrs: float
    woman_assisted_yrs: float
    roi_seed: int
    inflation_seed: int
    apy_roi: float
    apy_cpi: float
    roi_one_dollar_at_end: float
    cpi_one_dollar_at_end: float
    norm_one_dollar_at_end: float
    roi_mean_shift: float
    roi_vol_multiplier: float
    roi_mean_reversion: float
    inflation_mean_shift: float
    inflation_vol_multiplier: float
    inflation_mean_reversion: float
    man_goes_to_al_seed: int
    woman_goes_to_al_seed: int
    man_goes_to_al: bool
    woman_goes_to_al: bool
    man_age_to_al: float | str
    woman_age_to_al: float | str
    man_age_at_death: float
    woman_age_at_death: float
    exp_norm_al_cc: float
    exp_norm_cc: float
    exp_norm_lc: float
    exp_norm_non_taylor: float
    exp_norm_total_cc: float
    exp_norm_total_lc: float
    entrance_fee_cc: float
    entrance_fee_lc: float
    earn_norm_cc: float
    earn_norm_lc: float
    cum_mo_earn_lc_norm: float
    cum_mo_earn_cc_norm: float
    cum_mo_earn_ss_man_norm: float
    cum_mo_earn_ss_woman_norm: float
    cum_mo_earn_ss_norm: float
    cum_mo_earn_pen_man_norm: float
    cum_mo_earn_pen_woman_norm: float
    cum_mo_earn_pen_norm: float
    cum_mo_exp_lc_norm: float
    cum_mo_exp_cc_norm: float
    cum_mo_exp_al_cc_norm: float
    cum_mo_exp_non_taylor_norm: float
    cum_mo_exp_total_lc_norm: float
    cum_mo_exp_total_cc_norm: float
    start_pile: float
    final_worth_norm_cc: float
    final_worth_norm_lc: float
    worth_norm_lc: int
    worth_norm_cc: int
    added_lc_worth_norm: int
    yrs_il_single: float
    yrs_il_double: float
    yrs_sum_al: float
    total_living_yrs: float
    elapsed_time_yrs: float
    earning_potential: float
    earning_potential_cc: float
    man_age_at_start: float
    woman_age_at_start: float
    # Context constants
    ticker: str
    current_date: str
    history_years: int
    al_cum_running_avg_yrs: int | float
    start_clock: str
    man_dob: str
    woman_dob: str
    constant_monthly_roi: float | str | None
    constant_monthly_cpi: float | str | None


@dataclass(frozen=True)
class ScenarioRunContext:
    ticker: str = TICKER
    current_date: pd.Timestamp | str = DEFAULT_CURRENT_DATE
    history_years: int = HISTORY_YEARS
    al_cum_running_avg_yrs: int | float = AL_ESC_RUNNING_AVG_YRS
    start_clock: str = START_CLOCK
    man_dob: str = MAN_DOB
    woman_dob: str = WOMAN_DOB
    constant_monthly_roi: float | None = CONSTANT_MONTHLY_ROI
    constant_monthly_cpi: float | None = CONSTANT_MONTHLY_CPI


class TaylorLife:
    def __init__(
        self,
        roi: Roi,
        cpi: Inflation,
        man_dob: str = MAN_DOB,
        woman_dob: str = WOMAN_DOB,
        man_independent_yrs: float = MAN_INDEPENDENT_YRS,
        man_assisted_yrs: float = MAN_ASSISTED_YRS,
        woman_independent_yrs: float = WOMAN_INDEPENDENT_YRS,
        woman_assisted_yrs: float = WOMAN_ASSISTED_YRS,
        worth_at_start: float = PILE_AT_START,
        al_cc_2: float = AL_CC_2,
        al_cc_1: float = AL_CC_1,
        cc_2: float = CC_2,
        cc_1: float = CC_1,
        lc_2: float = LC_2,
        lc_1: float = LC_1,
        non_taylor_2: float = NON_TAYLOR_2,
        non_taylor_1: float = NON_TAYLOR_1,
        entrance_fee_cc: float = ENTRANCE_FEE_CC,
        entrance_fee_lc: float = ENTRANCE_FEE_LC,
        ss_man: float = SS_MAN,
        ss_woman: float = SS_WOMAN,
        pen_man: float = PEN_MAN,
        pen_woman: float = PEN_WOMAN,
        man_goes_to_al: bool = True,
        woman_goes_to_al: bool = True,
    ) -> None:
        self.roi = roi
        self.cpi = cpi
        self.start_clock = pd.Timestamp(roi.start_clock if roi.start_clock is not None else cpi.start_clock)
        self.dates = roi.life_horizon_dates
        self.man_dob = man_dob
        self.woman_dob = woman_dob
        self.man_independent_yrs = man_independent_yrs
        self.man_assisted_yrs = man_assisted_yrs if man_goes_to_al else 0.0
        self.man_move_to_al_date = date_after_years(self.start_clock, self.man_independent_yrs)
        self.man_age_to_al = age(self.man_move_to_al_date, self.man_dob)
        self.man_death_date = (
            date_after_years(self.man_move_to_al_date, self.man_assisted_yrs)
            if man_goes_to_al
            else self.man_move_to_al_date
        )
        self.man_age_at_death = age(self.man_death_date, self.man_dob)
        self.woman_independent_yrs = woman_independent_yrs
        self.woman_assisted_yrs = woman_assisted_yrs if woman_goes_to_al else 0.0
        self.woman_move_to_al_date = date_after_years(self.start_clock, self.woman_independent_yrs)
        self.woman_age_to_al = age(self.woman_move_to_al_date, self.woman_dob)
        self.woman_death_date = (
            date_after_years(self.woman_move_to_al_date, self.woman_assisted_yrs)
            if woman_goes_to_al
            else self.woman_move_to_al_date
        )
        self.woman_age_at_death = age(self.woman_death_date, self.woman_dob)
        self.worth_at_start = worth_at_start
        self.man_goes_to_al = man_goes_to_al
        self.woman_goes_to_al = woman_goes_to_al
        self.initial_al_cc_2 = al_cc_2
        self.initial_al_cc_1 = al_cc_1
        self.initial_cc_2 = cc_2
        self.initial_cc_1 = cc_1
        self.initial_lc_2 = lc_2
        self.initial_lc_1 = lc_1
        self.initial_non_taylor_2 = non_taylor_2
        self.initial_non_taylor_1 = non_taylor_1
        self.entrance_fee_cc = entrance_fee_cc
        self.entrance_fee_lc = entrance_fee_lc
        self.ss_man = ss_man
        self.ss_woman = ss_woman
        self.pen_man = pen_man
        self.pen_woman = pen_woman
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
        self.mo_earn_lc_norm: list[float] = []
        self.cum_mo_earn_lc_norm: list[float] = []
        self.mo_earn_cc_norm: list[float] = []
        self.cum_mo_earn_cc_norm: list[float] = []
        self.mo_earn_ss_man_norm: list[float] = []
        self.cum_mo_earn_ss_man_norm: list[float] = []
        self.mo_earn_ss_woman_norm: list[float] = []
        self.cum_mo_earn_ss_woman_norm: list[float] = []
        self.mo_earn_ss_norm: list[float] = []
        self.cum_mo_earn_ss_norm: list[float] = []
        self.mo_earn_pen_man_norm: list[float] = []
        self.cum_mo_earn_pen_man_norm: list[float] = []
        self.mo_earn_pen_woman_norm: list[float] = []
        self.cum_mo_earn_pen_woman_norm: list[float] = []
        self.mo_earn_pen_norm: list[float] = []
        self.cum_mo_earn_pen_norm: list[float] = []
        self.mo_exp_lc_norm: list[float] = []
        self.cum_mo_exp_lc_norm: list[float] = []
        self.mo_exp_cc_norm: list[float] = []
        self.cum_mo_exp_cc_norm: list[float] = []
        self.mo_exp_al_cc_norm: list[float] = []
        self.cum_mo_exp_al_cc_norm: list[float] = []
        self.mo_exp_non_taylor_norm: list[float] = []
        self.cum_mo_exp_non_taylor_norm: list[float] = []
        self.mo_exp_total_lc_norm: list[float] = []
        self.cum_mo_exp_total_lc_norm: list[float] = []
        self.mo_exp_total_cc_norm: list[float] = []
        self.cum_mo_exp_total_cc_norm: list[float] = []
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
        self.num_non_taylor: list[float] = []
        self.al_cc_2 = al_cc_2
        self.al_cc_1 = al_cc_1
        self.mo_al_cc_del_2 = 0.0
        self.mo_al_cc_del_1 = 0.0
        self.mo_al_cc = 0.0
        self.exp_al_cc = 0.0
        self.exp_al_cc_history: list[float] = []
        self.exp_norm_al_cc: list[float] = []
        self.exp_al_lc_history: list[float] = []
        self.exp_norm_al_lc: list[float] = []
        self.cc_2 = cc_2
        self.cc_1 = cc_1
        self.mo_cc_del_2 = 0.0
        self.mo_cc_del_1 = 0.0
        self.mo_cc = 0.0
        self.exp_cc = 0.0
        self.exp_cc_history: list[float] = []
        self.exp_norm_cc: list[float] = []
        self.exp_total_cc = 0.0
        self.exp_total_cc_history: list[float] = []
        self.exp_norm_total_cc: list[float] = []
        self.lc_2 = lc_2
        self.lc_1 = lc_1
        self.mo_lc_del_2 = 0.0
        self.mo_lc_del_1 = 0.0
        self.mo_lc = 0.0
        self.exp_lc = 0.0
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
        self.mo_non_taylor_del_2 = 0.0
        self.mo_non_taylor_del_1 = 0.0
        self.mo_non_taylor = 0.0
        self.exp_non_taylor = 0.0
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
        man_move_to_al_date = date_after_years(run_context.start_clock, scenario.man_independent_yrs)
        woman_move_to_al_date = date_after_years(run_context.start_clock, scenario.woman_independent_yrs)
        man_death_date = (
            date_after_years(man_move_to_al_date, scenario.man_assisted_yrs)
            if scenario.man_goes_to_al
            else man_move_to_al_date
        )
        woman_death_date = (
            date_after_years(woman_move_to_al_date, scenario.woman_assisted_yrs)
            if scenario.woman_goes_to_al
            else woman_move_to_al_date
        )
        man_age_at_death = age(man_death_date, run_context.man_dob)
        woman_age_at_death = age(woman_death_date, run_context.woman_dob)

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
        cls.apply_constant_roi(roi, run_context.constant_monthly_roi)

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
        cls.apply_constant_cpi(cpi, run_context.constant_monthly_cpi)

        return cls(
            roi=roi,
            cpi=cpi,
            man_dob=run_context.man_dob,
            woman_dob=run_context.woman_dob,
            man_independent_yrs=scenario.man_independent_yrs,
            man_assisted_yrs=scenario.man_assisted_yrs,
            woman_independent_yrs=scenario.woman_independent_yrs,
            woman_assisted_yrs=scenario.woman_assisted_yrs,
            man_goes_to_al=scenario.man_goes_to_al,
            woman_goes_to_al=scenario.woman_goes_to_al,
        )

    @staticmethod
    def apply_constant_roi(roi: Roi, constant_monthly_roi: float | None) -> None:
        if constant_monthly_roi is None:
            return
        roi.life_horizon_roi = np.full(len(roi.life_horizon_dates), constant_monthly_roi, dtype=float)
        roi.life_horizon_roi_cum = np.cumprod(1.0 + roi.life_horizon_roi)
        roi.monthly_roi = [
            MonthlyRoiPoint(
                month=pd.Timestamp(date),
                roi=constant_monthly_roi,
                rolling_average_12m=constant_monthly_roi,
            )
            for date in pd.DatetimeIndex(roi.life_horizon_dates)
        ]

    @staticmethod
    def apply_constant_cpi(cpi: Inflation, constant_monthly_cpi: float | None) -> None:
        if constant_monthly_cpi is None:
            return
        cpi.life_horizon_inflation = np.full(len(cpi.life_horizon_dates), constant_monthly_cpi, dtype=float)
        cpi.life_horizon_inflation_cum = np.cumprod(1.0 + cpi.life_horizon_inflation)
        cpi.life_horizon_cpi_running_avg = np.full(len(cpi.life_horizon_dates), constant_monthly_cpi, dtype=float)
        cpi.monthly_inflation = [
            MonthlyInflationPoint(
                month=pd.Timestamp(date),
                inflation=constant_monthly_cpi,
                rolling_average_12m=constant_monthly_cpi,
                lower_bound=constant_monthly_cpi,
                upper_bound=constant_monthly_cpi,
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
            inflation_factor=AL_INFLATION_FACTOR,
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
            inflation_factor=LC_INFLATION_FACTOR,
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
            + self.entrance_fee_cc
        ).tolist()
        self.exp_total_lc_history = (
            np.asarray(self.exp_lc_history, dtype=float)
            + np.asarray(self.exp_non_taylor_history, dtype=float)
            + np.asarray(self.exp_al_lc_history, dtype=float)
            + self.entrance_fee_lc
        ).tolist()
        worth_lc_history = self.build_worth_history(self.worth_at_start, self.exp_total_lc_history)
        worth_cc_history = self.build_worth_history(self.worth_at_start, self.exp_total_cc_history)
        earn_lc_history = self.build_earn_history(self.worth_at_start, worth_lc_history)
        earn_cc_history = self.build_earn_history(self.worth_at_start, worth_cc_history)
        self.earn_lc_history = earn_lc_history.tolist()
        self.earn_cc_history = earn_cc_history.tolist()
        self.earn_norm_lc_history = (earn_lc_history / inflation_cum).tolist()
        self.earn_norm_cc_history = (earn_cc_history / inflation_cum).tolist()
        mo_earn_lc_norm_base, _ = self._monthly_norm(earn_lc_history, inflation_cum)
        mo_earn_cc_norm_base, _ = self._monthly_norm(earn_cc_history, inflation_cum)
        # SS grows with CPI: nominal = ss * inflation_cum → norm = ss (constant)
        n = inflation_cum.size
        ss_man_mo = np.full(n, self.ss_man)
        ss_woman_mo = np.full(n, self.ss_woman)
        self.mo_earn_ss_man_norm = ss_man_mo.tolist()
        self.cum_mo_earn_ss_man_norm = np.cumsum(ss_man_mo).tolist()
        self.mo_earn_ss_woman_norm = ss_woman_mo.tolist()
        self.cum_mo_earn_ss_woman_norm = np.cumsum(ss_woman_mo).tolist()
        ss_mo = ss_man_mo + ss_woman_mo
        self.mo_earn_ss_norm = ss_mo.tolist()
        self.cum_mo_earn_ss_norm = np.cumsum(ss_mo).tolist()
        # Pension is fixed nominal: norm = pen / inflation_cum
        pen_man_mo = self.pen_man / inflation_cum
        pen_woman_mo = self.pen_woman / inflation_cum
        self.mo_earn_pen_man_norm = pen_man_mo.tolist()
        self.cum_mo_earn_pen_man_norm = np.cumsum(pen_man_mo).tolist()
        self.mo_earn_pen_woman_norm = pen_woman_mo.tolist()
        self.cum_mo_earn_pen_woman_norm = np.cumsum(pen_woman_mo).tolist()
        pen_mo = pen_man_mo + pen_woman_mo
        self.mo_earn_pen_norm = pen_mo.tolist()
        self.cum_mo_earn_pen_norm = np.cumsum(pen_mo).tolist()
        # Fold SS and PEN into earn norm series
        ss_pen_mo = ss_man_mo + ss_woman_mo + pen_man_mo + pen_woman_mo
        mo_earn_lc_norm_full = np.asarray(mo_earn_lc_norm_base) + ss_pen_mo
        mo_earn_cc_norm_full = np.asarray(mo_earn_cc_norm_base) + ss_pen_mo
        self.mo_earn_lc_norm = mo_earn_lc_norm_full.tolist()
        self.cum_mo_earn_lc_norm = np.cumsum(mo_earn_lc_norm_full).tolist()
        self.mo_earn_cc_norm = mo_earn_cc_norm_full.tolist()
        self.cum_mo_earn_cc_norm = np.cumsum(mo_earn_cc_norm_full).tolist()
        self.mo_exp_lc_norm, self.cum_mo_exp_lc_norm = self._monthly_norm(np.asarray(self.exp_lc_history, dtype=float), inflation_cum)
        self.mo_exp_cc_norm, self.cum_mo_exp_cc_norm = self._monthly_norm(np.asarray(self.exp_cc_history, dtype=float), inflation_cum)
        self.mo_exp_al_cc_norm, self.cum_mo_exp_al_cc_norm = self._monthly_norm(np.asarray(self.exp_al_cc_history, dtype=float), inflation_cum)
        self.mo_exp_non_taylor_norm, self.cum_mo_exp_non_taylor_norm = self._monthly_norm(np.asarray(self.exp_non_taylor_history, dtype=float), inflation_cum)
        self.mo_exp_total_lc_norm, self.cum_mo_exp_total_lc_norm = self._monthly_norm(np.asarray(self.exp_total_lc_history, dtype=float), inflation_cum)
        self.mo_exp_total_cc_norm, self.cum_mo_exp_total_cc_norm = self._monthly_norm(np.asarray(self.exp_total_cc_history, dtype=float), inflation_cum)
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
        # Normalize only the ongoing (inflation-varying) portion, then add the entrance
        # fees back as present-value constants so they are never deflated below face value.
        exp_ongoing_cc = (
            np.asarray(self.exp_cc_history, dtype=float)
            + np.asarray(self.exp_non_taylor_history, dtype=float)
            + np.asarray(self.exp_al_cc_history, dtype=float)
        )
        exp_ongoing_lc = (
            np.asarray(self.exp_lc_history, dtype=float)
            + np.asarray(self.exp_non_taylor_history, dtype=float)
            + np.asarray(self.exp_al_lc_history, dtype=float)
        )
        self.exp_norm_total_cc = [
            v + self.entrance_fee_cc
            for v in self.normalize_history(exp_ongoing_cc.tolist(), il_active | al_active, inflation_cum)
        ]
        self.exp_norm_total_lc = [
            v + self.entrance_fee_lc
            for v in self.normalize_history(exp_ongoing_lc.tolist(), il_active | non_taylor_active, inflation_cum)
        ]

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
            date_ts = pd.Timestamp(date)
            man_in_al = (self.man_move_to_al_date <= date_ts < self.man_death_date) and self.man_goes_to_al
            woman_in_al = (self.woman_move_to_al_date <= date_ts < self.woman_death_date) and self.woman_goes_to_al
            man_pre_al = date_ts < self.man_move_to_al_date
            woman_pre_al = date_ts < self.woman_move_to_al_date

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
    def _monthly_norm(cum_arr: np.ndarray, inflation_cum: np.ndarray) -> tuple[list[float], list[float]]:
        """Return (mo_norm, cum_mo_norm): monthly increments of cum_arr each deflated by
        their own inflation_cum, plus the running total of those deflated increments."""
        if cum_arr.size == 0:
            return [], []
        mo = np.concatenate([[cum_arr[0]], np.diff(cum_arr)])
        mo_norm = mo / inflation_cum
        return mo_norm.tolist(), np.cumsum(mo_norm).tolist()

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
        man_deceased = date_ts >= self.man_death_date
        woman_deceased = date_ts >= self.woman_death_date
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


@dataclass(frozen=True)
class TaylorLifeResult:
    worth_lc: int
    worth_norm_lc: int
    worth_cc: int
    worth_norm_cc: int


