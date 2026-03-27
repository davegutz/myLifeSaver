import argparse
from dataclasses import dataclass
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from Inflation import Inflation, plot_inflation_views
from Roi import TICKER, Roi, plot_projection_views
from utils import age

#  Fixed parameters
HISTORY_YEARS = 25
AL_ESC_RUNNING_AVG_YRS = 2
START_CLOCK = "2026-07-01"
DEFAULT_CURRENT_DATE = "2026-03-27"
MAN_DOB = "1957-07-26"
WOMAN_DOB = "1956-04-11"
PILE_AT_START = 5700000.
NON_TAYLOR_2 = 9612.
NON_TAYLOR_1 = 5492.
AL_CC_1 = 9200.
AL_CC_2 = AL_CC_1 * 2.
CC_1 = 3150.
CC_2 = 3750.
LC_1 = 8100.
LC_2 = 9600.

# To be varied
MAN_AGE_TO_AL = 71.
WOMAN_AGE_TO_AL = 71.
MAN_LINGER = 6.
WOMAN_LINGER = 6.
DEFAULT_SEED = 0
ROI_MEAN_SHIFT = 0.0
ROI_VOL_MULTIPLIER = 1.0
ROI_MEAN_REVERSION = 0.15
INFLATION_MEAN_SHIFT = 0.0
INFLATION_VOL_MULTIPLIER = 1.0
INFLATION_MEAN_REVERSION = 0.15


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
    principal_lc: int
    principal_norm_lc: int
    principal_cc: int
    principal_norm_cc: int

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
        pile_at_start: float = PILE_AT_START,
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
        self.pile_at_start = pile_at_start
        self.initial_al_cc_2 = al_cc_2
        self.initial_al_cc_1 = al_cc_1
        self.initial_cc_2 = cc_2
        self.initial_cc_1 = cc_1
        self.initial_lc_2 = lc_2
        self.initial_lc_1 = lc_1
        self.initial_non_taylor_2 = non_taylor_2
        self.initial_non_taylor_1 = non_taylor_1
        self.pile_lc = self.pile_at_start
        self.pile_cc = self.pile_at_start
        self.pile_norm_lc = self.pile_lc
        self.pile_norm_cc = self.pile_cc
        self.roi_lc = 0.
        self.roi_cc = 0.
        self.exp_al_lc = 0.0
        self.num_il_2: list[float] = []
        self.num_il_1: list[float] = []
        self.num_al_2: list[float] = []
        self.num_al_1: list[float] = []
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
        self.lc_2 = lc_2
        self.lc_1 = lc_1
        self.mo_lc_del_2 = 0.
        self.mo_lc_del_1 = 0.
        self.mo_lc = 0.
        self.exp_lc = 0.
        self.exp_lc_history: list[float] = []
        self.exp_norm_lc: list[float] = []
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

    def calc_result(self):
        self.pile_lc = self.pile_at_start
        self.pile_cc = self.pile_at_start
        self.pile_norm_lc = self.pile_lc
        self.pile_norm_cc = self.pile_cc
        self.al_cc_2 = self.initial_al_cc_2
        self.al_cc_1 = self.initial_al_cc_1
        self.cc_2 = self.initial_cc_2
        self.cc_1 = self.initial_cc_1
        self.lc_2 = self.initial_lc_2
        self.lc_1 = self.initial_lc_1
        self.non_taylor_2 = self.initial_non_taylor_2
        self.non_taylor_1 = self.initial_non_taylor_1
        self.exp_al_cc = 0.0
        self.exp_cc = 0.0
        self.exp_lc = 0.0
        self.exp_non_taylor = 0.0
        self.exp_al_lc = 0.0
        self.exp_al_cc_history = []
        self.exp_norm_al_cc = []
        self.exp_al_lc_history = []
        self.exp_norm_al_lc = []
        self.exp_cc_history = []
        self.exp_norm_cc = []
        self.exp_lc_history = []
        self.exp_norm_lc = []
        self.exp_non_taylor_history = []
        self.exp_norm_non_taylor = []
        self.count_all()
        self.exp_al_cc_history = self.build_expense_history(
            self.initial_al_cc_2,
            self.initial_al_cc_1,
            self.num_al_2,
            self.num_al_1,
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
        n = len(self.cpi.life_horizon_dates)
        for i in range(n):
            self.exp_al_cc = self.exp_al_cc_history[i]
            self.exp_cc = self.exp_cc_history[i]
            self.exp_lc = self.exp_lc_history[i]
            self.exp_non_taylor = self.exp_non_taylor_history[i]

            self.roi_lc = self.pile_lc * self.roi.life_horizon_roi[i]
            self.pile_lc += self.roi_lc - self.exp_lc - self.exp_non_taylor - self.exp_al_lc

            self.roi_cc = self.pile_cc * self.roi.life_horizon_roi[i]
            self.pile_cc += self.roi_cc - self.exp_cc - self.exp_non_taylor - self.exp_al_cc

        self.exp_norm_al_cc = self.normalize_history(self.exp_al_cc_history, al_active, inflation_cum)
        self.exp_norm_cc = self.normalize_history(self.exp_cc_history, il_active, inflation_cum)
        self.exp_norm_lc = self.normalize_history(self.exp_lc_history, il_active, inflation_cum)
        self.exp_norm_non_taylor = self.normalize_history(
            self.exp_non_taylor_history,
            non_taylor_active,
            inflation_cum,
        )

        self.pile_norm_lc = self.pile_lc * self.de_cumalate(self.cpi.life_horizon_dates[-1])
        self.pile_norm_cc = self.pile_cc * self.de_cumalate(self.cpi.life_horizon_dates[-1])

        result = int(self.pile_lc), int(self.pile_norm_lc), int(self.pile_cc), int(self.pile_norm_cc)

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
    ) -> list[float]:
        inflation_growth = np.cumprod(1.0 + np.asarray(self.cpi.life_horizon_inflation, dtype=float))
        cost_2_path = initial_cost_2 * inflation_growth
        cost_1_path = initial_cost_1 * inflation_growth
        monthly_expense = cost_2_path * np.asarray(num_2, dtype=float) + cost_1_path * np.asarray(num_1, dtype=float)
        return np.cumsum(monthly_expense).tolist()

    def deceased(self, date: str | pd.Timestamp) -> bool:
        date_ts = pd.Timestamp(date)
        man_deceased = age(date_ts, self.man_dob) >= self.man_age_at_death
        woman_deceased = age(date_ts, self.woman_dob) >= self.woman_age_at_death
        return man_deceased and woman_deceased

    def de_cumalate(
        self,
        date: str | pd.Timestamp,
        projection: Roi | None = None,
    ) -> float:
        date_ts = pd.Timestamp(date)
        if self.start_clock is None:
            raise ValueError("start_clock must be set on roi or cpi before calling de_cumalate.")
        start_ts = pd.Timestamp(self.start_clock)
        projection_to_use = projection if projection is not None else self.roi

        if date_ts <= start_ts:
            return 1.0

        full_growth_since_start = 1.0
        for item in projection_to_use.monthly_roi:
            if start_ts < item.month <= date_ts:
                full_growth_since_start *= 1 + item.roi

        return 1 / full_growth_since_start


def evaluate_lhs_scenario(
    scenario: LhsScenario,
    context: ScenarioRunContext | None = None,
) -> tuple[TaylorLife, TaylorLifeResult]:
    model = TaylorLife.from_lhs_scenario(scenario=scenario, context=context)
    principal_lc, principal_norm_lc, principal_cc, principal_norm_cc = model.calc_result()
    return model, TaylorLifeResult(
        principal_lc=principal_lc,
        principal_norm_lc=principal_norm_lc,
        principal_cc=principal_cc,
        principal_norm_cc=principal_norm_cc,
    )


def plot_taylor_life_exp_non_taylor(this_life: TaylorLife, show: bool = True) -> None:
    if not this_life.exp_non_taylor_history:
        this_life.calc_result()

    figure, axis = plt.subplots(figsize=(14, 7))
    axis.plot(
        pd.DatetimeIndex(this_life.dates),
        this_life.exp_non_taylor_history,
        linewidth=2.0,
        label="exp_non_taylor",
    )
    axis.plot(
        pd.DatetimeIndex(this_life.dates),
        this_life.exp_norm_non_taylor,
        linewidth=2.0,
        linestyle="--",
        label="exp_norm_non_taylor",
    )
    axis.plot(
        pd.DatetimeIndex(this_life.dates),
        this_life.exp_al_cc_history,
        linewidth=2.0,
        label="exp_al_cc",
    )
    axis.plot(
        pd.DatetimeIndex(this_life.dates),
        this_life.exp_norm_al_cc,
        linewidth=2.0,
        linestyle="--",
        label="exp_norm_al_cc",
    )
    axis.plot(
        pd.DatetimeIndex(this_life.dates),
        this_life.exp_cc_history,
        linewidth=2.0,
        label="exp_cc",
    )
    axis.plot(
        pd.DatetimeIndex(this_life.dates),
        this_life.exp_norm_cc,
        linewidth=2.0,
        linestyle="--",
        label="exp_norm_cc",
    )
    axis.plot(
        pd.DatetimeIndex(this_life.dates),
        this_life.exp_lc_history,
        linewidth=2.0,
        label="exp_lc",
    )
    axis.plot(
        pd.DatetimeIndex(this_life.dates),
        this_life.exp_norm_lc,
        linewidth=2.0,
        linestyle="--",
        label="exp_norm_lc",
    )
    axis.set_xlabel("Date")
    axis.set_ylabel("exp_non_taylor")
    axis.set_title("Taylor Life exp_non_taylor Over Time")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="upper left")
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
    principal_lc = result.principal_lc
    principal_norm_lc = result.principal_norm_lc
    principal_cc = result.principal_cc
    principal_norm_cc = result.principal_norm_cc

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
    print(f"LC Plan A {principal_lc=} {principal_norm_lc=} {principal_lc/principal_norm_lc:5.2f}"
          f"\nCC Plan B {principal_cc=} {principal_norm_cc=} {float(principal_cc)/float(principal_norm_cc):5.2f}")
    if roi.return_frame is None:
        raise ValueError("ROI history was not loaded during projection.")
    plot_projection_views(roi.return_frame, roi, show=False)
    plot_inflation_views(inflation_frame, cpi, show=False)
    plot_taylor_life_exp_non_taylor(this_life, show=False)
    plt.show()


if __name__ == "__main__":
    main()
