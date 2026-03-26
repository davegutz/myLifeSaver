import argparse

import matplotlib.pyplot as plt
import pandas as pd

from Inflation import Inflation, plot_inflation_views
from Roi import TICKER, Roi, plot_projection_views
from utils import age


HISTORY_YEARS = 25
AL_ESC_RUNNING_AVG_YRS = 2
START_CLOCK = "2026-07-01"
MAN_DOB = "1957-07-26"
WOMAN_DOB = "1956-04-11"
MAN_AGE_TO_AL = 71
WOMAN_AGE_TO_AL = 71
MAN_AGE_AT_DEATH = 71
WOMAN_AGE_AT_DEATH = 71
PILE_AT_START = 5700000
NON_TAYLOR_2 = 9612
NON_TAYLOR_1 = 5492

class TaylorLife:
    def __init__(
        self,
        roi: Roi,
        cpi: Inflation,
        man_dob: str = MAN_DOB,
        woman_dob: str = WOMAN_DOB,
        man_age_to_al: float = MAN_AGE_TO_AL,
        man_age_at_death: float = MAN_AGE_AT_DEATH,
        woman_age_to_al: float = WOMAN_AGE_TO_AL,
        woman_age_at_death: float = WOMAN_AGE_AT_DEATH,
        pile_at_start: float = PILE_AT_START,
        esc: float = 1,
        exp: float = 1000,
        non_taylor_2: float = NON_TAYLOR_2,
        non_taylor_1: float = NON_TAYLOR_1,
    ) -> None:
        if roi is None or cpi is None:
            raise ValueError("roi and an cpi vectors must be supplied by the caller")
        self.roi = roi
        self.cpi = cpi
        self.start_clock = roi.start_clock if roi.start_clock is not None else cpi.start_clock
        self.dates = roi.life_horizon_dates
        self.man_dob = man_dob
        self.woman_dob = woman_dob
        self.man_age_to_al = man_age_to_al
        self.man_age_at_death = man_age_at_death
        self.woman_age_to_al = woman_age_to_al
        self.woman_age_at_death = woman_age_at_death
        self.pile_at_start = pile_at_start
        self.cpi_cum = 0.0
        self.roi_cum = 0.0
        # self.esc_al = esc_al
        # self.esc_cc = esc_cc
        # self.non_Taylor_exp = non_Taylor_exp
        # self.exp_2 = exp_2
        # self.exp_1 = exp_1
        self.num_non_taylor = 0
        self.non_taylor_2 = non_taylor_2
        self.non_taylor_1 = non_taylor_1
        self.mo_non_taylor_del_2 = 0.
        self.mo_non_taylor_del_1 = 0.
        self.mo_non_taylor = 0.
        self.exp_non_taylor = 0.
        self.exp_non_taylor_history: list[float] = []
        self.exp_norm_taylor: list[float] = []
        self.exp_norm_non_taylor: list[float] = []

    def calc_result(self):
        self.num_non_taylor = 0.0
        self.mo_non_taylor = 0.0
        self.exp_non_taylor = 0.0
        self.exp_non_taylor_history = []
        self.exp_norm_taylor = []
        self.exp_norm_non_taylor = []
        n = len(self.cpi.life_horizon_dates)
        for i in range(n):
            self.cpi_cum = self.cpi.life_horizon_inflation_cum[i]
            self.roi_cum = self.roi.life_horizon_roi_cum[i]
            self.count_non_taylor(i)
            self.exp_non_taylor_history.append(self.exp_non_taylor)
            normalized_exp_non_taylor = (
                self.exp_non_taylor / self.cpi_cum
            )
            self.exp_norm_taylor.append(normalized_exp_non_taylor)
            self.exp_norm_non_taylor.append(normalized_exp_non_taylor)

        result = self.exp_non_taylor

        return result

    def count_al(self, date: str | pd.Timestamp) -> int:
        date_ts = pd.Timestamp(date)
        man_in_al = self.man_age_to_al <= age(date_ts, self.man_dob) < self.man_age_at_death
        woman_in_al = self.woman_age_to_al <= age(date_ts, self.woman_dob) < self.woman_age_at_death
        return int(man_in_al) + int(woman_in_al)

    def count_lc(self, date: str | pd.Timestamp) -> int:
        date_ts = pd.Timestamp(date)
        man_alive = age(date_ts, self.man_dob) < self.man_age_at_death
        woman_alive = age(date_ts, self.woman_dob) < self.woman_age_at_death
        return int(man_alive) + int(woman_alive)
    
    def count_non_taylor(self, i: int = 0) -> None:
        date = self.roi.life_horizon_dates[i]
        man_non_taylor = age(date, self.man_dob) < self.man_age_to_al
        woman_non_taylor = age(date, self.woman_dob) < self.woman_age_to_al
        self.num_non_taylor = int(man_non_taylor) + int(woman_non_taylor)
        self.mo_non_taylor_del_2 = self.cpi.life_horizon_inflation[i] * self.non_taylor_2
        self.mo_non_taylor_del_1 = self.cpi.life_horizon_inflation[i] * self.non_taylor_1
        self.non_taylor_2 += self.mo_non_taylor_del_2
        self.non_taylor_1 += self.mo_non_taylor_del_1
        if self.num_non_taylor == 2:
            self.mo_non_taylor = self.mo_non_taylor_del_2
            if i == 0:
                self.exp_non_taylor = self.non_taylor_2
        elif self.num_non_taylor == 1:
            self.mo_non_taylor = self.mo_non_taylor_del_1
            if i == 0:
                self.exp_non_taylor = self.non_taylor_1
        else:
            self.mo_non_taylor = 0
            if i == 0:
                self.exp_non_taylor = 0.
        self.exp_non_taylor += self.mo_non_taylor
        i = 1

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
        this_life.exp_norm_taylor,
        linewidth=2.0,
        linestyle="--",
        label="exp_norm_taylor",
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
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed for reproducible runs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    current_date = pd.Timestamp.today().normalize()
    roi = Roi(
        ticker=args.ticker,
        current_date=current_date,
        history_years=HISTORY_YEARS,
        start_clock=START_CLOCK,
        man_dob=MAN_DOB,
        woman_dob=WOMAN_DOB,
        man_age_at_death=MAN_AGE_AT_DEATH,
        woman_age_at_death=WOMAN_AGE_AT_DEATH,
    )
    roi.train(ticker=args.ticker)
    roi.project(ticker=args.ticker, seed=args.seed)
    cpi = Inflation(
        history_years=HISTORY_YEARS,
        al_cum_running_avg_yrs=AL_ESC_RUNNING_AVG_YRS,
        start_clock=START_CLOCK,
        man_dob=MAN_DOB,
        woman_dob=WOMAN_DOB,
        man_age_at_death=MAN_AGE_AT_DEATH,
        woman_age_at_death=WOMAN_AGE_AT_DEATH,
    )
    cpi.train(current_date=current_date)
    cpi.project(current_date=current_date, seed=args.seed)
    annualized_inflation = cpi.annualized_inflation
    if cpi.inflation_frame is None:
        raise ValueError("Inflation history was not loaded during projection.")
    inflation_frame = cpi.inflation_frame
    this_life = TaylorLife(roi=roi, cpi=cpi)
    this_life.calc_result()

    annualized_mean = (1 + roi.monthly_mean_return) ** 12 - 1
    annualized_mean_cpi = annualized_inflation
    print(
        f"Ticker: {args.ticker}\n"
        f"Historical monthly mean return: {roi.monthly_mean_return:.2%}\n"
        f"Implied annualized return: {annualized_mean:.2%}\n"
        f"Monthly volatility: {roi.monthly_volatility:.2%}\n"
        f"Seed: {args.seed}\n"
        f"CPI current date: {current_date.date()}\n"
        f"Implied annualized CPI inflation: {annualized_mean_cpi:.2%}"
    )
    print(roi)
    print(cpi)
    if roi.return_frame is None:
        raise ValueError("ROI history was not loaded during projection.")
    plot_projection_views(roi.return_frame, roi, show=False)
    plot_inflation_views(inflation_frame, cpi, show=False)
    plot_taylor_life_exp_non_taylor(this_life, show=False)
    plt.show()


if __name__ == "__main__":
    main()
