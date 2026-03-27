import argparse
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
        self.pile_lc = self.pile_at_start
        self.pile_cc = self.pile_at_start
        self.pile_norm_lc = self.pile_lc
        self.pile_norm_cc = self.pile_cc
        self.return_lc = 0.
        self.return_cc = 0.
        self.cpi_cum = 0.
        self.roi_cum = 0.

        self.num_al_cc = 0.
        self.al_cc_2 = al_cc_2
        self.al_cc_1 = al_cc_1
        self.mo_al_cc_del_2 = 0.
        self.mo_al_cc_del_1 = 0.
        self.mo_al_cc = 0.
        self.exp_al_cc = 0.
        self.exp_al_cc_history: list[float] = []
        self.exp_norm_al_cc: list[float] = []

        self.exp_al_lc_history = None
        self.exp_norm_al_lc = None

        self.num_cc = 0
        self.cc_2 = cc_2
        self.cc_1 = cc_1
        self.mo_cc_del_2 = 0.
        self.mo_cc_del_1 = 0.
        self.mo_cc = 0.
        self.exp_cc = 0.
        self.exp_cc_history: list[float] = []
        self.exp_norm_cc: list[float] = []

        self.num_lc = 0
        self.lc_2 = lc_2
        self.lc_1 = lc_1
        self.mo_lc_del_2 = 0.
        self.mo_lc_del_1 = 0.
        self.mo_lc = 0.
        self.exp_lc = 0.
        self.exp_lc_history: list[float] = []
        self.exp_norm_lc: list[float] = []

        self.num_non_taylor = 0
        self.non_taylor_2 = non_taylor_2
        self.non_taylor_1 = non_taylor_1
        self.mo_non_taylor_del_2 = 0.
        self.mo_non_taylor_del_1 = 0.
        self.mo_non_taylor = 0.
        self.exp_non_taylor = 0.
        self.exp_non_taylor_history: list[float] = []
        self.exp_norm_non_taylor: list[float] = []

        self.exp_norm_taylor: list[float] = []

    def calc_result(self):
        self.pile_lc = self.pile_at_start
        self.pile_cc = self.pile_at_start
        self.pile_norm_lc = self.pile_lc
        self.pile_norm_cc = self.pile_cc
        self.return_lc = 0.0
        self.return_cc = 0.0
        self.num_lc = 0.0
        self.mo_lc = 0.0
        self.exp_lc = 0.0
        self.num_non_taylor = 0.0
        self.mo_non_taylor = 0.0
        self.exp_non_taylor = 0.0
        self.exp_al_cc_history = []
        self.exp_norm_al_cc = []
        self.exp_cc_history = []
        self.exp_norm_cc = []
        self.exp_lc_history = []
        self.exp_norm_lc = []
        self.exp_non_taylor_history = []
        self.exp_norm_non_taylor = []
        n = len(self.cpi.life_horizon_dates)
        for i in range(n):
            self.cpi_cum = self.cpi.life_horizon_inflation_cum[i]
            self.roi_cum = self.roi.life_horizon_roi_cum[i]

            self.count_al_cc(i)
            self.exp_al_cc_history.append(self.exp_al_cc)
            if self.num_al_cc > 0:
                normalized_exp_al_cc = self.exp_al_cc / self.cpi_cum
            elif len(self.exp_norm_al_cc) > 0:
                normalized_exp_al_cc = self.exp_norm_al_cc[-1]
            else:
                normalized_exp_al_cc = 0.
            self.exp_norm_al_cc.append(normalized_exp_al_cc)

            # There are no assisted living expenses for plan lc
            self.exp_al_lc = 0.
            self.exp_norm_al_lc = 0.

            self.count_cc(i)
            self.exp_cc_history.append(self.exp_cc)
            if self.num_cc > 0:
                normalized_exp_cc = self.exp_cc / self.cpi_cum
            elif len(self.exp_norm_cc) > 0:
                normalized_exp_cc = self.exp_norm_cc[-1]
            else:
                normalized_exp_cc = 0.
            self.exp_norm_cc.append(normalized_exp_cc)

            self.count_lc(i)
            self.exp_lc_history.append(self.exp_lc)
            if self.num_lc > 0:
                normalized_exp_lc = self.exp_lc / self.cpi_cum
            elif len(self.exp_norm_lc) > 0:
                normalized_exp_lc = self.exp_norm_lc[-1]
            else:
                normalized_exp_lc = 0.
            self.exp_norm_lc.append(normalized_exp_lc)

            self.count_non_taylor(i)
            self.exp_non_taylor_history.append(self.exp_non_taylor)
            if self.num_non_taylor > 0:
                normalized_exp_non_taylor = self.exp_non_taylor / self.cpi_cum
            else:
                normalized_exp_non_taylor = self.exp_norm_non_taylor[-1]
            self.exp_norm_non_taylor.append(normalized_exp_non_taylor)

            self.return_lc = self.pile_lc * self.roi.life_horizon_roi[i]
            self.pile_lc += self.return_lc - self.exp_lc - self.exp_non_taylor - self.exp_al_lc

            self.return_cc = self.pile_cc * self.roi.life_horizon_roi[i]
            self.pile_cc += self.return_cc - self.exp_cc - self.exp_non_taylor - self.exp_al_cc

        self.exp_al_lc_history = np.array(self.exp_al_cc).copy()*0.
        self.exp_norm_al_lc = np.array(self.exp_norm_al_cc).copy()*0.

        self.pile_norm_lc = self.pile_lc * self.de_cumalate(self.cpi.life_horizon_dates[-1])
        self.pile_norm_cc = self.pile_cc * self.de_cumalate(self.cpi.life_horizon_dates[-1])

        result = int(self.pile_lc), int(self.pile_norm_lc), int(self.pile_cc), int(self.pile_norm_cc)

        return result

    def count_al_cc(self, i: int = 0) -> None:
        date = self.roi.life_horizon_dates[i]
        man_in_al_cc = self.man_age_to_al <= age(date, self.man_dob) < self.man_age_at_death
        woman_in_al_cc = self.woman_age_to_al <= age(date, self.woman_dob) < self.woman_age_at_death
        self.num_al_cc = int(man_in_al_cc) + int(woman_in_al_cc)
        self.mo_al_cc_del_2 = self.cpi.life_horizon_inflation[i] * self.al_cc_2
        self.mo_al_cc_del_1 = self.cpi.life_horizon_inflation[i] * self.al_cc_1
        self.al_cc_2 += self.mo_al_cc_del_2
        self.al_cc_1 += self.mo_al_cc_del_1
        if self.num_al_cc == 2:
            self.mo_al_cc = self.mo_al_cc_del_2
            if i == 0:
                self.exp_al_cc = self.al_cc_2
        elif self.num_al_cc == 1:
            self.mo_al_cc = self.mo_al_cc_del_1
            if i == 0:
                self.exp_al_cc = self.al_cc_1
        else:
            self.mo_al_cc = 0
            if i == 0:
                self.exp_al_cc = 0.
        self.exp_al_cc += self.mo_al_cc

    def count_cc(self, i: int = 0) -> None:
        date = self.roi.life_horizon_dates[i]
        man_in_cc = age(date, self.man_dob) < self.man_age_to_al
        woman_in_cc = age(date, self.man_dob) < self.woman_age_to_al
        self.num_cc = int(man_in_cc) + int(woman_in_cc)
        self.mo_cc_del_2 = self.cpi.life_horizon_inflation[i] * self.cc_2
        self.mo_cc_del_1 = self.cpi.life_horizon_inflation[i] * self.cc_1
        self.cc_2 += self.mo_cc_del_2
        self.cc_1 += self.mo_cc_del_1
        if self.num_cc == 2:
            self.mo_cc = self.mo_cc_del_2
            if i == 0:
                self.exp_cc = self.cc_2
        elif self.num_cc == 1:
            self.mo_cc = self.mo_cc_del_1
            if i == 0:
                self.exp_cc = self.cc_1
        else:
            self.mo_cc = 0
            if i == 0:
                self.exp_cc = 0.
        self.exp_cc += self.mo_cc

    def count_lc(self, i: int = 0) -> None:
        date = self.roi.life_horizon_dates[i]
        man_in_lc = age(date, self.man_dob) < self.man_age_to_al
        woman_in_lc = age(date, self.man_dob) < self.woman_age_to_al
        self.num_lc = int(man_in_lc) + int(woman_in_lc)
        self.mo_lc_del_2 = self.cpi.life_horizon_inflation[i] * self.lc_2
        self.mo_lc_del_1 = self.cpi.life_horizon_inflation[i] * self.lc_1
        self.lc_2 += self.mo_lc_del_2
        self.lc_1 += self.mo_lc_del_1
        if self.num_lc == 2:
            self.mo_lc = self.mo_lc_del_2
            if i == 0:
                self.exp_lc = self.lc_2
        elif self.num_lc == 1:
            self.mo_lc = self.mo_lc_del_1
            if i == 0:
                self.exp_lc = self.lc_1
        else:
            self.mo_lc = 0
            if i == 0:
                self.exp_lc = 0.
        self.exp_lc += self.mo_lc
    
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
        man_age_at_death=MAN_AGE_TO_AL + MAN_LINGER,
        woman_age_at_death=WOMAN_AGE_TO_AL + WOMAN_LINGER,
    )
    roi.train(ticker=args.ticker)
    roi.project(ticker=args.ticker, seed=args.seed)
    cpi = Inflation(
        history_years=HISTORY_YEARS,
        al_cum_running_avg_yrs=AL_ESC_RUNNING_AVG_YRS,
        start_clock=START_CLOCK,
        man_dob=MAN_DOB,
        woman_dob=WOMAN_DOB,
        man_age_at_death=MAN_AGE_TO_AL + MAN_LINGER,
        woman_age_at_death=WOMAN_AGE_TO_AL + WOMAN_LINGER,
    )
    cpi.train(current_date=current_date)
    cpi.project(current_date=current_date, seed=args.seed)
    annualized_inflation = cpi.annualized_inflation
    if cpi.inflation_frame is None:
        raise ValueError("Inflation history was not loaded during projection.")
    inflation_frame = cpi.inflation_frame
    this_life = TaylorLife(roi=roi, cpi=cpi)
    principal_lc, principal_norm_lc, principal_cc, principal_norm_cc = this_life.calc_result()

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
    # print(roi)
    # print(cpi)
    print(f"LC Plan A {principal_lc=} {principal_norm_lc=} {float(principal_lc)/float(principal_norm_lc)}"
          f"\nCC Plan B {principal_cc=} {principal_norm_cc=} {float(principal_cc)/float(principal_norm_cc)}")
    if roi.return_frame is None:
        raise ValueError("ROI history was not loaded during projection.")
    plot_projection_views(roi.return_frame, roi, show=False)
    plot_inflation_views(inflation_frame, cpi, show=False)
    plot_taylor_life_exp_non_taylor(this_life, show=False)
    plt.show()


if __name__ == "__main__":
    main()
