import argparse

import matplotlib.pyplot as plt
import pandas as pd

from Inflation import MONTHS_TO_PROJECT as INFLATION_MONTHS_TO_PROJECT
from Inflation import plot_inflation_views, prep_inflation
from Roi import MONTHS_TO_PROJECT, TICKER, Roi, plot_projection_views, prep_projection


START_CLOCK = "2026-07-01"
MAN_DOB = "1957-07-26"
WOMAN_DOB = "1956-04-11"
MAN_AGE_TO_AL = 71
WOMAN_AGE_TO_AL = 71
MAN_AGE_AT_DEATH = 77
WOMAN_AGE_AT_DEATH = 77
PILE_AT_START = 5700000


class Taylor_life:
    def __init__(
        self,
        projection: Roi,
        start_clock: str = START_CLOCK,
        man_dob: str = MAN_DOB,
        woman_dob: str = WOMAN_DOB,
        man_age_to_al: float = MAN_AGE_TO_AL,
        man_age_at_death: float = MAN_AGE_AT_DEATH,
        woman_age_to_al: float = WOMAN_AGE_TO_AL,
        woman_age_at_death: float = WOMAN_AGE_AT_DEATH,
        pile_at_start: float = PILE_AT_START,
        esc: float = 1,
        exp: float = 1000,
    ) -> None:
        self.projection = projection
        self.start_clock = start_clock
        self.man_dob = man_dob
        self.woman_dob = woman_dob
        self.man_age_to_al = man_age_to_al
        self.man_age_at_death = man_age_at_death
        self.woman_age_to_al = woman_age_to_al
        self.woman_age_at_death = woman_age_at_death
        self.pile_at_start = pile_at_start
        self.esc_al = esc_al
        self.esc_cc = esc_cc
        self.non_Taylor_exp = non_Taylor_exp
        self.exp_2 = exp_2
        self.exp_1 = exp_1

    def age(self, date: str | pd.Timestamp, birth_date: str | pd.Timestamp) -> float:
        date_ts = pd.Timestamp(date)
        birth_ts = pd.Timestamp(birth_date)
        return float((date_ts - birth_ts).days / 365.2425)

    def deceased(self, date: str | pd.Timestamp) -> bool:
        date_ts = pd.Timestamp(date)
        man_deceased = self.age(date_ts, self.man_dob) >= self.man_age_at_death
        woman_deceased = self.age(date_ts, self.woman_dob) >= self.woman_age_at_death
        return man_deceased and woman_deceased

    def al(self, date: str | pd.Timestamp) -> bool:
        date_ts = pd.Timestamp(date)
        man_in_al = self.man_age_to_al <= self.age(date_ts, self.man_dob) < self.man_age_at_death
        woman_in_al = self.woman_age_to_al <= self.age(date_ts, self.woman_dob) < self.woman_age_at_death
        return man_in_al or woman_in_al

    def cc_lc(self, date: str | pd.Timestamp) -> int:
        date_ts = pd.Timestamp(date)
        man_alive = self.age(date_ts, self.man_dob) < self.man_age_at_death
        woman_alive = self.age(date_ts, self.woman_dob) < self.woman_age_at_death
        return int(man_alive) + int(woman_alive)

    def de_escalate(
        self,
        date: str | pd.Timestamp,
        projection: Roi | None = None,
    ) -> float:
        date_ts = pd.Timestamp(date)
        start_ts = pd.Timestamp(self.start_clock)
        projection_to_use = projection if projection is not None else self.projection

        if date_ts <= start_ts:
            return 1.0

        full_growth_since_start = 1.0
        for item in projection_to_use.monthly_roi:
            if start_ts < item.month <= date_ts:
                full_growth_since_start *= 1 + item.roi

        return 1 / full_growth_since_start


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monte Carlo monthly ROI projection anchored to historical long-run growth."
    )
    parser.add_argument("--ticker", default=TICKER, help="Ticker symbol to download, default: SPY")
    parser.add_argument(
        "--months",
        type=int,
        default=min(MONTHS_TO_PROJECT, INFLATION_MONTHS_TO_PROJECT),
        help="Months to project",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed for reproducible runs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    roi, monthly_mean_return, monthly_volatility, return_frame = prep_projection(
        ticker=args.ticker,
        months_to_project=args.months,
        seed=args.seed,
    )
    current_date = pd.Timestamp.today().normalize()
    inflation, annualized_inflation, inflation_frame = prep_inflation(
        current_date=current_date,
        months_to_project=args.months,
        seed=args.seed,
    )

    annualized_mean = (1 + monthly_mean_return) ** 12 - 1
    annualized_mean_cpi = annualized_inflation
    print(
        f"Ticker: {args.ticker}\n"
        f"Historical monthly mean return: {monthly_mean_return:.2%}\n"
        f"Implied annualized return: {annualized_mean:.2%}\n"
        f"Monthly volatility: {monthly_volatility:.2%}\n"
        f"Seed: {args.seed}\n"
        f"CPI current date: {current_date.date()}\n"
        f"Implied annualized CPI inflation: {annualized_mean_cpi:.2%}"
    )
    print(roi)
    print(inflation)
    plot_projection_views(return_frame, roi, show=False)
    plot_inflation_views(inflation_frame, inflation, show=False)
    plt.show()


if __name__ == "__main__":
    main()
