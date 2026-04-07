import argparse
import matplotlib.pyplot as plt
from Roi import TICKER
from Run_one_Taylor import configurate_local_run, configurate_base_run, run_one, merge_run_config
from default_case import (
    DEFAULT_CURRENT_DATE,
    DEFAULT_SEED,
    PILE_AT_START,
    load_default_case,
)


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


def plot_run_many(results: list[dict], titl='') -> None:
    groups = {}
    for row in results:
        key = f"{row['roi_fixed']}_{row['cpi_fixed']}"
        groups.setdefault(key, []).append(row)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("LC benefit and Worth" + titl)
    plt.text(0.2, 1.2, titl,
             horizontalalignment='left',
             verticalalignment='top',
             fontsize=14,
             bbox=dict(facecolor='lightblue', alpha=0.5, pad=5))
            # bbox = dict(facecolor='yellow', alpha=0.5, pad=5))


    for label, rows in groups.items():
        rows_al = sorted(rows, key=lambda r: r["yrs_al_total"])

        x_al = [r["yrs_al_total"] for r in rows_al]
        diff_al = [r["worth_norm_lc"] - r["worth_norm_cc"] for r in rows_al]
        cc_al = [r["worth_norm_cc"] for r in rows_al]
        lc_al = [r["worth_norm_lc"] for r in rows_al]

        ax1.plot(x_al, diff_al, marker="o", label=f"lc_benefit {label}")

        ax2.plot(x_al, cc_al, marker="o", linestyle="--", color="orange", label=f"worth {label} cc")
        ax2.plot(x_al, lc_al, marker="o", color="lightblue", label=f"worth {label} lc")

    for ax in [ax1, ax2]:
        ax.set_xlabel("yrs_al_total")
        ax.set_ylabel("worth norm ($M)")
        ax.legend()
        ax.grid(True)
        plt.text(0.2, 0.9, "Key:  ROI_CPI ",
                 horizontalalignment='left',
                 verticalalignment='top',
                 transform=plt.gca().transAxes,
                 fontsize=12,
                 color='lightblue',
                 bbox=dict(alpha=0.5, pad=5))
                # bbox = dict(facecolor='yellow', alpha=0.5, pad=5))
    plt.tight_layout()
    plt.show()


def main() -> None:
    args = parse_args()
    active_case_name = RUN_ONE_CASE_NAME
    base_run_config = configurate_base_run(args)

    case_run_config = None
    if active_case_name is not None:
        case_scenario_kwargs, case_context_kwargs = load_default_case(active_case_name)
        case_run_config = {
            "scenario": case_scenario_kwargs,
            "context": case_context_kwargs,
        }

    # yrs_il_w, yrs_il_m = [15, 13.]
    yrs_il_woman, yrs_il_man = [10, 8.8]
    # yrs_il_w, yrs_il_m = [5, 4.4]
    ins = [
        [0, 8, 4],
        [1, 8, 4],
        [2, 8, 4],
        [4, 8, 4],
        [8, 8, 4],
        [16, 8, 4],
        [0, 4, 4],
        [1, 4, 4],
        [2, 4, 4],
        [4, 4, 4],
        [8, 4, 4],
        [16, 4, 4],
        [0, 2, 4],
        [1, 2, 4],
        [2, 2, 4],
        [4, 2, 4],
        [8, 2, 4],
        [16, 2, 4],
        [0, 0, 4],
        [1, 0, 4],
        [2, 0, 4],
        [4, 0, 4],
        [8, 0, 4],
        [16, 0, 4],
        [0, 0, 0],
        [1, 0, 0],
        [2, 0, 0],
        [0, 0, 0],
        [8, 0, 0],
        [16, 0, 0],
        [0, 4, 0],
        [1, 4, 0],
        [2, 4, 0],
        [4, 4, 0],
        [8, 4, 0],
        [16, 4, 0],
        [0, 4, 8],
        [1, 4, 8],
        [2, 4, 8],
        [4, 8, 4],
        [8, 4, 8],
        [16, 4, 8],
        [0, 4, 2],
        [1, 4, 2],
        [2, 4, 2],
        [4, 2, 4],
        [8, 4, 2],
        [16, 4, 2],
        # [0, 0, 0],
    ]

    cc_worth_norm = []
    results = []
    csv_file = open('run_many_taylor.csv', 'w')
    header = (
        f"yrs_al_total,"
        f"roi_fixed,"
        f"cpi_fixed,"
        f"yrs_il_man,"
        f"yrs_il_woman,"
        f"yrs_al_total,"

        f" |*|,"
        f"start_pile,"
        f" |,"
        f"cum_mo_earn_ss_norm,"
        f"cum_mo_earn_pen_norm,"
        f"cum_mo_earn_inv_cc_norm,"
        f" |,"
        f"cum_mo_earn_total_cc_norm,"
        f"cum_mo_earn_cc_norm,"
        f" |,"
        f"entrance_fee_cc,"
        f"cum_mo_exp_cc_norm,"
        f"cum_mo_exp_al_cc_norm,"
        f"cum_mo_exp_non_taylor_norm,"
        f" |,"
        f"cum_mo_exp_total_cc_norm,"
        f" |,"
        f"worth_norm_cc,"

        f" |*|,"
        f"start_pile,"
        f" |,"
        f"cum_mo_earn_ss_norm,"
        f"cum_mo_earn_pen_norm,"
        f"cum_mo_earn_inv_lc_norm,"
        f" |,"
        f"cum_mo_earn_total_lc_norm,"
        f"cum_mo_earn_lc_norm,"
        f" |,"
        f"entrance_fee_lc,"
        f"cum_mo_exp_lc_norm,"
        f"cum_mo_exp_al_lc_norm,"
        f"cum_mo_exp_non_taylor_norm,"
        f" |,"
        f"cum_mo_exp_total_lc_norm,"
        f" |,"
        f"worth_norm_lc,"
    )
    print(header)
    csv_file.write(header + "\n")
    for [yrs_al_total, roi_fixed, cpi_fixed] in ins:
        local_run_overrides = configurate_local_run(yrs_al_total=yrs_al_total, roi_fixed=roi_fixed, cpi_fixed=cpi_fixed,
                                                    yrs_il_man=yrs_il_man, yrs_il_woman=yrs_il_woman)
        run_config = merge_run_config(base_run_config, case_run_config, local_run_overrides)
        r = run_one(run_config=run_config, active_case_name=active_case_name, plot=False, printing=False)


        row = (
            f" {yrs_al_total:2d},"
            f" {roi_fixed:6.3f},"
            f" {cpi_fixed:6.3f},"
            f" {yrs_il_man:5.2f},"
            f" {yrs_il_woman:5.2f},"
            f" {yrs_al_total:5.2f},"

            f" |*|,"
            f" {r['start_pile']:6.3f},"
            f" |,"
            f" {r['cum_mo_earn_ss_norm']:6.3f},"
            f" {r['cum_mo_earn_pen_norm']:6.3f},"
            f" {r['cum_mo_earn_inv_cc_norm']:6.3f},"
            f" |,"
            f" {r['cum_mo_earn_total_cc_norm']:6.3f},"
            f" {r['cum_mo_earn_cc_norm']:6.3f},"
            f" |,"
            f" {r['entrance_fee_cc']:6.3f},"
            f" {r['cum_mo_exp_cc_norm']:6.3f},"
            f" {r['cum_mo_exp_al_cc_norm']:6.3f},"
            f" {r['cum_mo_exp_non_taylor_norm']:6.3f},"
            f" |,"
            f" {r['cum_mo_exp_total_cc_norm']:6.3f},"
            f" |,"
            f" {r['worth_norm_cc']:6.3f},"

            f" |*|,"
            f" {r['start_pile']:6.3f},"
            f" |,"
            f" {r['cum_mo_earn_ss_norm']:6.3f},"
            f" {r['cum_mo_earn_pen_norm']:6.3f},"
            f" {r['cum_mo_earn_inv_lc_norm']:6.3f},"
            f" |,"
            f" {r['cum_mo_earn_total_lc_norm']:6.3f},"
            f" {r['cum_mo_earn_lc_norm']:6.3f},"
            f" |,"
            f" {r['entrance_fee_lc']:6.3f},"
            f" {r['cum_mo_exp_lc_norm']:6.3f},"
            f" {r['cum_mo_exp_al_lc_norm']:6.3f},"
            f" {r['cum_mo_exp_non_taylor_norm']:6.3f},"
            f" |,"
            f" {r['cum_mo_exp_total_lc_norm']:6.3f},"
            f" |,"
            f" {r['worth_norm_lc']:6.3f},"
        )
        print(row)
        csv_file.write(row + "\n")
        cc_worth_norm.append(0)
        results.append({
            "yrs_al_total": yrs_al_total,
            "roi_fixed": roi_fixed,
            "cpi_fixed": cpi_fixed,
            "yrs_il_man": yrs_il_man,
            "yrs_il_woman": yrs_il_woman,
            "worth_norm_cc": r["worth_norm_cc"],
            "worth_norm_lc": r["worth_norm_lc"],
        })
    csv_file.close()
    print(f"\nResults written to run_many_taylor.csv")
    plot_run_many(results, titl=f"   {max(yrs_il_woman, yrs_il_man)} yrs Independent Living" + f" ${PILE_AT_START/1e6}M start")

if __name__ == "__main__":
    main()
