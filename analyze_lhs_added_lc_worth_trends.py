#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SCENARIO_COLUMNS = [
    "man_independent_yrs",
    "woman_independent_yrs",
    "man_assisted_yrs",
    "woman_assisted_yrs",
    "roi_mean_shift",
    "roi_vol_multiplier",
    "roi_mean_reversion",
    "inflation_mean_shift",
    "inflation_vol_multiplier",
    "inflation_mean_reversion",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze stochastic LHS rows near a target combined assisted-years value "
            "and compare low added_lc_worth_norm rows to nearby better rows."
        )
    )
    parser.add_argument(
        "--csv",
        default="lhs_taylor_results.csv",
        help="Path to the LHS results CSV. Default: lhs_taylor_results.csv",
    )
    parser.add_argument(
        "--target-x",
        type=float,
        default=10.0,
        help="Target combined assisted years on the x-axis. Default: 10.0",
    )
    parser.add_argument(
        "--band",
        type=float,
        default=0.75,
        help="Half-width of the assisted-years band around target-x. Default: 0.75",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1_000_000.0,
        help="Low-performance threshold for added_lc_worth_norm. Default: 1e6",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=12,
        help="How many rows to show in the low-case table. Default: 12",
    )
    return parser.parse_args()


def load_stochastic_rows(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    run_id_str = df["run_id"].astype(str)
    df = df[run_id_str.str.fullmatch(r"\d+")].copy()
    df["run_id"] = df["run_id"].astype(int)
    df["combined_assisted_yrs"] = df["man_assisted_yrs"] + df["woman_assisted_yrs"]
    df["combined_independent_yrs"] = df["man_independent_yrs"] + df["woman_independent_yrs"]
    df["assisted_split"] = df["man_assisted_yrs"] - df["woman_assisted_yrs"]
    df["independent_split"] = df["man_independent_yrs"] - df["woman_independent_yrs"]
    return df


def standardized_mean_diff(full_band: pd.DataFrame, low: pd.DataFrame, high: pd.DataFrame) -> pd.Series:
    values: dict[str, float] = {}
    numeric_cols = [
        "combined_independent_yrs",
        "assisted_split",
        "independent_split",
        *SCENARIO_COLUMNS,
        "exp_norm_lc",
        "exp_norm_al_cc",
        "exp_norm_total_cc",
        "worth_norm_lc",
        "worth_norm_cc",
        "added_lc_worth_norm",
    ]
    for column in numeric_cols:
        pooled_std = full_band[column].std(ddof=0)
        if pooled_std and not np.isnan(pooled_std):
            values[column] = (low[column].mean() - high[column].mean()) / pooled_std
    return pd.Series(values).sort_values()


def nearest_comparison(low: pd.DataFrame, high: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    if low.empty or high.empty:
        return pd.DataFrame()

    for _, low_row in low.iterrows():
        candidates = high.copy()
        candidates["distance"] = (
            (candidates["combined_assisted_yrs"] - low_row["combined_assisted_yrs"]).abs()
            + 0.25 * (candidates["combined_independent_yrs"] - low_row["combined_independent_yrs"]).abs()
        )
        best = candidates.sort_values(["distance", "added_lc_worth_norm"]).iloc[0]
        row: dict[str, float | int] = {
            "run_id": int(low_row["run_id"]),
            "match_run_id": int(best["run_id"]),
            "combined_assisted_yrs": float(low_row["combined_assisted_yrs"]),
            "added_lc_worth_norm": float(low_row["added_lc_worth_norm"]),
            "match_added_lc_worth_norm": float(best["added_lc_worth_norm"]),
            "gap_to_match": float(best["added_lc_worth_norm"] - low_row["added_lc_worth_norm"]),
        }
        for column in SCENARIO_COLUMNS + ["combined_independent_yrs"]:
            row[f"{column}_delta"] = float(low_row[column] - best[column])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("added_lc_worth_norm")


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    df = load_stochastic_rows(csv_path)

    band_rows = df[df["combined_assisted_yrs"].between(args.target_x - args.band, args.target_x + args.band)].copy()
    low = band_rows[band_rows["added_lc_worth_norm"] < args.threshold].copy()
    high = band_rows[band_rows["added_lc_worth_norm"] >= args.threshold].copy()

    print(f"CSV: {csv_path}")
    print(f"stochastic rows: {len(df)}")
    print(
        f"band: combined_assisted_yrs in [{args.target_x - args.band:.3f}, {args.target_x + args.band:.3f}]"
    )
    print(f"rows in band: {len(band_rows)}")
    print(f"low rows: added_lc_worth_norm < {args.threshold:,.0f} => {len(low)}")
    print()

    if band_rows.empty:
        print("No rows found in the requested band.")
        return
    if low.empty:
        print("No low rows found under the requested threshold.")
        return
    if high.empty:
        print("No comparison rows above the threshold were found in the requested band.")
        return

    corr_columns = [
        "added_lc_worth_norm",
        "combined_assisted_yrs",
        "combined_independent_yrs",
        "assisted_split",
        "independent_split",
        *SCENARIO_COLUMNS,
        "exp_norm_lc",
        "exp_norm_al_cc",
        "exp_norm_total_cc",
        "worth_norm_lc",
        "worth_norm_cc",
    ]
    corr = band_rows[corr_columns].corr(numeric_only=True)["added_lc_worth_norm"].sort_values()
    print("Correlations with added_lc_worth_norm inside band:")
    print(corr.to_string())
    print()

    compare_columns = [
        "combined_independent_yrs",
        "assisted_split",
        "independent_split",
        *SCENARIO_COLUMNS,
        "exp_norm_lc",
        "exp_norm_al_cc",
        "exp_norm_total_cc",
        "worth_norm_lc",
        "worth_norm_cc",
        "added_lc_worth_norm",
    ]
    comparison = pd.DataFrame(
        {
            "low_mean": low[compare_columns].mean(numeric_only=True),
            "high_mean": high[compare_columns].mean(numeric_only=True),
        }
    )
    comparison["low_minus_high"] = comparison["low_mean"] - comparison["high_mean"]
    comparison["std_mean_diff"] = standardized_mean_diff(band_rows, low, high)
    print("Low vs high mean comparison:")
    print(comparison.sort_values("std_mean_diff").to_string())
    print()

    show_columns = [
        "run_id",
        "combined_assisted_yrs",
        "combined_independent_yrs",
        "man_independent_yrs",
        "woman_independent_yrs",
        "man_assisted_yrs",
        "woman_assisted_yrs",
        "roi_mean_shift",
        "roi_vol_multiplier",
        "roi_mean_reversion",
        "inflation_mean_shift",
        "inflation_vol_multiplier",
        "inflation_mean_reversion",
        "added_lc_worth_norm",
    ]
    print(f"Worst {min(args.top_n, len(low))} low rows:")
    print(low.sort_values("added_lc_worth_norm")[show_columns].head(args.top_n).to_string(index=False))
    print()

    nearest = nearest_comparison(low, high)
    print("Nearest higher-performing comparison rows:")
    print(nearest.head(args.top_n).to_string(index=False))


if __name__ == "__main__":
    main()

