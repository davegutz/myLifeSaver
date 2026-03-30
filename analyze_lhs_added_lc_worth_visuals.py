#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
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
            "Visual trend analyzer for low added_lc_worth_norm cases near a target "
            "combined assisted-years band in lhs_taylor_results.csv."
        )
    )
    parser.add_argument("--csv", default="lhs_taylor_results.csv", help="Path to results CSV")
    parser.add_argument("--target-x", type=float, default=10.0, help="Target combined assisted years")
    parser.add_argument("--band", type=float, default=0.75, help="Half-width around target-x")
    parser.add_argument(
        "--threshold",
        type=float,
        default=1_000_000.0,
        help="Low-case threshold for added_lc_worth_norm",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many top separator parameters to include in the pairplot",
    )
    parser.add_argument(
        "--pair-max-rows",
        type=int,
        default=300,
        help="Max rows sampled into pairplot for readability",
    )
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed for pairplot")
    parser.add_argument("--outdir", default="analysis_outputs", help="Directory for output files")
    parser.add_argument("--prefix", default="lhs_added_lc_worth", help="Output filename prefix")
    parser.add_argument("--no-show", action="store_true", help="Do not open plot windows")
    return parser.parse_args()


def load_stochastic_rows(csv_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    run_id_str = frame["run_id"].astype(str)
    frame = frame[run_id_str.str.fullmatch(r"\d+")].copy()
    frame["run_id"] = frame["run_id"].astype(int)

    # --- combined totals ---
    frame["combined_assisted_yrs"] = frame["man_assisted_yrs"] + frame["woman_assisted_yrs"]
    frame["combined_independent_yrs"] = frame["man_independent_yrs"] + frame["woman_independent_yrs"]

    # --- LC-duration metrics (keep assisted_split for intermediate use only) ---
    frame["assisted_split"] = frame["man_assisted_yrs"] - frame["woman_assisted_yrs"]
    frame["independent_split"] = frame["man_independent_yrs"] - frame["woman_independent_yrs"]

    # solo_lc_yrs: years one person is in LC while the other is NOT
    # = |man_assisted − woman_assisted|  (identical to abs(assisted_split))
    # Since LC assisted-living costs are 0, the sign of the split carries no extra information;
    # solo_lc_yrs is the non-redundant primary metric.
    frame["solo_lc_yrs"] = frame["assisted_split"].abs()

    # overlap_lc_yrs: years BOTH are in LC simultaneously
    # combined = solo + 2*overlap  →  overlap = (combined − solo) / 2
    frame["overlap_lc_yrs"] = (frame["combined_assisted_yrs"] - frame["solo_lc_yrs"]) / 2

    return frame


def separator_table(band_rows: pd.DataFrame, low: pd.DataFrame, high: pd.DataFrame) -> pd.DataFrame:
    # Note: assisted_split is DROPPED — it is perfectly correlated with solo_lc_yrs
    # (solo_lc_yrs = abs(assisted_split); since LC costs are 0 the sign is uninformative).
    candidates = [
        "combined_independent_yrs",
        "combined_assisted_yrs",
        "solo_lc_yrs",       # primary LC-imbalance metric
        "overlap_lc_yrs",    # complementary: time both are in LC together
        "independent_split",
        *SCENARIO_COLUMNS,
    ]
    rows: list[dict[str, float | str]] = []
    for col in candidates:
        pooled_std = float(band_rows[col].std(ddof=0)) if col in band_rows else 0.0
        low_mean = float(low[col].mean())
        high_mean = float(high[col].mean())
        diff = low_mean - high_mean
        std_diff = diff / pooled_std if pooled_std > 0 else np.nan
        rows.append(
            {
                "parameter": col,
                "low_mean": low_mean,
                "high_mean": high_mean,
                "low_minus_high": diff,
                "std_mean_diff": std_diff,
                "abs_std_mean_diff": abs(std_diff) if not np.isnan(std_diff) else np.nan,
                "low_q25": float(low[col].quantile(0.25)),
                "low_q75": float(low[col].quantile(0.75)),
                "high_q25": float(high[col].quantile(0.25)),
                "high_q75": float(high[col].quantile(0.75)),
            }
        )
    table = pd.DataFrame(rows).sort_values("abs_std_mean_diff", ascending=False)
    return table


def plot_scatter(
    full: pd.DataFrame,
    low: pd.DataFrame,
    high: pd.DataFrame,
    target_x: float,
    band: float,
    threshold: float,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.scatter(
        full["combined_assisted_yrs"],
        full["added_lc_worth_norm"],
        s=14,
        alpha=0.2,
        color="gray",
        label="stochastic",
    )
    if not high.empty:
        ax.scatter(
            high["combined_assisted_yrs"],
            high["added_lc_worth_norm"],
            s=30,
            alpha=0.7,
            color="#2ca02c",
            label="in-band high",
        )
    if not low.empty:
        ax.scatter(
            low["combined_assisted_yrs"],
            low["added_lc_worth_norm"],
            s=40,
            alpha=0.9,
            color="#d62728",
            label="in-band low",
            edgecolor="black",
            linewidth=0.3,
        )

    ax.axhline(threshold, color="#d62728", linestyle="--", linewidth=1.2, label=f"threshold={threshold:,.0f}")
    ax.axvline(target_x, color="black", linestyle=":", linewidth=1.2, label=f"target x={target_x:.2f}")
    ax.axvspan(target_x - band, target_x + band, color="#1f77b4", alpha=0.08, label=f"band +/-{band}")

    ax.set_xlabel("combined_assisted_yrs")
    ax.set_ylabel("added_lc_worth_norm")
    ax.set_title("Stochastic points with low-cluster highlight")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)


def plot_pair(
    band_rows: pd.DataFrame,
    sep_table: pd.DataFrame,
    top_k: int,
    pair_max_rows: int,
    seed: int,
    out_path: Path,
) -> None:
    if band_rows.empty:
        return

    top_features = sep_table["parameter"].head(max(1, top_k)).tolist()
    plot_cols = [*top_features, "added_lc_worth_norm", "group"]
    pair_data = band_rows[plot_cols].copy()

    if len(pair_data) > pair_max_rows:
        pair_data = pair_data.sample(n=pair_max_rows, random_state=seed)

    # Try seaborn pairplot first. Fallback to pandas scatter matrix if seaborn is unavailable.
    try:
        import seaborn as sns  # type: ignore

        grid = sns.pairplot(
            pair_data,
            vars=[*top_features, "added_lc_worth_norm"],
            hue="group",
            corner=True,
            diag_kind="hist",
            plot_kws={"s": 16, "alpha": 0.6},
            diag_kws={"alpha": 0.6},
        )
        grid.figure.suptitle("Pairplot of top separator parameters", y=1.02)
        grid.figure.savefig(out_path, dpi=160, bbox_inches="tight")
        plt.close(grid.figure)
    except Exception:
        from pandas.plotting import scatter_matrix

        color_map = pair_data["group"].map({"low": "#d62728", "high": "#2ca02c"}).fillna("gray")
        axes = scatter_matrix(
            pair_data[[*top_features, "added_lc_worth_norm"]],
            figsize=(11, 11),
            diagonal="hist",
            alpha=0.6,
            color=color_map,
        )
        fig = axes[0, 0].figure
        fig.suptitle("Pair matrix of top separator parameters", y=0.92)
        fig.tight_layout()
        fig.savefig(out_path, dpi=160)
        plt.close(fig)


def plot_solo_lc_driver(
    full: pd.DataFrame,
    band_rows: pd.DataFrame,
    low: pd.DataFrame,
    high: pd.DataFrame,
    threshold: float,
    out_path: Path,
) -> None:
    """Side-by-side comparison: combined_assisted_yrs vs solo_lc_yrs as the real driver."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    fig.suptitle(
        "Solo-LC Imbalance is the Key Driver — Not Total LC Duration",
        fontsize=13,
        fontweight="bold",
    )

    panel_specs = [
        ("combined_assisted_yrs", "Combined Assisted Years\n(total couple LC duration)"),
        ("solo_lc_yrs", "Solo-LC Years\n(|man_assisted − woman_assisted|)"),
    ]

    for ax, (xcol, xlabel) in zip(axes, panel_specs):
        # background: all stochastic rows
        ax.scatter(
            full[xcol],
            full["added_lc_worth_norm"],
            s=10,
            alpha=0.15,
            color="gray",
            label="all stochastic",
        )
        # in-band high
        if not high.empty:
            ax.scatter(
                high[xcol],
                high["added_lc_worth_norm"],
                s=35,
                alpha=0.7,
                color="#2ca02c",
                label="in-band high",
                zorder=3,
            )
        # in-band low
        if not low.empty:
            ax.scatter(
                low[xcol],
                low["added_lc_worth_norm"],
                s=50,
                alpha=0.9,
                color="#d62728",
                edgecolor="black",
                linewidth=0.4,
                label="in-band low",
                zorder=4,
            )

        # threshold line
        ax.axhline(threshold, color="#d62728", linestyle="--", linewidth=1.2,
                   label=f"threshold {threshold:,.0f}")

        # trend lines for in-band low and high separately
        for subset, color, label in [
            (high, "#2ca02c", "high trend"),
            (low, "#d62728", "low trend"),
        ]:
            if len(subset) >= 3:
                x_vals = subset[xcol].values
                y_vals = subset["added_lc_worth_norm"].values
                z = np.polyfit(x_vals, y_vals, 1)
                p = np.poly1d(z)
                x_line = np.linspace(x_vals.min(), x_vals.max(), 120)
                ax.plot(x_line, p(x_line), color=color, linewidth=2.0,
                        linestyle="-", alpha=0.85, label=label, zorder=5)

        # Pearson r annotation
        if not band_rows.empty and band_rows[xcol].std() > 0:
            r = float(band_rows[[xcol, "added_lc_worth_norm"]].corr().iloc[0, 1])
            ax.annotate(
                f"r = {r:+.3f}\n(in-band rows)",
                xy=(0.97, 0.04),
                xycoords="axes fraction",
                ha="right",
                va="bottom",
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.85),
            )

        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel("added_lc_worth_norm" if ax is axes[0] else "", fontsize=11)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)

    axes[0].set_title("What you might expect to drive outcomes", fontsize=10, style="italic")
    axes[1].set_title("The actual primary driver of bad outcomes", fontsize=10, style="italic",
                       color="#b22222")

    # annotation box on right panel explaining the insight
    axes[1].annotate(
        "Low-worth cases cluster at\nhigh solo_lc_yrs: one person\nin LC alone for far longer.",
        xy=(0.03, 0.97),
        xycoords="axes fraction",
        ha="left",
        va="top",
        fontsize=9,
        color="#7f0000",
        bbox=dict(boxstyle="round,pad=0.35", fc="#fff5f5", ec="#d62728", alpha=0.9),
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_solo_lc_vs_added_worth(
    full: pd.DataFrame,
    band_rows: pd.DataFrame,
    low: pd.DataFrame,
    high: pd.DataFrame,
    threshold: float,
    out_path: Path,
) -> None:
    """Dedicated scatter of added_lc_worth_norm vs solo_lc_yrs across all stochastic rows,
    with in-band low/high groups highlighted and a full-dataset trend line."""
    fig, ax = plt.subplots(figsize=(11, 7))
    fig.suptitle(
        "added_lc_worth_norm  vs  solo_lc_yrs\n"
        "(years one person is in LC while the other is not)",
        fontsize=12,
        fontweight="bold",
    )

    # background: all stochastic rows
    ax.scatter(
        full["solo_lc_yrs"],
        full["added_lc_worth_norm"],
        s=10,
        alpha=0.18,
        color="gray",
        label="all stochastic",
        zorder=1,
    )

    # in-band high
    if not high.empty:
        ax.scatter(
            high["solo_lc_yrs"],
            high["added_lc_worth_norm"],
            s=35,
            alpha=0.75,
            color="#2ca02c",
            label="in-band high",
            zorder=3,
        )

    # in-band low
    if not low.empty:
        ax.scatter(
            low["solo_lc_yrs"],
            low["added_lc_worth_norm"],
            s=50,
            alpha=0.90,
            color="#d62728",
            edgecolor="black",
            linewidth=0.4,
            label="in-band low",
            zorder=4,
        )

    # trend line across ALL stochastic rows
    if len(full) >= 3 and full["solo_lc_yrs"].std() > 0:
        z = np.polyfit(full["solo_lc_yrs"].values, full["added_lc_worth_norm"].values, 1)
        p = np.poly1d(z)
        x_line = np.linspace(full["solo_lc_yrs"].min(), full["solo_lc_yrs"].max(), 200)
        r_full = float(full[["solo_lc_yrs", "added_lc_worth_norm"]].corr().iloc[0, 1])
        ax.plot(x_line, p(x_line), color="steelblue", linewidth=2.0,
                linestyle="--", alpha=0.85,
                label=f"full-dataset trend  (r={r_full:+.3f})", zorder=5)

    # threshold line
    ax.axhline(threshold, color="#d62728", linestyle="--", linewidth=1.2,
               label=f"threshold {threshold:,.0f}")

    # Pearson r annotation (in-band rows)
    if not band_rows.empty and band_rows["solo_lc_yrs"].std() > 0:
        r_band = float(band_rows[["solo_lc_yrs", "added_lc_worth_norm"]].corr().iloc[0, 1])
        ax.annotate(
            f"in-band r = {r_band:+.3f}\n({len(band_rows)} rows)",
            xy=(0.97, 0.06),
            xycoords="axes fraction",
            ha="right",
            va="bottom",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.88),
        )

    ax.set_xlabel("solo_lc_yrs  (|man_assisted − woman_assisted|)", fontsize=11)
    ax.set_ylabel("added_lc_worth_norm", fontsize=11)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_separator_effect_sizes(sep_table: pd.DataFrame, top_k: int, out_path: Path) -> None:
    table = sep_table[["parameter", "std_mean_diff"]].dropna().copy()
    if table.empty:
        return
    top = table.reindex(table["std_mean_diff"].abs().sort_values(ascending=False).index).head(max(1, top_k))
    top = top.iloc[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#d62728" if value < 0 else "#2ca02c" for value in top["std_mean_diff"]]
    ax.barh(top["parameter"], top["std_mean_diff"], color=colors, alpha=0.85)
    ax.axvline(0.0, color="black", linewidth=1.0)
    ax.set_xlabel("standardized mean difference (low - high)")
    ax.set_ylabel("parameter")
    ax.set_title("Top separator effect sizes")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    full = load_stochastic_rows(csv_path)
    band_rows = full[full["combined_assisted_yrs"].between(args.target_x - args.band, args.target_x + args.band)].copy()
    if band_rows.empty:
        print("No stochastic rows in requested x-band.")
        return

    band_rows["group"] = np.where(band_rows["added_lc_worth_norm"] < args.threshold, "low", "high")
    low = band_rows[band_rows["group"] == "low"].copy()
    high = band_rows[band_rows["group"] == "high"].copy()

    print(f"CSV: {csv_path}")
    print(f"stochastic rows: {len(full)}")
    print(f"x band: [{args.target_x - args.band:.3f}, {args.target_x + args.band:.3f}]")
    print(f"rows in band: {len(band_rows)}")
    print(f"low rows (< {args.threshold:,.0f}): {len(low)}")
    print(f"high rows: {len(high)}")

    if low.empty or high.empty:
        print("Need both low and high rows in-band for separator analysis.")
        return

    sep = separator_table(band_rows, low, high)
    sep_csv = outdir / f"{args.prefix}_separator_table.csv"
    sep_txt = outdir / f"{args.prefix}_separator_table.txt"
    sep.to_csv(sep_csv, index=False)
    sep_txt.write_text(sep.to_string(index=False), encoding="utf-8")

    scatter_png = outdir / f"{args.prefix}_scatter.png"
    pair_png = outdir / f"{args.prefix}_pairplot.png"
    effect_png = outdir / f"{args.prefix}_separator_effect_sizes.png"
    solo_lc_png = outdir / f"{args.prefix}_solo_lc_driver.png"
    solo_lc_worth_png = outdir / f"{args.prefix}_solo_lc_vs_worth.png"

    plot_scatter(full, low, high, args.target_x, args.band, args.threshold, scatter_png)
    plot_pair(band_rows, sep, args.top_k, args.pair_max_rows, args.seed, pair_png)
    plot_separator_effect_sizes(sep, args.top_k, effect_png)
    plot_solo_lc_driver(full, band_rows, low, high, args.threshold, solo_lc_png)
    plot_solo_lc_vs_added_worth(full, band_rows, low, high, args.threshold, solo_lc_worth_png)

    # --- console report for solo_lc_yrs ---
    r_full = float(full[["solo_lc_yrs", "added_lc_worth_norm"]].corr().iloc[0, 1])
    r_band = float(band_rows[["solo_lc_yrs", "added_lc_worth_norm"]].corr().iloc[0, 1]) if band_rows["solo_lc_yrs"].std() > 0 else float("nan")
    print(f"\nsolo_lc_yrs Pearson r (full dataset): {r_full:+.4f}")
    print(f"solo_lc_yrs Pearson r (in-band rows):  {r_band:+.4f}")
    print(f"  in-band low  solo_lc_yrs mean: {low['solo_lc_yrs'].mean():.3f}  median: {low['solo_lc_yrs'].median():.3f}")
    print(f"  in-band high solo_lc_yrs mean: {high['solo_lc_yrs'].mean():.3f}  median: {high['solo_lc_yrs'].median():.3f}")

    print("\nTop separators by |std_mean_diff|:")
    print(sep[["parameter", "std_mean_diff", "low_minus_high"]].head(args.top_k).to_string(index=False))
    print("\nWrote:")
    print(f"  {sep_csv}")
    print(f"  {sep_txt}")
    print(f"  {scatter_png}")
    print(f"  {pair_png}")
    print(f"  {effect_png}")
    print(f"  {solo_lc_png}")
    print(f"  {solo_lc_worth_png}")

    if not args.no_show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
'''
garbage.  ai found corellation between two inputs already known to be perfectly corellated.  Need a different study using SageMath / R
'''