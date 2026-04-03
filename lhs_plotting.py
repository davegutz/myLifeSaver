from __future__ import annotations

from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

AnnotationFormatter = Callable[[pd.Series], str]
ReferenceLineAdder = Callable[[plt.Axes], None]


def _lhs_rows(results: pd.DataFrame) -> pd.DataFrame:
    return results[results["run_id"].apply(lambda v: isinstance(v, int))]


def _centerpoint_rows(results: pd.DataFrame) -> pd.DataFrame:
    return results[results["run_id"].apply(lambda v: str(v) == "CENTERPOINT")]


def compute_extreme_indices(lhs_rows: pd.DataFrame) -> dict[str, int]:
    if lhs_rows.empty:
        return {}
    return {
        "f1_min": int(lhs_rows["added_lc_worth_norm"].idxmin()),
        "f1_max": int(lhs_rows["added_lc_worth_norm"].idxmax()),
        "f2_lc_min": int(lhs_rows["worth_norm_lc"].idxmin()),
        "f2_lc_max": int(lhs_rows["worth_norm_lc"].idxmax()),
        "f2_cc_min": int(lhs_rows["worth_norm_cc"].idxmin()),
        "f2_cc_max": int(lhs_rows["worth_norm_cc"].idxmax()),
    }


def _append_index_tag(index_tags: dict[int, list[str]], index: int | None, tag: str) -> None:
    if index is None:
        return
    tags = index_tags.setdefault(index, [])
    if tag not in tags:
        tags.append(tag)


def _annotate_tagged_points(
    axis: plt.Axes,
    lhs_rows: pd.DataFrame,
    index_tags: dict[int, list[str]],
    y_column: str,
    annotation_formatter: AnnotationFormatter | None,
) -> None:
    offsets = [(10, 10), (10, -18), (-90, 12), (-90, -20), (12, 28), (-102, 28)]
    for i, (idx, tags) in enumerate(index_tags.items()):
        if idx not in lhs_rows.index:
            continue
        row = lhs_rows.loc[idx]
        label = ", ".join(tags)
        if annotation_formatter is not None:
            label = f"{label}\n{annotation_formatter(row)}"
        axis.annotate(
            label,
            xy=(float(row["yrs_sum_al"]), float(row[y_column])),
            xytext=offsets[i % len(offsets)],
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="red", alpha=0.9),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0", color="red", lw=1.2),
            fontsize=8,
            ha="left",
        )


def _annotate_tagged_points_outside(
    axis: plt.Axes,
    lhs_rows: pd.DataFrame,
    index_tags: dict[int, list[str]],
    y_column: str,
    annotation_formatter: AnnotationFormatter | None,
    *,
    side: str,
) -> None:
    if side == "left":
        x_text = -0.42
        ha = "right"
    else:
        x_text = 1.04
        ha = "left"

    y_positions = [0.90, 0.73, 0.56, 0.39, 0.22, 0.05]
    for i, (idx, tags) in enumerate(index_tags.items()):
        if idx not in lhs_rows.index:
            continue
        row = lhs_rows.loc[idx]
        label = ", ".join(tags)
        if annotation_formatter is not None:
            label = f"{label}\n{annotation_formatter(row)}"
        axis.annotate(
            label,
            xy=(float(row["yrs_sum_al"]), float(row[y_column])),
            xycoords="data",
            xytext=(x_text, y_positions[i % len(y_positions)]),
            textcoords="axes fraction",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="red", alpha=0.9),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0", color="red", lw=1.2),
            fontsize=8,
            ha=ha,
            va="center",
            annotation_clip=False,
            clip_on=False,
        )


def _scatter_figure1_points(axis: plt.Axes, lhs_rows: pd.DataFrame, color_mode: str) -> None:
    x_vals = lhs_rows["yrs_sum_al"].to_numpy(dtype=float)
    y_vals = lhs_rows["added_lc_worth_norm"].to_numpy(dtype=float)
    positive_mask = y_vals > 0.0
    non_positive_mask = ~positive_mask

    if color_mode == "worth_override":
        worth_norm_lc_vals = lhs_rows["worth_norm_lc"].to_numpy(dtype=float)
        red_mask = non_positive_mask & (worth_norm_lc_vals <= 0.0)
        green_mask = non_positive_mask & (worth_norm_lc_vals > 0.0)
        black_mask = positive_mask & (worth_norm_lc_vals >= 0.0)
        orange_mask = positive_mask & (worth_norm_lc_vals < 0.0)
        if np.any(red_mask):
            axis.scatter(x_vals[red_mask], y_vals[red_mask], alpha=0.8, color="red", marker="x", s=18, label="stochastic LHS (added <= 0, LC <= 0)")
        if np.any(green_mask):
            axis.scatter(x_vals[green_mask], y_vals[green_mask], alpha=0.8, color="green", marker="x", s=18, label="stochastic LHS (added <= 0, LC > 0)")
        if np.any(black_mask):
            axis.scatter(x_vals[black_mask], y_vals[black_mask], alpha=0.8, color="black", marker="x", s=18, label="stochastic LHS (added > 0, LC >= 0)")
        if np.any(orange_mask):
            axis.scatter(x_vals[orange_mask], y_vals[orange_mask], alpha=0.8, color="orange", marker="x", s=18, label="stochastic LHS (added > 0, LC < 0)")
        return

    if np.any(non_positive_mask):
        axis.scatter(x_vals[non_positive_mask], y_vals[non_positive_mask], alpha=0.8, color="red", marker="x", s=18, label="stochastic LHS (<= 0)")
    if np.any(positive_mask):
        axis.scatter(x_vals[positive_mask], y_vals[positive_mask], alpha=0.8, color="black", marker="x", s=18, label="stochastic LHS (> 0)")


def plot_lhs_figure1(
    results: pd.DataFrame,
    *,
    main_title: str,
    subtitle: str,
    add_reference_line: ReferenceLineAdder,
    annotation_formatter: AnnotationFormatter | None,
    subtitle_y: float,
    color_mode: str,
    annotate_centerpoint: bool,
    show: bool,
) -> tuple[plt.Figure, plt.Axes, dict[str, int]]:
    lhs_rows = _lhs_rows(results)
    centerpoint_rows = _centerpoint_rows(results)
    extreme_indices = compute_extreme_indices(lhs_rows)

    figure, axis = plt.subplots(figsize=(12, 7))
    if not lhs_rows.empty:
        _scatter_figure1_points(axis, lhs_rows, color_mode)

        index_tags: dict[int, list[str]] = {}
        _append_index_tag(index_tags, extreme_indices.get("f1_max"), "Norm MAX")
        _append_index_tag(index_tags, extreme_indices.get("f1_min"), "Norm MIN")
        _append_index_tag(index_tags, extreme_indices.get("f2_lc_max"), "Abs LC MAX")
        _append_index_tag(index_tags, extreme_indices.get("f2_lc_min"), "Abs LC MIN")
        _append_index_tag(index_tags, extreme_indices.get("f2_cc_max"), "Abs CC MAX")
        _append_index_tag(index_tags, extreme_indices.get("f2_cc_min"), "Abs CC MIN")
        _annotate_tagged_points(axis, lhs_rows, index_tags, "added_lc_worth_norm", annotation_formatter)

    if not centerpoint_rows.empty:
        centerpoint_row = centerpoint_rows.iloc[0]
        axis.scatter(
            [float(centerpoint_row["yrs_sum_al"])],
            [float(centerpoint_row["added_lc_worth_norm"])],
            color="blue",
            marker="*",
            s=360,
            edgecolors="black",
            linewidths=0.9,
            zorder=6,
            label="CENTERPOINT",
        )
        if annotate_centerpoint and annotation_formatter is not None:
            axis.annotate(
                f"CENTERPOINT\n{annotation_formatter(centerpoint_row)}",
                xy=(float(centerpoint_row["yrs_sum_al"]), float(centerpoint_row["added_lc_worth_norm"])),
                xytext=(14, 14),
                textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#dbeafe", alpha=0.85),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0", color="blue"),
                fontsize=8,
                ha="left",
            )

    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(handles=handles, labels=labels, loc="best", fontsize=9)
    axis.set_xlabel("Sum of Assisted Living Years: yrs_sum_al (Years)")
    axis.set_ylabel("Added Worth (normalized to 2026 dollars)")
    axis.set_title(main_title, fontweight="bold", pad=20)
    axis.text(0.5, subtitle_y, subtitle, transform=axis.transAxes, ha="center", va="bottom", fontsize=9)
    axis.grid(True, alpha=0.3)
    add_reference_line(axis)

    if show:
        plt.show()
    return figure, axis, extreme_indices


def plot_lhs_figure2_worth_subplots(
    results: pd.DataFrame,
    *,
    main_title: str,
    add_reference_line: ReferenceLineAdder | None,
    annotation_formatter: AnnotationFormatter | None,
    show: bool,
) -> tuple[plt.Figure, np.ndarray]:
    lhs_rows = _lhs_rows(results)
    centerpoint_rows = _centerpoint_rows(results)
    extreme_indices = compute_extreme_indices(lhs_rows)

    figure, axes = plt.subplots(1, 2, figsize=(22, 7))
    figure.subplots_adjust(left=0.23, right=0.77, wspace=0.70, top=0.82)
    figure.suptitle(f"{main_title}\nNormalized Worth vs yrs_sum_al", fontsize=14)

    subplot_configs = [
        ("worth_norm_lc", "Worth (normalized) for Lifecare", "f2_lc_min", "f2_lc_max", "left"),
        ("worth_norm_cc", "Worth (normalized) for Continuing Care", "f2_cc_min", "f2_cc_max", "right"),
    ]

    for axis, (column, title, own_min_key, own_max_key, side) in zip(axes, subplot_configs):
        if not lhs_rows.empty:
            x_vals = lhs_rows["yrs_sum_al"].to_numpy(dtype=float)
            y_vals = lhs_rows[column].to_numpy(dtype=float)
            positive_mask = y_vals > 0.0
            negative_mask = y_vals < 0.0
            zero_mask = y_vals == 0.0

            if np.any(positive_mask):
                axis.scatter(x_vals[positive_mask], y_vals[positive_mask], alpha=0.8, color="black", marker="x", s=18, label=f"{column} > 0")
            if np.any(negative_mask):
                axis.scatter(x_vals[negative_mask], y_vals[negative_mask], alpha=0.8, color="red", marker="x", s=18, label=f"{column} < 0")
            if np.any(zero_mask):
                axis.scatter(x_vals[zero_mask], y_vals[zero_mask], alpha=0.8, color="gray", marker="x", s=18, label=f"{column} = 0")

            index_tags: dict[int, list[str]] = {}
            _append_index_tag(index_tags, extreme_indices.get(own_max_key), "Abs MAX")
            _append_index_tag(index_tags, extreme_indices.get(own_min_key), "Abs MIN")
            _append_index_tag(index_tags, extreme_indices.get("f1_max"), "Norm MAX")
            _append_index_tag(index_tags, extreme_indices.get("f1_min"), "Norm MIN")
            _annotate_tagged_points_outside(
                axis,
                lhs_rows,
                index_tags,
                column,
                annotation_formatter,
                side=side,
            )

        if not centerpoint_rows.empty:
            centerpoint_row = centerpoint_rows.iloc[0]
            axis.scatter(
                [float(centerpoint_row["yrs_sum_al"])],
                [float(centerpoint_row[column])],
                color="blue",
                marker="*",
                s=300,
                edgecolors="black",
                linewidths=0.8,
                zorder=6,
                label="CENTERPOINT",
            )

        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(handles=handles, labels=labels, loc="best", fontsize=9)
        axis.set_xlabel("Sum of Assisted Living Years: yrs_sum_al (Years)")
        axis.set_ylabel("Worth (normalized to 2026 dollars)")
        axis.set_title(title)
        axis.grid(True, alpha=0.3)
        if add_reference_line is not None:
            add_reference_line(axis)

    if show:
        plt.show()
    return figure, axes

