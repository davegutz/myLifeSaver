from default_case import DEFAULT_SEED, MAN_DOB, START_CLOCK, WOMAN_DOB, load_default_case
from Taylor import LhsScenario
from utils import age
from replay_case import REPLAY_CASES


MAN_BASELINE_INDEPENDENT_YRS = 2./12.
WOMAN_BASELINE_INDEPENDENT_YRS = MAN_BASELINE_INDEPENDENT_YRS


def _build_edge_case(
    man_independent_yrs: float,
    woman_independent_yrs: float,
    man_assisted_yrs: float,
    woman_assisted_yrs: float,
    roi_mean_shift: float = 0.01,
    roi_vol_multiplier: float = 0.5,
    roi_mean_reversion: float = 0.5,
    inflation_mean_shift: float = -0.005,
    inflation_vol_multiplier: float = 0.5,
    inflation_mean_reversion: float = 0.5,
) -> LhsScenario:
    return LhsScenario(
        man_independent_yrs=man_independent_yrs,
        woman_independent_yrs=woman_independent_yrs,
        man_assisted_yrs=man_assisted_yrs,
        woman_assisted_yrs=woman_assisted_yrs,
        roi_seed=DEFAULT_SEED,
        inflation_seed=DEFAULT_SEED,
        roi_mean_shift=roi_mean_shift,
        roi_vol_multiplier=roi_vol_multiplier,
        roi_mean_reversion=roi_mean_reversion,
        inflation_mean_shift=inflation_mean_shift,
        inflation_vol_multiplier=inflation_vol_multiplier,
        inflation_mean_reversion=inflation_mean_reversion,
    )


def monthly_fraction_to_apy_percent(monthly_fraction: float) -> float:
    return ((1.0 + monthly_fraction) ** 12 - 1.0) * 100.0


def format_apy_suffix(apy_percent: float | None) -> str:
    if apy_percent is None:
        return "S"
    return f"{apy_percent:.3f}".rstrip("0").rstrip(".")


def build_edge_case_scenarios(
    roi_apy_percent: float | None = None,
    cpi_apy_percent: float | None = None,
) -> list[tuple[str, LhsScenario]]:
    """
    Define explicit edge case scenarios for testing.
    Returns a list of tuples (case_name, scenario).

    NOTE: The ROI and CPI behavior for each edge case depends on scenario fields
    and downstream model settings.
    """
    rate_suffix = f"_{format_apy_suffix(roi_apy_percent)}_{format_apy_suffix(cpi_apy_percent)}"
    scenarios: list[tuple[str, LhsScenario]] = []
    for independent_years in (0.0, 5.0, 10.0):
        for assisted_years in (0.0, 5.0, 10.0):
            scenarios.append(
                (
                    f"EC_{int(independent_years)}_{int(assisted_years)}{rate_suffix}",
                    _build_edge_case(
                        man_independent_yrs=MAN_BASELINE_INDEPENDENT_YRS + independent_years,
                        woman_independent_yrs=WOMAN_BASELINE_INDEPENDENT_YRS + independent_years,
                        man_assisted_yrs=assisted_years,
                        woman_assisted_yrs=assisted_years,
                    ),
                )
            )

    scenarios.append(
        (
            f"EC_85_1{rate_suffix}",
            _build_edge_case(
                man_independent_yrs=85.0 - age(START_CLOCK, MAN_DOB),
                woman_independent_yrs=85.0 - age(START_CLOCK, WOMAN_DOB),
                man_assisted_yrs=1.0,
                woman_assisted_yrs=1.0,
                roi_mean_shift=0.005,
                roi_vol_multiplier=1.2,
                roi_mean_reversion=0.15,
                inflation_mean_shift=0.002,
                inflation_vol_multiplier=1.1,
                inflation_mean_reversion=0.15,
            ),
        )
    )

    run_one_present_scenario_kwargs, _ = load_default_case("RUN_ONE_PRESENT")
    scenarios.append((f"RUN_ONE_PRESENT{rate_suffix}", LhsScenario(**run_one_present_scenario_kwargs)))

    return scenarios


def build_replay_case_scenarios() -> list[tuple[str, LhsScenario]]:
    """Build replay scenarios once (not per ROI/CPI edge-case combination)."""
    scenarios: list[tuple[str, LhsScenario]] = []

    for replay_name, kwargs in REPLAY_CASES.items():
        full_name = replay_name
        scenario = LhsScenario(
            man_independent_yrs=float(kwargs["man_independent_yrs"]),
            woman_independent_yrs=float(kwargs["woman_independent_yrs"]),
            man_assisted_yrs=float(kwargs["man_assisted_yrs"]),
            woman_assisted_yrs=float(kwargs["woman_assisted_yrs"]),
            roi_seed=int(kwargs["roi_seed"]),
            inflation_seed=int(kwargs["inflation_seed"]),
            roi_mean_shift=float(kwargs["roi_mean_shift"]),
            roi_vol_multiplier=float(kwargs["roi_vol_multiplier"]),
            roi_mean_reversion=float(kwargs["roi_mean_reversion"]),
            inflation_mean_shift=float(kwargs["inflation_mean_shift"]),
            inflation_vol_multiplier=float(kwargs["inflation_vol_multiplier"]),
            inflation_mean_reversion=float(kwargs["inflation_mean_reversion"]),
        )
        print(f"  Replay case '{full_name}':")
        print(
            f"    ind_yrs=(man={float(kwargs['man_independent_yrs']):.3f}, "
            f"woman={float(kwargs['woman_independent_yrs']):.3f})  "
            f"ast_yrs=(man={float(kwargs['man_assisted_yrs']):.3f}, "
            f"woman={float(kwargs['woman_assisted_yrs']):.3f})  "
            f"seeds=(roi={int(kwargs['roi_seed'])}, inf={int(kwargs['inflation_seed'])})"
        )
        scenarios.append((full_name, scenario))

    return scenarios


# ============================================================================
# CUSTOM GUTZ EDGE CASES (for Run_LHS_Gutz_Taylor.py)
# ============================================================================

CUSTOM_EDGE_CASES_GUTZ = None  # Set to a dict to define custom Gutz edge cases


def build_custom_edge_cases_gutz(
    centerpoint_man_independent_yrs: float,
    centerpoint_woman_independent_yrs: float,
    centerpoint_man_assisted_yrs: float,
    centerpoint_woman_assisted_yrs: float,
    centerpoint_roi_seed: int,
    centerpoint_inflation_seed: int,
) -> list[tuple[str, LhsScenario]]:
    """
    Convert CUSTOM_EDGE_CASES_GUTZ dict into (name, LhsScenario) tuples.
    Falls back to centerpoint values for any missing fields.
    """
    if CUSTOM_EDGE_CASES_GUTZ is None:
        return []
    cases = []
    for case_name, params in CUSTOM_EDGE_CASES_GUTZ.items():
        scenario = LhsScenario(
            man_independent_yrs=params.get("man_independent_yrs", centerpoint_man_independent_yrs),
            woman_independent_yrs=params.get("woman_independent_yrs", centerpoint_woman_independent_yrs),
            man_assisted_yrs=params.get("man_assisted_yrs", centerpoint_man_assisted_yrs),
            woman_assisted_yrs=params.get("woman_assisted_yrs", centerpoint_woman_assisted_yrs),
            roi_seed=int(params.get("roi_seed", centerpoint_roi_seed)),
            inflation_seed=int(params.get("inflation_seed", centerpoint_inflation_seed)),
            roi_mean_shift=float(params.get("roi_mean_shift", 0.0)),
            roi_vol_multiplier=float(params.get("roi_vol_multiplier", 1.0)),
            roi_mean_reversion=float(params.get("roi_mean_reversion", 0.0)),
            inflation_mean_shift=float(params.get("inflation_mean_shift", 0.0)),
            inflation_vol_multiplier=float(params.get("inflation_vol_multiplier", 1.0)),
            inflation_mean_reversion=float(params.get("inflation_mean_reversion", 0.0)),
        )
        cases.append((case_name, scenario))
    return cases


def get_edge_cases_gutz(
    roi_apy: float,
    cpi_apy: float,
    centerpoint_man_independent_yrs: float,
    centerpoint_woman_independent_yrs: float,
    centerpoint_man_assisted_yrs: float,
    centerpoint_woman_assisted_yrs: float,
    centerpoint_roi_seed: int,
    centerpoint_inflation_seed: int,
) -> list[tuple[str, LhsScenario]]:
    """
    Return either custom Gutz edge cases (if CUSTOM_EDGE_CASES_GUTZ is set) or standard ones.
    When using custom cases, apply the ROI/CPI fixed rates by appending suffix to case names.
    """
    if CUSTOM_EDGE_CASES_GUTZ is not None:
        # Use custom cases and append ROI/CPI suffix to case names
        custom_cases = build_custom_edge_cases_gutz(
            centerpoint_man_independent_yrs=centerpoint_man_independent_yrs,
            centerpoint_woman_independent_yrs=centerpoint_woman_independent_yrs,
            centerpoint_man_assisted_yrs=centerpoint_man_assisted_yrs,
            centerpoint_woman_assisted_yrs=centerpoint_woman_assisted_yrs,
            centerpoint_roi_seed=centerpoint_roi_seed,
            centerpoint_inflation_seed=centerpoint_inflation_seed,
        )
        roi_cpi_suffix = f"_{format_apy_suffix(roi_apy)}_{format_apy_suffix(cpi_apy)}"
        return [(name + roi_cpi_suffix, scenario) for name, scenario in custom_cases]
    else:
        # Fall back to standard edge cases
        return build_edge_case_scenarios(roi_apy_percent=roi_apy, cpi_apy_percent=cpi_apy)


