from default_case import DEFAULT_SEED, MAN_DOB, START_CLOCK, WOMAN_DOB, load_default_case
from Taylor import LhsScenario
from utils import age


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
    for offset_years in (0.0, 5.0, 10.0, 15.0):
        for assisted_years in (0.0, 5.0, 10.0, 15.0):
            scenarios.append(
                (
                    f"EC_{int(offset_years)}_{int(assisted_years)}{rate_suffix}",
                    _build_edge_case(
                        man_independent_yrs=MAN_BASELINE_INDEPENDENT_YRS + offset_years,
                        woman_independent_yrs=WOMAN_BASELINE_INDEPENDENT_YRS + offset_years,
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

