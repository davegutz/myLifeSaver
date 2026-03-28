from default_case import DEFAULT_SEED, MAN_DOB, START_CLOCK, WOMAN_DOB
from Taylor import LhsScenario
from utils import age


MAN_BASELINE_INDEPENDENCE_YRS = 69.0 - age(START_CLOCK, MAN_DOB)
WOMAN_BASELINE_INDEPENDENCE_YRS = 70.29 - age(START_CLOCK, WOMAN_DOB)


def _build_edge_case(
    man_independence_yrs: float,
    woman_independence_yrs: float,
    man_linger: float,
    woman_linger: float,
    roi_mean_shift: float = 0.01,
    roi_vol_multiplier: float = 0.5,
    roi_mean_reversion: float = 0.5,
    inflation_mean_shift: float = -0.005,
    inflation_vol_multiplier: float = 0.5,
    inflation_mean_reversion: float = 0.5,
) -> LhsScenario:
    return LhsScenario(
        man_independence_yrs=man_independence_yrs,
        woman_independence_yrs=woman_independence_yrs,
        man_linger=man_linger,
        woman_linger=woman_linger,
        roi_seed=DEFAULT_SEED,
        inflation_seed=DEFAULT_SEED,
        roi_mean_shift=roi_mean_shift,
        roi_vol_multiplier=roi_vol_multiplier,
        roi_mean_reversion=roi_mean_reversion,
        inflation_mean_shift=inflation_mean_shift,
        inflation_vol_multiplier=inflation_vol_multiplier,
        inflation_mean_reversion=inflation_mean_reversion,
    )


def build_edge_case_scenarios() -> list[tuple[str, LhsScenario]]:
    """
    Define explicit edge case scenarios for testing.
    Returns a list of tuples (case_name, scenario).

    NOTE: The ROI and CPI behavior for each edge case depends on scenario fields
    and downstream model settings.
    """
    scenarios: list[tuple[str, LhsScenario]] = []
    for offset_years in (0.0, 5.0, 10.0, 15.0):
        for linger_years in (0.0, 5.0, 10.0, 15.0):
            scenarios.append(
                (
                    f"EC_{int(offset_years)}_{int(linger_years)}",
                    _build_edge_case(
                        man_independence_yrs=MAN_BASELINE_INDEPENDENCE_YRS + offset_years,
                        woman_independence_yrs=WOMAN_BASELINE_INDEPENDENCE_YRS + offset_years,
                        man_linger=linger_years,
                        woman_linger=linger_years,
                    ),
                )
            )

    scenarios.append(
        (
            "EC_85_1",
            _build_edge_case(
                man_independence_yrs=85.0 - age(START_CLOCK, MAN_DOB),
                woman_independence_yrs=85.0 - age(START_CLOCK, WOMAN_DOB),
                man_linger=1.0,
                woman_linger=1.0,
                roi_mean_shift=0.005,
                roi_vol_multiplier=1.2,
                roi_mean_reversion=0.15,
                inflation_mean_shift=0.002,
                inflation_vol_multiplier=1.1,
                inflation_mean_reversion=0.15,
            ),
        )
    )

    return scenarios

