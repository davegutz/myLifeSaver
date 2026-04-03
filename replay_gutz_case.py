"""Gutz-only replay cases used by Run_LHS_Gutz_Taylor and Replay_LHS_Gutz_Case."""
from Center_LHS_Gutz_Taylor import *

REPLAY_CASES_GUTZ: dict[str, dict[str, float | int | None]] = {
    "REPLAY_CENTERPOINT": {
        "man_independent_yrs": CENTERPOINT_MAN_INDEPENDENT_YRS,
        "woman_independent_yrs": CENTERPOINT_WOMAN_INDEPENDENT_YRS,
        "man_assisted_yrs": CENTERPOINT_MAN_ASSISTED_YRS,
        "woman_assisted_yrs": CENTERPOINT_WOMAN_ASSISTED_YRS,
        "roi_seed": CENTERPOINT_ROI_SEED,
        "inflation_seed": CENTERPOINT_INFLATION_SEED,
        "man_goes_to_al_seed": CENTERPOINT_MAN_GOES_TO_AL_SEED,
        "woman_goes_to_al_seed": CENTERPOINT_WOMAN_GOES_TO_AL_SEED,
        "roi_mean_shift": 0.0068122507422182,
        "roi_vol_multiplier": 0.8934722576907159,
        "roi_mean_reversion": 0.3971721322072199,
        "inflation_mean_shift": -0.0048161705509744,
        "inflation_vol_multiplier": 1.0502745708323007,
        "inflation_mean_reversion": 0.0459961806916466,
        "constant_monthly_roi": CENTERPOINT_CONSTANT_MONTHLY_ROI,
        "constant_monthly_cpi": CENTERPOINT_CONSTANT_MONTHLY_CPI,
    },
    "REPLAY_GUTZ_1000": {
        "man_independent_yrs": 5.306884467305709,
        "woman_independent_yrs": 12.408712456918552,
        "man_assisted_yrs": 2.1964107432458477,
        "woman_assisted_yrs": 4.895850739937967,
        "roi_seed": 324873,
        "inflation_seed": 943640,
        "man_goes_to_al_seed": 890953,
        "woman_goes_to_al_seed": 869299,
        "roi_mean_shift": -0.0091983000776078,
        "roi_vol_multiplier": 0.6027639046007233,
        "roi_mean_reversion": 0.1517062989042333,
        "inflation_mean_shift": 0.0024895712562442,
        "inflation_vol_multiplier": 0.791577426233915,
        "inflation_mean_reversion": 0.2017712301852744,
    },

}

