"""Gutz-only replay cases used by Run_LHS_Gutz_Taylor and Replay_LHS_Gutz_Case."""
from Center_LHS_Gutz_Taylor import *
# CENTERPOINT_MAN_INDEPENDENT_YRS = 10.0
# CENTERPOINT_WOMAN_INDEPENDENT_YRS = 15.5
# CENTERPOINT_MAN_ASSISTED_YRS = 2.35
# CENTERPOINT_WOMAN_ASSISTED_YRS = 5.5
# CENTERPOINT_ROI_SEED = 740264
# CENTERPOINT_INFLATION_SEED = 898910
#
# # If None, ROI/CPI are stochastic.
# # If float and abs(value) <= 1.0, it is treated as a fixed monthly fraction.
# # If float and abs(value) > 1.0, it is treated as APY percent and converted to monthly.
# CENTERPOINT_CONSTANT_MONTHLY_ROI = 4.
# CENTERPOINT_CONSTANT_MONTHLY_CPI = 5.


REPLAY_CASES_GUTZ: dict[str, dict[str, float | int]] = {
    "REPLAY_CENTERPOINT": {
        "man_independent_yrs": CENTERPOINT_MAN_INDEPENDENT_YRS,
        "woman_independent_yrs": CENTERPOINT_WOMAN_INDEPENDENT_YRS,
        "man_assisted_yrs": CENTERPOINT_MAN_ASSISTED_YRS,
        "woman_assisted_yrs": CENTERPOINT_WOMAN_ASSISTED_YRS,
        "roi_seed": CENTERPOINT_ROI_SEED,
        "inflation_seed": CENTERPOINT_INFLATION_SEED,
        "roi_mean_shift": 0.0068122507422182,
        "roi_vol_multiplier": 0.8934722576907159,
        "roi_mean_reversion": 0.3971721322072199,
        "inflation_mean_shift": -0.0048161705509744,
        "inflation_vol_multiplier": 1.0502745708323007,
        "inflation_mean_reversion": 0.0459961806916466,
    },
}

