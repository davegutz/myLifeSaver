"""
replay_case.py

Define stochastic LHS cases to be re-run as edge cases inside Run_LHS_Taylor.
Populate REPLAY_CASES by copying the scenario parameters directly from the
columns of lhs_taylor_results.csv for whichever run_id you want to replay.

Each key is the name prefix used in the run_id (e.g. "REPLAY_42").
Each value is a dict of LhsScenario keyword arguments.

Required keys:
    man_independent_yrs, woman_independent_yrs
    man_assisted_yrs,    woman_assisted_yrs
    roi_seed,            inflation_seed,  man_goes_to_al_seed, woman_goes_to_al_seed
    roi_mean_shift,      roi_vol_multiplier,      roi_mean_reversion
    inflation_mean_shift, inflation_vol_multiplier, inflation_mean_reversion

Example — uncomment and fill in values from your CSV row:
    REPLAY_CASES["REPLAY_42"] = {
        "man_independent_yrs":      6.744,
        "woman_independent_yrs":    5.477,
        "man_assisted_yrs":         9.035,
        "woman_assisted_yrs":       9.322,
        "roi_seed":                 716049,
        "inflation_seed":           154002,
        "roi_mean_shift":          -0.002701,
        "roi_vol_multiplier":       1.4342,
        "roi_mean_reversion":       0.3316,
        "inflation_mean_shift":    -0.001234,
        "inflation_vol_multiplier": 0.8765,
        "inflation_mean_reversion": 0.2100,
    }
"""

REPLAY_CASES: dict[str, dict[str, float | int]] = {
    "REPLAY_GUTZ_1": {
        "man_independent_yrs": 11.119851304450926,
        "woman_independent_yrs": 23.151157732325107,
        "man_assisted_yrs": 2.749282500878669,
        "woman_assisted_yrs": 7.201015734520627,
        "roi_seed": 493950,
        "inflation_seed": 6660,
        "man_goes_to_al_seed": 0,
        "woman_goes_to_al_seed": 0,
        "roi_mean_shift": 0.0068122507422182,
        "roi_vol_multiplier": 0.8934722576907159,
        "roi_mean_reversion": 0.3971721322072199,
        "inflation_mean_shift": -0.0048161705509744,
        "inflation_vol_multiplier": 1.0502745708323007,
        "inflation_mean_reversion": 0.0459961806916466,
    },
    "REPLAY_1": {
        "man_independent_yrs": 13.44391071880088,
        "woman_independent_yrs": 5.173895360036171,
        "man_assisted_yrs": 0.4097352393619469,
        "woman_assisted_yrs": 0.1652763552852909,
        "roi_seed": 813270,
        "inflation_seed": 912756,
        "man_goes_to_al_seed": 857404,
        "woman_goes_to_al_seed": 33586,
        "roi_mean_shift": 0.0021327155153435,
        "roi_vol_multiplier": 1.2294965609839985,
        "roi_mean_reversion": 0.2718124957327114,
        "inflation_mean_shift": 0.0043507242378776,
        "inflation_vol_multiplier": 1.3158535541215322,
        "inflation_mean_reversion": 0.001369250085074,
    },

}

