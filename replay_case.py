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
    roi_seed,            inflation_seed
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
    "REPLAY_27": {
        "man_independent_yrs": 3.079697193420813,
        "woman_independent_yrs": 0.668271894101048,
        "man_assisted_yrs": 1.6989124057009777,
        "woman_assisted_yrs": 3.1047997266971694,
        "roi_seed": 401824,
        "inflation_seed": 161476,
        "roi_mean_shift": 0.0090405753569154,
        "roi_vol_multiplier": 0.8557312082496091,
        "roi_mean_reversion": 0.0005045962227744,
        "inflation_mean_shift": -0.0021532902971554,
        "inflation_vol_multiplier": 1.181460569786066,
        "inflation_mean_reversion": 0.3474971012197136,
    },
    "REPLAY_26": {
        "man_independent_yrs": 15.5890994444694,
        "woman_independent_yrs": 17.32155020719376,
        "man_assisted_yrs": 2.290869507153896,
        "woman_assisted_yrs": 7.858870673526574,
        "roi_seed": 941395,
        "inflation_seed": 870484,
        "roi_mean_shift": 0.009360766391067,
        "roi_vol_multiplier": 0.7607165851084816,
        "roi_mean_reversion": 0.3411801968729518,
        "inflation_mean_shift": -0.0015887473766767,
        "inflation_vol_multiplier": 0.8520188896651917,
        "inflation_mean_reversion": 0.2777939831535874,
    },

}

