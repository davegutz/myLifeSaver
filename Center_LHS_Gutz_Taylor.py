"""Centerpoint constants for Run_LHS_Gutz_Taylor scenarios."""

CENTERPOINT_MAN_INDEPENDENT_YRS = 10.0
CENTERPOINT_WOMAN_INDEPENDENT_YRS = 15.5
CENTERPOINT_MAN_ASSISTED_YRS = 2.35
CENTERPOINT_WOMAN_ASSISTED_YRS = 5.5
CENTERPOINT_ROI_SEED = 740264
CENTERPOINT_INFLATION_SEED = 898910

# If None, ROI/CPI are stochastic.
# If float and abs(value) <= 1.0, it is treated as a fixed monthly fraction.
# If float and abs(value) > 1.0, it is treated as APY percent and converted to monthly.
CENTERPOINT_CONSTANT_MONTHLY_ROI = 4.
CENTERPOINT_CONSTANT_MONTHLY_CPI = 5.

# Toggle for CENTERPOINT row only in Run_LHS_Gutz_Taylor:
#   False -> centerpoint row uses stochastic ROI/CPI
#   True  -> centerpoint row uses CENTERPOINT_CONSTANT_MONTHLY_ROI/CPI above
CENTERPOINT_USE_CONSTANT_RATES = False

