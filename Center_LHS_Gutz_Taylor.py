"""Centerpoint constants for Run_LHS_Gutz_Taylor scenarios."""
from default_case import P_WOMAN_AL, P_MAN_AL
CENTERPOINT_MAN_INDEPENDENT_YRS = 10.0  # 75 - 84 both genders --> 79
CENTERPOINT_WOMAN_INDEPENDENT_YRS = 8.8  # 75 - 84 both genders --> 79
CENTERPOINT_MAN_ASSISTED_YRS = 2.35*P_MAN_AL  # forcing P=1 and compensating here
CENTERPOINT_WOMAN_ASSISTED_YRS = 5.5*P_WOMAN_AL  # forcing P=1 and compensating here
CENTERPOINT_ROI_SEED = 740264
CENTERPOINT_INFLATION_SEED = 898910
CENTERPOINT_MAN_GOES_TO_AL_SEED = 314159
CENTERPOINT_WOMAN_GOES_TO_AL_SEED = 271828
CENTERPOINT_MAN_GOES_TO_AL = True   # set False to let seed determine result
CENTERPOINT_WOMAN_GOES_TO_AL = True  # set False to let seed determine result

# If None, ROI/CPI are stochastic.
# If float and abs(value) <= 1.0, it is treated as a fixed monthly fraction.
# If float and abs(value) > 1.0, it is treated as APY percent and converted to monthly.
CENTERPOINT_CONSTANT_MONTHLY_ROI = 8.
CENTERPOINT_CONSTANT_MONTHLY_CPI = 4.

# Toggle for CENTERPOINT row only in Run_LHS_Gutz_Taylor:
#   False -> centerpoint row uses stochastic ROI/CPI
#   True  -> centerpoint row uses CENTERPOINT_CONSTANT_MONTHLY_ROI/CPI above
CENTERPOINT_USE_CONSTANT_RATES = True

