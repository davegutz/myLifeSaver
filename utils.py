import pandas as pd


def date_at_age(birth_date: str | pd.Timestamp, age_years: float) -> pd.Timestamp:
    birth_ts = pd.Timestamp(birth_date)
    whole_years = int(age_years)
    remaining_months = int(round((age_years - whole_years) * 12))
    return birth_ts + pd.DateOffset(years=whole_years, months=remaining_months)
