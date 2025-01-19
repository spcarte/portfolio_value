"""
Defines some date utility functions for the FundValue and PortfolioValue 
objects. 
"""

import datetime as dt
import holidays
import numpy as np

def is_weekday(date):
  return date.weekday() < 5  # Monday to Friday

def business_days_between(start_date, end_date, country_code='US'):
  us_holidays = holidays.country_holidays(country_code)
  business_days = 0
  date = start_date
  while date <= end_date:
    if is_weekday(date) and date not in us_holidays:
      business_days += 1
    date += dt.timedelta(days=1)
  return business_days

def business_day_vector_to_target(start_date, end_date):
    us_holidays = holidays.UnitedStates()
    business_days = []
    current_date = start_date
    one_day = dt.timedelta(days=1)

    while current_date <= end_date:
        if is_weekday(current_date) and current_date not in us_holidays:
            business_days.append(current_date)
        current_date += one_day
    return business_days

def interpolate_time_value(original_dates, original_value, want_dates):
    """
    Interpolates the given values from the original dates to the desired dates.

    Parameters
    ----------
    original_dates : datetime
        A list of datetime values that correspond to the original values.
    original_values : ndarray
        A 1d array of values that correspond to the original dates. 
    want_dates : datetime
        A list of datetime values to interpolate the original values to.

    Returns 
    -------    
    want_values : ndarray
        A 1d array of values that correspond to the want_dates
    """
    original_dates_between = np.cumsum(np.insert(np.array([(original_dates[ii+1]-original_dates[ii]).days for ii in range(len(original_dates)-1)]), 0, 0))
    want_dates_between = np.cumsum(np.insert(np.array([(want_dates[ii+1]-want_dates[ii]).days for ii in range(len(want_dates)-1)]), 0, 0))
    return np.interp(want_dates_between, original_dates_between, original_value)