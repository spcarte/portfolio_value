"""
Defines some date utility functions for the FundValue and PortfolioValue 
objects. 
"""

import datetime as dt
import holidays
import numpy as np
from scipy.interpolate import interp1d

def is_weekday(date):
    return date.weekday() < 5  # Monday to Friday

def business_days_between(start_date, end_date, country_code='US'):
    """
    Computes the number of business days between the start and end dates.

    Parameters
    ----------
    start_date : datetime
        The start date.
    start_date : datetime
        The start date.
    country_code : str
        The country code for determining if a date is a holiday. The 
        default is 'US'.
    """
    holidays = holidays.country_holidays(country_code)
    business_days = 0
    date = start_date
    while date <= end_date:
        if is_weekday(date) and date not in holidays:
        business_days += 1
        date += dt.timedelta(days=1)
    return business_days

def business_day_vector_to_target(start_date, end_date):
    """
    Creates a list of datetime values at the business days between the 
    provided dates.

    Parameters
    ----------
    start_date : datetime
        The start date of the date time list.
    end_date : datetime
        The end date of the date time list.

    Returns
    -------
    business_days : datetime
        A list of datetime values with the business days between the 
        start and end dates.
    """
    us_holidays = holidays.UnitedStates()
    business_days = []
    current_date = start_date
    one_day = dt.timedelta(days=1)

    while current_date <= end_date:
        if is_weekday(current_date) and current_date not in us_holidays:
            business_days.append(current_date)
        current_date += one_day
    return business_days

def find_nearest_date_index(want_date, date_range):
    """
    Finds the index where a list of dates is closest to a desired date.

    Parameters
    ----------
    want_date : datetime
        The desired date to find in the list of dates.
    date_range : datetime
        A list of datetime values to search through

    Returns
    -------
    day_ind : int
        The index in date_range that corresponds to the date that is 
        closest to want_date.
    """
    date_deltas = np.array([date_range[kk].timestamp()-want_date.timestamp() for kk in range(len(date_range))])
    return np.where(date_deltas>0)[0].min()

def interpolate_time_value(original_dates, original_value, want_dates, interpolation_type='linear'):
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
    interpolation_type : str
        The type of interpolation to use, which corresponds to the 'kind' argument
        in scipy.interpolate.interp1d. The default is 'linear'. 

    Returns 
    -------    
    want_values : ndarray
        A 1d array of values that correspond to the want_dates
    """
    original_dates_between = np.cumsum(np.insert(np.array([(original_dates[ii+1]-original_dates[ii]).days for ii in range(len(original_dates)-1)]), 0, 0))
    want_dates_between = np.cumsum(np.insert(np.array([(want_dates[ii+1]-want_dates[ii]).days for ii in range(len(want_dates)-1)]), 0, 0))
    interpolation_function = interp1d(original_dates_between, original_value, kind=interpolation_type, bounds_error=False, fill_value=(original_value[0], original_value[-1]))
    return interpolation_function(want_dates_between)