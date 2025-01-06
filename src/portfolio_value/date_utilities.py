"""
Defines some date utility functions for the FundValue and PortfolioValue 
objects. 
"""

import datetime as dt
import holidays

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