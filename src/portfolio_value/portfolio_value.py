"""
Defines the PortfolioValue object for predicting the future value of a 
portfolio that contains multiple assets. 
"""

import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from scipy.stats import norm
from .date_utilities import (business_days_between, business_day_vector_to_target)

class PortfolioValue:
    """
    Creates an object to predict the future value of a stock portfolio. 

    Properties
    ----------
    predicted_value_date_range : datetime
        A datetime vector for the predicted time-value of the fund.
    predicted_time_value_realizations : ndarray
        The predicted future time-value realizations of the fund that is 
        shaped [days, realizations]
    expected_future_fund_value : ndarray
        The expected (i.e., the mean) future time-value of the fund.
    """
