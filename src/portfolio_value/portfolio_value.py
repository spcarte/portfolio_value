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
    def __init__(self, historical_data, initial_value=0):
        """
        Initializes the FundValue object

        Parameters
        ----------
        historical_data : DataFrame
            A Pandas DataFrame with the historical data for the stocks in
            the portfolio (with a separate column for each stock), which 
            will be the basis for the prediction of the future value. The
            columns should be labeled with the stock ticker for each stock
            in the fund. 
        fund_allocation : DataFrame
            A Pandas DataFrame with the fund allocation (through the 
            target date for the portfolio), where The columns are labeled 
            with the stock ticker for each stock in the fund. 
        initial_value : float
            The initial amount that is invested in the portfolio (the sum 
            for all the stocks).  
        """   