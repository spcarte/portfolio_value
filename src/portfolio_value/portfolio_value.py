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
    lots of missing properties 

    predicted_value_date_range : datetime
        A datetime vector for the predicted time-value of the fund.
    predicted_time_value_realizations : ndarray
        The predicted future time-value realizations of the fund that is 
        shaped [days, realizations]
    expected_future_fund_value : ndarray
        The expected (i.e., the mean) future time-value of the fund.
    """

    def __init__(self, historical_data, fund_allocations, initial_value):
        """
        Initializes the PortfolioValue object

        Parameters
        ----------
        historical_data : DataFrame
            A Pandas DataFrame with the historical data for the stocks in
            the portfolio (with a separate column for each stock), which 
            will be the basis for the prediction of the future value. The
            columns should be labeled with the stock ticker for each stock
            in the fund. 
        fund_allocations : DataFrame
            A Pandas DataFrame with the fund allocation (through the 
            target date for the portfolio), where The columns are labeled 
            with the stock ticker for each stock in the fund. The simulations
            assume that the portfolio allocation on the provided dates (i.e.,
            the allocations use a "previous" interpolation from the provided
            dates to the daily dates for a simulation).
        initial_value : float
            The initial amount that is invested in the portfolio (the sum 
            for all the investment).  
        """ 
        self._tickers_ = None 

        # Initial values that are determined from the historical data and allocations
        self.historical_data = historical_data
        self.allocations = fund_allocations[self._tickers_]
        self._initial_value_ = float(initial_value)
        self._log_return_ = np.log(1+self._historical_data_.pct_change().to_numpy())[1:]

        # Data for future value predictions
        self._average_return_ = np.zeros(len(self._tickers_), dtype=float)
        self._return_variance_ = np.zeros(len(self._tickers_), dtype=float)
        self._return_std_dev_ = np.zeros(len(self._tickers_), dtype=float)
        self._drift_ = np.zeros(len(self._tickers_), dtype=float)
        for ii in range(len(self._tickers_)):
            ticker_rows = ~np.isnan(self._log_return_[:,ii])
            self._average_return_[ii] = self._log_return_[ticker_rows,ii].mean()
            self._return_variance_[ii] = self._log_return_[ticker_rows,ii].var()
            self._return_std_dev_[ii] = self._log_return_[ticker_rows,ii].std()
        self._drift_ = self._average_return_ - (0.5*self._return_variance_)

        # Predicted time-value information, initialized to none
        self._predicted_value_date_range_ = None
        self._predicted_time_value_realizations_ = None

    @property
    def historical_data(self):
        """
        A Pandas DataFrame of the historical value of the fund.
        """
        return self._historical_data_

    @historical_data.setter
    def historical_data(self, data):
        if self._tickers_ is not None and set(self._tickers_) != set(list(data)):
            raise ValueError('The historical tickers (the DataFrame headers) do not match the object')
        if self._tickers_ is None:
            self._tickers_ = list(data)
        self._historical_data_ = data

    @property
    def allocations(self):
        """
        A Pandas DataFrame of the portfolio allocations. 
        """
        allocation_dict = {}
        for ii in range(len(self._tickers_)):
            allocation_dict[self._tickers_[ii]] = self._allocation_values_[:,ii]
        return pd.DataFrame(data = allocation_dict, index = self._allocation_dates_)
    
    @allocations.setter
    def allocations(self, data):
        if self._tickers_ is not None and set(self._tickers_) != set(list(data)):
            raise ValueError('The fund allocation tickers (the DataFrame headers) do not match the object')
        self._allocation_dates_ = list(data.index.to_pydatetime())
        self._allocation_values_ = data.to_numpy() 