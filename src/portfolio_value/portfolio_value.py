"""
Defines the PortfolioValue object for predicting the future value of a 
portfolio that contains multiple assets. 
"""

import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from scipy.stats import norm
from .date_utilities import (business_days_between, business_day_vector_to_target, 
                             find_nearest_date_index, interpolate_time_value)

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

    def predict_value_monte_carlo(self, end_date, start_date=None, 
                                  number_of_realizations=10000,
                                  transactions=None):
        """
        Predicts the future potential time-value path of the fund over a number of 
        realizations using the brownian motion formula. 

        Parameters
        ----------
        end_date : datetime
            The end date for the time-value prediction.
        start_dae : datetime, optional 
            The start date for the time-value prediction. Defaults to today if a 
            date is not provided. 
        number_of_realizations : int, optional
            The number of realizations to simulate the time-value path of the fund.
            Defaults to 10,000 if a value is not provided. 
        transactions : DataFrame, optional
            A Pandas DataFrame with transactions (either deposits or withdrawals)
            into the investment account. The column label for the transaction 
            amount should be 'Amount' (cap sensitive).

        Returns
        -------
        self : FundValue
            Updated FundValue object with predicted time-value realizations. It is 
            sized [number of funds + 1, number of days, number of realizations].

        Notes
        -----
        The predicted_time_value_realizations is organized such that the zeroth 
        entry on the first axis is the total fund value and the remaining entries on 
        the first axis correspond to the tickers in the object.
        """
        if start_date is None:
            start_date = dt.datetime.today()
        
        if not isinstance(start_date, dt.datetime):
            raise ValueError('The start_date should be a datetime object.')
        if not isinstance(end_date, dt.datetime):
            raise ValueError('The end_date should be a datetime object.')

        business_days_to_end = business_days_between(start_date,end_date)
        self._predicted_value_date_range_ = business_day_vector_to_target(start_date,end_date)

        # Compute the random realizations for the daily returns
        daily_returns = np.zeros((self._drift_.shape[0], business_days_to_end, int(number_of_realizations)), dtype=float)
        allocation_values = np.zeros((self._drift_.shape[0], business_days_to_end), dtype=float)       
        for ii in range(self._drift_.shape[0]):
            z = norm.ppf(np.random.rand(business_days_to_end, int(number_of_realizations)))
            daily_returns[ii, ...] = np.exp(self._drift_[ii]+self._return_std_dev_[ii]*z)
            allocation_values[ii, ...] = interpolate_time_value(self._allocation_dates_, self._allocation_values_[:, ii], 
                                                                self._predicted_value_date_range_, interpolation_type='previous')/100

        self._predicted_time_value_realizations_ = np.zeros((self._drift_.shape[0]+1, len(self._predicted_value_date_range_), int(number_of_realizations)), dtype=float)
        self._predicted_time_value_realizations_[0, 0] = self._initial_value_
        self._predicted_time_value_realizations_[1:,0,:] = allocation_values[:,0][...,np.newaxis]*self._predicted_time_value_realizations_[0, 0,:]
        if transactions is not None:
            for ii in range(transactions.shape[0]):
                day_ind = find_nearest_date_index(transactions.index[ii].to_pydatetime(), self._predicted_value_date_range_)
                self._predicted_time_value_realizations_[0, day_ind, :] += transactions['Amount'][ii]

        for ii in range(1,business_days_to_end):
            for jj in range(self._drift_.shape[0]):
                self._predicted_time_value_realizations_[0,ii,:] += self._predicted_time_value_realizations_[jj+1,ii-1,:]*daily_returns[jj,ii,:]
            self._predicted_time_value_realizations_[1:,ii,:] += allocation_values[:,ii][...,np.newaxis]*self._predicted_time_value_realizations_[0,ii,:]  
        return self
    
    def future_portfolio_value_quantile(self, quantile):
        """
        Computes the given quantile of the predicted future time-value
        of the portfolio from the Monte Carlo simulations. 

        Parameters
        ----------
        quantile : float
            The desired quantile, ranging from 0-1 (so the 5th 
            percentile is given as 0.05).
        
        Returns
        -------
        DataFrame
            The quantile for the predicted future time-value in a DataFrame
            with a columns for each ticker and the total value.
        """
        value_dict = {'Total':np.quantile(self._predicted_time_value_realizations_[0,...], quantile, axis=-1)}
        for ii in range(len(test_object._tickers_)):
            value_dict[self._tickers_[ii]] = np.quantile(self._predicted_time_value_realizations_[ii,...], quantile, axis=-1)
        
        return pd.DataFrame(data=value_dict, index=self._predicted_value_date_range_)