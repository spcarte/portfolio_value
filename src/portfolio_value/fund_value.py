"""
Defines the FundValue object for predicting the future value of a 
mutual fund or stock
"""

import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy.stats import norm
from .date_utilities import (business_days_between, business_day_vector_to_target)

class FundValue:
    """
    Creates an object to predict the future value of a single stock or mutual fund. 

    Properties
    ----------
    historical_data : DataFrame
        A Pandas DataFrame with the historical value of the fund. 
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
            A Pandas DataFrame with the historical data for the fund, which 
            will be the basis for the prediction of the future value. 
        initial_value : float
            The initial amount that is invested in the fund.  
        """        
        # Initial values that are determined from the historical data
        self._historical_data_ = historical_data
        self._initial_value_ = float(initial_value)
        self._log_return_ = np.log(1+self._historical_data_.pct_change().to_numpy())[1:]
        self._average_return_ = self._log_return_.mean()
        self._return_variance_ = self._log_return_.var()
        self._return_std_dev_ = self._log_return_.std()
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
        if data.shape[1] != 1:
            raise ValueError('The historical_data DataFrame should have a single column')
        self._historical_data_ = data
    
    @property
    def predicted_value_date_range(self):
        """
        A datetime vector that corresponds to the predicted values of the fund. 
        """
        if self._predicted_value_date_range_ is None:
            raise AttributeError('A predicted_value_date_range has not been set for this FundValue object.')
        return self._predicted_value_date_range_

    @property
    def predicted_time_value_realizations(self):
        """
        The predicted future time-value realizations for the fund as a Numpy 
        array that is arranged [days, realizations].
        """
        if self._predicted_time_value_realizations_ is None:
            raise AttributeError('The predicted_time_value_realizations have not been computed for this FundValue object.')
        return self._predicted_time_value_realizations_
    
    def plot_historical_data(self):
        """
        Plots the historical data for the the fund. 
        """
        plt.figure()
        plt.plot(self._historical_data_)
        plt.grid()
        plt.xlabel('Date')
        plt.ylabel('Fund Value ($)')
        plt.xlim(left=self._historical_data_.index.min(), right=self._historical_data_.index.max())
        plt.tight_layout()

    def plot_historical_return(self, versus_time=False, histogram_bins=100):
        """
        Plots the historical log return of the fund. 

        Parameters
        ----------
        versus_time : bool, optional 
            Plots the log return versus time if true. Otherwise, the histogram of the log
            return will be plotted. The default is false (to plot the histogram). 
        histogram_bins : int, optional 
            The number of bins to use when creating the histogram. The default is 100. 
        """
        if versus_time:
            plt.figure()
            plt.plot(self._historical_data_.index[1:], self._log_return_)
            plt.xlabel('Date')
            plt.ylabel('Log Return')
            plt.grid()
            plt.xlim(left=self._historical_data_.index.min(), right=self._historical_data_.index.max())
            plt.hlines(self._average_return_, xmin=self._historical_data_.index.min(), xmax=self._historical_data_.index.max(),
                       colors='k', linestyles='--')
            plt.tight_layout()
        else:
            plt.figure()
            histogram = plt.hist(self._log_return_, bins=int(histogram_bins))
            plt.xlabel('Log Return')
            plt.ylabel('Frequency')
            plt.grid()
            plt.vlines(self._average_return_, ymin=0, ymax=histogram[0].max()*1.1, colors='k', linestyles='--')
            plt.ylim(top = histogram[0].max()*1.1)
            plt.tight_layout()
            
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
            Updated FundValue object with predicted time-value realizations.
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
        z = norm.ppf(np.random.rand(business_days_to_end, int(number_of_realizations)))
        daily_returns = np.exp(self._drift_+self._return_std_dev_*z)
        
        # Compute the predicted time-value realizations
        self._predicted_time_value_realizations_ = np.zeros_like(daily_returns)
        self._predicted_time_value_realizations_[0,:] = self._initial_value_
        if transactions is not None:
            for ii in range(transactions.shape[0]):
                transaction_date = transactions.index[ii].to_pydatetime()
                transaction_date_deltas = np.array([self._predicted_value_date_range_[kk].timestamp()-transaction_date.timestamp() for kk in range(len(self._predicted_value_date_range_))])
                day_ind = np.where(transaction_date_deltas>0)[0].min()
                self._predicted_time_value_realizations_[day_ind,:] += transactions['Amount'][ii]
        for ii in range(1,business_days_to_end):
            self._predicted_time_value_realizations_[ii,:] += self._predicted_time_value_realizations_[ii-1,:]*daily_returns[ii,:]
                
        return self
    
    @property
    def expected_future_fund_value(self):
        """
        Computes the expected (i.e., mean) future time-value of the fund
        as an ndarray.
        """
        return self._predicted_time_value_realizations_.mean(axis=1)
    
    def future_fund_value_quantile(self, quantile):
        """
        Computes the given quantile of the predicted future time-value
        of the fund from the Monte Carlo simulations. 

        Parameters
        ----------
        quantile : float
            The desired quantile, ranging from 0-1 (so the 5th 
            percentile is given as 0.05).
        
        Returns
        -------
        ndarray
            The quantile for the predicted future time-value
        """
        return np.quantile(self._predicted_time_value_realizations_, quantile, axis=1)

    def future_fund_value_std_dev(self, sigma):
        """
        ***This is not a good method to use because the future value follows a log normal distribution***

        Computes the expected (i.e., mean) future time-value of the fund
        with the specified number of standard deviations applied.

        Parameters
        ----------
        sigma : float
            The number of standard deviations to add (or subtract) from
            the expected future value of the fund.

        Returns
        -------
        future_value : ndarray
            The expected future value of the fund with the number of 
            standard deviations applied. 
        """
        std_dev = self._predicted_time_value_realizations_.std(axis=1)
        return self._predicted_time_value_realizations_.mean(axis=1) + sigma*std_dev
    
    def plot_expected_value(self, quantiles=None):
        """
        Plots the expected value of the fund with the desired quantiles.

        Parameters
        ----------
        quantiles : iterable, optional
            An iterable of the quantiles to plot along with the expected value. 
        """
        fig, ax =plt.subplots()
        ax.plot(self.predicted_value_date_range, self.expected_future_fund_value, label = 'Expected Value: ${:,.2f}'.format(np.round(self.expected_future_fund_value[-1])))
        if quantiles is not None:
            for quantile in quantiles:
                quantile_value = self.future_fund_value_quantile(quantile)
                ax.plot(self.predicted_value_date_range, quantile_value, label = '{:} Quantile: ${:,.2f}'.format(quantile, quantile_value.max()))
        ax.grid()
        ax.set_ylabel('Mutual Fund Value')
        ax.set_xlabel('Date')
        ax.set_xlim(left=self.predicted_value_date_range[0], right=self.predicted_value_date_range[-1])
        ax.set_ylim(bottom=0)
        ax.legend(loc='upper left')
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:,.0f}'))
        fig.tight_layout()
        return fig, ax