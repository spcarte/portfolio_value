"""
Performs tests to make sure the statistics for the FundValue work as expected. 
"""
import numpy as np
import datetime as dt
import holidays
import portfolio_value as pv
import pandas as pd
from scipy.stats import normaltest, skewtest, kurtosistest
import pytest

def work_days(start_date, number_days):
    holiday_calendar = holidays.country_holidays('US')
    workdays = [start_date]
    current_date = start_date + dt.timedelta(days=1)
    while len(workdays) < number_days:
        if current_date.weekday() < 5 and current_date not in holiday_calendar:
            workdays.append(current_date)
        current_date += dt.timedelta(days=1)
    return workdays

@pytest.fixture()
def simulated_data():
    """
    Creates a single sample path for a two stock portfolio from the geometric 
    brownian motion equation.
    """
    seed_pv_returns = 500
    pv_rng_returns = np.random.default_rng(seed_pv_returns)

    pv_initial_value = np.array([200, 200])
    pv_std = np.array([0.01, 0.02])
    pv_average = np.array([0.005, 0.003])
    number_samples = 500

    pv_return_shocks = pv_rng_returns.normal(loc=0, scale=1, size=(2,number_samples-1))
    pv_return_shocks *= pv_std[:,np.newaxis]/pv_return_shocks.std(axis=1, ddof=1)[:,np.newaxis]
    pv_return_shocks -= pv_return_shocks.mean(axis=1)[:,np.newaxis]
    pv_return_shocks += pv_average[:,np.newaxis] - 0.5*(pv_std[:,np.newaxis]**2)

    sample_portfolio = np.zeros((2,number_samples), dtype=float)
    sample_portfolio[:,0] = pv_initial_value
    for ii in range(1,number_samples):
        sample_portfolio[:,ii] = sample_portfolio[:,ii-1]*np.exp(pv_return_shocks[:, ii-1])

    index_days = work_days(dt.datetime(2026,8,3), number_samples)
    return pd.DataFrame({'test':sample_portfolio[0,:], 'test2':sample_portfolio[1,:]}, 
                         index=index_days)

@pytest.fixture()
def simulated_data_known_stats():
    """
    Creates sample data for a two "stocks" with known statistical properties.
    """
    seed_pv = 400
    pv_rng = np.random.default_rng(seed_pv)

    pv_initial_value = np.array([100, 300])
    pv_std = np.array([0.01, 0.02])
    pv_average = np.array([0.005, 0.003])
    number_samples = 500

    pv_shocks = pv_rng.normal(loc=0, scale=1, size=(2,number_samples-1))
    pv_shocks = pv_shocks*pv_std[:,np.newaxis]/pv_shocks.std(axis=1, ddof=1)[:,np.newaxis]
    pv_shocks = (pv_shocks - pv_shocks.mean(axis=1)[:,np.newaxis] + pv_average[:,np.newaxis])

    pv_sample_data = np.zeros((2,number_samples), dtype=float)
    pv_sample_data[:,0] = pv_initial_value
    for ii in range(1,number_samples):
        pv_sample_data[:,ii] = pv_sample_data[:,ii-1]*np.exp(pv_shocks[:,ii-1])

    index_days = work_days(dt.datetime(2026,8,3), number_samples)
    simulated_portfolio_fixed_props = pd.DataFrame({'test':pv_sample_data[0,:],
                                                    'test2':pv_sample_data[1,:]}, 
                                                    index=index_days)

    return simulated_portfolio_fixed_props

def test_portfolio_value_stats(simulated_data_known_stats):
    """
    Tests that the portfolio value computes some known statistical properties 
    from a set of sample data
    """
    # need to spoof allocations to build the PortfolioValue object
    allocations = pd.DataFrame({'test':50, 'test2':50}, 
                                index=[dt.datetime(2026,8,1), dt.datetime(2027,5,1)])

    # should match the stats in the sample stock data
    std = np.array([0.01, 0.02])
    average = np.array([0.005, 0.003])
    drift = average + 0.5*(std**2)

    simulated_portfolio = simulated_data_known_stats
    test_pv = pv.PortfolioValue(simulated_portfolio, allocations, initial_value=100)

    assert test_pv._initial_value_ == 100
    assert np.allclose(test_pv._log_drift_, average)
    assert np.allclose(np.sqrt(test_pv._asset_covariance_.diagonal()), std)
    assert np.allclose(test_pv._drift_, drift)

def test_portfolio_value_expectation(simulated_data):
    """
    Tests that the expected value from a monte carlo simulation matches 
    the value from the exponential equation (v(t-1)*exp(drift)).
    """
    simulated_portfolio = simulated_data
    pv_initial_value = np.array([200, 200])
    allocations = pd.DataFrame({'test':50, 'test2':50}, 
                            index=[dt.datetime(2026,8,1), dt.datetime(2027,5,1)])
    
    test_pv = pv.PortfolioValue(simulated_portfolio, allocations, 
                                initial_value=pv_initial_value.sum())

    # Manually set the covariance to have zero correlation between the stocks
    covariance = test_pv._asset_covariance_.copy()
    covariance[0,1] = 1e-16
    covariance[1,0] = 1e-16
    test_pv._asset_covariance_ = covariance

    test_pv.predict_value_monte_carlo(start_date=simulated_portfolio.index[0],
                                    end_date=simulated_portfolio.index[-1],
                                    number_of_realizations=10000, rebalance=False)

    expected_value_test = test_pv.future_portfolio_expected_value()['test']
    expected_value_test2 = test_pv.future_portfolio_expected_value()['test2']
    expected_value_total = test_pv.future_portfolio_expected_value()['Total']

    no_volatility = np.zeros((2,test_pv.future_portfolio_expected_value().shape[0]), dtype=float)
    no_volatility[:,0] = pv_initial_value
    for ii in range(1,test_pv.future_portfolio_expected_value().shape[0]):
        no_volatility[:,ii] = no_volatility[:,ii-1]*np.exp(test_pv._drift_)

    max_test_error = np.max(100*np.abs((expected_value_test - no_volatility[0,:])/expected_value_test))
    assert max_test_error < 0.5

    max_test2_error = np.max(100*np.abs((expected_value_test2 - no_volatility[1,:])/expected_value_test2))
    assert max_test2_error < 0.5

    max_total_error = np.max(100*np.abs((expected_value_total - no_volatility.sum(axis=0))/expected_value_total))
    assert max_total_error < 0.5

def test_portfolio_value_predicted_distribution(simulated_data):
    """
    Tests that the predicted future values follow a log-normal distribution
    """
    simulated_portfolio = simulated_data
    pv_initial_value = np.array([200, 200])
    allocations = pd.DataFrame({'test':50, 'test2':50}, 
                            index=[dt.datetime(2026,8,1), dt.datetime(2027,5,1)])
    
    test_pv = pv.PortfolioValue(simulated_portfolio, allocations, 
                                initial_value=pv_initial_value.sum())

    # Manually set the covariance to have zero correlation between the stocks
    covariance = test_pv._asset_covariance_.copy()
    covariance[0,1] = 1e-16
    covariance[1,0] = 1e-16
    test_pv._asset_covariance_ = covariance

    test_pv.predict_value_monte_carlo(start_date=simulated_portfolio.index[0],
                                      end_date=dt.datetime(2048,1,1),
                                      number_of_realizations=10000, rebalance=False)

    # Tests the total value
    total_returns = test_pv._predicted_time_value_realizations_[...,0]
    total_log_returns = np.log(total_returns[1:,:]/total_returns[:-1,:])

    _, total_normal_pvalue = normaltest(total_log_returns, axis=0)
    _, total_kurtosis_pvalue = kurtosistest(total_log_returns, axis=0)
    _, total_skew_pvalue = skewtest(total_log_returns, axis=0)

    # Decided that a p-value greater than 5% passes. 90% of paths should pass the normality tests.
    assert round(sum(100*total_normal_pvalue >= 5)/total_normal_pvalue.shape[0],2) >= 0.9
    assert round(sum(100*total_kurtosis_pvalue >= 5)/total_kurtosis_pvalue.shape[0],2) >= 0.9
    assert round(sum(100*total_skew_pvalue >= 5)/total_skew_pvalue.shape[0],2) >= 0.9

    # Tests the "test" value
    test_returns = test_pv._predicted_time_value_realizations_[...,1]
    test_log_returns = np.log(test_returns[1:,:]/test_returns[:-1,:])

    _, test_normal_pvalue = normaltest(test_log_returns, axis=0)
    _, test_kurtosis_pvalue = kurtosistest(test_log_returns, axis=0)
    _, test_skew_pvalue = skewtest(test_log_returns, axis=0)

    # Decided that a p-value greater than 5% passes. 90% of paths should pass the normality tests.
    assert round(sum(100*test_normal_pvalue >= 5)/test_normal_pvalue.shape[0],2) >= 0.9
    assert round(sum(100*test_kurtosis_pvalue >= 5)/test_kurtosis_pvalue.shape[0],2) >= 0.9
    assert round(sum(100*test_skew_pvalue >= 5)/test_skew_pvalue.shape[0],2) >= 0.9

    # Tests the "test2" value
    test2_returns = test_pv._predicted_time_value_realizations_[...,2]
    test2_log_returns = np.log(test2_returns[1:,:]/test2_returns[:-1,:])

    _, test2_normal_pvalue = normaltest(test2_log_returns, axis=0)
    _, test2_kurtosis_pvalue = kurtosistest(test2_log_returns, axis=0)
    _, test2_skew_pvalue = skewtest(test2_log_returns, axis=0)

    # Decided that a p-value greater than 5% passes. 90% of paths should pass the normality tests.
    assert round(sum(100*test2_normal_pvalue >= 5)/test2_normal_pvalue.shape[0],2) >= 0.9 
    assert round(sum(100*test2_kurtosis_pvalue >= 5)/test2_kurtosis_pvalue.shape[0],2) >= 0.9
    assert round(sum(100*test2_skew_pvalue >= 5)/test2_skew_pvalue.shape[0],2) >= 0.9

    