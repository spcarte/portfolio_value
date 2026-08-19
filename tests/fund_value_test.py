"""
Performs tests to make sure the statistics for the FundValue work as expected. 
"""
import numpy as np
import datetime as dt
import holidays
import portfolio_value as pv
import pandas as pd
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
    Creates a single sample path from the geometric brownian motion equation.
    """
    seed = 200
    single_rng = np.random.default_rng(seed)

    initial_value = 100
    std = 0.01
    average = np.log(1+0.0005)
    number_samples = 5000
    test_path = np.exp(average-0.5*(std**2)+std*single_rng.normal(loc=0, scale=1, size=number_samples))

    simulated_stock = np.zeros(number_samples)
    simulated_stock[0] = initial_value
    for ii in range(1,number_samples):
        simulated_stock[ii] += simulated_stock[ii-1]*test_path[ii]

    index_days = work_days(dt.datetime(2026,8,3), number_samples)
    return pd.DataFrame({'test':simulated_stock}, index=index_days)

@pytest.fixture()
def simulated_data_known_stats():
    """
    Creates sample data for a single "stock" with known statistical properties.
    """
    seed = 400
    single_rng = np.random.default_rng(seed)

    std=0.01
    average=0.005
    number_samples=500

    shocks = single_rng.normal(loc=0, scale=1, size=number_samples-1)
    shocks = shocks*std/shocks.std()
    shocks = shocks - shocks.mean()

    sample_data = np.zeros(number_samples, dtype=float)
    sample_data[0] = 100
    for ii in range(1,number_samples):
        # Don't use the actual GBM formulation to make sure the simulated data 
        # has the exact stats
        sample_data[ii] = sample_data[ii-1]*np.exp(average+shocks[ii-1])

    index_days = work_days(dt.datetime(2026,8,3), number_samples)
    return pd.DataFrame({'test':sample_data}, index=index_days)

# Want to check for normality on the predicted log returns

def test_fund_value_expectation(simulated_data):
    """
    Tests that the expected value from the monte carlo returns meets the
    value from the exponential equation (v(t-1)*exp(drift)).
    """
    initial_value = 1000
    simulated_stock = simulated_data

    stock_fv = pv.FundValue(simulated_stock['test'], initial_value=initial_value)
    stock_fv.predict_value_monte_carlo(start_date=simulated_stock.index.to_pydatetime()[0],
                                       end_date=simulated_stock.index.to_pydatetime()[-1],
                                       number_of_realizations=10000)

    simulated_stock_no_volatility = np.zeros(stock_fv.expected_future_fund_value.shape[0], dtype=float)
    simulated_stock_no_volatility[0] = initial_value
    for ii in range(1,stock_fv.expected_future_fund_value.shape[0]):
        simulated_stock_no_volatility[ii] = simulated_stock_no_volatility[ii-1]*np.exp(stock_fv._drift_)

    predicted_expectation = stock_fv.expected_future_fund_value
    max_error_from_expectation = np.max((predicted_expectation-simulated_stock_no_volatility)/predicted_expectation)*100

    assert max_error_from_expectation < 0.5

def test_fund_value_stats(simulated_data_known_stats):
    """
    Tests that the fund value computes some known statistical properties 
    from a set of sample data
    """
    # should match the stats in the sample stock data
    std=0.01
    average=0.005
    drift = average + 0.5*(std**2)

    simulated_stock = simulated_data_known_stats
    stock_fv = pv.FundValue(simulated_stock['test'], initial_value=100)

    assert stock_fv._initial_value_ == 100
    assert np.isclose(stock_fv._log_drift_, average)
    assert np.isclose(stock_fv._volatility_, std)
    assert np.isclose(stock_fv._drift_, drift)