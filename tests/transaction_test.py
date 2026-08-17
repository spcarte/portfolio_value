"""
Performs tests to make sure the asset allocations and transactions work as expected. 
"""
import numpy as np
import datetime as dt
import portfolio_value as pv
import pandas as pd
import pytest

@pytest.fixture()
def spoofed_single_asset_data():
    """
    Generates a spoofed data set for a single asset that has zero return mean and variance.
    """
    spoofed_data = pd.DataFrame({'SPC':np.array([100]*100)},
        index=[dt.datetime(2025,1,1)+dt.timedelta(days=ii) for ii in range(100)])
    return spoofed_data

@pytest.fixture()
def spoofed_multivariate_data():
    """
    Generates a spoofed data set for two assets that has zero return mean and variance.
    """
    spoofed_data = pd.DataFrame({'SPC':np.array([100]*100),
                                 'FVK':np.array([200]*100)},
        index=[dt.datetime(2025,1,1)+dt.timedelta(days=ii) for ii in range(100)])
    return spoofed_data

@pytest.fixture()
def sample_transactions():
    """
    Generates a set of sample (positive and negative) transactions. These transactions
    occur on business days in the 5/1/2025-2/1/2026 date range, so the monte carlo
    simulation must include this date range.
    """
    transactions = pd.DataFrame({'Amount':[100*(ii+1) for ii in range(10)]+[-100*(ii+1) for ii in range(10)]},
                                    index=[dt.datetime(2025,5,6)+dt.timedelta(days=ii*14) for ii in range(20)])
    return transactions

def test_single_asset_transactions(spoofed_single_asset_data, sample_transactions):
    """
    Checks that the transactions work as expected in the FundValue 
    `predict_value_monte_carlo` method. The sample data has zero 
    mean and variance, so the only value change is from the transactions.
    """
    spoofed_data = spoofed_single_asset_data
    transactions = sample_transactions

    test_fv = pv.FundValue(spoofed_data['SPC'], initial_value=1000)
    test_fv.predict_value_monte_carlo(start_date=dt.datetime(2025,5,1),
                                    end_date=dt.datetime(2026,5,1),
                                    transactions=transactions,
                                    number_of_realizations=1)

    # Checking that the value change matches the transaction
    value_change = np.diff(test_fv._predicted_time_value_realizations_, axis=0)
    for ii in range(transactions.shape[0]):
        loop_value = value_change[test_fv._predicted_value_date_range_.index(transactions.index[ii])]
        assert(transactions.iloc[ii]['Amount']==loop_value)

    # Checking that the last predicted value is back to the initial value since the 
    # transactions cancel each other out.
    assert(test_fv._predicted_time_value_realizations_[-1,0] == 1000)

def test_two_asset_transactions(spoofed_multivariate_data, sample_transactions):
    """
    Checks that the transactions work as expected in the PortfolioValue 
    `predict_value_monte_carlo` method. The sample data has zero 
    mean and variance, so the only value change is from the transactions.
    """
    spoofed_data = spoofed_multivariate_data
    transactions = sample_transactions

    allocations = pd.DataFrame({'SPC':50, 'FVK':50}, 
                            index=[dt.datetime(2025,5,1), dt.datetime(2026,5,1)])
    test_pv = pv.PortfolioValue(spoofed_data, 
                                fund_allocations=allocations, 
                                initial_value=1000)

    # Manually resetting the covariance from the data so it is non-zero (needed for cholesky 
    # decomposition in the monte carlo method)
    test_pv._asset_covariance_=np.array([[1e-16, 0],
                                        [0, 1e-16]])
    
    test_pv.predict_value_monte_carlo(start_date=dt.datetime(2025,5,1),
                                      end_date=dt.datetime(2026,5,1),
                                      transactions=transactions,
                                      number_of_realizations=1, 
                                      rebalance=False)

    # Checking that the value change matches the transaction
    value_change = np.round(np.diff(test_pv._predicted_time_value_realizations_[0,:,:], axis=0),2)
    for ii in range(transactions.shape[0]):
        loop_value = value_change[test_pv._predicted_value_date_range_.index(transactions.index[ii]),:]
        assert(transactions.iloc[ii]['Amount']==loop_value[0])
        assert(transactions.iloc[ii]['Amount']*0.5==loop_value[1])
        assert(transactions.iloc[ii]['Amount']*0.5==loop_value[2])

    # Checking that the last predicted value is back to the initial value since the 
    # transactions cancel each other out.
    assert(np.round(test_pv._predicted_time_value_realizations_[0,-1,0],2) == 1000)
    assert(np.round(test_pv._predicted_time_value_realizations_[0,-1,1],2) == 500)
    assert(np.round(test_pv._predicted_time_value_realizations_[0,-1,2],2) == 500)

def test_two_asset_transactions_with_rebalance(spoofed_multivariate_data, sample_transactions):
    """
    Checks that the transactions work as expected in the PortfolioValue 
    `predict_value_monte_carlo` method. The sample data has zero 
    mean and variance, so the only value change is from the transactions.
    """
    spoofed_data = spoofed_multivariate_data
    transactions = sample_transactions

    allocations = pd.DataFrame({'SPC':50, 'FVK':50}, 
                            index=[dt.datetime(2025,5,1), dt.datetime(2026,5,1)])
    test_pv = pv.PortfolioValue(spoofed_data, 
                                fund_allocations=allocations, 
                                initial_value=1000)

    # Manually resetting the covariance from the data so it is non-zero (needed for cholesky 
    # decomposition in the monte carlo method)
    test_pv._asset_covariance_=np.array([[1e-16, 0],
                                        [0, 1e-16]])
    
    test_pv.predict_value_monte_carlo(start_date=dt.datetime(2025,5,1),
                                      end_date=dt.datetime(2026,5,1),
                                      transactions=transactions,
                                      number_of_realizations=1,
                                      rebalance=True)

    # Checking that the value change matches the transaction
    value_change = np.round(np.diff(test_pv._predicted_time_value_realizations_[0,:,:], axis=0),2)
    for ii in range(transactions.shape[0]):
        loop_value = value_change[test_pv._predicted_value_date_range_.index(transactions.index[ii]),:]
        assert(transactions.iloc[ii]['Amount']==loop_value[0])
        assert(transactions.iloc[ii]['Amount']*0.5==loop_value[1])
        assert(transactions.iloc[ii]['Amount']*0.5==loop_value[2])

    # Checking that the last predicted value is back to the initial value since the 
    # transactions cancel each other out.
    assert(np.round(test_pv._predicted_time_value_realizations_[0,-1,0],2) == 1000)
    assert(np.round(test_pv._predicted_time_value_realizations_[0,-1,1],2) == 500)
    assert(np.round(test_pv._predicted_time_value_realizations_[0,-1,2],2) == 500)

def test_portfolio_transaction_allocations(spoofed_multivariate_data, sample_transactions):
    """
    Checks that the allocations on the work as expected using only the transactions 
    (the transactions should be split per the allocations).
    """
    spoofed_data = spoofed_multivariate_data
    transactions = sample_transactions

    spc_allocations = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 80, 70, 60, 50])
    allocations = pd.DataFrame({'SPC':spc_allocations, 'FVK':100-spc_allocations}, 
                                index=[dt.datetime(2025,5,1), dt.datetime(2025,6,1),
                                       dt.datetime(2025,7,1), dt.datetime(2025,8,1),
                                       dt.datetime(2025,9,1), dt.datetime(2025,10,1),
                                       dt.datetime(2025,11,1), dt.datetime(2025,12,1),
                                       dt.datetime(2026,1,1), dt.datetime(2026,2,1), 
                                       dt.datetime(2026,3,1), dt.datetime(2026,4,1),
                                       dt.datetime(2026,5,1)])

    test_pv = pv.PortfolioValue(spoofed_data, 
                                fund_allocations=allocations, 
                                initial_value=1000)

    # Manually resetting the covariance from the data so it is non-zero (needed for cholesky 
    # decomposition in the monte carlo method)
    test_pv._asset_covariance_=np.array([[1e-16, 0],
                                        [0, 1e-16]])

    test_pv.predict_value_monte_carlo(start_date=dt.datetime(2025,5,1),
                                      end_date=dt.datetime(2026,5,1),
                                      transactions=transactions,
                                      number_of_realizations=1,
                                      rebalance=False)

    # The ratios in the different funds should always match the allocations
    value_change = np.round(np.diff(test_pv._predicted_time_value_realizations_[0,:,:], axis=0),2)
    spc_ratios = 100*value_change[:,1]/value_change[:,0]
    fvk_ratios = 100*value_change[:,2]/value_change[:,0]
    for ii in range(transactions.shape[0]):
        time_deltas = transactions.index.to_pydatetime()[ii] - allocations.index.to_pydatetime()
        allocation_idx = min((delta.days, idx) for idx, delta in enumerate(time_deltas) if delta.days >= 0)[1]

        value_idx = test_pv._predicted_value_date_range_.index(transactions.index.to_pydatetime()[ii])
        assert np.allclose(spc_ratios[value_idx],allocations['SPC'].iloc[allocation_idx])
        assert np.allclose(fvk_ratios[value_idx],allocations['FVK'].iloc[allocation_idx])

def test_portfolio_allocations_rebalance(spoofed_multivariate_data, sample_transactions):
    """
    Checks that the allocations on the work as expected using only the transactions 
    (the value should always be split per the allocations) when the value is rebalanced.
    """
    spoofed_data = spoofed_multivariate_data
    transactions = sample_transactions

    spc_allocations = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 80, 70, 60, 50])
    allocations = pd.DataFrame({'SPC':spc_allocations, 'FVK':100-spc_allocations}, 
                                index=[dt.datetime(2025,5,1), dt.datetime(2025,6,1),
                                       dt.datetime(2025,7,1), dt.datetime(2025,8,1),
                                       dt.datetime(2025,9,1), dt.datetime(2025,10,1),
                                       dt.datetime(2025,11,1), dt.datetime(2025,12,1),
                                       dt.datetime(2026,1,1), dt.datetime(2026,2,1), 
                                       dt.datetime(2026,3,1), dt.datetime(2026,4,1),
                                       dt.datetime(2026,5,1)])

    test_pv = pv.PortfolioValue(spoofed_data, 
                                fund_allocations=allocations, 
                                initial_value=1000)

    # Manually resetting the covariance from the data so it is non-zero (needed for cholesky 
    # decomposition in the monte carlo method)
    test_pv._asset_covariance_=np.array([[1e-16, 0],
                                        [0, 1e-16]])

    test_pv.predict_value_monte_carlo(start_date=dt.datetime(2025,5,1),
                                      end_date=dt.datetime(2026,5,1),
                                      transactions=transactions,
                                      number_of_realizations=1,
                                      rebalance=True)

    # The ratios in the different funds should always match the allocations
    spc_ratios = 100*test_pv._predicted_time_value_realizations_[0,:,1]/test_pv._predicted_time_value_realizations_[0,:,0]
    fvk_ratios = 100*test_pv._predicted_time_value_realizations_[0,:,2]/test_pv._predicted_time_value_realizations_[0,:,0]
    for ii in range(1,allocations.shape[0]):
        date_inds = [jj for jj in range(len(test_pv._predicted_value_date_range_)) if (test_pv._predicted_value_date_range_[jj] < allocations.index[ii].to_pydatetime() and test_pv._predicted_value_date_range_[jj] > allocations.index[ii-1].to_pydatetime())]
        assert np.allclose(spc_ratios[date_inds],allocations['SPC'].iloc[ii-1])
        assert np.allclose(fvk_ratios[date_inds],allocations['FVK'].iloc[ii-1])

    assert np.isclose(spc_ratios[-1], allocations['SPC'].iloc[-1])
    assert np.isclose(fvk_ratios[-1], allocations['FVK'].iloc[-1])