"""Tests for the Phoenix Autocall Monte Carlo pricer."""

import numpy as np
import pytest

from src.models.schemas import AutocallSpecs, MarketDataInput, PricingResult
from src.tools.pricer import (
    calculate_greeks,
    simulate_autocall_payoffs,
    solve_target_coupon,
)


@pytest.fixture
def mock_specs():
    """Mock AutocallSpecs for testing."""
    return AutocallSpecs(
        underlying="DUMMY",
        maturity_years=5,
        frequency="quarterly",
        autocall_barrier=1.0,
        coupon_barrier=0.7,
        protection_barrier=0.6,
        memory_effect=True,
    )


@pytest.fixture
def mock_market_data():
    """Mock MarketDataInput for testing."""
    return MarketDataInput(
        spot_price=100.0,
        risk_free_rate=0.03,
        dividend_yield=0.02,
        historical_vol_proxy=0.20,
    )


def test_zero_volatility_pricing(mock_specs, mock_market_data):
    """Test pricing with near-zero volatility returns valid payoffs and metrics."""
    np.random.seed(42)
    
    low_vol_market = MarketDataInput(
        spot_price=mock_market_data.spot_price,
        risk_free_rate=mock_market_data.risk_free_rate,
        dividend_yield=mock_market_data.dividend_yield,
        historical_vol_proxy=0.0001,
    )
    
    payoffs, metrics = simulate_autocall_payoffs(
        mock_specs, low_vol_market, coupon_rate=0.05, n_paths=5_000
    )
    
    assert isinstance(payoffs, np.ndarray), "Payoffs should be np.ndarray"
    assert "fair_value" in metrics, "Metrics should contain 'fair_value'"
    assert isinstance(metrics["fair_value"], (float, np.floating)), "Fair value should be numeric"


def test_solve_target_coupon(mock_specs, mock_market_data):
    """Test coupon solver returns valid PricingResult with positive coupon."""
    np.random.seed(42)
    
    result = solve_target_coupon(
        mock_specs, mock_market_data, target_fair_value=98.5, n_paths=10_000
    )
    
    assert isinstance(result, PricingResult), "Result should be a PricingResult instance"
    assert result.optimized_coupon > 0, "Optimized coupon should be strictly positive"
    assert isinstance(result.fair_value, (float, np.floating)), "Fair value should be numeric"


def test_calculate_greeks(mock_specs, mock_market_data):
    """Test Greeks calculation returns valid Delta and Vega as floats."""
    np.random.seed(42)
    
    delta, vega = calculate_greeks(
        mock_specs, mock_market_data, coupon_rate=8.0, n_paths=5_000
    )
    
    assert isinstance(delta, float), "Delta should be a float"
    assert isinstance(vega, float), "Vega should be a float"
    assert isinstance((delta, vega), tuple), "Return value should be a tuple"
