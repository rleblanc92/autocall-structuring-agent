import pytest
from pydantic import ValidationError

from src.models.schemas import AutocallSpecs, PricingResult


def test_autocall_specs_valid() -> None:
    specs = AutocallSpecs(
        underlying="^STOXX50E",
        maturity_years=5,
        frequency="quarterly",
        autocall_barrier=1.0,
        coupon_barrier=0.7,
        protection_barrier=0.6,
        target_coupon=8.0,
        memory_effect=True,
    )

    assert specs.underlying == "^STOXX50E"
    assert specs.maturity_years == 5


def test_autocall_specs_invalid_barrier() -> None:
    with pytest.raises(ValidationError):
        AutocallSpecs(
            underlying="^STOXX50E",
            maturity_years=5,
            protection_barrier=0.2,
        )

    with pytest.raises(ValidationError):
        AutocallSpecs(
            underlying="^STOXX50E",
            maturity_years=5,
            autocall_barrier=1.5,
        )


def test_pricing_result_probabilities() -> None:
    with pytest.raises(ValidationError):
        PricingResult(
            fair_value=100.0,
            optimized_coupon=8.0,
            autocall_probability=1.1,
            capital_loss_probability=0.2,
            delta=0.8,
            vega=0.2,
        )

    with pytest.raises(ValidationError):
        PricingResult(
            fair_value=100.0,
            optimized_coupon=8.0,
            autocall_probability=0.4,
            capital_loss_probability=-0.1,
            delta=0.8,
            vega=0.2,
        )
