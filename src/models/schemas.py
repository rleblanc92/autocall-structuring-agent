"""Pydantic schemas and state types for autocall structuring workflows."""

from typing import Literal, Optional, TypedDict

from pydantic import BaseModel, Field


class AutocallSpecs(BaseModel):
    """Specifications for an autocall structured product."""

    underlying: str = Field(
        ...,
        description="Underlying instrument ticker, for example ^STOXX50E.",
    )
    maturity_years: int = Field(..., ge=1, le=10)
    frequency: Literal["monthly", "quarterly", "semi-annual", "annual"] = "quarterly"
    autocall_barrier: float = Field(
        default=1.0,
        ge=0.5,
        le=1.2,
        description="Niveau de rappel anticipé (1.0 = 100%).",
    )
    coupon_barrier: float = Field(
        default=0.7,
        ge=0.3,
        le=1.0,
        description="Niveau d'activation du coupon (0.7 = 70%).",
    )
    protection_barrier: float = Field(
        default=0.6,
        ge=0.3,
        le=0.9,
        description="Barrière PDI à maturité (0.6 = 60%).",
    )
    target_coupon: Optional[float] = Field(
        default=None,
        description="Coupon cible annuel en %.",
    )
    memory_effect: bool = True


class MarketDataInput(BaseModel):
    """Market inputs required for pricing."""

    spot_price: float = Field(..., gt=0)
    risk_free_rate: float
    dividend_yield: float
    historical_vol_proxy: float = Field(..., gt=0, description="Using 1Y historical volatility as a proxy for implied volatility due to free API constraints")


class PricingResult(BaseModel):
    """Output metrics produced by the pricing workflow."""

    fair_value: float = Field(..., description="Fair Value en % du nominal.")
    optimized_coupon: float = Field(..., description="Coupon annuel résolu en %.")
    autocall_probability: float = Field(..., ge=0.0, le=1.0)
    capital_loss_probability: float = Field(..., ge=0.0, le=1.0)
    delta: float
    vega: float


class StructurerState(TypedDict, total=False):
    """Shared state passed between structuring workflow nodes."""

    client_request: str
    specs: dict
    market_data: dict
    pricing_result: dict
    term_sheet_md: str
    status: str
    error_message: str
