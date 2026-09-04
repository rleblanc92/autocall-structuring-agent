"""Market data retrieval and volatility calculation for autocall structuring."""

import asyncio
import numpy as np
import yfinance as yf
from src.models.schemas import MarketDataInput


async def get_market_data(ticker_symbol: str, risk_free_rate: float = 0.03) -> MarketDataInput:
    def _fetch_data() -> tuple[float, float, float]:
        ticker = yf.Ticker(ticker_symbol)
        history = ticker.history(period="1y", auto_adjust=False)
        if history.empty or len(history) < 30:
            raise ValueError(f"Insufficient or missing market history for '{ticker_symbol}'.")

        closes = history["Close"].dropna()
        spot_price = float(closes.iloc[-1])

        returns = np.log(closes / closes.shift(1)).dropna()
        annualized_vol = float(np.std(returns, ddof=1) * np.sqrt(252))

        dividend_yield = 0.02
        if ticker.info and ticker.info.get("dividendYield"):
            dividend_yield = float(ticker.info["dividendYield"])

        return spot_price, annualized_vol, dividend_yield

    spot_price, implied_vol, div_yield = await asyncio.to_thread(_fetch_data)

    return MarketDataInput(
        spot_price=spot_price,
        risk_free_rate=risk_free_rate,
        dividend_yield=div_yield,
        historical_vol_proxy=implied_vol,
    )