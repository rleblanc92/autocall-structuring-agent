"""Vectorized Monte Carlo pricer for Phoenix Autocall structures."""

import numpy as np
from scipy.optimize import brentq
from src.models.schemas import AutocallSpecs, MarketDataInput, PricingResult


def simulate_autocall_payoffs(
    specs: AutocallSpecs,
    market_data: MarketDataInput,
    coupon_rate: float,
    n_paths: int = 50_000,
    seed: int = 42,
) -> tuple[np.ndarray, dict]:
    np.random.seed(seed)

    S0 = market_data.spot_price
    r = market_data.risk_free_rate
    q = market_data.dividend_yield
    sigma = market_data.historical_vol_proxy
    T = specs.maturity_years

    freq_map = {"monthly": 12, "quarterly": 4, "semi-annual": 2, "annual": 1}
    freq = freq_map[specs.frequency]
    n_obs = freq * int(T)

    times = np.linspace(0, T, n_obs + 1)[1:]
    dt = T / n_obs

    drift = (r - q - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)

    n_half = n_paths // 2
    Z = np.random.standard_normal((n_half, n_obs))
    Z_full = np.vstack([Z, -Z])

    log_returns = drift + diffusion * Z_full
    prices = S0 * np.exp(np.cumsum(log_returns, axis=1))

    payoffs = np.zeros(n_paths)
    alive = np.ones(n_paths, dtype=bool)
    recalled = np.zeros(n_paths, dtype=bool)
    memory_coupons = np.zeros(n_paths)

    autocall_lvl = specs.autocall_barrier * S0
    coupon_lvl = specs.coupon_barrier * S0
    protection_lvl = specs.protection_barrier * S0
    coupon_periodic = coupon_rate / freq

    for i, t in enumerate(times):
        S_t = prices[:, i]
        is_maturity = i == (n_obs - 1)
        discount_t = np.exp(-r * t)

        # 1. Autocall Check
        autocall_event = alive & (S_t >= autocall_lvl)
        if np.any(autocall_event):
            redemption = 100.0 + coupon_periodic + memory_coupons[autocall_event]
            payoffs[autocall_event] += redemption * discount_t
            recalled[autocall_event] = True
            alive[autocall_event] = False

        # 2. Coupon Check (trajectoires encore en vie)
        if not is_maturity:
            coupon_event = alive & (S_t >= coupon_lvl)
            if np.any(coupon_event):
                coupon_paid = coupon_periodic + memory_coupons[coupon_event]
                payoffs[coupon_event] += coupon_paid * discount_t
                memory_coupons[coupon_event] = 0.0

            if specs.memory_effect:
                missed_coupon = alive & (S_t < coupon_lvl)
                memory_coupons[missed_coupon] += coupon_periodic

    # 3. Payoff à Maturité (trajectoires non rappelées)
    S_T = prices[:, -1]
    discount_T = np.exp(-r * T)
    
    # Coupon à maturité
    maturity_coupon = alive & (S_T >= coupon_lvl)
    payoffs[maturity_coupon] += (coupon_periodic + memory_coupons[maturity_coupon]) * discount_T

    # Remboursement capital à maturité
    capital_protected = alive & (S_T >= protection_lvl)
    payoffs[capital_protected] += 100.0 * discount_T

    capital_loss = alive & (S_T < protection_lvl)
    payoffs[capital_loss] += ((S_T[capital_loss] / S0) * 100.0) * discount_T

    metrics = {
        "autocall_probability": float(np.mean(recalled)),
        "capital_loss_probability": float(np.mean(capital_loss)),
        "fair_value": float(np.mean(payoffs)),
    }
    return payoffs, metrics


def calculate_greeks(
    specs: AutocallSpecs,
    market_data: MarketDataInput,
    coupon_rate: float,
    n_paths: int = 50_000,
) -> tuple[float, float]:
    _, base = simulate_autocall_payoffs(specs, market_data, coupon_rate, n_paths)
    base_fv = base["fair_value"]

    # Delta (+1% spot)
    h_s = market_data.spot_price * 0.01
    md_spot = market_data.model_copy(update={"spot_price": market_data.spot_price + h_s})
    _, s_bump = simulate_autocall_payoffs(specs, md_spot, coupon_rate, n_paths)
    delta = (s_bump["fair_value"] - base_fv) / h_s

    # Vega (+1% absolute vol)
    h_v = 0.01
    md_vol = market_data.model_copy(update={"historical_vol_proxy": market_data.historical_vol_proxy + h_v})
    _, v_bump = simulate_autocall_payoffs(specs, md_vol, coupon_rate, n_paths)
    vega = (v_bump["fair_value"] - base_fv) / 100.0

    return float(delta), float(vega)


def solve_target_coupon(
    specs: AutocallSpecs,
    market_data: MarketDataInput,
    target_fair_value: float = 98.5,
    n_paths: int = 50_000,
) -> PricingResult:
    def objective(rate: float) -> float:
        _, m = simulate_autocall_payoffs(specs, market_data, rate, n_paths)
        return m["fair_value"] - target_fair_value

    try:
        optimized_coupon = float(brentq(objective, 0.0, 40.0, xtol=1e-3))
    except ValueError as exc:
        raise ValueError(f"Solver failed to bracket target fair value {target_fair_value}%: {exc}") from exc

    _, final_metrics = simulate_autocall_payoffs(specs, market_data, optimized_coupon, n_paths)
    delta, vega = calculate_greeks(specs, market_data, optimized_coupon, n_paths)

    return PricingResult(
        fair_value=final_metrics["fair_value"],
        optimized_coupon=optimized_coupon,
        autocall_probability=final_metrics["autocall_probability"],
        capital_loss_probability=final_metrics["capital_loss_probability"],
        delta=delta,
        vega=vega,
    )