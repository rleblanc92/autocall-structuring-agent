"""Term Sheet and risk analysis generation using LLM."""

from openai import AsyncOpenAI

from src.models.schemas import AutocallSpecs, MarketDataInput, PricingResult


async def generate_term_sheet_markdown(
    specs: AutocallSpecs,
    market: MarketDataInput,
    pricing: PricingResult,
) -> str:
    """
    Generate a professional Term Sheet in Markdown format.

    Parameters
    ----------
    specs : AutocallSpecs
        Autocall product specifications.
    market : MarketDataInput
        Market data at pricing date.
    pricing : PricingResult
        Pricing results including coupon and Greeks.

    Returns
    -------
    str
        Markdown-formatted Term Sheet document.

    Raises
    ------
    ValueError
        If OpenAI API call fails.
    """
    client = AsyncOpenAI()

    system_prompt = """You are a Senior Structurer at a leading Investment Bank, responsible for 
creating professional Term Sheets and risk analyses for structured products. Your documents must 
be institutional-grade, clear, and suitable for client presentation. Use Markdown formatting 
for readability. Include all relevant quantitative details and risk metrics."""

    specs_json = specs.model_dump_json(indent=2)
    market_json = market.model_dump_json(indent=2)
    pricing_json = pricing.model_dump_json(indent=2)

    user_prompt = f"""Please generate a comprehensive Term Sheet in Markdown for the following Autocall structured product:

## Product Specifications
```json
{specs_json}
```

## Market Data (at pricing date)
```json
{market_json}
```

## Pricing Results
```json
{pricing_json}
```

The Term Sheet must include:
1. **Product Overview**: Underlying, Maturity, Observation Frequency, Barriers (autocall, coupon, protection).
2. **Market Conditions**: Current spot price, implied volatility, risk-free rate, dividend yield.
3. **Optimized Coupon**: Prominently display the annual coupon rate and fair value at 98.5% target.
4. **Risk Analysis**: Include autocall probability, capital loss probability, and their implications.
5. **Greeks**: Display Delta and Vega for risk management context.
6. **Disclaimer**: Add a regulatory disclaimer ("Not Investment Advice").

Format as professional Markdown suitable for institutional use."""

    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            max_tokens=2048,
        )

        term_sheet = response.choices[0].message.content
        if not term_sheet:
            raise ValueError("Empty response from OpenAI API")

        return term_sheet

    except Exception as exc:
        raise ValueError(f"Failed to generate Term Sheet: {exc}") from exc
