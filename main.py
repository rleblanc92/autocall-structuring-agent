"""Main entry point for the Autocall Structurer agent."""

import asyncio
import logging
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI, AuthenticationError

from src.agents.structurer_graph import build_graph
from src.models.schemas import StructurerState

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


async def verify_openai_credentials(api_key: str) -> bool:
    """
    Verify OpenAI API key validity with a lightweight test call.

    Parameters
    ----------
    api_key : str
        OpenAI API key to verify.

    Returns
    -------
    bool
        True if credentials are valid, False otherwise.
    """
    try:
        client = AsyncOpenAI(api_key=api_key)
        await client.models.list()
        return True
    except AuthenticationError:
        return False


def mask_api_key(api_key: str) -> str:
    """
    Mask API key for display, showing only prefix and last 4 chars.

    Parameters
    ----------
    api_key : str
        Full API key to mask.

    Returns
    -------
    str
        Masked API key.
    """
    if len(api_key) > 10:
        return f"{api_key[:7]}...{api_key[-4:]}"
    return "***"


async def main() -> None:
    """
    Execute the Autocall Structurer workflow end-to-end.

    Loads OpenAI API key from environment, builds the agent graph,
    and processes a client request through the full workflow.
    """
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not set in environment")
        return

    masked_key = mask_api_key(api_key)
    logger.info(f"OpenAI API Key loaded: {masked_key}")

    logger.info("Running pre-flight check...")
    if not await verify_openai_credentials(api_key):
        logger.error(
            "Échec d'authentification OpenAI : vérifie la validité de OPENAI_API_KEY dans ton fichier .env"
        )
        return

    logger.info("✓ Authentication verified")

    client_request = (
        "Bonjour, je gère un fonds actions et je cherche du rendement. "
        "Je voudrais pricer un Phoenix Autocall sur l'Euro Stoxx 50 (^STOXX50E) sur 5 ans "
        "avec observation trimestrielle. La barrière Autocall est à 100%, la barrière Coupon à 70% "
        "et la protection du capital à maturité à 60% (européenne). Effet mémoire activé. "
        "Calculez-moi le coupon faisable."
    )

    logger.info("=" * 80)
    logger.info("AUTOCALL STRUCTURER AGENT - START")
    logger.info("=" * 80)
    logger.info(f"Client Request:\n{client_request}")

    app = build_graph()

    initial_state: StructurerState = {
        "client_request": client_request,
        "status": "initialized",
    }

    try:
        logger.info("Executing workflow...")
        logger.info("-" * 80)

        final_state = await app.ainvoke(initial_state)

        logger.info("Workflow Execution Summary:")
        logger.info(f"Final Status: {final_state.get('status', 'UNKNOWN')}")

        if final_state.get("error_message"):
            logger.error(f"Error: {final_state['error_message']}")

        if final_state.get("specs"):
            logger.info("✓ Specs parsed successfully")

        if final_state.get("market_data"):
            logger.info("✓ Market data fetched successfully")

        if final_state.get("pricing_result"):
            pricing = final_state["pricing_result"]
            logger.info(f"✓ Pricing complete (Fair Value: {pricing.get('fair_value'):.2f}%)")

        if final_state.get("term_sheet_md"):
            logger.info("✓ Term Sheet generated successfully")

        logger.info("=" * 80)
        logger.info("GENERATED TERM SHEET")
        logger.info("=" * 80)
        print(final_state.get("term_sheet_md", "No term sheet generated"))

        logger.info("=" * 80)
        logger.info("AUTOCALL STRUCTURER AGENT - COMPLETE")
        logger.info("=" * 80)

    except Exception as exc:
        logger.error(f"Workflow failed with exception: {exc}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
