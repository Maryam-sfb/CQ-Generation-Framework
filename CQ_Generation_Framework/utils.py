from pathlib import Path
from dotenv import load_dotenv
import os
import openai


def load_environment_variables() -> None:
    """
    Load environment variables from the .env file
    """
    dotenv_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(dotenv_path=dotenv_path)


def initialize_clients(debug: bool = False) -> tuple[str, str]:
    """
    Initialize OpenAI and SerpAPI clients using
    environment variables.

    Args:
        debug: If True, prints debug information about
               the loaded environment variables.

    Returns:
        tuple: (deployment_name, serpapi_api_key)
    """
    openai.api_key = os.getenv("OPENAI_API_KEY_4o")
    openai.api_type = os.getenv("OPENAI_API_TYPE_4o")
    openai.api_version = os.getenv("OPENAI_API_VERSION_4o")
    openai.azure_endpoint = os.getenv("OPENAI_API_BASE_4o")

    deployment_name = os.getenv("DEPLOYMENT_NAME_4o") or "gpt-4o"
    serpapi_api_key = os.getenv("SERPAPI_API_KEY")

    # Validate and set OpenAI configuration
    if openai.azure_endpoint:
        setattr(openai, "api_base", openai.azure_endpoint)
    if openai.api_type:
        if openai.api_type not in ("openai", "azure"):
            raise ValueError(
                f"Invalid OPENAI_API_TYPE_4o value: {openai.api_type}"
            )
        setattr(openai, "api_type", openai.api_type)
    if openai.api_version:
        setattr(openai, "api_version", openai.api_version)

    # Debug output (only printed when debug=True)
    if debug:
        print("[DEBUG] OpenAI ENV loaded:")
        print("KEY:", bool(openai.api_key))
        print("BASE:", openai.azure_endpoint)
        print("DEPLOYMENT:", deployment_name)

    return deployment_name, serpapi_api_key