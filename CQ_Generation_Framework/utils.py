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


def initialize_clients() -> tuple[str, str]:
    """
    Initialize OpenAI and SerpAPI clients using
    environment variables.

    Returns:
        tuple: (deployment_name, serpapi_api_key)
    """
    openai.api_key = os.getenv("OPENAI_API_KEY_4o")
    openai.api_type = os.getenv("OPENAI_API_TYPE_4o")
    openai.api_version = os.getenv("OPENAI_API_VERSION_4o")
    openai.azure_endpoint = os.getenv("OPENAI_API_BASE_4o")

    deployment_name = os.getenv("DEPLOYMENT_NAME_4o") or "gpt-4o"
    serpapi_api_key = os.getenv("SERPAPI_API_KEY")

    return deployment_name, serpapi_api_key