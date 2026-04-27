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
    Initialize OpenAI and SerpAPI clients using environment variables.

    Returns:
        tuple: (deployment_name, serpapi_api_key)

    Raises:
        ValueError: If required environment variables are missing or invalid.
    """
    openai.api_key = os.getenv("OPENAI_API_KEY_4o")
    openai.api_type = os.getenv("OPENAI_API_TYPE_4o")
    openai.api_version = os.getenv("OPENAI_API_VERSION_4o")
    openai.azure_endpoint = os.getenv("OPENAI_API_BASE_4o")

    deployment_name = os.getenv("DEPLOYMENT_NAME_4o") or "gpt-4o"
    serpapi_api_key = os.getenv("SERPAPI_API_KEY")

    # Validate required variables
    missing = []
    if not openai.api_key:
        missing.append("OPENAI_API_KEY_4o")
    if not openai.azure_endpoint:
        missing.append("OPENAI_API_BASE_4o")
    if not openai.api_version:
        missing.append("OPENAI_API_VERSION_4o")
    if not serpapi_api_key:
        missing.append("SERPAPI_API_KEY")
    if missing:
        raise ValueError(
            f"Missing required environment variables: "
            f"{', '.join(missing)}. "
            f"Please check your .env file."
        )

    # Validate API type
    if openai.api_type not in ("openai", "azure"):
        raise ValueError(
            f"Invalid OPENAI_API_TYPE_4o value: '{openai.api_type}'. "
            f"Must be 'openai' or 'azure'."
        )

    # Set OpenAI configuration
    if openai.azure_endpoint:
        setattr(openai, "api_base", openai.azure_endpoint)
    if openai.api_type:
        setattr(openai, "api_type", openai.api_type)
    if openai.api_version:
        setattr(openai, "api_version", openai.api_version)

    # Confirm successful initialization
    print("[INFO] OpenAI client initialized successfully.")
    print(f"[INFO] Using deployment: {deployment_name}")
    print(f"[INFO] API type: {openai.api_type}")

    return deployment_name, serpapi_api_key