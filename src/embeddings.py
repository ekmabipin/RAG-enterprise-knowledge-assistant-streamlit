import os

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

from config import load_config
from src.logger import get_logger


logger = get_logger(__name__)

load_dotenv()


def get_embedding_model():
    config = load_config()

    model_name = config["embeddings"]["model_name"]
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        logger.error("OPENAI_API_KEY was not found in the environment.")
        raise ValueError(
            "OPENAI_API_KEY was not found. Please check your .env file."
        )

    logger.info(
        "Initializing embedding model: %s",
        model_name,
    )

    try:
        embedding_model = OpenAIEmbeddings(
            model=model_name,
            api_key=api_key,
        )

        logger.info(
            "Embedding model initialized successfully: %s",
            model_name,
        )

        return embedding_model

    except Exception:
        logger.exception(
            "Failed to initialize embedding model: %s",
            model_name,
        )
        raise