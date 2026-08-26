from src.vector_store import create_vector_store
from src.logger import get_logger


logger = get_logger(__name__)


def main():
    logger.info("Starting document ingestion and indexing.")

    try:
        create_vector_store()

        logger.info(
            "Document ingestion and indexing completed successfully."
        )

    except Exception:
        logger.exception(
            "Document ingestion and indexing failed."
        )
        raise


if __name__ == "__main__":
    main()