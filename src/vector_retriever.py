from langchain_chroma import Chroma

from config import load_config
from src.embeddings import get_embedding_model
from src.logger import get_logger


logger = get_logger(__name__)


def get_vector_store():
    config = load_config()

    collection_name = config["vector_store"]["collection_name"]
    persist_directory = config["vector_store"]["persist_directory"]

    logger.debug(
        "Loading Chroma vector store. Collection=%s, Persist directory=%s",
        collection_name,
        persist_directory,
    )

    try:
        vector_store = Chroma(
            collection_name=collection_name,
            persist_directory=persist_directory,
            embedding_function=get_embedding_model(),
        )

        logger.info(
            "Chroma vector store loaded successfully. Collection=%s",
            collection_name,
        )

        return vector_store

    except Exception:
        logger.exception(
            "Failed to load Chroma vector store. Collection=%s",
            collection_name,
        )
        raise


def search_vector_store(query: str):
    config = load_config()

    top_k = config["retrieval"]["vector_top_k"]

    logger.debug(
        "Starting vector search. Query=%r, top_k=%d",
        query,
        top_k,
    )

    try:
        vector_store = get_vector_store()

        results = vector_store.similarity_search_with_score(
            query=query,
            k=top_k,
        )

        logger.info(
            "Vector search completed. Results=%d",
            len(results),
        )

        for index, (document, score) in enumerate(results, start=1):
            logger.debug(
                "Vector result %d | chunk_id=%s | source=%s | page=%s | score=%.4f",
                index,
                document.metadata.get("chunk_id"),
                document.metadata.get("source"),
                document.metadata.get("page", "N/A"),
                score,
            )

        return results

    except Exception:
        logger.exception(
            "Vector search failed for query=%r",
            query,
        )
        raise


if __name__ == "__main__":
    query = "How many annual leave days can I carry forward?"

    results = search_vector_store(query)

    logger.info(
        "Manual vector retrieval test completed. Query=%r, Results=%d",
        query,
        len(results),
    )