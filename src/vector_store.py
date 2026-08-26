from langchain_chroma import Chroma

from config import load_config
from src.chunking import split_documents
from src.document_loader import load_documents
from src.embeddings import get_embedding_model
from src.logger import get_logger


logger = get_logger(__name__)


def create_vector_store():
    config = load_config()

    persist_directory = config["vector_store"]["persist_directory"]
    collection_name = config["vector_store"]["collection_name"]

    logger.info(
        "Starting vector store creation. Collection=%s, Persist directory=%s",
        collection_name,
        persist_directory,
    )

    try:
        documents = load_documents()

        logger.info(
            "Loaded %d documents/pages for indexing.",
            len(documents),
        )

        chunks = split_documents(documents)

        logger.info(
            "Prepared %d chunks for embedding and indexing.",
            len(chunks),
        )

        embedding_model = get_embedding_model()

        ids = [
            chunk.metadata["chunk_id"]
            for chunk in chunks
        ]

        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            ids=ids,
            collection_name=collection_name,
            persist_directory=persist_directory,
        )

        logger.info(
            "Vector store created successfully. Collection=%s, Chunks stored=%d",
            collection_name,
            len(chunks),
        )

        return vector_store

    except Exception:
        logger.exception(
            "Failed to create vector store. Collection=%s",
            collection_name,
        )
        raise


if __name__ == "__main__":
    create_vector_store()