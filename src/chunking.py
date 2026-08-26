from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import load_config
from src.document_loader import load_documents
from src.logger import get_logger

logger = get_logger(__name__)


def split_documents(documents):
    config = load_config()

    chunk_size = config["chunking"]["chunk_size"]
    chunk_overlap = config["chunking"]["chunk_overlap"]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    for index, chunk in enumerate(chunks):
        source = chunk.metadata.get("source", "unknown")
        chunk.metadata["chunk_id"] = f"{source}_{index}"

    return chunks


if __name__ == "__main__":
    documents = load_documents()
    chunks = split_documents(documents)

    logger.info(
    "Chunking completed: %d documents/pages -> %d chunks",
    len(documents),
    len(chunks),
    )

  

    for index, chunk in enumerate(chunks[:5], start=1):
        logger.debug(
            "Sample chunk %d | id=%s | source=%s | page=%s | size=%d",
            index,
            chunk.metadata.get("chunk_id"),
            chunk.metadata.get("source"),
            chunk.metadata.get("page", "N/A"),
            len(chunk.page_content),
        )

        logger.debug(
            "Chunk text: %s",
            chunk.page_content[:500],
        )