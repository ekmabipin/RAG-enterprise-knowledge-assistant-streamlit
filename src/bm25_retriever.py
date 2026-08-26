import re

from rank_bm25 import BM25Okapi

from config import load_config
from src.chunking import split_documents
from src.document_loader import load_documents
from src.logger import get_logger


logger = get_logger(__name__)


class BM25Retriever:
    def __init__(self):
        config = load_config()

        self.top_k = config["retrieval"]["bm25_top_k"]

        logger.info(
            "Initializing BM25 retriever. top_k=%d",
            self.top_k,
        )

        try:
            documents = load_documents()
            self.chunks = split_documents(documents)

            self.tokenized_corpus = [
                self._tokenize(chunk.page_content)
                for chunk in self.chunks
            ]

            self.bm25 = BM25Okapi(self.tokenized_corpus)

            logger.info(
                "BM25 retriever initialized successfully. Chunks indexed=%d",
                len(self.chunks),
            )

        except Exception:
            logger.exception(
                "Failed to initialize BM25 retriever."
            )
            raise

    def _tokenize(self, text: str):
        return re.findall(
            r"\b[a-zA-Z0-9]+\b",
            text.lower(),
        )

    def search(self, query: str):
        logger.debug(
            "Starting BM25 search. Query=%r, top_k=%d",
            query,
            self.top_k,
        )

        try:
            tokenized_query = self._tokenize(query)

            scores = self.bm25.get_scores(tokenized_query)

            ranked_results = sorted(
                zip(self.chunks, scores),
                key=lambda item: item[1],
                reverse=True,
            )

            results = ranked_results[: self.top_k]

            logger.info(
                "BM25 search completed. Results=%d",
                len(results),
            )

            for index, (document, score) in enumerate(
                results,
                start=1,
            ):
                logger.debug(
                    "BM25 result %d | chunk_id=%s | source=%s | "
                    "page=%s | score=%.4f",
                    index,
                    document.metadata.get("chunk_id"),
                    document.metadata.get("source"),
                    document.metadata.get("page", "N/A"),
                    score,
                )

            return results

        except Exception:
            logger.exception(
                "BM25 search failed for query=%r",
                query,
            )
            raise


if __name__ == "__main__":
    query = "How many annual leave days can I carry forward?"

    retriever = BM25Retriever()
    results = retriever.search(query)

    logger.info(
        "Manual BM25 retrieval test completed. "
        "Query=%r, Results=%d",
        query,
        len(results),
    )