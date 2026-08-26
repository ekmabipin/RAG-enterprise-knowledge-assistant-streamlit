from sentence_transformers import CrossEncoder

from config import load_config
from src.logger import get_logger


logger = get_logger(__name__)


class Reranker:
    def __init__(self):
        config = load_config()

        self.enabled = config["reranker"]["enabled"]
        self.model_name = config["reranker"]["model_name"]
        self.top_n = config["reranker"]["top_n"]

        logger.info(
            "Initializing reranker. enabled=%s, model=%s, top_n=%d",
            self.enabled,
            self.model_name,
            self.top_n,
        )

        if self.enabled:
            try:
                self.model = CrossEncoder(self.model_name)

                logger.info(
                    "Reranker model initialized successfully: %s",
                    self.model_name,
                )

            except Exception:
                logger.exception(
                    "Failed to initialize reranker model: %s",
                    self.model_name,
                )
                raise
        else:
            self.model = None
            logger.info("Reranker is disabled.")

    def rerank(self, query: str, hybrid_results):
        if not hybrid_results:
            logger.warning(
                "Reranking skipped because no hybrid results were provided."
            )
            return []

        if not self.enabled:
            logger.debug(
                "Reranking disabled. Returning top %d hybrid results.",
                self.top_n,
            )
            return hybrid_results[: self.top_n]

        logger.debug(
            "Starting reranking. Query=%r, Candidates=%d",
            query,
            len(hybrid_results),
        )

        try:
            pairs = [
                (
                    query,
                    result["document"].page_content,
                )
                for result in hybrid_results
            ]

            rerank_scores = self.model.predict(pairs)

            reranked_results = []

            for result, rerank_score in zip(
                hybrid_results,
                rerank_scores,
            ):
                reranked_results.append(
                    {
                        **result,
                        "rerank_score": float(rerank_score),
                    }
                )

            reranked_results.sort(
                key=lambda item: item["rerank_score"],
                reverse=True,
            )

            results = reranked_results[: self.top_n]

            logger.info(
                "Reranking completed. Candidates=%d, Returned=%d",
                len(hybrid_results),
                len(results),
            )

            for index, result in enumerate(
                results,
                start=1,
            ):
                document = result["document"]

                logger.debug(
                    "Reranked result %d | chunk_id=%s | source=%s | "
                    "hybrid_score=%.4f | rerank_score=%.4f",
                    index,
                    document.metadata.get("chunk_id"),
                    document.metadata.get("source"),
                    result.get("hybrid_score", 0.0),
                    result["rerank_score"],
                )

            return results

        except Exception:
            logger.exception(
                "Reranking failed for query=%r",
                query,
            )
            raise


if __name__ == "__main__":
    from src.hybrid_retriever import HybridRetriever

    query = "How many annual leave days can I carry forward?"

    hybrid_retriever = HybridRetriever()
    reranker = Reranker()

    hybrid_results = hybrid_retriever.search(query)
    reranked_results = reranker.rerank(
        query,
        hybrid_results,
    )

    logger.info(
        "Manual reranker test completed. Query=%r, Results=%d",
        query,
        len(reranked_results),
    )