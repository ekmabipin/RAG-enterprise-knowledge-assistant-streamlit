from collections import defaultdict

from config import load_config
from src.bm25_retriever import BM25Retriever
from src.vector_retriever import search_vector_store
from src.logger import get_logger


logger = get_logger(__name__)


class HybridRetriever:
    def __init__(self):
        config = load_config()

        self.vector_weight = config["hybrid_search"]["vector_weight"]
        self.keyword_weight = config["hybrid_search"]["keyword_weight"]
        self.hybrid_top_k = config["retrieval"]["hybrid_top_k"]

        logger.info(
            "Initializing hybrid retriever. "
            "vector_weight=%.2f, keyword_weight=%.2f, top_k=%d",
            self.vector_weight,
            self.keyword_weight,
            self.hybrid_top_k,
        )

        self.bm25_retriever = BM25Retriever()

    def _normalize_scores(
        self,
        values,
        higher_is_better=True,
    ):
        if not values:
            return []

        minimum = min(values)
        maximum = max(values)

        if maximum == minimum:
            return [1.0 for _ in values]

        normalized = [
            (value - minimum) / (maximum - minimum)
            for value in values
        ]

        if not higher_is_better:
            normalized = [
                1.0 - value
                for value in normalized
            ]

        return normalized

    def search(self, query: str):
        logger.debug(
            "Starting hybrid retrieval. Query=%r",
            query,
        )

        try:
            vector_results = search_vector_store(query)
            bm25_results = self.bm25_retriever.search(query)

            vector_raw_scores = [
                score
                for _, score in vector_results
            ]

            bm25_raw_scores = [
                score
                for _, score in bm25_results
            ]

            # Chroma similarity_search_with_score returns
            # distance-like values, so lower is better.
            vector_scores = self._normalize_scores(
                vector_raw_scores,
                higher_is_better=False,
            )

            # BM25 scores are higher-is-better.
            bm25_scores = self._normalize_scores(
                bm25_raw_scores,
                higher_is_better=True,
            )

            combined = defaultdict(
                lambda: {
                    "document": None,
                    "vector_score": 0.0,
                    "bm25_score": 0.0,
                    "hybrid_score": 0.0,
                }
            )

            for (document, _), normalized_score in zip(
                vector_results,
                vector_scores,
            ):
                chunk_id = document.metadata["chunk_id"]

                combined[chunk_id]["document"] = document
                combined[chunk_id]["vector_score"] = normalized_score

            for (document, _), normalized_score in zip(
                bm25_results,
                bm25_scores,
            ):
                chunk_id = document.metadata["chunk_id"]

                combined[chunk_id]["document"] = document
                combined[chunk_id]["bm25_score"] = normalized_score

            for item in combined.values():
                item["hybrid_score"] = (
                    self.vector_weight * item["vector_score"]
                    + self.keyword_weight * item["bm25_score"]
                )

            ranked_results = sorted(
                combined.values(),
                key=lambda item: item["hybrid_score"],
                reverse=True,
            )

            results = ranked_results[: self.hybrid_top_k]

            logger.info(
                "Hybrid retrieval completed. "
                "Vector results=%d, BM25 results=%d, "
                "Combined candidates=%d, Returned=%d",
                len(vector_results),
                len(bm25_results),
                len(combined),
                len(results),
            )

            for index, result in enumerate(
                results,
                start=1,
            ):
                document = result["document"]

                logger.debug(
                    "Hybrid result %d | chunk_id=%s | source=%s | "
                    "vector_score=%.4f | bm25_score=%.4f | "
                    "hybrid_score=%.4f",
                    index,
                    document.metadata.get("chunk_id"),
                    document.metadata.get("source"),
                    result["vector_score"],
                    result["bm25_score"],
                    result["hybrid_score"],
                )

            return results

        except Exception:
            logger.exception(
                "Hybrid retrieval failed for query=%r",
                query,
            )
            raise


if __name__ == "__main__":
    query = "How many annual leave days can I carry forward?"

    retriever = HybridRetriever()
    results = retriever.search(query)

    logger.info(
        "Manual hybrid retrieval test completed. "
        "Query=%r, Results=%d",
        query,
        len(results),
    )