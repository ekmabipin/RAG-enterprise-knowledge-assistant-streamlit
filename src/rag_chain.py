from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from config import load_config
from src.hybrid_retriever import HybridRetriever
from src.reranker import Reranker
from src.prompts import RAG_SYSTEM_PROMPT
from src.memory import ConversationMemory
from src.logger import get_logger


logger = get_logger(__name__)


class RAGChain:
    def __init__(self):
        config = load_config()

        self.model = ChatOpenAI(
            model=config["llm"]["model_name"],
            temperature=config["llm"]["temperature"],
            max_tokens=config["llm"]["max_tokens"],
        )

        self.hybrid_retriever = HybridRetriever()
        self.reranker = Reranker()

        self.fallback = config["rag"]["hallucination_fallback"]

        self.memory = ConversationMemory(
            max_history_messages=config["memory"]["max_history_messages"]
        )

        self.final_ranking = config["final_ranking"]
        self.source_priority = config["source_priority"]
        self.final_selection = config["final_selection"]

        logger.info("RAG chain initialized successfully.")

    def _rewrite_query(self, query: str):
        history = self.memory.get_messages()

        if not history:
            logger.debug(
                "No conversation history. Using original query=%r",
                query,
            )
            return query

        history_text = "\n".join(
            f"{message.__class__.__name__}: {message.content}"
            for message in history
        )

        rewrite_prompt = f"""
Rewrite the user's latest question as a standalone question using the
conversation history.

Do not answer the question.
Do not add facts that are not present in the conversation.
Return only the rewritten question.

Conversation history:
{history_text}

Latest question:
{query}
"""

        try:
            response = self.model.invoke(
                [
                    SystemMessage(
                        content=(
                            "You rewrite follow-up questions into standalone "
                            "questions for document retrieval."
                        )
                    ),
                    HumanMessage(content=rewrite_prompt),
                ]
            )

            rewritten_query = response.content.strip()

            logger.debug(
                "Query rewritten. Original=%r, Rewritten=%r",
                query,
                rewritten_query,
            )

            return rewritten_query

        except Exception:
            logger.exception(
                "Failed to rewrite query=%r",
                query,
            )
            raise

    def _build_context(self, results):
        context_parts = []

        for index, result in enumerate(results, start=1):
            document = result["document"]

            source = document.metadata.get("source", "unknown")
            page = document.metadata.get("page", "N/A")

            context_parts.append(
                f"""
Context {index}
Source: {source}
Page: {page}

{document.page_content}
""".strip()
            )

        logger.debug(
            "Built LLM context from %d chunks.",
            len(results),
        )

        return "\n\n".join(context_parts)

    def _extract_sources(self, results):
        sources = []

        for result in results:
            source = result["document"].metadata.get("source")

            if source and source not in sources:
                sources.append(source)

        return sources

    def _final_rank(self, results):
        if not results:
            return []

        rerank_weight = self.final_ranking["rerank_weight"]
        hybrid_weight = self.final_ranking["hybrid_weight"]
        source_weight = self.final_ranking["source_priority_weight"]

        rerank_scores = [
            result.get("rerank_score", 0.0)
            for result in results
        ]

        hybrid_scores = [
            result.get("hybrid_score", 0.0)
            for result in results
        ]

        def normalize(values):
            minimum = min(values)
            maximum = max(values)

            if maximum == minimum:
                return [1.0] * len(values)

            return [
                (value - minimum) / (maximum - minimum)
                for value in values
            ]

        normalized_rerank = normalize(rerank_scores)
        normalized_hybrid = normalize(hybrid_scores)

        for result, rerank_score, hybrid_score in zip(
            results,
            normalized_rerank,
            normalized_hybrid,
        ):
            source = result["document"].metadata.get("source")

            source_score = self.source_priority.get(
                source,
                0.0,
            )

            result["final_score"] = (
                rerank_weight * rerank_score
                + hybrid_weight * hybrid_score
                + source_weight * source_score
            )

        ranked_results = sorted(
            results,
            key=lambda result: result["final_score"],
            reverse=True,
        )

        logger.debug(
            "Final ranking completed for %d candidates.",
            len(ranked_results),
        )

        return ranked_results

    def _select_final_context(
        self,
        hybrid_results,
        reranked_results,
    ):
        hybrid_keep = self.final_selection["hybrid_keep_top_n"]
        rerank_keep = self.final_selection["rerank_keep_top_n"]
        max_chunks = self.final_selection["max_context_chunks"]

        selected = []
        seen_chunk_ids = set()

        for result in hybrid_results[:hybrid_keep]:
            chunk_id = result["document"].metadata.get("chunk_id")

            if chunk_id not in seen_chunk_ids:
                selected.append(result)
                seen_chunk_ids.add(chunk_id)

        for result in reranked_results[:rerank_keep]:
            chunk_id = result["document"].metadata.get("chunk_id")

            if chunk_id not in seen_chunk_ids:
                selected.append(result)
                seen_chunk_ids.add(chunk_id)

        selected = self._final_rank(selected)
        final_results = selected[:max_chunks]

        logger.info(
            "Final context selected. Hybrid candidates=%d, "
            "Reranked candidates=%d, Final chunks=%d",
            len(hybrid_results),
            len(reranked_results),
            len(final_results),
        )

        for index, result in enumerate(
            final_results,
            start=1,
        ):
            document = result["document"]

            logger.debug(
                "Final context %d | chunk_id=%s | source=%s | "
                "hybrid_score=%.4f | rerank_score=%.4f | "
                "final_score=%.4f",
                index,
                document.metadata.get("chunk_id"),
                document.metadata.get("source"),
                result.get("hybrid_score", 0.0),
                result.get("rerank_score", 0.0),
                result.get("final_score", 0.0),
            )

        return final_results

    def answer(self, query: str):
        logger.info(
            "Processing user query."
        )

        logger.debug(
            "User query=%r",
            query,
        )

        try:
            retrieval_query = self._rewrite_query(query)

            hybrid_results = self.hybrid_retriever.search(
                retrieval_query
            )

            reranked_results = self.reranker.rerank(
                retrieval_query,
                hybrid_results,
            )

            final_results = self._select_final_context(
                hybrid_results,
                reranked_results,
            )

            if not final_results:
                answer = self.fallback

                self.memory.add_user_message(query)
                self.memory.add_ai_message(answer)

                logger.warning(
                    "No retrieval context found for query=%r",
                    query,
                )

                return {
                    "answer": answer,
                    "sources": [],
                    "retrieval_query": retrieval_query,
                }

            context = self._build_context(final_results)

            conversation_history = self.memory.get_messages()

            user_prompt = f"""
Use only the retrieved company context to answer the user's question.

Retrieved context:
{context}

Current user question:
{query}
"""

            messages = [
                SystemMessage(content=RAG_SYSTEM_PROMPT),
                *conversation_history,
                HumanMessage(content=user_prompt),
            ]

            response = self.model.invoke(messages)

            answer = response.content
            sources = self._extract_sources(final_results)

            self.memory.add_user_message(query)
            self.memory.add_ai_message(answer)

            logger.info(
                "Query processed successfully. Sources=%d",
                len(sources),
            )

            return {
                "answer": answer,
                "sources": sources,
                "retrieval_query": retrieval_query,
            }

        except Exception:
            logger.exception(
                "RAG processing failed for query=%r",
                query,
            )
            raise

    def clear_memory(self):
        self.memory.clear()
        logger.info("Conversation memory cleared.")

if __name__ == "__main__":
    rag = RAGChain()

    query = "What is the annual leave policy?"

    result = rag.answer(query)

    logger.info(
        "Manual RAG test completed. Query=%r",
        query,
    )

    logger.info(
        "Answer: %s",
        result["answer"],
    )

    logger.info(
        "Sources: %s",
        result["sources"],
    )   