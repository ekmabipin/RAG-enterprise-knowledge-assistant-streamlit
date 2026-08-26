from src.hybrid_retriever import HybridRetriever


def test_hybrid_retriever_initializes():
    retriever = HybridRetriever()

    assert retriever is not None
    assert retriever.bm25_retriever is not None


def test_hybrid_search_returns_results():
    retriever = HybridRetriever()

    results = retriever.search(
        "How many annual leave days can I carry forward?"
    )

    assert results is not None
    assert len(results) > 0


def test_hybrid_results_have_expected_fields():
    retriever = HybridRetriever()

    results = retriever.search(
        "How many annual leave days can I carry forward?"
    )

    for result in results:
        assert "document" in result
        assert "vector_score" in result
        assert "bm25_score" in result
        assert "hybrid_score" in result


def test_hybrid_results_are_sorted_by_score():
    retriever = HybridRetriever()

    results = retriever.search(
        "How many annual leave days can I carry forward?"
    )

    scores = [
        result["hybrid_score"]
        for result in results
    ]

    assert scores == sorted(scores, reverse=True)


def test_hybrid_search_finds_leave_policy():
    retriever = HybridRetriever()

    results = retriever.search(
        "How many annual leave days can I carry forward?"
    )

    sources = [
        result["document"].metadata.get("source")
        for result in results
    ]

    assert "leave_policy.pdf" in sources