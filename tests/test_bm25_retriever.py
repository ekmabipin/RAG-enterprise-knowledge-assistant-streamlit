from src.bm25_retriever import BM25Retriever


def test_bm25_retriever_initializes():
    retriever = BM25Retriever()

    assert retriever is not None
    assert len(retriever.chunks) > 0


def test_bm25_search_returns_results():
    retriever = BM25Retriever()

    results = retriever.search(
        "How many annual leave days can I carry forward?"
    )

    assert results is not None
    assert len(results) > 0


def test_bm25_results_have_scores():
    retriever = BM25Retriever()

    results = retriever.search(
        "How many annual leave days can I carry forward?"
    )

    for document, score in results:
        assert document is not None
        assert isinstance(score, (int, float))


def test_bm25_finds_leave_policy():
    retriever = BM25Retriever()

    results = retriever.search(
        "How many annual leave days can I carry forward?"
    )

    sources = [
        document.metadata.get("source")
        for document, _ in results
    ]

    assert "leave_policy.pdf" in sources