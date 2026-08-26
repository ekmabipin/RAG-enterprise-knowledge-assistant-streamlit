from src.rag_chain import RAGChain


def test_rag_chain_initializes():
    rag = RAGChain()

    assert rag is not None
    assert rag.hybrid_retriever is not None
    assert rag.reranker is not None
    assert rag.memory is not None


def test_rag_chain_returns_answer_and_sources():
    rag = RAGChain()

    result = rag.answer(
        "What is the annual leave policy?"
    )

    assert "answer" in result
    assert "sources" in result
    assert "retrieval_query" in result

    assert result["answer"]
    assert len(result["sources"]) > 0


def test_rag_chain_finds_leave_policy_source():
    rag = RAGChain()

    result = rag.answer(
        "How many annual leave days can I carry forward?"
    )

    assert "leave_policy.pdf" in result["sources"]


def test_rag_chain_conversation_memory():
    rag = RAGChain()

    rag.answer(
        "What is the annual leave policy?"
    )

    result = rag.answer(
        "What about carry-forward?"
    )

    assert result["retrieval_query"]
    assert "carry" in result["retrieval_query"].lower()


def test_rag_chain_clear_memory():
    rag = RAGChain()

    rag.answer(
        "What is the annual leave policy?"
    )

    assert len(rag.memory.get_messages()) > 0

    rag.clear_memory()

    assert rag.memory.get_messages() == []