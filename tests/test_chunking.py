from src.document_loader import load_documents
from src.chunking import split_documents


def test_chunking_returns_chunks():
    documents = load_documents()
    chunks = split_documents(documents)

    assert chunks is not None
    assert len(chunks) > 0


def test_chunk_count_is_greater_than_document_count():
    documents = load_documents()
    chunks = split_documents(documents)

    assert len(chunks) >= len(documents)


def test_chunk_metadata_contains_source():
    documents = load_documents()
    chunks = split_documents(documents)

    for chunk in chunks:
        assert "source" in chunk.metadata
        assert chunk.metadata["source"]


def test_chunk_metadata_contains_chunk_id():
    documents = load_documents()
    chunks = split_documents(documents)

    chunk_ids = []

    for chunk in chunks:
        assert "chunk_id" in chunk.metadata
        assert chunk.metadata["chunk_id"]

        chunk_ids.append(chunk.metadata["chunk_id"])

    assert len(chunk_ids) == len(set(chunk_ids))


def test_chunks_have_content():
    documents = load_documents()
    chunks = split_documents(documents)

    for chunk in chunks:
        assert chunk.page_content.strip()