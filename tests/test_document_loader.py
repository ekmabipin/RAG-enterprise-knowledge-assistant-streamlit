from src.document_loader import load_documents


def test_load_documents_returns_documents():
    documents = load_documents()

    assert documents is not None
    assert len(documents) > 0


def test_supported_file_types_are_loaded():
    documents = load_documents()

    file_types = {
        document.metadata.get("file_type")
        for document in documents
    }

    assert "pdf" in file_types
    assert "docx" in file_types
    assert "txt" in file_types


def test_source_metadata_exists():
    documents = load_documents()

    for document in documents:
        assert "source" in document.metadata
        assert document.metadata["source"]


def test_pdf_documents_have_page_metadata():
    documents = load_documents()

    pdf_documents = [
        document
        for document in documents
        if document.metadata.get("file_type") == "pdf"
    ]

    assert len(pdf_documents) > 0

    for document in pdf_documents:
        assert "page" in document.metadata