from pathlib import Path

from docx import Document as DocxDocument
from pypdf import PdfReader
from langchain_core.documents import Document

from config import load_config
from src.logger import get_logger

logger = get_logger(__name__)


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


def load_pdf(file_path: Path):
    documents = []

    reader = PdfReader(str(file_path))

    for page_number, page in enumerate(reader.pages):
        text = page.extract_text() or ""

        if not text.strip():
            continue

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": file_path.name,
                    "file_type": "pdf",
                    "page": page_number,
                },
            )
        )

    return documents


def load_docx(file_path: Path):
    docx_file = DocxDocument(str(file_path))

    paragraphs = [
        paragraph.text
        for paragraph in docx_file.paragraphs
        if paragraph.text.strip()
    ]

    text = "\n".join(paragraphs)

    if not text.strip():
        return []

    return [
        Document(
            page_content=text,
            metadata={
                "source": file_path.name,
                "file_type": "docx",
            },
        )
    ]


def load_txt(file_path: Path):
    text = file_path.read_text(encoding="utf-8")

    if not text.strip():
        return []

    return [
        Document(
            page_content=text,
            metadata={
                "source": file_path.name,
                "file_type": "txt",
            },
        )
    ]


def load_documents(data_dir: str | None = None):
    """
    Load PDF, DOCX, and TXT files from the configured data directory.

    Returns:
        list[Document]: LangChain Document objects.
    """

    config = load_config()

    if data_dir is None:
        data_dir = config["paths"]["data_dir"]

    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Data directory does not exist: {data_path.resolve()}"
        )

    if not data_path.is_dir():
        raise NotADirectoryError(
            f"Expected directory but found: {data_path.resolve()}"
        )

    documents = []

    for file_path in sorted(data_path.iterdir()):
        if not file_path.is_file():
            continue

        extension = file_path.suffix.lower()

        if file_path.name == ".DS_Store":
            continue

        if extension not in SUPPORTED_EXTENSIONS:
            logger.info(f"Skipping unsupported file: {file_path.name}")
            continue

                

        # if extension not in SUPPORTED_EXTENSIONS:
        #     print(f"Skipping unsupported file: {file_path.name}")
        #     continue

        try:
            if extension == ".pdf":
                loaded_documents = load_pdf(file_path)

            elif extension == ".docx":
                loaded_documents = load_docx(file_path)

            else:
                loaded_documents = load_txt(file_path)

            documents.extend(loaded_documents)

            logger.info(
                f"Loaded: {file_path.name} "
                f"({len(loaded_documents)} document/page(s))"
            )

        except Exception as error:
            logger.info(
                "Loaded: %s (%d document/page(s))",
                file_path.name,
                len(loaded_documents),
            )

    logger.info("\nDocument loading completed.")
    logger.info(f"Total LangChain documents/pages loaded: {len(documents)}")

    return documents


if __name__ == "__main__":
    docs = load_documents()

    print("\nSample loaded documents")
    print("=" * 60)

    for index, doc in enumerate(docs[:5], start=1):
        logger.info(f"\nDocument {index}")
        logger.info(f"Source: {doc.metadata.get('source')}")
        logger.info(f"File type: {doc.metadata.get('file_type')}")
        logger.info(f"Page: {doc.metadata.get('page', 'N/A')}")
        logger.info("-" * 60)
        logger.info(doc.page_content[:500])