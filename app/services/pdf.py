import io
import logging
from dataclasses import dataclass
from pypdf import PdfReader

logger = logging.getLogger(__name__)


@dataclass
class PDFMetadata:
    title: str | None
    author: str | None
    page_count: int


@dataclass
class ParsedPage:
    page_number: int
    text: str


@dataclass
class ParsedPDF:
    metadata: PDFMetadata
    pages: list[ParsedPage]


class PDFParserService:
    @staticmethod
    def parse_pdf(file_bytes: bytes) -> ParsedPDF:
        stream = io.BytesIO(file_bytes)
        reader = PdfReader(stream)

        meta = reader.metadata
        title = None
        author = None

        if meta:
            if meta.title:
                title = str(meta.title).replace("\x00", "").strip() or None
            if meta.author:
                author = str(meta.author).replace("\x00", "").strip() or None

        pages: list[ParsedPage] = []
        page_count = len(reader.pages)

        for index, page in enumerate(reader.pages):
            try:
                extracted = page.extract_text() or ""
            except Exception as e:
                logger.warning(f"Failed to extract text from page {index + 1}: {e}")
                extracted = ""

            cleaned_text = extracted.replace("\x00", "")
            pages.append(ParsedPage(page_number=index + 1, text=cleaned_text))

        return ParsedPDF(
            metadata=PDFMetadata(
                title=title,
                author=author,
                page_count=page_count,
            ),
            pages=pages,
        )
