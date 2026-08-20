from dataclasses import dataclass
from typing import List, Sequence


@dataclass
class ChunkResult:
    chunk_index: int
    page_id: int | None
    page_number: int
    text: str
    char_count: int


class RecursiveTextSplitter:
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: Sequence[str] | None = None,
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", " ", ""]

    def split_text(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []
        return self._split_text(text, self.separators)

    def _split_text(self, text: str, separators: Sequence[str]) -> List[str]:
        final_chunks: List[str] = []

        separator = separators[-1]
        new_separators: Sequence[str] = []
        for i, s in enumerate(separators):
            if s == "":
                separator = s
                break
            if s in text:
                separator = s
                new_separators = separators[i + 1 :]
                break

        splits = text.split(separator) if separator else list(text)

        good_splits: List[str] = []
        _separator = separator if separator else ""

        for s in splits:
            if len(s) < self.chunk_size:
                good_splits.append(s)
            else:
                if good_splits:
                    merged = self._merge_splits(good_splits, _separator)
                    final_chunks.extend(merged)
                    good_splits = []

                if not new_separators:
                    final_chunks.append(s[: self.chunk_size])
                else:
                    recursive_chunks = self._split_text(s, new_separators)
                    final_chunks.extend(recursive_chunks)

        if good_splits:
            merged = self._merge_splits(good_splits, _separator)
            final_chunks.extend(merged)

        return final_chunks

    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        docs: List[str] = []
        current_doc: List[str] = []
        total = 0
        sep_len = len(separator)

        for d in splits:
            _len = len(d)
            if total + _len + (sep_len if current_doc else 0) > self.chunk_size:
                if total > 0:
                    doc_str = separator.join(current_doc).strip()
                    if doc_str:
                        docs.append(doc_str)

                    while total > 0 and (
                        total + _len + (sep_len if current_doc else 0) > self.chunk_size
                        or (total > self.chunk_overlap and total > 0)
                    ):
                        popped = current_doc.pop(0)
                        total -= len(popped) + (sep_len if current_doc else 0)
                        if total < 0:
                            total = 0

            current_doc.append(d)
            total += _len + (sep_len if len(current_doc) > 1 else 0)

        if current_doc:
            doc_str = separator.join(current_doc).strip()
            if doc_str:
                docs.append(doc_str)

        return docs


class ChunkingService:
    @staticmethod
    def chunk_pages(
        pages: List[tuple[int, int, str]],  # (page_id, page_number, page_text)
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ) -> List[ChunkResult]:
        splitter = RecursiveTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        results: List[ChunkResult] = []
        chunk_index = 0

        for page_id, page_number, text in pages:
            raw_chunks = splitter.split_text(text)
            for chunk_text in raw_chunks:
                cleaned = chunk_text.strip()
                if cleaned:
                    results.append(
                        ChunkResult(
                            chunk_index=chunk_index,
                            page_id=page_id,
                            page_number=page_number,
                            text=cleaned,
                            char_count=len(cleaned),
                        )
                    )
                    chunk_index += 1

        return results
