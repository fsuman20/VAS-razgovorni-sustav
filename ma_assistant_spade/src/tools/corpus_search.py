from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Korpusi su ai generirani: https://chatgpt.com/s/t_696d5791085c8191b8ecba099705f2eb
#Ukredano iz vlastitog zavrsnog rada dostuponog na foi radovi

@dataclass
class Chunk:
    doc_id: str
    chunk_id: int
    text: str


class CorpusIndex:
    """Vrlo mali lokalni indeks (TF-IDF preko chunka teksta).

    Namijenjeno za prototip kolegija: stavi izvore kao .txt datoteke u data/corpus/.
    """

    def __init__(self, corpus_dir: str, chunk_chars: int = 900, overlap: int = 150):
        self.corpus_dir = Path(corpus_dir)
        self.chunk_chars = chunk_chars
        self.overlap = overlap

        self.chunks: List[Chunk] = []
        self._vectorizer = TfidfVectorizer(stop_words=None)
        self._X = None

    def build(self) -> None:
        files = sorted(self.corpus_dir.glob("*.txt"))
        if not files:
            # Kreiraj placeholder datoteku kako bi program radio.
            self.corpus_dir.mkdir(parents=True, exist_ok=True)
            placeholder = self.corpus_dir / "README_ADD_SOURCES.txt"
            if not placeholder.exists():
                placeholder.write_text(
                    "Dodaj vlastite izvore kao .txt datoteke u ovaj direktorij (mini-korpus).\n"
                    "Npr. izvatci iz skripte/PDF-a ili iz znanstvenih radova.\n",
                    encoding="utf-8",
                )
            files = [placeholder]

        self.chunks = []
        for f in files:
            text = f.read_text(encoding="utf-8", errors="ignore")
            text = self._normalize(text)
            for i, chunk_text in enumerate(self._chunk(text)):
                self.chunks.append(Chunk(doc_id=f.stem, chunk_id=i, text=chunk_text))

        corpus_texts = [c.text for c in self.chunks]
        self._X = self._vectorizer.fit_transform(corpus_texts)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        if self._X is None:
            self.build()
        q = self._vectorizer.transform([self._normalize(query)])
        sims = cosine_similarity(q, self._X)[0]
        idxs = sims.argsort()[::-1][:top_k]
        out: List[Tuple[Chunk, float]] = []
        for idx in idxs:
            out.append((self.chunks[int(idx)], float(sims[int(idx)])))
        return out

    @staticmethod #-> Github copilot je predlozio staticmethod ovdje 
    def _normalize(text: str) -> str:
        text = text.replace("\r\n", "\n")
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _chunk(self, text: str) -> List[str]:
        chunks: List[str] = []
        n = len(text)
        start = 0
        while start < n:
            end = min(n, start + self.chunk_chars)
            chunk = text[start:end]
            chunks.append(chunk)
            if end == n:
                break
            start = max(0, end - self.overlap)
        return chunks
