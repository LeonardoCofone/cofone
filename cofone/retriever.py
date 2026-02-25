import json
from pathlib import Path
from collections import defaultdict
from .chunker import chunk_text
import logging
import transformers

transformers.logging.set_verbosity_error()

class Retriever:
    def __init__(self, use_faiss=False, embedding_model="all-MiniLM-L6-v2", chunk_mode="smart", persist_path=None):
        self.chunks = []
        self.use_faiss = use_faiss
        self.embedding_model = embedding_model
        self.chunk_mode = chunk_mode
        self.persist_path = Path(persist_path) if persist_path else None
        self._index = defaultdict(list)
        self._faiss_index = None
        self._embedder = None

    def _get_embedder(self):
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer(self.embedding_model)
            except ImportError:
                raise ImportError("pip install sentence-transformers")
        return self._embedder

    def index(self, docs):
        if self.persist_path and self._load_from_disk():
            return

        for doc in docs:
            for chunk in chunk_text(doc, mode=self.chunk_mode):
                self.chunks.append(chunk)

        if self.use_faiss:
            self._build_faiss()
            if self.persist_path:
                self._save_to_disk()
        else:
            self._build_bm25()

    def _build_bm25(self):
        for idx, chunk in enumerate(self.chunks):
            for word in set(chunk.lower().split()):
                self._index[word].append(idx)

    def _build_faiss(self):
        try:
            import faiss
            import numpy as np
        except ImportError:
            raise ImportError("pip install faiss-cpu")

        embedder = self._get_embedder()
        vectors = embedder.encode(self.chunks, convert_to_numpy=True).astype("float32")
        dim = vectors.shape[1]
        self._faiss_index = faiss.IndexFlatL2(dim)
        self._faiss_index.add(vectors)

    def _save_to_disk(self):
        import faiss
        self.persist_path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._faiss_index, str(self.persist_path / "index.faiss"))
        (self.persist_path / "chunks.json").write_text(json.dumps(self.chunks), encoding="utf-8")
        print(f"[cofone] index saved to {self.persist_path}")

    def _load_from_disk(self):
        if not self.persist_path:
            return False
        index_file = self.persist_path / "index.faiss"
        chunks_file = self.persist_path / "chunks.json"
        if not index_file.exists() or not chunks_file.exists():
            return False
        try:
            import faiss
            self._faiss_index = faiss.read_index(str(index_file))
            self.chunks = json.loads(chunks_file.read_text(encoding="utf-8"))
            print(f"[cofone] index loaded from {self.persist_path} ({len(self.chunks)} chunks)")
            return True
        except Exception as e:
            print(f"[cofone] cache load failed: {e}")
            return False

    def query(self, text, top_k=5):
        if self.use_faiss:
            return self._query_faiss(text, top_k)
        return self._query_bm25(text, top_k)

    def _query_faiss(self, text, top_k):
        embedder = self._get_embedder()
        vec = embedder.encode([text], convert_to_numpy=True).astype("float32")
        _, indices = self._faiss_index.search(vec, min(top_k, len(self.chunks)))
        return [self.chunks[i] for i in indices[0] if i < len(self.chunks)]

    def _query_bm25(self, text, top_k):
        scores = defaultdict(int)
        for word in text.lower().split():
            for idx in self._index.get(word, []):
                scores[idx] += 1
        if scores:
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            return [self.chunks[i] for i, _ in ranked[:top_k]]
        return self.chunks[:top_k]