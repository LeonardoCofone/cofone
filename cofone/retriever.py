import json
import os
from pathlib import Path
from collections import defaultdict
from .chunker import chunk_text

try:
    import transformers
    transformers.logging.set_verbosity_error()
except ImportError:
    pass


# ── Embedding Providers ────────────────────────────────────────────────────────
#
# "local"      → sentence-transformers (runs on your machine, no key, no internet)
# "openai"     → OpenAI Embeddings API  (text-embedding-3-small, text-embedding-3-large)
# "gemini"     → Google Gemini          (text-embedding-004)
# "cohere"     → Cohere Embed API       (embed-multilingual-v3.0, embed-english-v3.0)
# "mistral"    → Mistral                (mistral-embed)
# "voyage"     → Voyage AI              (voyage-3, voyage-3-lite, voyage-code-3)
# "jina"       → Jina AI               (jina-embeddings-v3)
# "nvidia"     → NVIDIA                 (nvidia/nv-embed-v2)
# "together"   → Together AI            (BAAI/bge-large-en-v1.5)
# "openrouter" → OpenRouter             (routes to various embedding models)
# "ollama"     → local Ollama           (nomic-embed-text, mxbai-embed-large)
#
EMBEDDING_ENV_KEYS = {
    "openai":      "OPENAI_API_KEY",
    "gemini":      "GEMINI_API_KEY",
    "cohere":      "COHERE_API_KEY",
    "mistral":     "MISTRAL_API_KEY",
    "openrouter":  "OPENROUTER_API_KEY",
    "voyage":      "VOYAGE_API_KEY",
    "jina":        "JINA_API_KEY",
    "nvidia":      "NVIDIA_API_KEY",
    "together":    "TOGETHER_API_KEY",
    "ollama":      None,   # local, no key
    "local":       None,   # local, no key
}

EMBEDDING_URLS = {
    "openai":      "https://api.openai.com/v1",
    "gemini":      "https://generativelanguage.googleapis.com/v1beta/openai",
    "mistral":     "https://api.mistral.ai/v1",
    "openrouter":  "https://openrouter.ai/api/v1",
    "nvidia":      "https://integrate.api.nvidia.com/v1",
    "together":    "https://api.together.xyz/v1",
    "ollama":      "http://localhost:11434/v1",
}

LOCAL_EMBEDDING_PROVIDERS = {"local", "ollama"}


class Retriever:
    """
    Handles document chunking, indexing, and retrieval.

    Supports two retrieval modes:
    - BM25 (default): keyword-based, no extra dependencies.
    - FAISS: semantic vector search, requires faiss-cpu + an embedding model.

    Parameters
    ----------
    use_faiss : bool
        If True, uses FAISS semantic search. Default: False (BM25).
    embedding_provider : str
        Provider for text embeddings. Default: "local".
    embedding_model : str
        Embedding model identifier. Default: "all-MiniLM-L6-v2".
    embedding_api_key : str, optional
        API key for the embedding provider. Reads from env if not given.
    chunk_mode : str
        How to split documents. One of: smart, paragraphs, sentences, fixed.
    persist_path : str or Path, optional
        Folder to save/load the FAISS index. Skips recomputation on reload.
    """

    def __init__(
        self,
        use_faiss=False,
        embedding_provider="local",
        embedding_model="all-MiniLM-L6-v2",
        embedding_api_key=None,
        chunk_mode="smart",
        persist_path=None,
    ):
        self.chunks              = []
        self.use_faiss           = use_faiss
        self.embedding_provider  = embedding_provider
        self.embedding_model     = embedding_model
        self.embedding_api_key   = embedding_api_key
        self.chunk_mode          = chunk_mode
        self.persist_path        = Path(persist_path) if persist_path else None
        self._index              = defaultdict(list)  # BM25 inverted index
        self._faiss_index        = None
        self._local_embedder     = None               # lazy-loaded SentenceTransformer

    # ── Key resolution ─────────────────────────────────────────────────────────

    def _resolve_embedding_key(self):
        """Return the API key for the embedding provider, or raise if missing."""
        if self.embedding_api_key:
            return self.embedding_api_key
        if self.embedding_provider in LOCAL_EMBEDDING_PROVIDERS:
            return None
        env = EMBEDDING_ENV_KEYS.get(self.embedding_provider)
        if env:
            key = os.environ.get(env)
            if not key:
                raise ValueError(
                    f"[cofone] embedding API key not found for '{self.embedding_provider}'.\n"
                    f"Set {env} in your .env file and call load_dotenv(),\n"
                    f"or pass embedding_api_key='...' directly to RAG()."
                )
            return key
        return None

    # ── Embed dispatch ─────────────────────────────────────────────────────────

    def _embed(self, texts):
        """Dispatch embedding to the correct provider method."""
        p = self.embedding_provider
        if p == "local":
            return self._embed_local(texts)
        if p in ("openai", "gemini", "mistral", "openrouter", "nvidia", "together"):
            return self._embed_openai_compat(texts)
        if p == "ollama":
            return self._embed_ollama(texts)
        if p == "cohere":
            return self._embed_cohere(texts)
        if p == "voyage":
            return self._embed_voyage(texts)
        if p == "jina":
            return self._embed_jina(texts)
        raise ValueError(
            f"[cofone] unknown embedding provider '{p}'.\n"
            f"Valid options: local, openai, gemini, cohere, mistral, voyage, "
            f"jina, nvidia, together, openrouter, ollama."
        )

    def _embed_local(self, texts):
        """Embed using a local sentence-transformers model. No API key needed."""
        try:
            from sentence_transformers import SentenceTransformer
            if self._local_embedder is None:
                self._local_embedder = SentenceTransformer(self.embedding_model)
            return self._local_embedder.encode(texts, convert_to_numpy=True).astype("float32")
        except ImportError:
            raise ImportError(
                "[cofone] sentence-transformers not installed.\n"
                "Run: pip install \"cofone[faiss]\"  or  pip install sentence-transformers"
            )

    def _embed_openai_compat(self, texts):
        """Embed using any OpenAI-compatible embeddings endpoint."""
        try:
            from openai import OpenAI
            import numpy as np
            base_url = EMBEDDING_URLS.get(self.embedding_provider, EMBEDDING_URLS["openai"])
            api_key  = self._resolve_embedding_key()
            client   = OpenAI(api_key=api_key or "local", base_url=base_url)
            response = client.embeddings.create(model=self.embedding_model, input=texts)
            return np.array([d.embedding for d in response.data], dtype="float32")
        except ImportError:
            raise ImportError(
                "[cofone] 'openai' package not found.\n"
                "Run: pip install openai"
            )

    def _embed_ollama(self, texts):
        """Embed using a locally running Ollama model. No API key needed."""
        try:
            from openai import OpenAI
            import numpy as np
            client   = OpenAI(api_key="ollama", base_url=EMBEDDING_URLS["ollama"])
            response = client.embeddings.create(model=self.embedding_model, input=texts)
            return np.array([d.embedding for d in response.data], dtype="float32")
        except ImportError:
            raise ImportError(
                "[cofone] 'openai' package not found.\n"
                "Run: pip install openai"
            )

    def _embed_cohere(self, texts):
        """Embed using the Cohere Embed API. Requires COHERE_API_KEY."""
        try:
            import cohere
            import numpy as np
            client   = cohere.Client(self._resolve_embedding_key())
            response = client.embed(
                texts=texts,
                model=self.embedding_model,
                input_type="search_document",
            )
            return np.array(response.embeddings, dtype="float32")
        except ImportError:
            raise ImportError(
                "[cofone] 'cohere' package not found.\n"
                "Run: pip install cohere"
            )

    def _embed_voyage(self, texts):
        """Embed using the Voyage AI API. Requires VOYAGE_API_KEY."""
        try:
            import voyageai
            import numpy as np
            client   = voyageai.Client(api_key=self._resolve_embedding_key())
            response = client.embed(texts, model=self.embedding_model)
            return np.array(response.embeddings, dtype="float32")
        except ImportError:
            raise ImportError(
                "[cofone] 'voyageai' package not found.\n"
                "Run: pip install voyageai"
            )

    def _embed_jina(self, texts):
        """Embed using the Jina AI API. Requires JINA_API_KEY."""
        try:
            import httpx
            import numpy as np
            headers = {
                "Authorization": f"Bearer {self._resolve_embedding_key()}",
                "Content-Type": "application/json",
            }
            resp = httpx.post(
                "https://api.jina.ai/v1/embeddings",
                json={"model": self.embedding_model, "input": texts},
                headers=headers,
                timeout=30,
            )
            resp.raise_for_status()
            return np.array([d["embedding"] for d in resp.json()["data"]], dtype="float32")
        except ImportError:
            raise ImportError(
                "[cofone] 'httpx' package not found.\n"
                "Run: pip install httpx"
            )

    # ── Index ──────────────────────────────────────────────────────────────────

    def index(self, docs):
        """
        Chunk and index a list of document strings.

        If persist_path is set and a valid index exists on disk, loads
        from disk instead of recomputing embeddings.
        """
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
        """Build an inverted index for BM25 keyword retrieval."""
        for idx, chunk in enumerate(self.chunks):
            for word in set(chunk.lower().split()):
                self._index[word].append(idx)

    def _build_faiss(self):
        """Encode all chunks and build a FAISS L2 flat index."""
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "[cofone] faiss-cpu not installed.\n"
                "Run: pip install \"cofone[faiss]\"  or  pip install faiss-cpu"
            )
        vectors = self._embed(self.chunks)
        dim = vectors.shape[1]
        self._faiss_index = faiss.IndexFlatL2(dim)
        self._faiss_index.add(vectors)

    def _save_to_disk(self):
        """Save the FAISS index and chunks to disk."""
        import faiss
        self.persist_path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._faiss_index, str(self.persist_path / "index.faiss"))
        (self.persist_path / "chunks.json").write_text(
            json.dumps(self.chunks, ensure_ascii=False), encoding="utf-8"
        )
        print(f"[cofone] index saved to {self.persist_path} ({len(self.chunks)} chunks)")

    def _load_from_disk(self):
        """Load the FAISS index and chunks from disk. Returns True if successful."""
        index_file  = self.persist_path / "index.faiss"
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
            print(f"[cofone] cache load failed ({e}), rebuilding index...")
            return False

    # ── Query ──────────────────────────────────────────────────────────────────

    def query(self, text, top_k=5):
        """Return the top_k most relevant chunks for the given query text."""
        if self.use_faiss:
            return self._query_faiss(text, top_k)
        return self._query_bm25(text, top_k)

    def _query_faiss(self, text, top_k):
        """Encode the query and search the FAISS index."""
        vec = self._embed([text])
        _, indices = self._faiss_index.search(vec, min(top_k, len(self.chunks)))
        return [self.chunks[i] for i in indices[0] if i < len(self.chunks)]

    def _query_bm25(self, text, top_k):
        """Score chunks by word overlap frequency and return the top_k."""
        scores = defaultdict(int)
        for word in text.lower().split():
            for idx in self._index.get(word, []):
                scores[idx] += 1
        if scores:
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            return [self.chunks[i] for i, _ in ranked[:top_k]]
        # fallback: return first top_k chunks if no overlap found
        return self.chunks[:top_k]