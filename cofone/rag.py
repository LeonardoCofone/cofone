import os
from .loader import load_documents
from .retriever import Retriever
from .llm import ask, stream as llm_stream, DEFAULT_MODELS, _detect_provider


class RAG:
    def __init__(
        self,
        model=None,
        api_key=None,
        base_url=None,
        provider=None,
        faiss=False,
        embedding_model="all-MiniLM-L6-v2",
        chunk_mode="smart",
        persist_path=None,
        memory=False,
    ):
        self.provider = _detect_provider(provider, model, base_url)
        self.model = model or DEFAULT_MODELS[self.provider]
        self.api_key = api_key
        self.base_url = base_url
        self._retriever = Retriever(
            use_faiss=faiss,
            embedding_model=embedding_model,
            chunk_mode=chunk_mode,
            persist_path=persist_path,
        )
        self._tools = []
        self._debug = False
        self._memory = memory
        self._history = []

    def add_source(self, source):
        docs = load_documents(source)
        if self._debug:
            print(f"[cofone] loaded {len(docs)} doc(s) from: {source}")
        self._retriever.index(docs)
        return self

    def add_tool(self, fn):
        self._tools.append(fn)
        return self

    def debug(self):
        self._debug = True
        return self

    def _build_prompt(self, query, context):
        if self._memory and self._history:
            history_str = "\n".join(
                f"User: {q}\nAssistant: {a}" for q, a in self._history
            )
            return f"Conversation so far:\n{history_str}\n\nContext:\n{context}\n\nUser: {query}\nAssistant:"
        return f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

    def run(self, query, schema=None):
        chunks = self._retriever.query(query)

        if self._debug:
            print(f"\n[cofone] provider: {self.provider} | model: {self.model}")
            print(f"[cofone] query: {query}")
            print(f"[cofone] chunks found: {len(chunks)}")
            for i, c in enumerate(chunks):
                print(f"  [{i}] {c[:80]}...")

        context = "\n\n".join(chunks)
        prompt = self._build_prompt(query, context)
        answer = ask(prompt, model=self.model, api_key=self.api_key, base_url=self.base_url, provider=self.provider, schema=schema)

        if self._memory:
            self._history.append((query, answer if isinstance(answer, str) else str(answer)))

        return answer

    def chat(self, query):
        self._memory = True
        return self.run(query)

    def stream(self, query):
        chunks = self._retriever.query(query)
        context = "\n\n".join(chunks)
        prompt = self._build_prompt(query, context)

        if self._debug:
            print(f"\n[cofone] provider: {self.provider} | model: {self.model}")
            print(f"[cofone] query: {query}")
            print(f"[cofone] chunks found: {len(chunks)}")

        yield from llm_stream(prompt, model=self.model, api_key=self.api_key, base_url=self.base_url, provider=self.provider)

    def reset_memory(self):
        self._history = []
        return self