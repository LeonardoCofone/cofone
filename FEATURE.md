# Cofone — Complete Features Reference

```bash
pip install "cofone[all]"
```

Every example assumes you have a `.env` file loaded:
```python
from dotenv import load_dotenv
load_dotenv()
```

---

## Table of contents

1. [How RAG() works — parameter logic](#1-how-rag-works--parameter-logic)
2. [Sources](#2-sources)
3. [Chunking](#3-chunking)
4. [Retrieval — BM25 vs FAISS](#4-retrieval--bm25-vs-faiss)
5. [FAISS persistence](#5-faiss-persistence)
6. [LLM providers — complete list](#6-llm-providers--complete-list)
7. [Embedding providers — complete list](#7-embedding-providers--complete-list)
8. [Debug mode](#8-debug-mode)
9. [Chat memory](#9-chat-memory)
10. [System prompt](#10-system-prompt)
11. [Streaming](#11-streaming)
12. [Structured output (Pydantic)](#12-structured-output-pydantic)
13. [Custom tools](#13-custom-tools)
14. [API key configuration](#14-api-key-configuration)
15. [Full parameter reference](#15-full-parameter-reference)

---

## 1. How RAG() works — parameter logic

`RAG()` has two independent groups of parameters: one for the **LLM** and one for **embeddings**.
They are completely separate — you can mix any LLM provider with any embedding provider.

```
RAG(
    # ── LLM (what answers the question) ──────────────────
    model_provider = "openrouter",                  # which service to call
    model          = "arcee-ai/trinity-...:free",   # which model on that service
    model_api_key  = "sk-or-...",                   # key for that service

    # ── Embeddings (only when faiss=True) ─────────────────
    faiss              = True,
    embedding_provider = "openai",
    embedding_model    = "text-embedding-3-small",
    embedding_api_key  = "sk-...",

    # ── RAG behaviour ─────────────────────────────────────
    chunk_mode   = "smart",
    persist_path = "./my_db",
    memory       = False,
)
```

### When can you omit parameters?

| Situation | What you need |
|---|---|
| Default usage (OpenRouter, free model, BM25) | Nothing — `RAG()` works if `OPENROUTER_API_KEY` is in `.env` |
| Specific LLM provider | `model_provider` + `model` + key in `.env` or `model_api_key=` |
| Local LLM (Ollama) | `model_provider="ollama"` + `model="llama3"` — no key needed |
| FAISS with local embeddings | `faiss=True` — no extra key, uses sentence-transformers |
| FAISS with API embeddings | `faiss=True` + `embedding_provider` + `embedding_model` + key in `.env` or `embedding_api_key=` |
| Fully local (no internet, no keys) | `model_provider="ollama"` + `embedding_provider="ollama"` or `"local"` |

### Auto-detection

`model_provider` is auto-detected from `model` when possible (see section 6).
If you pass both `model_provider` and `model`, auto-detection is skipped.

### Keys: .env vs direct parameter

Keys are resolved in this order:
1. `model_api_key=` / `embedding_api_key=` passed directly
2. Environment variable from `.env` (e.g. `OPENAI_API_KEY`)
3. If provider is local → no key needed

---

## 2. Sources

All loaded via `.add_source()`. Chain as many calls as you want.

### Single text file (.txt or .md)
```python
RAG().add_source("notes.txt").run("Summarize")
RAG().add_source("README.md").run("What does this project do?")
```

### Folder — recursive
Scans all subfolders. Loads every `.txt`, `.md`, and `.pdf` file found.
```python
RAG().add_source("documents/").run("What is the main topic?")
RAG().add_source("C:/Users/me/docs/").run("Find everything about pricing")
```

### PDF
Requires `pip install "cofone[pdf]"` (uses `pypdf` internally).
```python
RAG().add_source("report.pdf").run("What are the key findings?")
RAG().add_source("contracts/").run("What are the payment terms?")  # folder of PDFs
```

### Web URL (generic)
Fetches the page, strips HTML tags, nav, footer, scripts. Returns visible text only.
```python
RAG().add_source("https://example.com/blog/post").run("Summarize this article")
RAG().add_source("https://docs.python.org/3/library/os.html").run("How do I list files?")
```

### Wikipedia
Auto-detected from the URL. Uses the `wikipedia` Python package — returns clean article text without markup.
Supports any language via the subdomain (`it.wikipedia`, `fr.wikipedia`, etc.).
```python
RAG().add_source("https://en.wikipedia.org/wiki/Artificial_intelligence").run("What is AI?")
RAG().add_source("https://it.wikipedia.org/wiki/Leonardo_da_Vinci").run("Chi è Leonardo?")
RAG().add_source("https://fr.wikipedia.org/wiki/Tour_Eiffel").run("Quand a-t-elle été construite?")
```

### YouTube
Fetches the official transcript/subtitles. Auto-generated captions work too.
Language priority: Italian first, then English. If neither is available, raises an error.
```python
RAG().add_source("https://www.youtube.com/watch?v=jNQXAC9IVRw").run("What is this about?")
RAG().add_source("https://youtu.be/jNQXAC9IVRw").run("Summarize")  # short URL also works
```

### Multiple sources chained
All documents are merged into a single index. Retrieval searches across all of them.
```python
answer = (
    RAG()
    .add_source("internal_notes.txt")
    .add_source("docs/")
    .add_source("https://en.wikipedia.org/wiki/Machine_learning")
    .add_source("https://www.youtube.com/watch?v=VIDEO_ID")
    .run("Give me a complete overview combining all sources")
)
```

---

## 3. Chunking

Controls how documents are split into pieces before indexing.
Choosing the right mode significantly impacts retrieval quality.

### `smart` — default
Algorithm:
1. Split document by blank lines → paragraphs
2. For each paragraph: if `len > 600` chars → split into sentences
3. Group sentences until ~500 chars, then start a new chunk

Best for: structured articles, documentation, notes, Wikipedia pages.

```python
RAG(chunk_mode="smart").add_source("docs/").run("question")
```

### `paragraphs`
Splits only on blank lines (`\n\n`). Each paragraph becomes one chunk, regardless of length.

Best for: short, self-contained paragraphs. Poor for long dense blocks.

```python
RAG(chunk_mode="paragraphs").add_source("docs/").run("question")
```

### `sentences`
Splits on `.`, `!`, `?` boundaries. Groups sentences together up to ~500 characters.

Best for: dense text where every sentence matters (legal docs, scientific papers).

```python
RAG(chunk_mode="sentences").add_source("docs/").run("question")
```

### `fixed`
Splits into fixed-length slices of exactly 500 characters, with 50-character overlap between consecutive chunks.
The overlap prevents losing context at boundaries.

Best for: raw data exports, logs, text with no structure.

```python
RAG(chunk_mode="fixed").add_source("docs/").run("question")
```

### Comparing modes
```python
for mode in ["smart", "paragraphs", "sentences", "fixed"]:
    rag = RAG(chunk_mode=mode).add_source("document.txt")
    print(f"[{mode:12s}] → {len(rag._retriever.chunks)} chunks")
```

---

## 4. Retrieval — BM25 vs FAISS

### BM25 — default
**How it works:** counts how many query words appear in each chunk, ranks by frequency + rarity of terms (TF-IDF-like). Pure keyword matching.

**Pros:** zero extra dependencies, instant, works well when query words match document words exactly.  
**Cons:** misses synonyms and paraphrases. "automobile" won't match a chunk about "cars".

```python
RAG().add_source("docs/").run("Who invented the telephone?")
# finds chunks containing "invented", "telephone"
```

### FAISS — semantic search
**How it works:** converts every chunk and the query into a dense vector (embedding). Retrieves the chunks whose vectors are closest to the query vector in geometric space.

**Pros:** finds conceptually related chunks even without exact word overlap. "vehicle" matches "car", "automobile", "transportation".  
**Cons:** requires `faiss-cpu` + an embedding model. First run is slower (computes embeddings).

```python
RAG(faiss=True).add_source("docs/").run("Who invented the telephone?")
# finds chunks about Bell even if "telephone" isn't the exact word used
```

Requires `pip install "cofone[faiss]"`.

---

## 5. FAISS persistence

Without `persist_path`, FAISS rebuilds the index from scratch every time the script runs.
For large document sets this can take minutes. `persist_path` saves the index to disk.

**First run** — builds embeddings, saves index:
```python
RAG(
    faiss=True,
    persist_path="./my_index"
).add_source("1000_page_manual.pdf").run("How do I reset the device?")
# [cofone] index saved to ./my_index
```

**Second run onwards** — loads from disk instantly:
```python
RAG(
    faiss=True,
    persist_path="./my_index"
).add_source("1000_page_manual.pdf").run("What is the warranty period?")
# [cofone] index loaded from ./my_index (843 chunks)
```

**What's saved in the folder:**
```
my_index/
├── index.faiss   ← the vector index (binary)
└── chunks.json   ← the original text chunks (UTF-8 JSON)
```

**To rebuild from scratch:** delete the folder and run again.

```python
import shutil
shutil.rmtree("./my_index")
```

---

## 6. LLM providers — complete list

`model_provider` controls which service handles the LLM call.
Auto-detection from `model` name is available for the most common providers.

---

### OpenRouter — default
One API key → 200+ models. Free tier available.  
Key env var: `OPENROUTER_API_KEY`

```python
# uses default model (arcee-ai/trinity-large-preview:free)
RAG()

# explicit
RAG(model_provider="openrouter", model="arcee-ai/trinity-large-preview:free")
RAG(model_provider="openrouter", model="meta-llama/llama-3.3-70b-instruct:free")
RAG(model_provider="openrouter", model="google/gemini-2.0-flash-exp:free")
RAG(model_provider="openrouter", model="openai/gpt-4o")
RAG(model_provider="openrouter", model="anthropic/claude-3-5-sonnet")
RAG(model_provider="openrouter", model="mistralai/mistral-large")
RAG(model_provider="openrouter", model="deepseek/deepseek-r1")
RAG(model_provider="openrouter", model="qwen/qwen-2.5-72b-instruct")
```
Full list: [openrouter.ai/models](https://openrouter.ai/models)

---

### OpenAI
Auto-detected from: `gpt-*`, `o1-*`, `o3-*`, `o4-*`  
Key env var: `OPENAI_API_KEY`

```python
RAG(model_provider="openai", model="gpt-4o-mini")          # cheap, fast
RAG(model_provider="openai", model="gpt-4o")               # best GPT-4 class
RAG(model_provider="openai", model="gpt-4-turbo")
RAG(model_provider="openai", model="o1-mini")              # reasoning
RAG(model_provider="openai", model="o3-mini")              # reasoning, faster
# auto-detected:
RAG(model="gpt-4o-mini")
RAG(model="o3-mini")
```

---

### Anthropic (Claude)
Auto-detected from: `claude-*`  
Key env var: `ANTHROPIC_API_KEY`

```python
RAG(model_provider="anthropic", model="claude-3-5-haiku-20241022")    # fast, cheap
RAG(model_provider="anthropic", model="claude-3-5-sonnet-20241022")   # best balance
RAG(model_provider="anthropic", model="claude-3-opus-20240229")       # most capable
# auto-detected:
RAG(model="claude-3-5-sonnet-20241022")
```

---

### Google Gemini
Auto-detected from: `gemini-*`  
Key env var: `GEMINI_API_KEY`

```python
RAG(model_provider="gemini", model="gemini-2.0-flash")                # fast, free tier
RAG(model_provider="gemini", model="gemini-1.5-pro")                  # long context (1M tokens)
RAG(model_provider="gemini", model="gemini-2.0-flash-thinking-exp")   # reasoning
# auto-detected:
RAG(model="gemini-2.0-flash")
```

---

### Mistral
Auto-detected from: `mistral-*`, `codestral-*`  
Key env var: `MISTRAL_API_KEY`

```python
RAG(model_provider="mistral", model="mistral-small-latest")    # cheap
RAG(model_provider="mistral", model="mistral-large-latest")    # best Mistral
RAG(model_provider="mistral", model="codestral-latest")        # code specialized
RAG(model_provider="mistral", model="mistral-nemo")            # 12B, very fast
# auto-detected:
RAG(model="mistral-large-latest")
```

---

### Groq — very fast inference
Auto-detected from: `llama-*`, `mixtral-*`  
Key env var: `GROQ_API_KEY`

```python
RAG(model_provider="groq", model="llama-3.1-8b-instant")        # fastest
RAG(model_provider="groq", model="llama-3.3-70b-versatile")     # best quality
RAG(model_provider="groq", model="mixtral-8x7b-32768")
RAG(model_provider="groq", model="gemma2-9b-it")
# auto-detected:
RAG(model="llama-3.1-8b-instant")
```

---

### Cohere
Auto-detected from: `command-*`  
Key env var: `COHERE_API_KEY`

```python
RAG(model_provider="cohere", model="command-r-plus")   # best, multilingual
RAG(model_provider="cohere", model="command-r")        # faster
RAG(model_provider="cohere", model="command-light")    # cheapest
# auto-detected:
RAG(model="command-r-plus")
```

---

### DeepSeek
Auto-detected from: `deepseek-*`  
Key env var: `DEEPSEEK_API_KEY`

```python
RAG(model_provider="deepseek", model="deepseek-chat")       # general purpose
RAG(model_provider="deepseek", model="deepseek-reasoner")   # reasoning (R1)
# auto-detected:
RAG(model="deepseek-chat")
```

---

### xAI (Grok)
Auto-detected from: `grok-*`  
Key env var: `XAI_API_KEY`

```python
RAG(model_provider="xai", model="grok-beta")
RAG(model_provider="xai", model="grok-2-latest")
RAG(model_provider="xai", model="grok-2-vision-latest")   # multimodal
# auto-detected:
RAG(model="grok-beta")
```

---

### Together AI
Key env var: `TOGETHER_API_KEY`

```python
RAG(model_provider="together", model="meta-llama/Llama-3-8b-chat-hf")
RAG(model_provider="together", model="mistralai/Mixtral-8x7B-Instruct-v0.1")
RAG(model_provider="together", model="google/gemma-2-27b-it")
RAG(model_provider="together", model="Qwen/Qwen2.5-72B-Instruct-Turbo")
```

---

### Perplexity — web-connected models
Key env var: `PERPLEXITY_API_KEY`

```python
RAG(model_provider="perplexity", model="llama-3.1-sonar-small-128k-online")
RAG(model_provider="perplexity", model="llama-3.1-sonar-large-128k-online")
RAG(model_provider="perplexity", model="llama-3.1-sonar-huge-128k-online")
```

---

### Fireworks AI
Key env var: `FIREWORKS_API_KEY`

```python
RAG(model_provider="fireworks", model="accounts/fireworks/models/llama-v3p1-8b-instruct")
RAG(model_provider="fireworks", model="accounts/fireworks/models/mixtral-8x22b-instruct")
```

---

### Cerebras — ultra-fast inference
Key env var: `CEREBRAS_API_KEY`

```python
RAG(model_provider="cerebras", model="llama3.1-8b")
RAG(model_provider="cerebras", model="llama3.1-70b")
```

---

### NVIDIA
Key env var: `NVIDIA_API_KEY`

```python
RAG(model_provider="nvidia", model="meta/llama-3.1-8b-instruct")
RAG(model_provider="nvidia", model="mistralai/mistral-large")
RAG(model_provider="nvidia", model="google/gemma-2-27b-it")
```

---

### Ollama — fully local, no key, no internet
Requires Ollama running at `localhost:11434`.

```python
RAG(model_provider="ollama", model="llama3")
RAG(model_provider="ollama", model="mistral")
RAG(model_provider="ollama", model="phi3")
RAG(model_provider="ollama", model="deepseek-r1")
RAG(model_provider="ollama", model="gemma2")
RAG(model_provider="ollama", model="qwen2.5")
RAG(model_provider="ollama", model="codellama")   # code specialized
```

---

### LM Studio — local GUI app, no key
Requires LM Studio running at `localhost:1234`.

```python
RAG(model_provider="lmstudio", model="local-model")
```

---

### llama.cpp server — raw local server, no key
Requires llama.cpp server running at `localhost:8080`.

```python
RAG(model_provider="llamacpp", model="local-model")
```

---

### Custom endpoint
For any OpenAI-compatible API not listed above.

```python
RAG(
    base_url="https://my-custom-endpoint/v1",
    model_api_key="my-key",
    model="my-model",
)
```

---

## 7. Embedding providers — complete list

Used only when `faiss=True`. Controls which model converts text to vectors.
`embedding_provider` + `embedding_model` are independent from `model_provider` + `model`.

---

### `local` — default, sentence-transformers, no API key

Runs entirely on your machine. No internet required after the first download.

```python
# English, fast, 80MB — default
RAG(faiss=True, embedding_provider="local", embedding_model="all-MiniLM-L6-v2")

# English, best quality, 420MB
RAG(faiss=True, embedding_provider="local", embedding_model="all-mpnet-base-v2")

# 50+ languages, 270MB
RAG(faiss=True, embedding_provider="local", embedding_model="paraphrase-multilingual-MiniLM-L12-v2")

# 50+ languages, best multilingual, 1GB
RAG(faiss=True, embedding_provider="local", embedding_model="paraphrase-multilingual-mpnet-base-v2")

# SOTA English (BGE), 1.3GB
RAG(faiss=True, embedding_provider="local", embedding_model="BAAI/bge-large-en-v1.5")

# SOTA multilingual (BGE-M3), supports 100+ languages
RAG(faiss=True, embedding_provider="local", embedding_model="BAAI/bge-m3")

# E5 family, English
RAG(faiss=True, embedding_provider="local", embedding_model="intfloat/e5-large-v2")

# E5 family, multilingual
RAG(faiss=True, embedding_provider="local", embedding_model="intfloat/multilingual-e5-large")
```

Requires `pip install "cofone[faiss]"` (includes `sentence-transformers`).

---

### `openai` — OpenAI Embeddings API
Key env var: `OPENAI_API_KEY`

```python
# cheap, fast, 1536 dimensions — recommended for most use cases
RAG(faiss=True, embedding_provider="openai", embedding_model="text-embedding-3-small")

# best quality, 3072 dimensions
RAG(faiss=True, embedding_provider="openai", embedding_model="text-embedding-3-large")

# legacy, 1536 dimensions
RAG(faiss=True, embedding_provider="openai", embedding_model="text-embedding-ada-002")

# pass key directly (overrides .env)
RAG(faiss=True, embedding_provider="openai", embedding_model="text-embedding-3-small",
    embedding_api_key="sk-...")
```

---

### `gemini` — Google Gemini Embeddings
Key env var: `GEMINI_API_KEY`

```python
# latest, 768 dimensions
RAG(faiss=True, embedding_provider="gemini", embedding_model="text-embedding-004")

# legacy
RAG(faiss=True, embedding_provider="gemini", embedding_model="embedding-001")
```

---

### `cohere` — Cohere Embed API
Key env var: `COHERE_API_KEY`  
Requires `pip install cohere`.

```python
# English, 1024 dimensions
RAG(faiss=True, embedding_provider="cohere", embedding_model="embed-english-v3.0")

# 100+ languages, 1024 dimensions
RAG(faiss=True, embedding_provider="cohere", embedding_model="embed-multilingual-v3.0")

# English, lighter
RAG(faiss=True, embedding_provider="cohere", embedding_model="embed-english-light-v3.0")

# multilingual, lighter
RAG(faiss=True, embedding_provider="cohere", embedding_model="embed-multilingual-light-v3.0")
```

---

### `mistral` — Mistral Embeddings
Key env var: `MISTRAL_API_KEY`

```python
# 1024 dimensions
RAG(faiss=True, embedding_provider="mistral", embedding_model="mistral-embed")
```

---

### `voyage` — Voyage AI (state-of-the-art retrieval quality)
Key env var: `VOYAGE_API_KEY`  
Requires `pip install voyageai`.

```python
# general purpose, 1024 dimensions
RAG(faiss=True, embedding_provider="voyage", embedding_model="voyage-3")

# faster and cheaper
RAG(faiss=True, embedding_provider="voyage", embedding_model="voyage-3-lite")

# optimized for code retrieval
RAG(faiss=True, embedding_provider="voyage", embedding_model="voyage-code-3")

# multilingual
RAG(faiss=True, embedding_provider="voyage", embedding_model="voyage-multilingual-2")

# for financial/legal documents
RAG(faiss=True, embedding_provider="voyage", embedding_model="voyage-finance-2")
RAG(faiss=True, embedding_provider="voyage", embedding_model="voyage-law-2")
```

---

### `jina` — Jina AI Embeddings
Key env var: `JINA_API_KEY`

```python
# latest, multilingual, 8192 token context
RAG(faiss=True, embedding_provider="jina", embedding_model="jina-embeddings-v3")

# English
RAG(faiss=True, embedding_provider="jina", embedding_model="jina-embeddings-v2-base-en")

# multilingual
RAG(faiss=True, embedding_provider="jina", embedding_model="jina-embeddings-v2-base-multilingual")

# ColBERT-style for late interaction
RAG(faiss=True, embedding_provider="jina", embedding_model="jina-colbert-v2")
```

---

### `nvidia` — NVIDIA Embeddings
Key env var: `NVIDIA_API_KEY`

```python
RAG(faiss=True, embedding_provider="nvidia", embedding_model="nvidia/nv-embed-v2")
RAG(faiss=True, embedding_provider="nvidia", embedding_model="nvidia/embed-qa-4")
RAG(faiss=True, embedding_provider="nvidia", embedding_model="baai/bge-m3")
```

---

### `together` — Together AI Embeddings
Key env var: `TOGETHER_API_KEY`

```python
RAG(faiss=True, embedding_provider="together", embedding_model="BAAI/bge-large-en-v1.5")
RAG(faiss=True, embedding_provider="together", embedding_model="WhereIsAI/UAE-Large-V1")
RAG(faiss=True, embedding_provider="together", embedding_model="togethercomputer/m2-bert-80M-8k-retrieval")
```

---

### `ollama` — local embeddings via Ollama, no key

```python
RAG(faiss=True, embedding_provider="ollama", embedding_model="nomic-embed-text")    # 768 dim
RAG(faiss=True, embedding_provider="ollama", embedding_model="mxbai-embed-large")   # 1024 dim
RAG(faiss=True, embedding_provider="ollama", embedding_model="all-minilm")          # 384 dim, fast
```

Pull before using:
```bash
ollama pull nomic-embed-text
ollama pull mxbai-embed-large
```

---

### Mixing LLM and embedding providers

```python
# OpenAI LLM + Cohere embeddings
RAG(
    model_provider="openai",    model="gpt-4o-mini",
    faiss=True,
    embedding_provider="cohere", embedding_model="embed-multilingual-v3.0",
)

# Groq LLM (fast) + OpenAI embeddings (quality)
RAG(
    model_provider="groq",  model="llama-3.3-70b-versatile",
    faiss=True,
    embedding_provider="openai", embedding_model="text-embedding-3-large",
)

# 100% local — Ollama LLM + Ollama embeddings, zero API cost
RAG(
    model_provider="ollama", model="llama3",
    faiss=True,
    embedding_provider="ollama", embedding_model="nomic-embed-text",
)

# 100% local — Ollama LLM + sentence-transformers embeddings
RAG(
    model_provider="ollama", model="llama3",
    faiss=True,
    embedding_provider="local", embedding_model="BAAI/bge-m3",
)
```

---

## 8. Debug mode

`.debug()` prints: provider, model, loaded docs count, retrieved chunks count, 80-char preview of each chunk.

```python
RAG().debug().add_source("docs/").run("question")
```

Output:
```
[cofone] loaded 2 doc(s) from: docs/
[cofone] model_provider: openrouter | model: arcee-ai/trinity-large-preview:free
[cofone] query: question
[cofone] chunks found: 5
  [0] First chunk preview text here, truncated at 80 chars...
  [1] Second chunk preview...
```

---

## 9. Chat memory

By default every `.run()` is stateless — no context between calls.
Chat memory keeps the full conversation history and injects it into each prompt.

### Using `.chat()`
Automatically enables memory. Best for interactive use.
```python
bot = RAG().add_source("docs/")

r1 = bot.chat("Who is Leonardo da Vinci?")
print(r1)

r2 = bot.chat("When was he born?")    # "he" = Leonardo, context is preserved
print(r2)

r3 = bot.chat("What are his most famous works?")
print(r3)
```

### Using `memory=True` in constructor
Same behavior, useful when using `.run()` in a loop.
```python
bot = RAG(memory=True).add_source("docs/")
bot.run("Who is Leonardo?")
bot.run("What did he invent?")   # has context
```

### Reset memory
Clears the entire history. Next call starts fresh.
```python
bot.reset_memory()
bot.chat("What are we talking about?")   # no context
```

### How memory is injected
The full history is prepended to the prompt:
```
Conversation so far:
User: Who is Leonardo da Vinci?
Assistant: He was a Renaissance polymath...

User: When was he born?
Assistant: He was born on April 15, 1452...

Context:
[retrieved chunks]

User: What are his most famous works?
Assistant:
```

---

## 10. System prompt

The system prompt tells the LLM **who it is and how to behave** — before any context or question.

By default cofone uses:
```
You are a helpful assistant. Answer the user's question using ONLY the information
provided in the context below. If the answer is not in the context, say you don't know.
Be concise, accurate, and respond in the same language as the user's question.
```

Override it with `system_prompt=` to customize tone, role, language, or format.

### Default (no system_prompt)
```python
RAG().add_source("docs/").run("What is this about?")
# → polite, concise, answers only from context, matches user language
```

### Custom role
```python
RAG(
    system_prompt="You are an expert lawyer. Answer formally and cite relevant clauses."
).add_source("contract.pdf").run("What are the termination conditions?")
```

### Force a language
```python
RAG(
    system_prompt="Rispondi sempre in italiano, in modo chiaro e sintetico."
).add_source("docs/").run("What is the main topic?")
# → always responds in Italian regardless of query language
```

### Restrict the scope
```python
RAG(
    system_prompt="You are an art historian. Answer ONLY about paintings and sculptures. "
                  "Refuse any question outside this topic."
).add_source("docs/").run("Tell me about Leonardo")
```

### Control format
```python
RAG(
    system_prompt="Answer always with a bullet-point list. Maximum 5 points. No prose."
).add_source("docs/").run("Summarize this document")
```

### Combined with memory and streaming
```python
bot = RAG(
    system_prompt="You are a friendly tutor. Explain concepts simply, use examples.",
    memory=True,
    max_history=5,
).add_source("docs/")

for token in bot.stream("Explain RAG to me like I'm 10"):
    print(token, end="", flush=True)
print()
```

### How it's injected
The system prompt is always the **first line** of the prompt sent to the LLM:
```
{system_prompt}

Context:
[retrieved chunks]

Question: {query}
Answer:
```

With memory:
```
{system_prompt}

Conversation so far:
User: ...
Assistant: ...

Context:
[retrieved chunks]

User: {query}
Assistant:
```

---

## 11. Streaming

`.stream()` is a generator — yields string tokens one by one as they arrive from the LLM.
No waiting for the full response.

```python
rag = RAG().add_source("docs/")
for token in rag.stream("Tell me about Leonardo's inventions"):
    print(token, end="", flush=True)
print()   # newline at the end
```

Works with all providers that support streaming (OpenRouter, OpenAI, Anthropic, Gemini, Mistral, Groq, Cohere, Ollama, ...).

```python
# streaming with debug (prints chunk info before streaming starts)
rag = RAG().debug().add_source("docs/")
for token in rag.stream("Describe the main topics"):
    print(token, end="", flush=True)
print()
```

---

## 12. Structured output (Pydantic)

Pass a Pydantic `BaseModel` as `schema=` to `.run()`. Instead of a string, you get back a validated Python object.

```python
from pydantic import BaseModel
from cofone import RAG

class Person(BaseModel):
    name: str
    birth_year: int
    nationality: str
    most_famous_work: str

data = RAG().add_source("docs/").run("Extract data about Leonardo da Vinci", schema=Person)

print(data.name)              # Leonardo da Vinci
print(data.birth_year)        # 1452
print(data.nationality)       # Italian
print(data.most_famous_work)  # Mona Lisa
print(type(data))             # <class '__main__.Person'>
```

### Lists and nested types
```python
from typing import List, Optional

class AISummary(BaseModel):
    definition: str
    main_subfields: List[str]
    biggest_challenge: str
    year_founded: Optional[int]

result = (
    RAG()
    .add_source("https://en.wikipedia.org/wiki/Artificial_intelligence")
    .run("Extract key info about AI", schema=AISummary)
)
print(result.model_dump_json(indent=2))
# {
#   "definition": "...",
#   "main_subfields": ["machine learning", "NLP", ...],
#   "biggest_challenge": "...",
#   "year_founded": 1956
# }
```

### How it works internally
1. The Pydantic schema is converted to JSON Schema and appended to the prompt
2. The LLM is asked to respond only with a JSON object
3. `response_format={"type": "json_object"}` is passed where supported
4. The response is parsed and validated with `schema.model_validate()`

---

## 13. Custom tools

Attach Python functions with `.add_tool()`. They are described to the LLM alongside the retrieved context.

```python
def calculate(expression: str) -> str:
    """Evaluate a math expression and return the result."""
    try:
        return f"Result: {eval(expression)}"
    except Exception as e:
        return f"Error: {e}"

def word_count(text: str) -> str:
    """Count the number of words in a text."""
    return f"Word count: {len(text.split())}"

answer = (
    RAG()
    .add_tool(calculate)
    .add_tool(word_count)
    .add_source("docs/")
    .run("Summarize the document and tell me what 144 divided by 12 is")
)
print(answer)
```

Multiple tools:
```python
RAG()
    .add_tool(calculate)
    .add_tool(web_search)
    .add_tool(send_email)
    .add_source("docs/")
    .run("question")
```

---

## 14. API key configuration

Three ways to pass keys, applied in this priority order:

### 1. Direct parameter — highest priority, overrides everything
```python
RAG(model_api_key="sk-or-...").add_source("docs/").run("question")

# embedding key separately
RAG(
    faiss=True,
    embedding_provider="openai",
    embedding_model="text-embedding-3-small",
    embedding_api_key="sk-...",
).add_source("docs/").run("question")
```

### 2. `.env` file — recommended for local development
Create a `.env` file in your project folder:
```
# LLM providers
OPENROUTER_API_KEY=sk-or-...
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=AI...
MISTRAL_API_KEY=...
GROQ_API_KEY=gsk_...
COHERE_API_KEY=...
DEEPSEEK_API_KEY=...
XAI_API_KEY=xai-...
TOGETHER_API_KEY=...
FIREWORKS_API_KEY=...
PERPLEXITY_API_KEY=pplx-...
CEREBRAS_API_KEY=...
NVIDIA_API_KEY=nvapi-...

# Embedding-only providers
VOYAGE_API_KEY=...
JINA_API_KEY=...
```

Load it at the top of every script:
```python
from dotenv import load_dotenv
load_dotenv()
```

### 3. System environment variable
```bash
# PowerShell (Windows)
$env:OPENROUTER_API_KEY="sk-or-..."

# bash / zsh (Linux / Mac)
export OPENROUTER_API_KEY="sk-or-..."
```

---

## 15. Full parameter reference

### `RAG()` constructor

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model_provider` | `str` | `"openrouter"` | LLM provider — see section 6 for full list |
| `model` | `str` | provider default | LLM model identifier |
| `model_api_key` | `str` | `None` | LLM API key — overrides `.env` |
| `base_url` | `str` | provider default | Custom LLM endpoint (any OpenAI-compatible API) |
| `faiss` | `bool` | `False` | Use FAISS semantic search instead of BM25 |
| `embedding_provider` | `str` | `"local"` | Embedding provider — see section 7 for full list |
| `embedding_model` | `str` | `"all-MiniLM-L6-v2"` | Embedding model identifier |
| `embedding_api_key` | `str` | `None` | Embedding API key — overrides `.env` |
| `chunk_mode` | `str` | `"smart"` | `"smart"` / `"paragraphs"` / `"sentences"` / `"fixed"` |
| `persist_path` | `str\|Path` | `None` | Folder path to save/load FAISS index |
| `system_prompt` | `str` | `None` | Custom system prompt. `None` uses built-in default |
| `memory` | `bool` | `False` | Enable conversation history across calls |

### Methods

| Method | Returns | Description |
|---|---|---|
| `.add_source(path_or_url)` | `self` | Load file, folder, URL, Wikipedia, YouTube |
| `.add_tool(fn)` | `self` | Attach a custom Python function |
| `.debug()` | `self` | Enable verbose logging |
| `.run(query, schema=None)` | `str` or Pydantic model | Single stateless query |
| `.chat(query)` | `str` | Stateful query with memory |
| `.stream(query)` | `Generator[str]` | Streaming query — yields tokens one by one |
| `.reset_memory()` | `self` | Clear conversation history |