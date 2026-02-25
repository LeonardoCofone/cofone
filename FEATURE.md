# Cofone — Features Reference

Complete guide to every feature in the library.

---

## Table of contents

1. [Basic usage](#1-basic-usage)
2. [Sources](#2-sources)
3. [Chunking](#3-chunking)
4. [Retrieval — BM25 vs FAISS](#4-retrieval--bm25-vs-faiss)
5. [FAISS persistence](#5-faiss-persistence)
6. [Providers & models](#6-providers--models)
7. [Debug mode](#7-debug-mode)
8. [Chat memory](#8-chat-memory)
9. [Streaming](#9-streaming)
10. [Structured output](#10-structured-output)
11. [Custom tools](#11-custom-tools)
12. [API key configuration](#12-api-key-configuration)
13. [Full parameter reference](#13-full-parameter-reference)

---

## 1. Basic usage

Load a source and ask a question. Everything chains fluently.

```python
from cofone import RAG

answer = RAG().add_source("docs/").run("Who is Leonardo?")
print(answer)
```

---

## 2. Sources

All sources are loaded via `.add_source()`. You can chain as many as you want.

### Text file
```python
RAG().add_source("notes.txt").run("Summarize")
```

### Folder (recursive)
Loads all `.txt`, `.md`, and `.pdf` files found recursively.
```python
RAG().add_source("documents/").run("What is the main topic?")
```

### PDF
Requires `pip install pypdf`.
```python
RAG().add_source("report.pdf").run("What are the key findings?")
```

### Web URL
Fetches and extracts the visible text from any public webpage.
```python
RAG().add_source("https://example.com/article").run("Summarize this article")
```

### Wikipedia
Automatically detected when the URL contains `wikipedia.org`. Uses the `wikipedia` package for clean content extraction — no HTML noise.
```python
RAG().add_source("https://en.wikipedia.org/wiki/Python_(programming_language)").run("What is Python?")
RAG().add_source("https://it.wikipedia.org/wiki/Leonardo_da_Vinci").run("Chi è Leonardo?")
```

### YouTube
Fetches the video transcript (subtitles must be available). Tries Italian first, falls back to English.
```python
RAG().add_source("https://www.youtube.com/watch?v=VIDEO_ID").run("What is this video about?")
```

### Multiple sources
```python
RAG()
  .add_source("notes.txt")
  .add_source("docs/")
  .add_source("https://en.wikipedia.org/wiki/Machine_learning")
  .run("Give me a complete overview")
```

---

## 3. Chunking

Chunking is how Cofone splits documents before indexing. The right mode improves retrieval quality significantly.

### smart (default)
Splits by paragraphs first. If a paragraph is longer than 600 characters, it is further split into sentences. Best for structured articles, notes, and documentation.
```python
RAG(chunk_mode="smart").add_source("docs/").run("question")
```

### paragraphs
Splits only on blank lines. Keeps entire paragraphs together. Best when paragraphs are short and self-contained.
```python
RAG(chunk_mode="paragraphs").add_source("docs/").run("question")
```

### sentences
Splits on sentence boundaries (`.`, `!`, `?`). Groups sentences up to ~500 characters. Best for dense, information-heavy text.
```python
RAG(chunk_mode="sentences").add_source("docs/").run("question")
```

### fixed
Splits into fixed-length slices of 500 characters with 50-character overlap. Best for raw data exports or unstructured text.
```python
RAG(chunk_mode="fixed").add_source("docs/").run("question")
```

---

## 4. Retrieval — BM25 vs FAISS

### BM25 (default)
Keyword-based ranking. Fast, no extra dependencies. Works well when query words appear in the document.
```python
RAG().add_source("docs/").run("Who invented the telephone?")
```

### FAISS
Semantic vector search using sentence-transformers embeddings. Finds conceptually related chunks even when exact words don't match. Requires `faiss-cpu` and `sentence-transformers`.
```python
RAG(faiss=True).add_source("docs/").run("Who invented the telephone?")
```

### Custom embedding model
Default embedding model is `all-MiniLM-L6-v2` (English, fast, ~80MB). For multilingual documents, use a multilingual model.
```python
RAG(faiss=True, embedding_model="paraphrase-multilingual-MiniLM-L12-v2").add_source("docs/").run("question")
```

---

## 5. FAISS persistence

By default, FAISS rebuilds the index every run — slow for large document sets. Use `persist_path` to save the index to disk and reload it instantly on subsequent runs.

```python
# First run: computes embeddings and saves to disk
RAG(faiss=True, persist_path="./my_db").add_source("big_manual.pdf").run("question")

# Second run: loads from disk instantly, no recompute
RAG(faiss=True, persist_path="./my_db").add_source("big_manual.pdf").run("another question")
```

The `persist_path` folder contains two files:
- `index.faiss` — the vector index
- `chunks.json` — the original text chunks

To rebuild the index from scratch, delete the folder.

---

## 6. Providers & models

Cofone supports four LLM providers through the same interface.

### OpenRouter (default)
Access to 200+ models with a single API key. Free tier available.
```python
RAG(model="arcee-ai/trinity-large-preview:free")
RAG(model="meta-llama/llama-3.3-70b-instruct:free")
RAG(model="google/gemini-2.0-flash-exp:free")
RAG(model="openai/gpt-4o")
RAG(model="anthropic/claude-3-5-sonnet")
```
Full model list: [openrouter.ai/models](https://openrouter.ai/models)

### OpenAI
```python
RAG(provider="openai", model="gpt-4o-mini")
RAG(provider="openai", model="gpt-4o")
RAG(model="gpt-4o-mini")  # auto-detected
```
Requires `OPENAI_API_KEY` in `.env`.

### Gemini
```python
RAG(provider="gemini", model="gemini-2.0-flash")
RAG(provider="gemini", model="gemini-1.5-pro")
RAG(model="gemini-2.0-flash")  # auto-detected
```
Requires `GEMINI_API_KEY` in `.env`.

### Ollama (local)
Runs entirely on your machine. No API key, no internet required.
```python
RAG(provider="ollama", model="llama3")
RAG(provider="ollama", model="mistral")
RAG(provider="ollama", model="phi3")
```
Requires Ollama running on `localhost:11434`. Install at [ollama.ai](https://ollama.ai).

### Auto-detection
Provider is automatically inferred from the model name:
- `gpt-*`, `o1-*`, `o3-*` → OpenAI
- `gemini-*` → Gemini
- `name/name` (slash) → OpenRouter
- everything else → OpenRouter

### Custom endpoint
```python
RAG(base_url="https://my-custom-endpoint/v1", api_key="my-key", model="my-model")
```

---

## 7. Debug mode

`.debug()` enables verbose logging: shows provider, model, loaded documents, retrieved chunk count, and a text preview of each chunk.

```python
RAG().debug().add_source("docs/").run("question")
```

Output:
```
[cofone] loaded 2 doc(s) from: docs/
[cofone] provider: openrouter | model: arcee-ai/trinity-large-preview:free
[cofone] query: question
[cofone] chunks found: 5
  [0] First chunk preview...
  [1] Second chunk preview...
```

---

## 8. Chat memory

By default, every `.run()` call is stateless. Chat memory keeps the conversation history so follow-up questions have full context.

### Using `.chat()`
Automatically enables memory.
```python
bot = RAG().add_source("docs/")
bot.chat("Who is Leonardo da Vinci?")
bot.chat("When was he born?")          # knows "he" = Leonardo
bot.chat("What are his best works?")   # still has full context
```

### Using `memory=True`
```python
bot = RAG(memory=True).add_source("docs/")
bot.run("Who is Leonardo?")
bot.run("What did he invent?")
```

### Reset memory
```python
bot.reset_memory()
bot.chat("What are we talking about?")  # no context anymore
```

---

## 9. Streaming

`.stream()` returns a generator that yields tokens as they arrive from the LLM. No waiting for the full response.

```python
rag = RAG().add_source("docs/")
for token in rag.stream("Tell me about Leonardo's inventions"):
    print(token, end="", flush=True)
print()
```

Works with all providers that support streaming (OpenRouter, OpenAI, Gemini, Ollama).

---

## 10. Structured output

Pass a Pydantic model as `schema` to `.run()` and get back a validated Python object instead of a string.

```python
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    birth_year: int
    nationality: str
    most_famous_work: str

data = RAG().add_source("docs/").run("Extract data about Leonardo", schema=Person)

print(data.name)            # Leonardo da Vinci
print(data.birth_year)      # 1452
print(data.nationality)     # Italian
print(data.most_famous_work)  # Mona Lisa
```

### Complex schemas
```python
from typing import List

class AISummary(BaseModel):
    definition: str
    main_subfields: List[str]
    biggest_challenge: str

result = RAG().add_source("https://en.wikipedia.org/wiki/Artificial_intelligence") \
    .run("Extract key info about AI", schema=AISummary)

print(result.model_dump_json(indent=2))
```

---

## 11. Custom tools

Attach Python functions with `.add_tool()`. They are passed to the LLM as context alongside the retrieved chunks.

```python
def calculate(expression: str) -> str:
    try:
        return f"Result: {eval(expression)}"
    except Exception as e:
        return f"Error: {e}"

RAG().add_tool(calculate).add_source("docs/").run("Summarize and tell me what is 144 / 12")
```

Multiple tools:
```python
RAG()
  .add_tool(calculate)
  .add_tool(my_search_function)
  .add_source("docs/")
  .run("question")
```

---

## 12. API key configuration

Three ways to pass keys, in order of priority:

### 1. Direct parameter (highest priority)
```python
RAG(api_key="sk-or-...").add_source("docs/").run("question")
```

### 2. `.env` file in project root
```
OPENROUTER_API_KEY=sk-or-...
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=AI...
```
Load it in your script:
```python
from dotenv import load_dotenv
load_dotenv()
```

### 3. System environment variable
```bash
# PowerShell
$env:OPENROUTER_API_KEY="sk-or-..."

# bash / zsh
export OPENROUTER_API_KEY="sk-or-..."
```

---

## 13. Full parameter reference

### `RAG()` constructor

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `str` | provider default | LLM model identifier |
| `provider` | `str` | `"openrouter"` | `"openrouter"` / `"openai"` / `"gemini"` / `"ollama"` |
| `api_key` | `str` | `None` | API key — overrides `.env` |
| `base_url` | `str` | provider default | Custom API endpoint URL |
| `faiss` | `bool` | `False` | Use FAISS semantic search instead of BM25 |
| `embedding_model` | `str` | `"all-MiniLM-L6-v2"` | sentence-transformers model for FAISS |
| `chunk_mode` | `str` | `"smart"` | `"smart"` / `"paragraphs"` / `"sentences"` / `"fixed"` |
| `persist_path` | `str\|Path` | `None` | Folder to save/load FAISS index |
| `memory` | `bool` | `False` | Enable conversation history across calls |

### Methods

| Method | Returns | Description |
|---|---|---|
| `.add_source(path_or_url)` | `self` | Load a file, folder, URL, Wikipedia page, or YouTube video |
| `.add_tool(fn)` | `self` | Attach a custom Python function |
| `.debug()` | `self` | Enable verbose logging |
| `.run(query, schema=None)` | `str` or Pydantic model | Single stateless query |
| `.chat(query)` | `str` | Query with memory enabled (stateful) |
| `.stream(query)` | `Generator[str]` | Streaming query — yields tokens one by one |
| `.reset_memory()` | `self` | Clear conversation history |