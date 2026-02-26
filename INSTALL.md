# Cofone — Installation Guide

## Requirements

- Python >= 3.9
- pip

---

## 1. Install from PyPI

```bash
pip install cofone
```

With optional extras:
```bash
pip install "cofone[pdf]"      # + PDF reading (pypdf)
pip install "cofone[faiss]"    # + FAISS semantic search (faiss-cpu + sentence-transformers)
pip install "cofone[web]"      # + Wikipedia and YouTube
pip install "cofone[all]"      # everything above
```

---

## 2. API key setup

Cofone requires at least one LLM provider API key.  
The default provider is **OpenRouter** — free tier, 200+ models, one key.

### Get a free OpenRouter key
1. Go to [openrouter.ai/keys](https://openrouter.ai/keys)
2. Sign up (free)
3. Create an API key (starts with `sk-or-...`)

### Configure the key

**Option A — `.env` file (recommended)**

Create a `.env` file in your **project folder** (not inside the `cofone` package):
```
OPENROUTER_API_KEY=sk-or-...
```

Load it at the top of every script:
```python
from dotenv import load_dotenv
load_dotenv()
```

**Option B — pass it directly to RAG()**
```python
from cofone import RAG
RAG(model_api_key="sk-or-...").add_source("docs/").run("question")
```

**Option C — system environment variable**
```bash
# PowerShell (Windows)
$env:OPENROUTER_API_KEY="sk-or-..."

# bash / zsh (Linux / Mac)
export OPENROUTER_API_KEY="sk-or-..."
```

---

## 3. All provider keys

Only add keys for the providers you plan to use.

```
# ── LLM providers ─────────────────────────────────────────────────────────────
OPENROUTER_API_KEY=sk-or-...       # openrouter  (default, free tier available)
OPENAI_API_KEY=sk-...              # openai
ANTHROPIC_API_KEY=sk-ant-...       # anthropic
GEMINI_API_KEY=AI...               # gemini
MISTRAL_API_KEY=...                # mistral
GROQ_API_KEY=gsk_...               # groq
COHERE_API_KEY=...                 # cohere
DEEPSEEK_API_KEY=...               # deepseek
XAI_API_KEY=xai-...                # xai (Grok)
FIREWORKS_API_KEY=...              # fireworks
TOGETHER_API_KEY=...               # together
PERPLEXITY_API_KEY=pplx-...        # perplexity
NVIDIA_API_KEY=nvapi-...           # nvidia
CEREBRAS_API_KEY=...               # cerebras
DEEPINFRA_API_KEY=...              # deepinfra
ANYSCALE_API_KEY=...               # anyscale

# ── Embedding providers (only needed when faiss=True + API embeddings) ─────────
VOYAGE_API_KEY=...                 # embedding_provider="voyage"
JINA_API_KEY=...                   # embedding_provider="jina"
# OpenAI/Gemini/Cohere/Mistral embeddings reuse the same key as above
```

**Get your keys:**

| Provider | Link | Free tier? |
|---|---|---|
| OpenRouter | [openrouter.ai/keys](https://openrouter.ai/keys) | ✅ Yes |
| OpenAI | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) | ❌ Paid |
| Anthropic | [console.anthropic.com](https://console.anthropic.com) | ❌ Paid |
| Gemini | [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) | ✅ Yes |
| Mistral | [console.mistral.ai](https://console.mistral.ai) | ✅ Trial |
| Groq | [console.groq.com](https://console.groq.com) | ✅ Yes |
| Cohere | [dashboard.cohere.com](https://dashboard.cohere.com) | ✅ Trial |
| DeepSeek | [platform.deepseek.com](https://platform.deepseek.com) | ✅ Trial |
| xAI | [console.x.ai](https://console.x.ai) | ❌ Paid |
| Voyage | [dash.voyageai.com](https://dash.voyageai.com) | ✅ Trial |
| Jina | [jina.ai](https://jina.ai) | ✅ Free tier |
| Ollama | [ollama.ai](https://ollama.ai) | ✅ Free (local) |

---

## 4. First script

```python
from dotenv import load_dotenv
from cofone import RAG

load_dotenv()  # reads your .env file

answer = RAG().add_source("my_document.txt").run("Summarize this document")
print(answer)
```

---

## 5. Ollama — fully local, no key, no internet

Ollama runs LLMs and embedding models entirely on your machine.

1. Install from [ollama.ai](https://ollama.ai)
2. Pull models:
```bash
ollama pull llama3               # LLM
ollama pull mistral              # alternative LLM
ollama pull phi3                 # small/fast LLM
ollama pull nomic-embed-text     # embedding model (for faiss=True)
ollama pull mxbai-embed-large    # higher quality embeddings
```
3. Make sure Ollama is running (it auto-starts on most systems)
4. Use it:

```python
from cofone import RAG

# local LLM only (BM25 retrieval)
RAG(model_provider="ollama", model="llama3").add_source("docs/").run("Summarize")

# fully offline: local LLM + local embeddings
RAG(
    model_provider="ollama",     model="llama3",
    faiss=True,
    embedding_provider="ollama", embedding_model="nomic-embed-text",
).add_source("docs/").run("Summarize")
```

---

## 6. Optional dependencies reference

| Package | Purpose | How to install |
|---|---|---|
| `faiss-cpu` | FAISS vector index | `pip install "cofone[faiss]"` |
| `sentence-transformers` | Local embeddings (for FAISS) | `pip install "cofone[faiss]"` |
| `pypdf` | Read PDF files | `pip install "cofone[pdf]"` |
| `wikipedia` | Fetch Wikipedia articles | `pip install "cofone[web]"` |
| `youtube-transcript-api` | Fetch YouTube transcripts | `pip install "cofone[web]"` |
| `cohere` | Cohere Embed API | `pip install cohere` |
| `voyageai` | Voyage AI embeddings | `pip install voyageai` |

---

## Troubleshooting

**`[cofone] API key not found for provider 'openrouter'`**  
Create a `.env` file containing `OPENROUTER_API_KEY=sk-or-...` and add `load_dotenv()` at the top of your script.

**`ModuleNotFoundError: No module named 'faiss'`**  
Run `pip install "cofone[faiss]"`.

**`ModuleNotFoundError: No module named 'pypdf'`**  
Run `pip install "cofone[pdf]"`.

**`faiss-cpu` install fails on Windows**  
Try `pip install faiss-cpu --no-cache-dir`. If it still fails, install from conda: `conda install -c conda-forge faiss-cpu`.

**Ollama: connection refused**  
Make sure the Ollama app is running. On Windows/Mac it starts as a background app after launch. Check `http://localhost:11434` in your browser.

**YouTube: transcript not found**  
The video must have subtitles or auto-generated captions enabled. Private videos are not supported.

**PDF returns no text**  
The PDF may be image-based (scanned). Cofone uses text extraction only — OCR is not included.

**Wikipedia: page not found / disambiguation error**  
Use the exact Wikipedia URL including the full article slug (e.g. `https://en.wikipedia.org/wiki/Python_(programming_language)`).