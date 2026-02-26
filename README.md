# Cofone

**Simple, fast, yours.** Turn documents, websites and videos into a queryable knowledge base in a few lines of Python.

```python
from dotenv import load_dotenv
from cofone import RAG

load_dotenv()  # reads OPENROUTER_API_KEY from .env

answer = RAG().add_source("docs/").run("Who is Leonardo?")
print(answer)
```

---

## What is Cofone?

Cofone is an open-source Python RAG (Retrieval-Augmented Generation) library.  
Load any document, ask questions in natural language, get precise answers — without complex setup or boilerplate.

**Key highlights:**
- Fluent DSL — chain everything in one expression
- BM25 (default) + FAISS semantic search
- **19 LLM providers:** OpenRouter, OpenAI, Anthropic, Gemini, Mistral, Groq, Cohere, DeepSeek, xAI, Together, Perplexity, Fireworks, Cerebras, NVIDIA, DeepInfra, Anyscale, Ollama, LM Studio, llama.cpp
- **10 embedding providers:** local sentence-transformers, OpenAI, Gemini, Cohere, Mistral, Voyage, Jina, NVIDIA, Together, Ollama
- Smart chunking that respects document structure
- Chat memory, streaming, structured output (Pydantic)
- FAISS index persistence to disk
- Sources: files, folders, PDFs, web URLs, Wikipedia, YouTube

---

## Installation

```bash
pip install cofone
```

With optional extras:
```bash
pip install "cofone[pdf]"      # PDF support (pypdf)
pip install "cofone[faiss]"    # FAISS semantic search (faiss-cpu + sentence-transformers)
pip install "cofone[web]"      # Wikipedia + YouTube
pip install "cofone[all]"      # everything above
```

---

## Setup — API key required

Cofone needs at least one LLM provider API key.  
The default provider is **OpenRouter** — free tier available, 200+ models, one key.

**Step 1:** Get a free key at [openrouter.ai/keys](https://openrouter.ai/keys)

**Step 2:** Create a `.env` file in your project folder:
```
OPENROUTER_API_KEY=sk-or-...
```

**Step 3:** Load it in your script:
```python
from dotenv import load_dotenv
load_dotenv()
```

Or pass the key directly (no `.env` needed):
```python
RAG(model_api_key="sk-or-...").add_source("docs/").run("question")
```

→ Full setup guide for all providers: [INSTALL.md](INSTALL.md)

---

## Examples

```python
from dotenv import load_dotenv
from cofone import RAG
load_dotenv()

# ── Sources ───────────────────────────────────────────────────────────────────

# single file
RAG().add_source("notes.txt").run("Summarize")

# folder — loads all .txt .md .pdf recursively
RAG().add_source("docs/").run("What is the main topic?")

# PDF (requires pip install "cofone[pdf]")
RAG().add_source("report.pdf").run("What are the conclusions?")

# Wikipedia
RAG().add_source("https://en.wikipedia.org/wiki/Python").run("What is Python?")

# YouTube transcript
RAG().add_source("https://www.youtube.com/watch?v=VIDEO_ID").run("Summarize this video")

# multiple sources combined
RAG().add_source("docs/").add_source("https://en.wikipedia.org/wiki/AI").run("Overview")

# ── LLM providers ─────────────────────────────────────────────────────────────

RAG(model_provider="openai",     model="gpt-4o-mini").add_source("docs/").run("question")
RAG(model_provider="anthropic",  model="claude-3-5-haiku-20241022").add_source("docs/").run("question")
RAG(model_provider="gemini",     model="gemini-2.0-flash").add_source("docs/").run("question")
RAG(model_provider="groq",       model="llama-3.1-8b-instant").add_source("docs/").run("question")
RAG(model_provider="ollama",     model="llama3").add_source("docs/").run("question")  # local, no key

# ── FAISS semantic search ─────────────────────────────────────────────────────

# local embeddings (no extra key)
RAG(faiss=True).add_source("docs/").run("Find concepts related to learning")

# OpenAI embeddings
RAG(faiss=True,
    embedding_provider="openai",
    embedding_model="text-embedding-3-small"
).add_source("docs/").run("question")

# fully local — Ollama LLM + Ollama embeddings, no internet, no keys
RAG(model_provider="ollama",    model="llama3",
    faiss=True,
    embedding_provider="ollama", embedding_model="nomic-embed-text"
).add_source("docs/").run("question")

# ── Chat memory ───────────────────────────────────────────────────────────────

bot = RAG().add_source("docs/")
bot.chat("Who is Leonardo da Vinci?")
bot.chat("When was he born?")          # knows the context — "he" = Leonardo
bot.chat("What are his best works?")

# ── Streaming ─────────────────────────────────────────────────────────────────

for token in RAG().add_source("docs/").stream("Tell me about this document"):
    print(token, end="", flush=True)
print()

# ── Structured output (Pydantic) ──────────────────────────────────────────────

from pydantic import BaseModel

class Person(BaseModel):
    name: str
    birth_year: int
    nationality: str

data = RAG().add_source("docs/").run("Extract data about Leonardo", schema=Person)
print(data.name)        # Leonardo da Vinci
print(data.birth_year)  # 1452
```

---

## LLM Providers (19 total)

| Provider | `model_provider=` | Key env var | Notes |
|---|---|---|---|
| OpenRouter | `"openrouter"` | `OPENROUTER_API_KEY` | **Default.** 200+ models, free tier |
| OpenAI | `"openai"` | `OPENAI_API_KEY` | GPT-4o, o3, etc. |
| Anthropic | `"anthropic"` | `ANTHROPIC_API_KEY` | Claude 3.5, Claude 3 |
| Gemini | `"gemini"` | `GEMINI_API_KEY` | Gemini 2.0 Flash, 1.5 Pro |
| Mistral | `"mistral"` | `MISTRAL_API_KEY` | Mistral Large, Codestral |
| Groq | `"groq"` | `GROQ_API_KEY` | Very fast inference |
| Cohere | `"cohere"` | `COHERE_API_KEY` | Command R+ |
| DeepSeek | `"deepseek"` | `DEEPSEEK_API_KEY` | DeepSeek-R1 reasoning |
| xAI | `"xai"` | `XAI_API_KEY` | Grok |
| Together | `"together"` | `TOGETHER_API_KEY` | Many open models |
| Perplexity | `"perplexity"` | `PERPLEXITY_API_KEY` | Web-connected |
| Fireworks | `"fireworks"` | `FIREWORKS_API_KEY` | Fast open models |
| Cerebras | `"cerebras"` | `CEREBRAS_API_KEY` | Ultra-fast |
| NVIDIA | `"nvidia"` | `NVIDIA_API_KEY` | NIM platform |
| DeepInfra | `"deepinfra"` | `DEEPINFRA_API_KEY` | Cheap open models |
| Anyscale | `"anyscale"` | `ANYSCALE_API_KEY` | Scalable inference |
| Ollama | `"ollama"` | none | **Local**, no internet |
| LM Studio | `"lmstudio"` | none | **Local**, no internet |
| llama.cpp | `"llamacpp"` | none | **Local**, no internet |

## Embedding Providers (10 total)

| Provider | `embedding_provider=` | Key env var | Notes |
|---|---|---|---|
| sentence-transformers | `"local"` | none | **Default.** Fully offline |
| OpenAI | `"openai"` | `OPENAI_API_KEY` | text-embedding-3-small/large |
| Gemini | `"gemini"` | `GEMINI_API_KEY` | text-embedding-004 |
| Cohere | `"cohere"` | `COHERE_API_KEY` | Multilingual, `pip install cohere` |
| Mistral | `"mistral"` | `MISTRAL_API_KEY` | mistral-embed |
| Voyage | `"voyage"` | `VOYAGE_API_KEY` | Top retrieval quality, `pip install voyageai` |
| Jina | `"jina"` | `JINA_API_KEY` | jina-embeddings-v3 |
| NVIDIA | `"nvidia"` | `NVIDIA_API_KEY` | nv-embed-v2 |
| Together | `"together"` | `TOGETHER_API_KEY` | BGE, UAE models |
| Ollama | `"ollama"` | none | **Local**, nomic-embed-text |

---

## Links

- 📦 PyPI: [pypi.org/project/cofone](https://pypi.org/project/cofone)
- 💻 GitHub: [github.com/LeonardoCofone/cofone](https://github.com/LeonardoCofone/cofone)
- 📖 Full feature reference: [FEATURE.md](FEATURE.md)
- 🔧 Installation guide: [INSTALL.md](INSTALL.md)

## License

MIT — see [LICENSE](LICENSE)