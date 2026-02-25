# Cofone — Installation Guide

## Requirements

- Python >= 3.9
- pip

---

## 1. Get the code

```bash
git clone https://github.com/yourname/cofone
cd cofone
```

---

## 2. Install

### Minimal (OpenRouter / OpenAI / Gemini / Ollama only)
```bash
pip install -e .
```

### With PDF support
```bash
pip install -e ".[pdf]"
```

### With FAISS semantic search
```bash
pip install -e ".[faiss]"
```

### With web sources (Wikipedia + YouTube)
```bash
pip install -e ".[web]"
```

### Everything
```bash
pip install -e ".[all]"
```

Or install all dependencies manually:
```bash
pip install -r requirements.txt
```

---

## 3. API keys

Create a `.env` file in the **project root** (same level as `setup.py`):

```
OPENROUTER_API_KEY=sk-or-...
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=AI...
```

You only need the keys for the providers you plan to use.

**Get your keys:**
- OpenRouter (free tier available): [openrouter.ai/keys](https://openrouter.ai/keys)
- OpenAI: [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
- Gemini: [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
- Ollama (no key): [ollama.ai](https://ollama.ai)

---

## 4. Verify installation

```bash
cd test
python test.py
```

You should see 13 tests running. Tests that require unavailable services (e.g. Ollama) are skipped automatically.

---

## 5. Ollama (local inference, no API key)

1. Download and install Ollama from [ollama.ai](https://ollama.ai)
2. Pull a model:
```bash
ollama pull llama3
# or
ollama pull mistral
ollama pull phi3
```
3. Make sure Ollama is running (it starts automatically on most systems)
4. Use it:
```python
RAG(provider="ollama", model="llama3").add_source("docs/").run("Summarize")
```

---

## 6. Optional dependencies

| Package | Purpose | Install |
|---|---|---|
| `faiss-cpu` | Semantic vector search | `pip install faiss-cpu` |
| `sentence-transformers` | Text embeddings for FAISS | `pip install sentence-transformers` |
| `pypdf` | Read PDF files | `pip install pypdf` |
| `wikipedia` | Fetch Wikipedia pages | `pip install wikipedia` |
| `youtube-transcript-api` | Fetch YouTube transcripts | `pip install youtube-transcript-api` |
| `pydantic` | Structured output validation | `pip install pydantic` |
| `httpx` + `beautifulsoup4` | Fetch web URLs | `pip install httpx beautifulsoup4` |

---

## 7. Project structure after install

```
cofone/
├── cofone/               ← the library
├── test/                 ← demo files and tests
├── .env                  ← your keys (never commit this)
├── cofone.egg-info/      ← created by pip install -e .
└── ...
```

The `cofone.egg-info/` folder is auto-generated — do not edit or commit it.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'cofone'`**  
Run `pip install -e .` from the project root (where `setup.py` is).

**`API key not found`**  
Make sure `.env` is in the project root and contains the right key name.

**`faiss-cpu` install fails on Windows**  
Try: `pip install faiss-cpu --no-cache-dir`

**Ollama connection refused**  
Make sure the Ollama app is running. On Windows/Mac it runs as a background process after launch.