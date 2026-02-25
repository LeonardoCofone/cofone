# Cofone

**Simple, fast, yours.** Turn documents, websites and videos into a queryable knowledge base in a few lines of Python.

```python
from cofone import RAG

answer = RAG().add_source("docs/").run("Who is Leonardo?")
print(answer)
```

---

## What is Cofone?

Cofone is an open-source Python RAG (Retrieval-Augmented Generation) library. Load any document, ask questions in natural language, get precise answers — without complex setup or boilerplate.

**Key highlights:**
- Fluent DSL — chain everything in one expression
- BM25 + FAISS semantic search
- 4 LLM providers: OpenRouter, OpenAI, Gemini, Ollama
- Smart chunking that respects document structure
- Chat memory, streaming, structured output (Pydantic)
- Sources: files, folders, PDFs, web URLs, Wikipedia, YouTube

---

## Quick start

```bash
# 1. clone and install
git clone https://github.com/yourname/cofone
cd cofone
pip install -e ".[all]"

# 2. create .env in project root
echo "OPENROUTER_API_KEY=sk-or-..." > .env

# 3. run
python -c "
from cofone import RAG
print(RAG().add_source('test/note_ex.txt').run('Summarize'))
"
```

For full installation instructions → see [INSTALL.md](INSTALL.md)  
For all features with examples → see [FEATURES.md](FEATURES.md)

---

## Minimal examples

```python
from cofone import RAG

# file
RAG().add_source("notes.txt").run("Summarize")

# folder
RAG().add_source("docs/").run("What is the main topic?")

# web
RAG().add_source("https://en.wikipedia.org/wiki/Python").run("What is Python?")

# YouTube
RAG().add_source("https://www.youtube.com/watch?v=VIDEO_ID").run("What is this about?")

# FAISS semantic search
RAG(faiss=True).add_source("docs/").run("Find concepts related to learning")

# chat memory
bot = RAG().add_source("docs/")
bot.chat("Who is Leonardo?")
bot.chat("When was he born?")  # knows the context

# streaming
for token in RAG().add_source("docs/").stream("Tell me a story"):
    print(token, end="", flush=True)

# structured output
from pydantic import BaseModel
class Person(BaseModel):
    name: str
    birth_year: int
RAG().add_source("docs/").run("Extract data", schema=Person)
```

---

## Providers

| Provider | Default model | Key |
|---|---|---|
| OpenRouter | `arcee-ai/trinity-large-preview:free` | `OPENROUTER_API_KEY` |
| OpenAI | `gpt-4o-mini` | `OPENAI_API_KEY` |
| Gemini | `gemini-2.0-flash` | `GEMINI_API_KEY` |
| Ollama | `llama3` | none (local) |

Provider is auto-detected from model name — `gpt-*` → openai, `gemini-*` → gemini.

---

## License

MIT