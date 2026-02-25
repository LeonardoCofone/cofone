from pathlib import Path
from dotenv import load_dotenv
from cofone import RAG

load_dotenv(Path(__file__).parent.parent / ".env")
BASE = Path(__file__).parent

def section(title):
    print("\n" + "=" * 55)
    print(f"  {title}")
    print("=" * 55)


# ── TEST 1 · BM25 + smart chunking ────────────────────────
section("TEST 1 · BM25 + smart chunking")
print(RAG(model="arcee-ai/trinity-large-preview:free", chunk_mode="smart")
    .debug()
    .add_source(BASE / "note_ex.txt")
    .run("Summarize"))


# ── TEST 2 · FAISS + paragraph chunking ───────────────────
section("TEST 2 · FAISS + paragraph chunking")
print(RAG(model="arcee-ai/trinity-large-preview:free", faiss=True, chunk_mode="paragraphs")
    .debug()
    .add_source(BASE / "docs_ex/")
    .run("Who is Leonardo?"))


# ── TEST 3 · Chunking modes comparison ────────────────────
section("TEST 3 · Chunking modes comparison")
for mode in ["smart", "paragraphs", "sentences", "fixed"]:
    rag = RAG(chunk_mode=mode).add_source(BASE / "note_ex.txt")
    print(f"  [{mode:12s}] → {len(rag._retriever.chunks)} chunk(s)")


# ── TEST 4 · Wikipedia URL ─────────────────────────────────
section("TEST 4 · Wikipedia URL")
print(RAG(model="arcee-ai/trinity-large-preview:free")
    .debug()
    .add_source("https://en.wikipedia.org/wiki/Artificial_intelligence")
    .run("What is artificial intelligence?"))


# ── TEST 5 · YouTube transcript ───────────────────────────
section("TEST 5 · YouTube transcript")
print(RAG(model="arcee-ai/trinity-large-preview:free")
    .debug()
    .add_source("https://www.youtube.com/watch?v=jNQXAC9IVRw")
    .run("What is this video about?"))


# ── TEST 6 · Multi-source ─────────────────────────────────
section("TEST 6 · Multi-source (file + folder)")
print(RAG(model="arcee-ai/trinity-large-preview:free")
    .add_source(BASE / "note_ex.txt")
    .add_source(BASE / "docs_ex/")
    .run("What do you know about Cofone and Leonardo?"))


# ── TEST 7 · Custom tools ─────────────────────────────────
section("TEST 7 · Custom tools")

def calculate(expression: str) -> str:
    try:
        return f"Result: {eval(expression)}"
    except Exception as e:
        return f"Error: {e}"

print(RAG(model="arcee-ai/trinity-large-preview:free")
    .add_tool(calculate)
    .add_source(BASE / "note_ex.txt")
    .run("Summarize and tell me what is 144 / 12"))


# ── TEST 8 · FAISS with persistence ───────────────────────
section("TEST 8 · FAISS persistence (cache_dir)")
db_path = BASE / ".cofone_cache"

print("  First run (builds index):")
RAG(model="arcee-ai/trinity-large-preview:free", faiss=True, persist_path=db_path) \
    .add_source(BASE / "docs_ex/") \
    .run("Who is Leonardo?")

print("\n  Second run (loads from disk, no embedding recomputed):")
answer = RAG(model="arcee-ai/trinity-large-preview:free", faiss=True, persist_path=db_path) \
    .add_source(BASE / "docs_ex/") \
    .run("What did Leonardo invent?")
print(answer)


# ── TEST 9 · Chat memory ──────────────────────────────────
section("TEST 9 · Chat memory")
bot = RAG(model="arcee-ai/trinity-large-preview:free").add_source(BASE / "docs_ex/")
r1 = bot.chat("Who is Leonardo da Vinci?")
print("Q1:", r1)
r2 = bot.chat("When was he born?")
print("Q2:", r2)
r3 = bot.chat("What are his most famous paintings?")
print("Q3:", r3)


# ── TEST 10 · Structured output (JSON) ───────────────────
section("TEST 10 · Structured output (Pydantic)")
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    birth_year: int
    nationality: str
    most_famous_work: str

data = RAG(model="arcee-ai/trinity-large-preview:free") \
    .add_source(BASE / "docs_ex/") \
    .run("Extract data about Leonardo da Vinci", schema=Person)
print(f"  name:             {data.name}")
print(f"  birth_year:       {data.birth_year}")
print(f"  nationality:      {data.nationality}")
print(f"  most_famous_work: {data.most_famous_work}")


# ── TEST 11 · Streaming ───────────────────────────────────
section("TEST 11 · Streaming response")
rag = RAG(model="arcee-ai/trinity-large-preview:free").add_source(BASE / "docs_ex/")
print("  ", end="")
for token in rag.stream("Tell me about Leonardo's inventions"):
    print(token, end="", flush=True)
print()


# ── TEST 12 · Auto provider detection ────────────────────
section("TEST 12 · Auto provider detection")
from cofone.llm import _detect_provider
for model, expected in [
    ("arcee-ai/trinity-large-preview:free", "openrouter"),
    ("gpt-4o-mini",                         "openai"),
    ("gemini-2.0-flash",                    "gemini"),
]:
    detected = _detect_provider(None, model, None)
    print(f"  {'✓' if detected == expected else '✗'} {model!r:45s} → {detected}")


# ── TEST 13 · Ollama (skipped if not running) ─────────────
section("TEST 13 · Ollama local")
try:
    import httpx
    httpx.get("http://localhost:11434", timeout=2)
    print(RAG(provider="ollama", model="llama3")
        .add_source(BASE / "note_ex.txt")
        .run("Summarize"))
except Exception:
    print("  [skipped] Ollama not running — install ollama.ai then: ollama pull llama3")