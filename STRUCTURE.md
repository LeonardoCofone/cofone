# Cofone — Project Structure

```
cofone/
├── cofone/                        ← the Python package
│   ├── __init__.py                → exposes RAG + __version__
│   ├── rag.py                     → RAG class: main interface, fluent DSL, memory, streaming
│   ├── loader.py                  → multi-source loading: txt/md/pdf, URL, Wikipedia, YouTube
│   ├── chunker.py                 → 4 chunking modes: smart / paragraphs / sentences / fixed
│   ├── retriever.py               → BM25 + FAISS, 10 embedding providers, disk persistence
│   └── llm.py                     → 19 LLM providers, streaming, structured output
├── test/
│   ├── docs_ex/
│   │   ├── leonardo.txt           → test document: Leonardo da Vinci biography
│   │   └── machine_learning.md    → test document: machine learning overview
│   ├── note_ex.txt                → test document: Cofone feature notes
│   ├── test.py                    → 13 integration tests covering every feature
│   └── cofone_demo.ipynb          → Jupyter notebook demo of all features
├── .env                           → API keys (hidden by .gitignore — never commit)
├── .gitignore                     → ignores .env, __pycache__, dist/, *.egg-info/, etc.
├── requirements.txt               → core deps (commented optional ones)
├── pyproject.toml                 → build config + PyPI metadata + optional extras
├── LICENSE                        → MIT
├── RELEASE.md                     → step-by-step guide to release new versions (gitignored)
├── STRUCTURE.md                   → this file
├── INSTALL.md                     → installation guide + API keys + troubleshooting
├── FEATURES.md                    → complete feature reference with all providers and examples
└── README.md                      → landing page: quick start, examples, provider tables
```