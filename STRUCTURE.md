# Cofone — Project Structure

```
cofone/
├── cofone/
│   ├── __init__.py       → exposes RAG only
│   ├── rag.py            → main class + fluent DSL
│   ├── loader.py         → reads txt, md, pdf, URL, Wikipedia, YouTube
│   ├── chunker.py        → smart / paragraphs / sentences / fixed
│   ├── retriever.py      → BM25 + FAISS with optional disk persistence
│   └── llm.py            → OpenRouter / OpenAI / Gemini / Ollama
├── test/
│   ├── docs_ex/
│   │   ├── leonardo.txt
│   │   └── machine_learning.md
│   ├── note_ex.txt
│   ├── test.py
│   └── cofone_demo.ipynb
├── .env                  → API keys (never commit)
├── .gitignore
├── requirements.txt
├── setup.py
├── STRUCTURE.md
├── INSTALL.md
├── FEATURES.md
└── README.md
```