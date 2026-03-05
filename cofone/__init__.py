"""
Cofone — Simple, fast RAG library for Python.

Turn documents, websites, and videos into a queryable knowledge base
in a few lines of Python.

Quick start
-----------
    from dotenv import load_dotenv
    from cofone import RAG

    load_dotenv()  # reads OPENROUTER_API_KEY from .env

    answer = RAG().add_source("docs/").run("Summarize this document")
    print(answer)

Links
-----
- PyPI:   https://pypi.org/project/cofone
- GitHub: https://github.com/LeonardoCofone/cofone
- Docs:   https://github.com/LeonardoCofone/cofone/blob/main/FEATURES.md
"""

from .rag import RAG

__version__ = "0.1.0"
__author__  = "Leonardo Cofone"
__license__ = "MIT"
__all__     = ["RAG"]