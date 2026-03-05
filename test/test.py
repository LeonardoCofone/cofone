from dotenv import load_dotenv
from cofone import RAG
import os

load_dotenv()
link = "https://leonardocofone.github.io/Leonardo_Cofone/"

system_prompt = "You are a helpful assistant that answer questions based on the provided sources. " \
"If the question is not related to the sources, answer normally. " \

rag = RAG(
    system_prompt=system_prompt,                                  # System prompt (Default = None)

    model_provider="openrouter",                                  # 19 providers + local (Default = openrouter)
    model="arcee-ai/trinity-large-preview:free",                  # any model for any provider (Default = arcee-ai/trinity-large-preview:free)
    model_api_key=os.getenv("OPEN"),

    faiss=True,            
    embedding_provider="local",                                   # 10 options/providers (Default = local)
    embedding_model="all-MiniLM-L6-v2",                           # auto-downloaded if not installed (Default = all-MiniLM-L6-v2)

    persist_path="./my_db",                                       # path to persist db (Default = None, in-memory only)
    memory=True,                                                  # memory for conversation (Default = False)
    max_history=5,                                                # How many changes to keep in memory (Default = unlimited)
).add_source(link).add_source("docs_ex/").add_source("note_ex.txt")

while True:
    print("\n--------------------------------------------------\n")
    dom = input("Enter a question: ")
    for token in rag.stream(dom):                                 # streaming response  
        print(token, end="", flush=True)
    print()