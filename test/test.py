from dotenv import load_dotenv
from cofone import RAG
import os

load_dotenv()
link = "https://leonardocofone.github.io/Leonardo_Cofone/"

rag = RAG(
    model_provider="openrouter",                                  # 18 providers + local
    model="arcee-ai/trinity-large-preview:free",                  # all the model for all providers
    model_api_key=os.getenv("OPEN"),

    faiss=True,            
    embedding_provider="local",                                   # 10 options/providers
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",     # If it's not installet yet, it will be installed

    persist_path="./my_db",                                       # path to persist db
    memory=True,                                                  # memory for conversation
).add_source(link).add_source("docs_ex/").add_source("note_ex.txt")

while True:
    print("\n--------------------------------------------------\n")
    dom = input("Enter a question: ")
    answer = rag.run(dom)
    print(answer)
    print("\n--------------------------------------------------\n")
