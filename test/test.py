from dotenv import load_dotenv
from cofone import RAG
import os

load_dotenv()
link = "https://leonardocofone.github.io/Leonardo_Cofone/"

rag = RAG(
    model_provider="openrouter",
    model="arcee-ai/trinity-large-preview:free",
    model_api_key=os.getenv("OPEN"),

    faiss=True,
    embedding_provider="local",

    chunk_mode="smart",
    persist_path="./my_db",
    memory=True
    
).add_source(link).add_source("docs/")

while True:
    print("\n--------------------------------------------------\n")
    dom = input("Enter a question: ")
    answer = rag.run(dom)
    print(answer)
    print("\n--------------------------------------------------\n")