import json
import os
import time
from datetime import datetime
from pathlib import Path

from flask_restx import Model
from langchain_community.vectorstores.faiss import FAISS, DistanceStrategy
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from langchain.chains.retrieval_qa.base import (RetrievalQA,
                                                StuffDocumentsChain,
                                                VectorDBQA)

model = AzureChatOpenAI(
    api_key=os.getenv("AZURE_API_KEY"),
    api_version=os.getenv("AZURE_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    model=os.getenv("AZURE_JARINA"),
    temperature=0.1,
    max_tokens=2048,
    streaming=False,
)

embeddings = AzureOpenAIEmbeddings(
    api_key=os.getenv("AZURE_API_KEY"),
    api_version=os.getenv("AZURE_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    model=os.getenv("AZURE_ADA"),

)

db = None
qa_chain = None
chain = None


# Load output from gpt crawler
path_to_gptcrawler = "/code/data/output-1.json"

def regenerate_faiss_db():
    global db, qa_chain, chain  # A globális változókat használjuk

    if os.path.exists(path_to_gptcrawler) and os.path.getsize(path_to_gptcrawler) > 0:
        with open(path_to_gptcrawler, 'r') as f:
            data = json.load(f)
    else:
        print(f"A {path_to_gptcrawler} fájl nem létezik vagy üres.")
        data = []

    docs = [
        Document(
            page_content=dict_["html"],
            metadata={"title": dict_["title"], "url": dict_["url"]},
        )
        for dict_ in data
    ]
    # Split
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    all_splits = text_splitter.split_documents(docs)

    # Generate FAISS index file name with date-time format
    current_datetime = datetime.now().strftime("%Y-%m-%d-%H")
    index_file = f"/code/faiss/{current_datetime}.faiss"

    # Create FAISS vectorstore from documents and embeddings
    db = FAISS.from_documents(documents=all_splits, embedding=embeddings, normalize_L2=True, distance_strategy=DistanceStrategy.COSINE)
    db.save_local(index_file)

    # RAG prompt
    template = """Válaszolj a kérdésre kizárólag a következő kontextus alapján:
    {context}

    Kérdés: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    hun = "A válaszokat magyar nyelven várjuk"
    qa_chain = RetrievalQA.from_chain_type(
        model,
        retriever=db.as_retriever(),
        chain_type_kwargs={"prompt": prompt + hun}
    )

    # RAG chain
    chain = (
        RunnableParallel({"context": db.as_retriever(), "question": RunnablePassthrough()})
        | prompt
        | model
        | StrOutputParser()
    )

    # Add typing for input
    class Question(BaseModel):
        __root__: str

    chain = chain.with_types(input_type=Question)

class FileChangeHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory and event.src_path == path_to_gptcrawler:
            # A JSON fájl létrejött, inicializáljuk a FAISS adatbázist
            regenerate_faiss_db()

    def on_modified(self, event):
        if not event.is_directory and event.src_path == path_to_gptcrawler:
            # A JSON fájl módosult, újrageneráljuk a FAISS adatbázist
            regenerate_faiss_db()

# Inicializáld a db változót a konténer indulásakor
if os.path.exists(path_to_gptcrawler) and os.path.getsize(path_to_gptcrawler) > 0:
    regenerate_faiss_db()
else:
    print(f"A {path_to_gptcrawler} fájl nem létezik vagy üres. Várakozás a fájl létrehozására...")

observer = Observer()
observer.schedule(FileChangeHandler(), path=os.path.dirname(path_to_gptcrawler), recursive=False)
observer.start()

# A szkript futásban tartása
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    observer.stop()
observer.join()
