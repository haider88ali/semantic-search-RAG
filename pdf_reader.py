import sys

from openai import OpenAI
from dotenv import load_dotenv
import chromadb
from pypdf import PdfReader

load_dotenv()

client=OpenAI()
chroma_client=chromadb.PersistentClient(
    path="./chroma_db"
)

COLLECTION_NAME="pdf_reader"
CHUNK_SIZE=500
CHUNK_DISTANCE=50

def get_collection():

    collection=chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=None,
        metadata={"hnsw:space":"cosine"}
    )

    return collection

def read_pdf(path):
    reader= PdfReader(path)
    pages=[]


    for page_numer,page in enumerate(reader.pages,start=1):
        text=" ".join((page.extract_text() or "").split())

        if text:
            pages.append((page_numer,text))

    return pages

def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_DISTANCE):
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + size, len(text))

        if end < len(text):
            last_space = text.rfind(" ", start, end)
            if last_space > start:
                end = last_space

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(end - overlap, start + 1)

    return chunks

def embedding(text):
    response= client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )

    return [item.embedding for item in response.data]

def injest_pdf(path):
    collection=get_collection()

    ids,text,metadata=[],[],[]
    for page_number,page_text in read_pdf(path):
        for index,chunk in enumerate(chunk_text(page_text)):
            ids.append(f"{path}-p{page_number}-{index}")
            text.append(chunk)
            metadata.append({
                "source":path,
                "index":index,
                "page":page_number
                })

    embeddings=embedding(text)      

    collection.upsert(
        ids=ids,
        documents=text,
        embeddings=embeddings,
        metadatas=metadata
    )

    print(collection)

if __name__ == "__main__":
    injest_pdf(sys.argv[1] if len(sys.argv) > 1 else "nimbusdb_handbook.pdf")
