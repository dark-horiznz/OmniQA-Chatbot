import time
import os
from tqdm import tqdm
from langchain_pinecone import PineconeVectorStore


def upsert(docs , embeddings , split_size = 100):
    for start in tqdm(range(0, len(docs), split_size)):
        doc = docs[start : start + split_size]
        vectorstore = PineconeVectorStore.from_documents(
            doc,
            embeddings,
            index_name= os.environ['PINECONE_ENV']
        )
        time.sleep(30)
    return vectorstore
    
def load_existing(embeddings):
    vectorstore = PineconeVectorStore.from_existing_index(
    embedding=embeddings,                   
    index_name=os.environ["PINECONE_ENV"],   
)
    return vectorstore