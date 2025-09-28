from src.db_utils import faiss_db, postgres_db
from src.embeddings import get_embedding_models
from src.utils.document_processor import load_docs_from_folder

import os
import google.generativeai as genai
from langchain.schema import Document

from datasets import load_dataset
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")
from dotenv import load_dotenv
load_dotenv()

def configure_modules():
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    embeddings = get_embedding_models()['GeminiEmbeddings'](api_key=os.environ["GEMINI_API_KEY"])
    return embeddings

def load_dataset_from_hf(hf_dataset_name , query_col , answer_col , split='train' , subset=None, shuffle=True):
    data = load_dataset(hf_dataset_name , split=split)
    if data is None:
        return pd.DataFrame()
    df = data.to_pandas()
    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)
    if subset:
        df = df.head(subset)
    if query_col not in df.columns or answer_col not in df.columns:
        raise ValueError(f"Columns {query_col} and {answer_col} must be present in the dataset.")
    df = df[[query_col, answer_col]].rename(columns={query_col: "question", answer_col: "answer"})
    return df

def load_data_from_postgres(query_col, answer_col , table_name):
    print(f"Fetching data from table: {table_name} with columns: {query_col}, {answer_col}")
    cols = postgres_db()['list_db_columns'](table_name)
    print(f"Columns in table {table_name}: {cols}")
    if query_col not in cols or answer_col not in cols:
        raise ValueError(f"Columns {query_col} and {answer_col} must be present in the table {table_name}, present columns are: {cols}")
    df = postgres_db()['fetch_qa_data'](query_col, answer_col , table_name)
    print(f"Data fetched from table {table_name} with shape: {df.shape}")
    return df

def load_data_to_vectorstore(df, embeddings):
    docs = [Document(page_content=f"Q: {row['question']}\nA: {row['answer']}", metadata={'Page index':i+1})
    for i, row in df.iterrows()]
    vectorstore = faiss_db()['upsert'](docs, embeddings)
    return vectorstore

def load_existing_vectorstore(embeddings):
    return faiss_db()['load_existing'](embeddings)

def load_docs_to_vectorstore(docs_path, embeddings, faiss_index_path="faiss_index"):
    vectorstore = load_existing_vectorstore(embeddings)
    
    if vectorstore is None:
        docs = load_docs_from_folder(docs_path)
        if not docs:
            print(f"No documents found in {docs_path}")
            return None
        
        vectorstore = faiss_db()['upsert'](docs, embeddings)
        faiss_db()['save'](vectorstore, faiss_index_path)
        print(f"Created and saved new FAISS index with {len(docs)} document chunks")
    else:
        print("Loaded existing FAISS index")
    
    return vectorstore