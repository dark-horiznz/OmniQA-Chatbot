from src.db_utils import pinecone_db
from src.embeddings import get_embedding_models

import os
import google.generativeai as genai
from pinecone import Pinecone
from langchain.schema import Document

from datasets import load_dataset
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")
from dotenv import load_dotenv
load_dotenv()

def configure_moudles():
    """
    Configure the necessary modules for the application.
    Returns:
        tuple: Contains the embeddings model and the Pinecone vector store.
    """
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    embeddings = get_embedding_models()['GeminiEmbeddings'](api_key=os.environ["GEMINI_API_KEY"])
    pc = Pinecone(os.environ['PINECONE_API_KEY'])
    return embeddings, pc

def load_dataset(hf_dataset_name , query_col , answer_col , split='train' , subset=None, shuffle=True):
    """
    Load a dataset from Hugging Face and return a DataFrame with specified columns.
    Args:
        hf_dataset_name (str): The name of the dataset on Hugging Face.
        query_col (str): The column name for the query.
        answer_col (str): The column name for the answer.
        split (str): The split of the dataset to load (default is 'train').
        subset (int, optional): If provided, only a subset of the dataset will be returned.
        shuffle (bool): Whether to shuffle the dataset before returning.
    Returns:
        pd.DataFrame: A DataFrame containing the specified columns.
    """
    data = load_dataset(hf_dataset_name , split=split)
    if subset and subset < len(data):
        data = data.shuffle(42)[:subset]
    df = pd.DataFrame(data)
    if query_col not in df.columns or answer_col not in df.columns:
        raise ValueError(f"Columns {query_col} and {answer_col} must be present in the dataset.")
    df = df[[query_col, answer_col]].rename(columns={query_col: "question", answer_col: "answer"})
    return df

def load_data_to_vectorstore(df, embeddings):
    """
    Load data from a DataFrame into a Pinecone vector store.
    Args:
        df (pd.DataFrame): DataFrame containing 'question' and 'answer' columns.
        embeddings (Embeddings): The embeddings model to use for vectorization.
    Returns:
        PineconeVectorStore: The vector store containing the upserted documents.
    """
    docs = [Document(page_content=f"Q: {row['question']}\nA: {row['answer']}", metadata={'Page index':i+1})
    for i, row in df.iterrows()]
    vectorstore = pinecone_db()['upsert'](docs, embeddings)
    return vectorstore

def load_existing_vectorstore(embeddings):
    """
    Load the vector store from Pinecone.
    Args:
        embeddings (Embeddings): The embeddings model to use for vectorization.
    Returns:
        PineconeVectorStore: The loaded vector store.
    """
    return pinecone_db()['load_existing'](embeddings)