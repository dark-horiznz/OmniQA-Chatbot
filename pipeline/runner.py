from .runner_service import configure_moudles , load_dataset_from_hf , load_data_to_vectorstore, load_existing_vectorstore, load_data_from_postgres
from src.chains import run_chains
import warnings
warnings.filterwarnings("ignore")

def run_pipeline(question , 
                 hf_dataset_name=None,
                 postgres_table=None,
                 postgres_query_col=None,
                 postgres_answer_col=None,
                 hf_query_col=None,
                 hf_answer_col=None,
                 subset=None,
                 split='train',
                 shuffle=True,
                 vectorstore=None,
                 embeddings=None,
                 max_queries=5,
                 k_retrieval=5,
                 web_mode=True):
    """
    Run the entire pipeline for question answering.
    
    Args:
        question (str): The question to be answered.
        hf_dataset_name (str, optional): The name of the Hugging Face dataset.
        query_col (str, optional): The column name for the query in the dataset.
        answer_col (str, optional): The column name for the answer in the dataset.
        subset (int, optional): If provided, only a subset of the dataset will be used.
        split (str): The split of the dataset to load (default is 'train').
        shuffle (bool): Whether to shuffle the dataset before returning.
        vectorstore (PineconeVectorStore, optional): Existing vector store to use.
        embeddings (Embeddings, optional): Embeddings model to use.
        max_queries (int): Maximum number of queries to run.
        k_retrieval (int): Number of documents to retrieve.
        web_mode (bool): Whether to use web search mode or not.

    Returns:
        dict: The result of the QA chain execution.
    """
    if not embeddings or not vectorstore:
        embeddings, pc = configure_moudles()

    if hf_dataset_name and hf_query_col and hf_answer_col:
        df = load_dataset_from_hf(hf_dataset_name, hf_query_col, hf_answer_col, subset, split, shuffle)
        if not vectorstore:
            vectorstore = load_data_to_vectorstore(df, embeddings)
    
    elif postgres_table and postgres_query_col and postgres_answer_col:
        df = load_data_from_postgres(postgres_query_col, postgres_answer_col, postgres_table)
        if not vectorstore:
            vectorstore = load_data_to_vectorstore(df, embeddings)
    
    elif not vectorstore:
        vectorstore = load_existing_vectorstore(embeddings)
    
    return run_chains(question, vectorstore, max_queries, k_retrieval, web_mode)
