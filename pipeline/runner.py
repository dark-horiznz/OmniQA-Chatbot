from .runner_service import configure_modules , load_dataset_from_hf , load_data_to_vectorstore, load_existing_vectorstore, load_data_from_postgres, load_docs_to_vectorstore
from src.chains import run_chains
import os
import warnings
warnings.filterwarnings("ignore")

def initialize_resources(embeddings=None, vectorstore=None):
    if not embeddings or not vectorstore:
        embeddings = configure_modules()
    return embeddings, vectorstore

def load_data(
    hf_dataset_name=None,
    postgres_table=None,
    postgres_query_col=None,
    postgres_answer_col=None,
    hf_query_col=None,
    hf_answer_col=None,
    subset=None,
    split='train',
    shuffle=True
):
    if hf_dataset_name and hf_query_col and hf_answer_col:
        return load_dataset_from_hf(
            hf_dataset_name, hf_query_col, hf_answer_col, subset, split, shuffle
        )

    if postgres_table and postgres_query_col and postgres_answer_col:
        return load_data_from_postgres(
            postgres_query_col, postgres_answer_col, postgres_table
        )

    return None

def select_vectorstore(
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
    docs_path=None,
    use_local_docs=False
):
    embeddings, vectorstore = initialize_resources(embeddings, vectorstore)

    # Check if local documents should be used
    if use_local_docs or docs_path:
        if not docs_path:
            docs_path = "docs"  # Default documents folder
        return load_docs_to_vectorstore(docs_path, embeddings)

    df = load_data(
        hf_dataset_name, postgres_table, postgres_query_col, postgres_answer_col,
        hf_query_col, hf_answer_col, subset, split, shuffle
    )

    if df is not None:
        if not vectorstore:
            vectorstore = load_data_to_vectorstore(df, embeddings)
    elif not vectorstore:
        vectorstore = load_existing_vectorstore(embeddings)

    return vectorstore

def run_pipeline(
    question,
    vectorstore,
    max_queries=5,
    k_retrieval=5,
    web_mode=True
):
    return run_chains(question, vectorstore, max_queries, k_retrieval, web_mode)


def run_main(
    question,
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
    web_mode=True,
    docs_path=None,
    use_local_docs=False
):
    # Auto-detect local docs mode if docs folder exists
    if not use_local_docs and docs_path is None:
        default_docs_path = "docs"
        if os.path.exists(default_docs_path) and os.listdir(default_docs_path):
            use_local_docs = True
            docs_path = default_docs_path
            print(f"Detected documents in {default_docs_path}, switching to local docs mode")

    vectorstore = select_vectorstore(
        hf_dataset_name, postgres_table, postgres_query_col, postgres_answer_col,
        hf_query_col, hf_answer_col, subset, split, shuffle, vectorstore, embeddings,
        docs_path, use_local_docs
    )

    return run_pipeline(question, vectorstore, max_queries, k_retrieval, web_mode)



