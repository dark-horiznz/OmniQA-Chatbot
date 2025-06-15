import pandas as pd
from typing import Union, List
import os
from sqlalchemy import create_engine, inspect

def list_db_columns(table_name):
    """
    Return a list of column names for the specified table.

    Args:
        table_name: The name of the table whose columns to list.

    Returns:
        A list of column name strings.
    """
    engine = create_engine(os.environ['POSTGRES_URI'])
    inspector = inspect(engine)
    if table_name not in inspector.get_table_names():
        raise ValueError(f"Table '{table_name}' does not exist in the database.")
    return [col['name'] for col in inspector.get_columns(table_name)]

def fetch_qa_data(query_col:str, answer_col: Union[str , List] , table_name):
    """
    Fetch specified columns from the given table and optionally rename them to ['question','answer'].

    Args:
        query_col: Name of the column to use as the question.
        answer_col: Name of the column to use as the answer.
        rename: Whether to rename the fetched columns to 'question' and 'answer'.
        table_name: The database table to query.

    Returns:
        A pandas DataFrame with columns ['question','answer'] (if rename=True),
        or DataFrame with original column names otherwise.
    """
    engine = create_engine(os.environ['POSTGRES_URI'])
    inspector = inspect(engine)
    if table_name not in inspector.get_table_names():
        raise ValueError(f"Table '{table_name}' does not exist in the database.")

    cols = [col['name'] for col in inspector.get_columns(table_name)]
    for c in (query_col, answer_col):
        if c not in cols:
            raise ValueError(f"Column '{c}' not found in table '{table_name}'.")

    q_col = f'"{query_col}"'
    a_col = f'"{answer_col}"'
    t_name = f'"{table_name}"'

    sql = f"SELECT {q_col}, {a_col} FROM {t_name}"
    df = pd.read_sql(sql, engine)

    df = df.rename(columns={query_col: 'question', answer_col: 'answer'})
    return df