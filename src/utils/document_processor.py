import os
from typing import List, Optional
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


def load_text_file(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_text(text)


def load_docs_from_folder(docs_path: str, 
                         supported_extensions: Optional[List[str]] = None,
                         chunk_size: int = 1000, 
                         chunk_overlap: int = 200) -> List[Document]:
    if supported_extensions is None:
        supported_extensions = ['.txt', '.md']
    
    if not os.path.exists(docs_path):
        print(f"Documents folder not found at {docs_path}")
        return []
    
    documents = []
    
    for filename in os.listdir(docs_path):
        file_path = os.path.join(docs_path, filename)
        
        if os.path.isdir(file_path):
            continue
            
        file_extension = os.path.splitext(filename)[1].lower()
        if file_extension not in supported_extensions:
            print(f"Skipping unsupported file: {filename}")
            continue
        
        try:
            content = load_text_file(file_path)
            chunks = chunk_text(content, chunk_size, chunk_overlap)
            
            for i, chunk in enumerate(chunks):
                metadata = {
                    'source': filename,
                    'chunk_id': i,
                    'file_path': file_path
                }
                documents.append(Document(page_content=chunk, metadata=metadata))
                
        except Exception as e:
            print(f"Error processing file {filename}: {str(e)}")
            continue
    
    print(f"Loaded {len(documents)} document chunks from {docs_path}")
    return documents


def process_uploaded_file(file_path: str, 
                         filename: str,
                         chunk_size: int = 1000, 
                         chunk_overlap: int = 200) -> List[Document]:
    try:
        content = load_text_file(file_path)
        chunks = chunk_text(content, chunk_size, chunk_overlap)
        
        documents = []
        for i, chunk in enumerate(chunks):
            metadata = {
                'source': filename,
                'chunk_id': i,
                'file_path': file_path
            }
            documents.append(Document(page_content=chunk, metadata=metadata))
        
        print(f"Processed {filename} into {len(documents)} chunks")
        return documents
        
    except Exception as e:
        print(f"Error processing file {filename}: {str(e)}")
        return []