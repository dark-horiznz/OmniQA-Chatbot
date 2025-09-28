"""
FastAPI service for OmniQA Chatbot with local document support.
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import os
import tempfile
import shutil
from pipeline.runner import run_main, initialize_resources
from pipeline.runner_service import configure_modules, load_docs_to_vectorstore
from src.utils.document_processor import process_uploaded_file
import warnings
warnings.filterwarnings("ignore")

# Initialize FastAPI app
app = FastAPI(
    title="OmniQA Chatbot API",
    description="A configurable RAG-based chatbot with support for local documents, HuggingFace datasets, and PostgreSQL",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state to store embeddings and vectorstore
class AppState:
    def __init__(self):
        self.embeddings = None
        self.vectorstore = None
        self.instructions = ""
        self.data_mode = "docs"  # docs, hf_dataset, postgres
        self.initialized = False

app_state = AppState()

# Pydantic models
class ChatRequest(BaseModel):
    question: str
    max_queries: Optional[int] = 5
    k_retrieval: Optional[int] = 5
    web_mode: Optional[bool] = False

class ChatResponse(BaseModel):
    answer: str
    data_mode: str
    instructions_used: Optional[str] = None

class LoadDataRequest(BaseModel):
    mode: str  # "docs", "hf_dataset", "postgres"
    docs_path: Optional[str] = None
    hf_dataset_name: Optional[str] = None
    hf_query_col: Optional[str] = None
    hf_answer_col: Optional[str] = None
    postgres_table: Optional[str] = None
    postgres_query_col: Optional[str] = None
    postgres_answer_col: Optional[str] = None
    instructions: Optional[str] = ""

class StatusResponse(BaseModel):
    status: str
    data_mode: str
    instructions: str
    initialized: bool

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    try:
        print("Initializing OmniQA Chatbot API...")
        
        # Initialize embeddings
        app_state.embeddings = configure_modules()
        
        # Check if docs folder exists and load it by default
        docs_path = "docs"
        if os.path.exists(docs_path) and os.listdir(docs_path):
            print(f"Found documents in {docs_path}, loading them...")
            app_state.vectorstore = load_docs_to_vectorstore(docs_path, app_state.embeddings)
            app_state.data_mode = "docs"
            app_state.initialized = True
            print("Successfully initialized with local documents")
        else:
            print("No documents found in docs/ folder. Ready to load data via API.")
            
    except Exception as e:
        print(f"Error during startup: {e}")
        # Continue startup even if initialization fails

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "OmniQA Chatbot API",
        "version": "1.0.0",
        "docs": "/docs",
        "status_endpoint": "/status"
    }

@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Get the current status of the chatbot."""
    return StatusResponse(
        status="ready" if app_state.initialized else "not_initialized",
        data_mode=app_state.data_mode,
        instructions=app_state.instructions,
        initialized=app_state.initialized
    )

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat with the bot using the loaded knowledge base.
    """
    if not app_state.initialized or not app_state.vectorstore:
        raise HTTPException(
            status_code=400, 
            detail="Chatbot not initialized. Please load data first using /load-data endpoint."
        )
    
    try:
        # Run the pipeline with current configuration
        result = run_main(
            question=request.question,
            vectorstore=app_state.vectorstore,
            embeddings=app_state.embeddings,
            max_queries=request.max_queries,
            k_retrieval=request.k_retrieval,
            web_mode=request.web_mode
        )
        
        return ChatResponse(
            answer=str(result),
            data_mode=app_state.data_mode,
            instructions_used=app_state.instructions if app_state.instructions else None
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

@app.post("/load-data")
async def load_data(request: LoadDataRequest, background_tasks: BackgroundTasks):
    """
    Load data from various sources.
    """
    try:
        if request.mode == "docs":
            docs_path = request.docs_path or "docs"
            if not os.path.exists(docs_path):
                raise HTTPException(status_code=400, detail=f"Documents folder not found: {docs_path}")
            
            # Load documents in background
            background_tasks.add_task(load_docs_background, docs_path, request.instructions or "")
            return {"message": f"Loading documents from {docs_path} in background..."}
            
        elif request.mode == "hf_dataset":
            if not all([request.hf_dataset_name, request.hf_query_col, request.hf_answer_col]):
                raise HTTPException(
                    status_code=400, 
                    detail="HuggingFace mode requires: hf_dataset_name, hf_query_col, hf_answer_col"
                )
            
            # Load HF dataset in background
            background_tasks.add_task(
                load_hf_dataset_background, 
                request.hf_dataset_name, 
                request.hf_query_col, 
                request.hf_answer_col,
                request.instructions or ""
            )
            return {"message": f"Loading HuggingFace dataset {request.hf_dataset_name} in background..."}
            
        elif request.mode == "postgres":
            if not all([request.postgres_table, request.postgres_query_col, request.postgres_answer_col]):
                raise HTTPException(
                    status_code=400,
                    detail="PostgreSQL mode requires: postgres_table, postgres_query_col, postgres_answer_col"
                )
            
            # Load PostgreSQL data in background
            background_tasks.add_task(
                load_postgres_background,
                request.postgres_table,
                request.postgres_query_col,
                request.postgres_answer_col,
                request.instructions or ""
            )
            return {"message": f"Loading PostgreSQL table {request.postgres_table} in background..."}
            
        else:
            raise HTTPException(status_code=400, detail="Invalid mode. Use: docs, hf_dataset, or postgres")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading data: {str(e)}")

@app.post("/upload-documents")
async def upload_documents(
    files: List[UploadFile] = File(...),
    instructions: Optional[str] = "",
    background_tasks: BackgroundTasks = None
):
    """
    Upload documents and add them to the knowledge base.
    """
    try:
        docs_path = "docs"
        os.makedirs(docs_path, exist_ok=True)
        
        uploaded_files = []
        for file in files:
            # Save uploaded file
            file_path = os.path.join(docs_path, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            uploaded_files.append(file.filename)
        
        # Reload documents in background
        background_tasks.add_task(load_docs_background, docs_path, instructions or "")
        
        return {
            "message": f"Uploaded {len(uploaded_files)} files. Processing in background...",
            "files": uploaded_files
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading files: {str(e)}")

# Background tasks
async def load_docs_background(docs_path: str, instructions: str):
    """Load documents in the background."""
    try:
        app_state.vectorstore = load_docs_to_vectorstore(docs_path, app_state.embeddings)
        app_state.data_mode = "docs"
        app_state.instructions = instructions
        app_state.initialized = True
        print(f"Successfully loaded documents from {docs_path}")
    except Exception as e:
        print(f"Error loading documents: {e}")

async def load_hf_dataset_background(dataset_name: str, query_col: str, answer_col: str, instructions: str):
    """Load HuggingFace dataset in the background."""
    try:
        result = run_main(
            question="dummy",  # We won't actually run this, just initialize
            hf_dataset_name=dataset_name,
            hf_query_col=query_col,
            hf_answer_col=answer_col,
            embeddings=app_state.embeddings,
            web_mode=False
        )
        app_state.data_mode = "hf_dataset"
        app_state.instructions = instructions
        app_state.initialized = True
        print(f"Successfully loaded HuggingFace dataset {dataset_name}")
    except Exception as e:
        print(f"Error loading HuggingFace dataset: {e}")

async def load_postgres_background(table: str, query_col: str, answer_col: str, instructions: str):
    """Load PostgreSQL data in the background."""
    try:
        result = run_main(
            question="dummy",  # We won't actually run this, just initialize
            postgres_table=table,
            postgres_query_col=query_col,
            postgres_answer_col=answer_col,
            embeddings=app_state.embeddings,
            web_mode=False
        )
        app_state.data_mode = "postgres"
        app_state.instructions = instructions
        app_state.initialized = True
        print(f"Successfully loaded PostgreSQL table {table}")
    except Exception as e:
        print(f"Error loading PostgreSQL data: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)