import streamlit as st
import os
import sys
import warnings
from datetime import datetime
import json
import uuid
import tempfile
import shutil

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

from pipeline.runner import run_main

st.set_page_config(
    page_title="OmniQA - Hybrid RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_css():
    return """
    <style>
    :root {
        --primary-blue: #2196f3;
        --primary-purple: #9c27b0;
        --gradient-start: #667eea;
        --gradient-end: #764ba2;
    }
    
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0f1419 0%, #1a1a2e 50%, #16213e 100%);
    }
    
    .main-header {
        background: linear-gradient(135deg, var(--gradient-start) 0%, var(--gradient-end) 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .chat-message {
        padding: 1.2rem;
        border-radius: 15px;
        margin-bottom: 1.2rem;
        display: flex;
        align-items: flex-start;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .user-message {
        background: linear-gradient(135deg, rgba(33, 150, 243, 0.15) 0%, rgba(33, 150, 243, 0.25) 100%);
        border-left: 4px solid var(--primary-blue);
        color: #e3f2fd;
    }
    
    .bot-message {
        background: linear-gradient(135deg, rgba(156, 39, 176, 0.15) 0%, rgba(156, 39, 176, 0.25) 100%);
        border-left: 4px solid var(--primary-purple);
        color: #f3e5f5;
    }
    
    .message-avatar {
        width: 45px;
        height: 45px;
        border-radius: 50%;
        margin-right: 1rem;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        font-size: 1.2rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    
    .user-avatar {
        background: linear-gradient(135deg, var(--primary-blue) 0%, #1976d2 100%);
        color: white;
    }
    
    .bot-avatar {
        background: linear-gradient(135deg, var(--primary-purple) 0%, #7b1fa2 100%);
        color: white;
    }
    
    .sidebar-section {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.05) 0%, rgba(255, 255, 255, 0.1) 100%);
        padding: 1.2rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .metric-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.08) 0%, rgba(255, 255, 255, 0.12) 100%);
        padding: 1.2rem;
        border-radius: 10px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
        margin-bottom: 1rem;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .upload-section {
        background: linear-gradient(135deg, rgba(67, 142, 185, 0.1) 0%, rgba(156, 39, 176, 0.1) 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        border: 2px dashed rgba(255, 255, 255, 0.3);
        text-align: center;
    }
    
    .success-message {
        background: linear-gradient(135deg, rgba(76, 175, 80, 0.2) 0%, rgba(76, 175, 80, 0.3) 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4caf50;
        color: #c8e6c9;
        margin: 0.5rem 0;
    }
    
    /* Dark mode text adjustments */
    .stMarkdown, .stText, p, span, div {
        color: rgba(255, 255, 255, 0.9) !important;
    }
    
    /* Form styling */
    .stTextInput > div > div > input {
        background-color: rgba(255, 255, 255, 0.1) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        color: white !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-blue) 0%, var(--primary-purple) 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    </style>
    """

def initialize_session_state():
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if 'total_questions' not in st.session_state:
        st.session_state.total_questions = 0

def save_session():
    session_data = {
        'session_id': st.session_state.session_id,
        'messages': st.session_state.messages,
        'timestamp': datetime.now().isoformat(),
        'total_questions': st.session_state.total_questions
    }
    
    os.makedirs('sessions', exist_ok=True)
    with open(f'sessions/{st.session_state.session_id}.json', 'w') as f:
        json.dump(session_data, f, indent=2)

def handle_uploaded_files(uploaded_files):
    """Process uploaded files and save them to the docs directory."""
    if not uploaded_files:
        return False, "No files uploaded."
    
    docs_dir = "docs"
    upload_dir = os.path.join(docs_dir, "uploaded")
    
    # Create directories if they don't exist
    os.makedirs(upload_dir, exist_ok=True)
    
    saved_files = []
    
    for uploaded_file in uploaded_files:
        # Save the uploaded file
        file_path = os.path.join(upload_dir, uploaded_file.name)
        
        try:
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            saved_files.append(uploaded_file.name)
        except Exception as e:
            return False, f"Error saving {uploaded_file.name}: {str(e)}"
    
    return True, f"Successfully uploaded {len(saved_files)} files: {', '.join(saved_files)}"

def get_uploaded_files_info():
    """Get information about uploaded files."""
    upload_dir = os.path.join("docs", "uploaded")
    if not os.path.exists(upload_dir):
        return []
    
    files_info = []
    for filename in os.listdir(upload_dir):
        file_path = os.path.join(upload_dir, filename)
        if os.path.isfile(file_path):
            file_size = os.path.getsize(file_path)
            file_size_kb = round(file_size / 1024, 2)
            files_info.append({
                'name': filename,
                'size': f"{file_size_kb} KB",
                'path': file_path
            })
    
    return files_info

def clear_uploaded_files():
    """Clear all uploaded files."""
    upload_dir = os.path.join("docs", "uploaded")
    if os.path.exists(upload_dir):
        shutil.rmtree(upload_dir)
        os.makedirs(upload_dir, exist_ok=True)
        return True
    return False

def display_message(message, is_user=True):
    message_class = "user-message" if is_user else "bot-message"
    avatar_class = "user-avatar" if is_user else "bot-avatar"
    avatar_text = "👤" if is_user else "🤖"
    
    st.markdown(f"""
    <div class="chat-message {message_class}">
        <div class="message-avatar {avatar_class}">
            {avatar_text}
        </div>
        <div style="flex: 1;">
            <strong>{'You' if is_user else 'OmniQA Bot'}:</strong><br>
            {message}
        </div>
    </div>
    """, unsafe_allow_html=True)

def display_sidebar():
    st.sidebar.markdown("### ⚙️ Configuration")
    
    with st.sidebar.expander("🔧 RAG Settings", expanded=True):
        web_mode = st.checkbox(
            "Enable Web Search", 
            value=True, 
            help="When enabled, the bot can search the web for additional information"
        )
        
        max_queries = st.slider(
            "Max Clarifying Queries", 
            min_value=1, 
            max_value=10, 
            value=5,
            help="Maximum number of follow-up questions the system can ask internally"
        )
        
        k_retrieval = st.slider(
            "Documents Retrieved", 
            min_value=1, 
            max_value=20, 
            value=5,
            help="Number of relevant documents to retrieve from the knowledge base"
        )
    
    with st.sidebar.expander("📊 Data Source", expanded=False):
        data_source = st.selectbox(
            "Select Data Source",
            ["Local Documents", "HuggingFace Dataset", "PostgreSQL"]
        )
        
        if data_source == "Local Documents":
            st.markdown("#### 📁 Document Management")
            
            # File upload section
            uploaded_files = st.file_uploader(
                "Upload Documents",
                accept_multiple_files=True,
                type=['txt', 'pdf', 'docx', 'md', 'json', 'csv'],
                help="Upload documents to add to your knowledge base"
            )
            
            if uploaded_files:
                if st.button("💾 Save Uploaded Files", type="primary"):
                    success, message = handle_uploaded_files(uploaded_files)
                    if success:
                        st.markdown(f"""
                        <div class="success-message">
                            ✅ {message}
                        </div>
                        """, unsafe_allow_html=True)
                        st.rerun()
                    else:
                        st.error(message)
            
            # Show uploaded files info
            uploaded_info = get_uploaded_files_info()
            if uploaded_info:
                st.markdown("**📋 Uploaded Files:**")
                for file_info in uploaded_info:
                    st.markdown(f"• {file_info['name']} ({file_info['size']})")
                
                if st.button("🗑️ Clear Uploaded Files", type="secondary"):
                    if clear_uploaded_files():
                        st.markdown("""
                        <div class="success-message">
                            ✅ Uploaded files cleared!
                        </div>
                        """, unsafe_allow_html=True)
                        st.rerun()
            
            # Show existing docs info
            docs_dir = "docs"
            if os.path.exists(docs_dir):
                all_files = []
                for root, dirs, files in os.walk(docs_dir):
                    for file in files:
                        if file.endswith(('.txt', '.pdf', '.docx', '.md', '.json', '.csv')):
                            all_files.append(file)
                
                if all_files:
                    st.markdown(f"**📚 Total Documents:** {len(all_files)}")
        
        elif data_source == "HuggingFace Dataset":
            hf_dataset = st.text_input("Dataset Name", placeholder="e.g., squad")
            hf_query_col = st.text_input("Question Column", placeholder="question")
            hf_answer_col = st.text_input("Answer Column", placeholder="answer")
        
        elif data_source == "PostgreSQL":
            pg_table = st.text_input("Table Name", placeholder="qa_table")
            pg_query_col = st.text_input("Question Column", placeholder="question")
            pg_answer_col = st.text_input("Answer Column", placeholder="answer")
    
    with st.sidebar.expander("🎨 Display Options", expanded=False):
        show_debug = st.checkbox("Show Debug Info", value=False)
        auto_scroll = st.checkbox("Auto-scroll to bottom", value=True)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📈 Session Stats")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Questions Asked", st.session_state.total_questions)
    with col2:
        st.metric("Messages", len(st.session_state.messages))
    
    if st.sidebar.button("🗑️ Clear Chat", type="secondary"):
        st.session_state.messages = []
        st.session_state.total_questions = 0
        st.rerun()
    
    if st.sidebar.button("💾 Save Session"):
        save_session()
        st.sidebar.markdown("""
        <div class="success-message">
            ✅ Session saved!
        </div>
        """, unsafe_allow_html=True)
    
    return {
        'web_mode': web_mode,
        'max_queries': max_queries,
        'k_retrieval': k_retrieval,
        'data_source': data_source,
        'show_debug': show_debug if 'show_debug' in locals() else False
    }

def main():
    st.markdown(load_css(), unsafe_allow_html=True)
    initialize_session_state()
    
    st.markdown("""
    <div class="main-header">
        <h1>🤖 OmniQA - Hybrid RAG Chatbot</h1>
        <p>🧠 Intelligent Question-Answering with Local Knowledge & Web Search</p>
        <small>Upload documents • Configure settings • Ask questions</small>
    </div>
    """, unsafe_allow_html=True)
    
    config = display_sidebar()
    
    # Configuration status indicator
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, rgba(255, 255, 255, 0.05) 0%, rgba(255, 255, 255, 0.1) 100%); 
         padding: 0.8rem; border-radius: 8px; margin-bottom: 1rem; 
         backdrop-filter: blur(10px); border: 1px solid rgba(255, 255, 255, 0.1);">
        <small>
            <strong>Current Config:</strong> 
            {'🌐 Web Search ON' if config['web_mode'] else '📚 Local Only'} | 
            📄 {config['k_retrieval']} docs | 
            🔄 {config['max_queries']} max queries |
            📂 {config['data_source']}
        </small>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 💬 Chat Interface")
    
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.messages:
            display_message(message['content'], message['is_user'])
    
    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([4, 1])
        
        with col1:
            user_input = st.text_input(
                "Ask a question:",
                placeholder="What would you like to know? 🤔",
                label_visibility="collapsed",
                help="Type your question and press Enter or click Send"
            )
        
        with col2:
            submitted = st.form_submit_button(
                "Send 🚀", 
                use_container_width=True,
                type="primary"
            )
    
    # Add helpful hints
    if not st.session_state.messages:
        st.markdown("""
        <div style="text-align: center; padding: 2rem; opacity: 0.7;">
            <h3>👋 Welcome to OmniQA!</h3>
            <p>Try asking questions like:</p>
            <ul style="list-style: none; padding: 0;">
                <li>💡 "What are the three types of machine learning?"</li>
                <li>🐍 "How do I optimize Python code performance?"</li>
                <li>🔍 "Search the web for recent AI developments"</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    if submitted and user_input:
        st.session_state.messages.append({
            'content': user_input,
            'is_user': True,
            'timestamp': datetime.now().isoformat()
        })
        st.session_state.total_questions += 1
        
        with st.spinner("🔍 Thinking..."):
            try:
                kwargs = {
                    'question': user_input,
                    'web_mode': config['web_mode'],
                    'max_queries': config['max_queries'],
                    'k_retrieval': config['k_retrieval']
                }
                
                if config['data_source'] == "HuggingFace Dataset":
                    kwargs.update({
                        'hf_dataset_name': st.session_state.get('hf_dataset'),
                        'hf_query_col': st.session_state.get('hf_query_col'),
                        'hf_answer_col': st.session_state.get('hf_answer_col')
                    })
                
                elif config['data_source'] == "PostgreSQL":
                    kwargs.update({
                        'postgres_table': st.session_state.get('pg_table'),
                        'postgres_query_col': st.session_state.get('pg_query_col'),
                        'postgres_answer_col': st.session_state.get('pg_answer_col')
                    })
                
                else:  # Local Documents
                    kwargs.update({
                        'use_local_docs': True,
                        'docs_path': 'docs'
                    })
                
                result = run_main(**kwargs)
                
                bot_response = result if isinstance(result, str) else str(result)
                
                st.session_state.messages.append({
                    'content': bot_response,
                    'is_user': False,
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                error_message = f"Sorry, I encountered an error: {str(e)}"
                st.session_state.messages.append({
                    'content': error_message,
                    'is_user': False,
                    'timestamp': datetime.now().isoformat()
                })
                st.error(f"Error: {str(e)}")
        
        st.rerun()
    
    if st.session_state.messages:
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📋 Export Chat"):
                chat_data = {
                    'session_id': st.session_state.session_id,
                    'messages': st.session_state.messages,
                    'timestamp': datetime.now().isoformat()
                }
                st.download_button(
                    "Download JSON",
                    json.dumps(chat_data, indent=2),
                    f"chat_export_{st.session_state.session_id[:8]}.json",
                    "application/json"
                )
        
        with col2:
            if st.button("🔄 New Session"):
                st.session_state.session_id = str(uuid.uuid4())
                st.session_state.messages = []
                st.session_state.total_questions = 0
                st.rerun()
    
    st.markdown("---")
    with st.expander("ℹ️ About OmniQA - Features & Capabilities"):
        st.markdown("""
        **OmniQA** is a sophisticated Hybrid RAG (Retrieval-Augmented Generation) system that combines:
        
        ### 🔥 Core Features
        - 🔍 **Local Knowledge Base**: Searches through your document collection
        - 🌐 **Web Search Integration**: Fetches real-time information from the internet  
        - 🧠 **Self-Clarifying QA**: Automatically asks follow-up questions for better answers
        - � **Document Upload**: Drag & drop files directly into the knowledge base
        - 🎨 **Dark Mode Optimized**: Beautiful interface that works great in dark themes
        
        ### 📊 Data Sources
        - 📁 **Local Documents**: Upload TXT, PDF, DOCX, MD, JSON, CSV files
        - 🤗 **HuggingFace Datasets**: Connect to any dataset with Q&A pairs
        - 🐘 **PostgreSQL**: Query data directly from your database
        
        ### ⚡ Performance Features  
        - **Fast Vector Search**: Uses FAISS for efficient similarity search
        - **Configurable Retrieval**: Adjust number of documents and queries
        - **Session Management**: Save and export conversation history
        - **Real-time Stats**: Track questions asked and performance metrics
        
        ### 🎯 Perfect for
        - Customer support systems
        - Knowledge management platforms  
        - Educational applications
        - Research and development tools
        - Personal knowledge assistants
        
        ### 📝 Supported File Types
        - **Text Files**: .txt, .md (Markdown)
        - **Documents**: .pdf, .docx  
        - **Data Files**: .json, .csv
        - **More formats**: Easily extensible
        """)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🚀 Response Speed", "< 3s", "avg")
        with col2:
            st.metric("📚 Doc Types", "6+", "formats")  
        with col3:
            st.metric("🔧 Config Options", "10+", "settings")

if __name__ == "__main__":
    main()