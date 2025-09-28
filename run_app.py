#!/usr/bin/env python3
"""
OmniQA Streamlit App Launcher
Run this script to start the OmniQA Hybrid RAG chatbot interface.
"""

import os
import sys
import subprocess

def main():
    print("🤖 Starting OmniQA - Hybrid RAG Chatbot")
    print("=" * 50)
    
    # Change to the project directory
    project_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_dir)
    
    print(f"📁 Working directory: {project_dir}")
    
    # Check if streamlit is installed
    try:
        import streamlit
        print("✅ Streamlit is installed")
    except ImportError:
        print("❌ Streamlit not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "streamlit"])
    
    # Start the streamlit app
    print("🚀 Launching OmniQA interface...")
    print("📱 Your app will open in your default web browser")
    print("🔗 Local URL: http://localhost:8501")
    print("\n🆕 NEW FEATURES:")
    print("   📤 Upload documents directly to knowledge base")  
    print("   🎨 Enhanced dark mode theme")
    print("   ⚙️ Better configuration interface")
    print("   📊 Real-time status indicators")
    print("\n💡 To stop the server, press Ctrl+C")
    print("=" * 50)
    
    try:
        subprocess.run([
            sys.executable, 
            "-m", 
            "streamlit", 
            "run", 
            "app.py",
            "--server.port=8501",
            "--server.address=localhost"
        ])
    except KeyboardInterrupt:
        print("\n👋 OmniQA chatbot stopped. Goodbye!")
    except Exception as e:
        print(f"❌ Error starting the application: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())