#!/usr/bin/env python3
"""
Start the DSPY Boss Backend API Server
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Start the FastAPI backend server"""
    # Change to backend directory
    backend_dir = Path(__file__).parent / "backend"
    os.chdir(backend_dir)
    
    print("🚀 Starting DSPY Boss Backend API Server...")
    print(f"📁 Working directory: {backend_dir}")
    
    # Install dependencies if needed
    try:
        print("📦 Installing backend dependencies...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True)
        print("✅ Backend dependencies installed")
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Warning: Could not install backend dependencies: {e}")
    
    # Start the API server
    print("🌐 Starting API server on http://localhost:8000")
    print("📊 API documentation available at http://localhost:8000/docs")
    print("🔌 WebSocket endpoint at ws://localhost:8000/ws")
    print("\nPress Ctrl+C to stop the server\n")
    
    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "dspy_boss.api_server:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload",
            "--log-level", "info"
        ], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Backend server stopped")
    except subprocess.CalledProcessError as e:
        print(f"❌ Backend server failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()