#!/usr/bin/env python3
"""
Startup script for DSPY Boss System
Coordinates backend API server and frontend development server
"""

import os
import sys
import subprocess
import time
import signal
import threading
from pathlib import Path

def run_backend():
    """Start the FastAPI backend server"""
    backend_dir = Path(__file__).parent / "backend"
    os.chdir(backend_dir)
    
    print("🚀 Starting DSPY Boss Backend API Server...")
    
    # Install dependencies if needed
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, capture_output=True)
        print("✅ Backend dependencies installed")
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Warning: Could not install backend dependencies: {e}")
    
    # Start the API server
    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "dspy_boss.api_server:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload"
        ], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Backend server stopped")
    except subprocess.CalledProcessError as e:
        print(f"❌ Backend server failed: {e}")

def run_frontend():
    """Start the Next.js frontend development server"""
    frontend_dir = Path(__file__).parent / "frontend"
    os.chdir(frontend_dir)
    
    print("🎨 Starting Frontend Development Server...")
    
    # Install dependencies if needed
    try:
        subprocess.run(["npm", "install"], check=True, capture_output=True)
        print("✅ Frontend dependencies installed")
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Warning: Could not install frontend dependencies: {e}")
    
    # Start the development server
    try:
        subprocess.run(["npm", "run", "dev"], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Frontend server stopped")
    except subprocess.CalledProcessError as e:
        print(f"❌ Frontend server failed: {e}")

def main():
    """Main startup coordinator"""
    print("🎯 DSPY Boss System Startup")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not (Path.cwd() / "backend" / "dspy_boss").exists():
        print("❌ Error: Please run this script from the hume-boss root directory")
        sys.exit(1)
    
    # Start backend in a separate thread
    backend_thread = threading.Thread(target=run_backend, daemon=True)
    backend_thread.start()
    
    # Give backend time to start
    print("⏳ Waiting for backend to initialize...")
    time.sleep(3)
    
    # Start frontend in main thread
    try:
        run_frontend()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down system...")
    
    print("👋 DSPY Boss System stopped")

if __name__ == "__main__":
    main()