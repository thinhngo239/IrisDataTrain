#!/usr/bin/env python3
"""
Iris ML Pipeline Launcher

A convenient script to start different components of the ML pipeline.
"""

import sys
import subprocess
import time
import argparse
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def run_command(command, background=False, cwd=None):
    """Run a command with optional background execution."""
    try:
        if background:
            print(f"🚀 Starting in background: {command}")
            process = subprocess.Popen(
                command.split(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=cwd
            )
            return process
        else:
            print(f"🚀 Running: {command}")
            result = subprocess.run(
                command.split(),
                check=True,
                cwd=cwd
            )
            return result
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running command: {e}")
        return None
    except FileNotFoundError:
        print(f"❌ Command not found: {command}")
        return None

def train_models():
    """Train ML models."""
    print("🎯 Training ML models...")
    return run_command("python -m iris_pipeline.models.training")

def start_api(port=8000):
    """Start FastAPI server."""
    print(f"🌐 Starting API server on port {port}...")
    return run_command(
        f"uvicorn iris_pipeline.api.server:app --host 0.0.0.0 --port {port} --reload",
        background=True
    )

def start_web(port=8501):
    """Start Streamlit web interface."""
    print(f"🎨 Starting web interface on port {port}...")
    return run_command(
        f"streamlit run apps/web_interface.py --server.port {port}",
        background=True
    )

def check_health():
    """Check if services are running."""
    import requests
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ API server is healthy")
            return True
        else:
            print(f"⚠️ API server returned {response.status_code}")
            return False
    except requests.exceptions.RequestException:
        print("❌ API server is not responding")
        return False

def install_dependencies():
    """Install required dependencies."""
    print("📦 Installing dependencies...")
    return run_command("pip install -r requirements.txt")

def main():
    """Main launcher function."""
    parser = argparse.ArgumentParser(description="Iris ML Pipeline Launcher")
    parser.add_argument(
        "action",
        choices=["train", "api", "web", "demo", "health", "install", "all"],
        help="Action to perform"
    )
    parser.add_argument(
        "--api-port", 
        type=int, 
        default=8000, 
        help="Port for API server (default: 8000)"
    )
    parser.add_argument(
        "--web-port", 
        type=int, 
        default=8501, 
        help="Port for web interface (default: 8501)"
    )
    parser.add_argument(
        "--wait", 
        type=int, 
        default=3, 
        help="Wait time between services (default: 3 seconds)"
    )
    
    args = parser.parse_args()
    
    # Change to project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    print(f"🌸 Iris ML Pipeline Launcher")
    print(f"📁 Working directory: {project_root}")
    print(f"🎯 Action: {args.action}")
    print("-" * 50)
    
    if args.action == "install":
        install_dependencies()
    
    elif args.action == "train":
        train_models()
    
    elif args.action == "api":
        start_api(args.api_port)
        print(f"🌐 API server started on http://localhost:{args.api_port}")
        print(f"📚 API docs: http://localhost:{args.api_port}/docs")
        
        # Keep the process running
        try:
            print("Press Ctrl+C to stop...")
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n👋 Stopping API server...")
    
    elif args.action == "web":
        start_web(args.web_port)
        print(f"🎨 Web interface started on http://localhost:{args.web_port}")
        
        # Keep the process running
        try:
            print("Press Ctrl+C to stop...")
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n👋 Stopping web interface...")
    
    elif args.action == "health":
        check_health()
    
    elif args.action == "demo":
        print("🎪 Starting demo environment...")
        
        # Start API server
        api_process = start_api(args.api_port)
        
        if api_process:
            print(f"⏳ Waiting {args.wait} seconds for API to start...")
            time.sleep(args.wait)
            
            # Check if API is ready
            if check_health():
                print("✅ API server is ready!")
                
                # Start web interface
                web_process = start_web(args.web_port)
                
                if web_process:
                    print(f"✅ Web interface started!")
                    print(f"🌐 API: http://localhost:{args.api_port}")
                    print(f"🎨 Web: http://localhost:{args.web_port}")
                    
                    try:
                        print("Press Ctrl+C to stop all services...")
                        while True:
                            time.sleep(1)
                    except KeyboardInterrupt:
                        print("\n👋 Stopping all services...")
                        if api_process and isinstance(api_process, subprocess.Popen):
                            api_process.terminate()
                        if web_process and isinstance(web_process, subprocess.Popen):
                            web_process.terminate()
                else:
                    print("❌ Failed to start web interface")
            else:
                print("❌ API server failed to start properly")
        else:
            print("❌ Failed to start API server")
    
    elif args.action == "all":
        print("🚀 Starting complete pipeline...")
        
        # Install dependencies
        if install_dependencies():
            print("✅ Dependencies installed")
            
            # Train models
            if train_models():
                print("✅ Models trained")
                
                # Start demo
                print("🎪 Starting demo...")
                main_args = argparse.Namespace(
                    action="demo",
                    api_port=args.api_port,
                    web_port=args.web_port,
                    wait=args.wait
                )
                
                # Change args and run demo
                sys.argv = [sys.argv[0], "demo", 
                           f"--api-port={args.api_port}",
                           f"--web-port={args.web_port}"]
                main()
            else:
                print("❌ Failed to train models")
        else:
            print("❌ Failed to install dependencies")
    
    print("\n🎉 Launcher finished!")

if __name__ == "__main__":
    main() 