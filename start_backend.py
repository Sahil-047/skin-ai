"""
Quick Backend Startup Script
Run this from the main project directory to start the API server
"""

import subprocess
import sys
from pathlib import Path
import os

def main():
    """Start the backend server."""
    
    print("🚀 Starting Skin Disease Classification API...")
    print("="*60)
    
    # Check if we're in the right directory
    if not Path("backend").exists():
        print("❌ Error: backend folder not found!")
        print("   Make sure you're in the main project directory.")
        return
    
    # Change to backend directory
    os.chdir("backend")
    
    print("📁 Changed to backend directory")
    print("🔧 Starting server...")
    print("\n🌐 API will be available at:")
    print("   • http://localhost:8000 (main API)")
    print("   • http://localhost:8000/docs (interactive docs)")
    print("   • http://localhost:8000/web_interface.html (web UI)")
    print("\n📱 For mobile app:")
    print("   • Find your IP with: ipconfig (Windows) or ifconfig (Mac/Linux)")
    print("   • Use: http://YOUR_IP:8000/predict")
    print("\n" + "="*60)
    
    try:
        # Start the server
        subprocess.run([sys.executable, "start_server.py"], check=True)
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped by user.")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error starting server: {e}")
    except FileNotFoundError:
        print("\n❌ Python not found. Make sure Python is installed and in PATH.")

if __name__ == "__main__":
    main()
