#!/usr/bin/env python
import os
import sys
import subprocess
import webbrowser
from pathlib import Path

def check_env():
    """Check if virtual environment exists, create if not"""
    venv_path = Path("venv")
    if not venv_path.exists():
        print("Creating virtual environment...")
        subprocess.check_call([sys.executable, "-m", "venv", "venv"])
        return False
    return True

def install_dependencies():
    """Install required packages"""
    print("Installing dependencies...")
    
    # Determine the pip path based on platform
    if sys.platform == "win32":
        pip_path = "venv\\Scripts\\pip"
    else:
        pip_path = "venv/bin/pip"
        
    subprocess.check_call([pip_path, "install", "-r", "requirements.txt"])

def check_env_file():
    """Check if .env file exists, create template if not"""
    env_path = Path(".env")
    if not env_path.exists():
        env_example = Path(".env.example")
        if env_example.exists():
            print("Creating .env file from example...")
            with open(env_example, "r") as src, open(env_path, "w") as dst:
                dst.write(src.read())
        else:
            print("Creating .env file...")
            with open(env_path, "w") as f:
                f.write("OPENAI_API_KEY=your_api_key_here\n")
        
        print("\nPlease update the .env file with your OpenAI API key before running the app.")
        
        # Try to open the .env file for editing
        try:
            if sys.platform == "win32":
                os.system(f"notepad {env_path}")
            elif sys.platform == "darwin":  # macOS
                subprocess.call(["open", "-t", env_path])
            else:  # Linux and other Unix
                if os.environ.get("EDITOR"):
                    subprocess.call([os.environ.get("EDITOR"), env_path])
                else:
                    subprocess.call(["nano", env_path])
        except:
            print(f"Please edit {env_path} manually to add your API key.")
            
        return False
    return True

def run_app():
    """Run the Streamlit app"""
    print("Starting AI Flashcard Generator...")
    
    # Determine the streamlit path based on platform
    if sys.platform == "win32":
        streamlit_path = "venv\\Scripts\\streamlit"
    else:
        streamlit_path = "venv/bin/streamlit"
    
    # Get the port number (default is 8501)
    port = 8501
    
    # Open browser after a short delay
    webbrowser.open(f"http://localhost:{port}")
    
    # Run the app
    subprocess.call([streamlit_path, "run", "app.py"])

def main():
    """Main function to setup and run the app"""
    print("Setting up AI Flashcard Generator...")
    
    env_exists = check_env()
    if not env_exists:
        install_dependencies()
    
    env_file_ok = check_env_file()
    if not env_file_ok:
        input("Press Enter to continue after updating the .env file...")
    
    run_app()

if __name__ == "__main__":
    main()
