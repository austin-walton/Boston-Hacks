#!/usr/bin/env python3
"""
Setup script for Pac-Man ML Project
Run this to set up the environment and create necessary directories
"""

import os
import subprocess
import sys

def create_directories():
    """Create necessary directories"""
    dirs = ['models', 'pacman_ai/levels', 'pacman_ai/assets']
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")

def install_dependencies():
    """Install required packages"""
    try:
        print("Installing dependencies...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Dependencies installed successfully!")
    except subprocess.CalledProcessError:
        print("Failed to install dependencies. Please run manually:")
        print("pip install -r requirements.txt")

def create_sample_level():
    """Create a sample level file"""
    try:
        from utils import create_level_file
        create_level_file("sample_level.txt")
        print("Sample level created!")
    except ImportError:
        print("Could not create sample level (utils.py not available yet)")

def main():
    print("Pac-Man ML Project Setup")
    print("=" * 30)
    
    print("\nCreating directories...")
    create_directories()
    
    print("\nInstalling dependencies...")
    install_dependencies()
    
    print("\nCreating sample level...")
    create_sample_level()
    
    print("\nSetup complete!")
    print("\nReady to run! Try these commands:")
    print("  python quick_demo.py          # Quick demo (2-3 minutes)")
    print("  python run_game.py            # Manual gameplay")
    print("  python panner.py --help       # See all options")

if __name__ == "__main__":
    main()
