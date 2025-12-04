#!/usr/bin/env python3
"""Launcher script for Streamlit app that ensures proper Python path."""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Now import and run the app
from src.app import main

if __name__ == "__main__":
    main()
