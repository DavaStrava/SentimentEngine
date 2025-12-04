#!/bin/bash
# Launcher script for Streamlit app

# Activate virtual environment
source venv/bin/activate

# Set Python path to include project root
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Run Streamlit
streamlit run src/app.py
