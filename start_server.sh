#!/bin/bash
# Script to start the FastAPI server

echo "Starting ML Training Pipeline Server..."

# Create necessary directories
mkdir -p models data logs

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

# Upgrade pip first
echo "Upgrading pip..."
python -m pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Load environment variables from .env
if [ -f ".env" ]; then
    echo "Loading environment variables..."
    set -a
    source .env
    set +a
else
    echo "Warning: .env file not found"
fi

# Start the server
echo "Starting server on http://localhost:8000"
echo "API documentation available at http://localhost:8000/docs"
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
