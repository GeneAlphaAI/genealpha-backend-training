#!/bin/bash
# Script to start the FastAPI server

echo "Starting ML Training Pipeline Server..."

# Create necessary directories
mkdir -p models data logs

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Export environment variables from .env if it exists
if [ -f ".env" ]; then
    export $(cat .env | xargs)
fi

# Start the server
echo "Starting server on http://localhost:8000"
echo "API documentation available at http://localhost:8000/docs"
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
