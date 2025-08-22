# Base image
FROM python:3.11-slim

# Set working directory
WORKDIR /opt/genealpha-backend-training

# Prevent Python from writing pyc files and enable stdout/stderr unbuffered
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/opt/genealpha-backend-training

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt uvicorn pydantic-settings

# Copy entire project
COPY . .

# Fix line endings and make scripts executable
#RUN sed -i 's/\r$//' start_server.sh && \
 #   chmod +x start_server.sh && \
  #  chmod +x scripts/run_training.py

# Clean up
RUN apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

EXPOSE 8000

# Directly run the server without shell script
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

