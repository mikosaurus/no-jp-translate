FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variables
ENV PYTHONPATH=/app
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface

# Create cache directory
RUN mkdir -p /app/.cache/huggingface

# Copy model download script and run it
COPY download_models.py .
RUN python download_models.py

# Copy application code
COPY main.py .

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "main.py"]
