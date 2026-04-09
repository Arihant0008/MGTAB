FROM python:3.11-slim

WORKDIR /code

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip first
RUN pip install --no-cache-dir --upgrade pip

# Step 1: Install PyTorch CPU FIRST (before anything else)
RUN pip install --no-cache-dir torch==2.2.0+cpu --index-url https://download.pytorch.org/whl/cpu

# Step 2: Install numpy (must be <2.0 for torch 2.1/2.2 compat)
RUN pip install --no-cache-dir "numpy<2.0.0"

# Step 3: Install torch-geometric
RUN pip install --no-cache-dir torch-geometric>=2.4.0

# Step 4: Install transformers and sentence-transformers
RUN pip install --no-cache-dir transformers==4.44.0 sentence-transformers==3.0.0

# Step 5: Copy requirements and install remaining deps (fastapi, uvicorn, etc.)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Verify torch is importable (fail fast if broken)
RUN python -c "import torch; print('PyTorch version:', torch.__version__); from transformers import AutoModel; print('Transformers OK')"

# Create a non-root user (Required by HuggingFace Spaces)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONPATH=/code

# Set working directory for app
WORKDIR $HOME/app

# Copy application files and model
COPY --chown=user . $HOME/app/

# Expose port (Hugging Face routes 7860 automatically)
EXPOSE 7860

# Command to run FastAPI server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
