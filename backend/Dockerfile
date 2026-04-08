FROM python:3.11-slim

WORKDIR /code

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install dependencies (install PyTorch CPU specifically to save space)
RUN pip install --no-cache-dir torch==2.1.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install --no-cache-dir torch-geometric
RUN pip install --no-cache-dir -r requirements.txt

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
