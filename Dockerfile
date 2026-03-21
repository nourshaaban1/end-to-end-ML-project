# Use Python slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (build-essential for some native python deps)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Optimize Layer Cache: Copy setup and requirements first
COPY requirements.txt setup.py ./

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY artifacts/*.pkl artifacts/

# Copy the rest of the application code
COPY application.py .
COPY src/ ./src/
COPY templates/ ./templates/

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Expose port (Internal documentation)
EXPOSE 5000

# Run the Flask app
CMD ["python", "application.py"]
