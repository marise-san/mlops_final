# Use of the slim Python runtime as specified
FROM python:3.9-slim

# Set working directory inside the container
WORKDIR /usr/src/app

# Set environment variables to optimize Python runtime
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install minimal OS packages if required by C-extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first to cache Python dependency layers
COPY requirements.txt .

# Install Python dependencies cleanly without cache
RUN pip install --no-cache-dir -r requirements.txt

# Copy the actual application components and trained model. 
# Exclusdes unneeded layers like .git or tests based on local setup logic
COPY app/ ./app/
COPY models/ ./models/

# Expose the API port targeted by FastAPI default
EXPOSE 8000

# Instruct Uvicorn to run and bind externally for Docker
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
