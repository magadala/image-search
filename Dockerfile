# Use slim Python image for smaller size
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV KMP_DUPLICATE_LIB_OK=TRUE

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project
COPY . .

# Expose FastAPI port
EXPOSE 8080

# Default command to run FastAPI API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
