FROM python:3.9-slim

WORKDIR /app

# Install system dependencies required for pygame and rendering
RUN apt-get update && apt-get install -y \
    libx11-6 \
    libxext6 \
    libxrender-dev \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .