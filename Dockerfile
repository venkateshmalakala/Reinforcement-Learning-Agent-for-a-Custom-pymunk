FROM python:3.9-slim

WORKDIR /app

# Install system dependencies for pygame, X11, and ffmpeg
# Using more generic package names to ensure compatibility
RUN apt-get update && apt-get install -y \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libxinerama1 \
    libxi6 \
    libxrandr2 \
    libxcursor1 \
    libxtst6 \
    libtk8.6 \
    python3-tk \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create directories for outputs
RUN mkdir -p logs models media

CMD ["python", "train.py"]