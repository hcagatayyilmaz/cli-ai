# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
  build-essential \
  curl \
  cmake \
  git \
  python3-dev \
  && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy only pyproject.toml / poetry.lock first so Docker can cache deps
COPY pyproject.toml poetry.lock ./

# Install Poetry
RUN pip install --no-cache-dir poetry \
  && poetry config virtualenvs.create false

# Install llama-cpp-python with CPU support
#   If you are on an x86 CPU host, you typically just need:
RUN pip install --no-cache-dir "llama-cpp-python[server]"

# Install project dependencies (will pull in what's in pyproject.toml)
RUN poetry install --no-root --only main

# Now copy your source code
COPY . .

# Expose the port for the server
EXPOSE 8000

# By default, run the server. Adjust if you have a different entrypoint
CMD ["python", "server.py"]
