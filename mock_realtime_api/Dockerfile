# syntax=docker/dockerfile:1.4
FROM python:3.11-slim

WORKDIR /app

# Step 1: Copy only requirements.txt first to leverage Docker cache
COPY requirements.txt .

# Step 2: Install dependencies with pip cache enabled
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

# Step 3: Copy the rest of the application
COPY . .

# Step 4: Start the API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
