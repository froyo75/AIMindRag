# Install from slim Python image
FROM python:3.13-slim

LABEL maintainer="froyo75@users.noreply.github.com"

# Set working directory
WORKDIR /app

# Copy requirements and project files
COPY requirements.txt .
COPY config/ ./config
COPY modules/ ./modules
COPY utils/ ./utils
COPY main.py .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run the app
ENTRYPOINT ["python", "main.py"]