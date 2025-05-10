FROM python:3.12-slim

# Set working directory
WORKDIR /Visualizer

# Install system dependencies needed for OpenCV
RUN apt-get update && \
    apt-get install -y libgl1 libglib2.0-0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy Python dependencies
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Run the app
CMD ["python", "app.py"]
