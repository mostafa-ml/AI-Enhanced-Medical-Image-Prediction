# Use a specific Python version image
FROM python:3.12.0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    unzip \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements.txt first to leverage Docker caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install gdown (Python tool for Google Drive)
RUN pip install gdown

# Download all models files from the Google Drive folder
RUN gdown --folder https://drive.google.com/drive/folders/15B9pvy0oSYqNBAY-Yi2XIyBXusbGjnEw -O models

# Copy application files
COPY . .

# Apply permissions correctly
RUN chmod -R 755 /app/models || echo "ERROR in Dockerfile >> models folder not found"

# Expose the port your app will run on
EXPOSE 8000

# Run the FastAPI app with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
