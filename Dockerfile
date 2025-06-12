# Use a specific Python version image
FROM python:3.12.0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    unzip \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install gdown (Python tool for Google Drive)
RUN pip install gdown

# Download the models.zip using gdown
RUN gdown https://drive.google.com/uc?id=1u4hNEYS-GbaqR1_T0-UpqtszVXlzKUcy

# Unzip the model
RUN unzip models.zip -d /models && rm models.zip

# Copy requirements.txt first to leverage Docker caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install faiss-cpu
RUN pip install faiss-cpu

# Copy application files
COPY . .

# Explicitly ensure models directory exists and has the right permissions
RUN mkdir -p /app/models/models && chmod -R 755 /app/models/models

# Expose the port your app will run on
EXPOSE 8000

# Run the FastAPI application with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
