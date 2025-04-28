# Use a specific Python version image
FROM python:3.12.0

# Install system dependencies (OpenCV needs libGL)
RUN apt-get update && apt-get install -y libgl1

# Set the working directory to /app
WORKDIR /app

# Copy requirements.txt first to leverage Docker caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the necessary application files (assuming your app is in the current directory)
COPY . .

# Expose the port your app will run on
EXPOSE 8000

# Run the FastAPI application with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
