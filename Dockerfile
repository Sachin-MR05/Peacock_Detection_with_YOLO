# Base image with Python
FROM python:3.11-slim

# Create working directory
WORKDIR /app

# Install system dependencies required for OpenCV / video handling
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (leverage Docker layer caching)
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Hugging Face Spaces expects the app to listen on this port
ENV PORT=7860

# Default to lite mode in the container (safe for free CPU tier).
# You can override this in the Space settings to "full" if desired.
ENV APP_MODE=lite

# Expose port (useful for local Docker runs)
EXPOSE 7860

# Start the Flask app bound to 0.0.0.0:$PORT so Spaces can reach it
CMD ["python", "-m", "flask", "run", "--host=0.0.0.0", "--port=7860", "--app=app:app"]
