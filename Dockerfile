# Use a stable Debian-based Python image
FROM python:3.10-slim

# Install system-level graphics drivers required by OpenCV/MediaPipe
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set up your app
WORKDIR /app
COPY . .

# Install your Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run the app
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501"]