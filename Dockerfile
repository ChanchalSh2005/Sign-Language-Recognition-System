FROM python:3.12-slim

# 1. Install Linux dependencies
RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 libgles2 libegl1 wget && rm -rf /var/lib/apt/lists/*

# 2. Set up the exact user Hugging Face requires (UID 1000)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# 3. Copy files and give our user permission to read them
COPY --chown=user . $HOME/app

# 4. Pre-download the MediaPipe models here to prevent read-only crashes
RUN wget -q https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
RUN wget -q https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task

# 5. FORCE the lightweight CPU version of PyTorch
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# 6. Install the rest of the packages
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 7860
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]
