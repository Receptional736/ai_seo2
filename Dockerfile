# ─────────────────────────────────────────────────────────────
# Build stage
# ─────────────────────────────────────────────────────────────
FROM python:3.11-slim

# system packages occasionally needed for wheels
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# put code in /app
WORKDIR /app

# install python deps first (leverages layer-caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# copy the rest of the source
COPY . .

# gradio must listen on all interfaces so Railway can reach it
ENV GRADIO_SERVER_NAME=0.0.0.0

# Railway injects a random $PORT at runtime; expose 7860 for local dev
EXPOSE 7860

# start the app
CMD ["python", "main.py"]
