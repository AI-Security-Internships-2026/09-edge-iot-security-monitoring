# Single image, used for both the server and the two clients.
# Which one runs is decided by the `command:` in docker-compose.yml.
FROM python:3.11-slim

WORKDIR /workspace

# System deps: tenseal ships manylinux wheels, but keep a compiler
# available as a fallback in case pip has to build from source on
# your architecture (e.g. Apple Silicon / arm64 hosts).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Torch CPU-only wheel — this is a research prototype for an IoT
# gateway, no GPU involved, and the CPU-only wheel is a fraction of
# the size of the default CUDA-bundled one.
RUN pip install --no-cache-dir torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

# App code (client/server orchestration + shared crypto modules)
COPY client.py server.py train_worker.py he_aggregation.py wire_format.py ./

# Model + defence code goes to /app, matching the sys.path.insert(0,"/app")
# used by client.py, server.py and train_worker.py
COPY src/ /app/

ENV PYTHONUNBUFFERED=1

# Default command is overridden per-service in docker-compose.yml
CMD ["python", "server.py"]
