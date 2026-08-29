# Constrained-client image. Deliberately does NOT install pandas/scikit-
# learn -- matches model_defs.py's documented "no pandas/sklearn in the
# constrained subprocess" constraint. Only the offline partition builder
# (run on the host, not in this image) needs those.
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-container.txt .
RUN pip install --no-cache-dir -r requirements-container.txt

COPY src/ /app/

ENTRYPOINT ["python"]
