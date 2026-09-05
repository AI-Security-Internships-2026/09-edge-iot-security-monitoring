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

# Without this, Python fully buffers stdout when it's not a real TTY
# (which it isn't under Docker's log capture) -- print() output would
# silently sit in a buffer instead of reaching `docker compose logs`
# until the process exits or the buffer fills. This makes every print
# show up immediately, which matters a lot for a script whose whole
# point is timing/progress visibility.
ENV PYTHONUNBUFFERED=1

COPY src/ /app/

ENTRYPOINT ["python"]
