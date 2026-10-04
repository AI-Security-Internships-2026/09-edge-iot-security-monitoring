# CPU-only image for the E8 harness. Replace the pins below with the versions from YOUR environment
# (run `pip freeze | grep -i -E "torch|opacus|numpy|pandas|scikit|scipy|psutil"`), so the container matches your experiments.
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN pip install --no-cache-dir torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir opacus==1.5.2 numpy==1.26.4 scipy==1.13.1 pandas==2.2.2 scikit-learn==1.5.0 psutil==5.9.8
WORKDIR /app
