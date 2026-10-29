# One image for both the ingest job and the API (docker-compose.yml runs it both ways).

# Build the Vue frontend; only frontend/dist goes into the final image.
FROM node:24-slim AS frontend
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models
WORKDIR /app
# The CPU build of torch, as in CI: the default wheel pulls several GB of CUDA libraries.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY reviewproof/ reviewproof/
COPY samples/ samples/
COPY --from=frontend /frontend/dist frontend/dist
# No data and no .env in the image: data/ is mounted by the ingest service, the API key comes from env_file.
CMD ["python", "-m", "reviewproof.api", "--host", "0.0.0.0", "--port", "8000"]
