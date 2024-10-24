FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src

COPY requirements.txt ./
COPY app/ ./app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000 5432

CMD ["sh", "-c", "quart run --host 0.0.0.0 --port 8000"]

