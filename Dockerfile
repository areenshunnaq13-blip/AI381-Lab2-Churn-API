
# Stage 1: Install dependencies
FROM python:3.11-slim@sha256:673719e071ecf27434fd2635bbd8d00f49f4fdd81f243143438a42ced8bedf60 AS builder

RUN apt-get update && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python -m pip install --no-cache-dir --upgrade \
    "setuptools>=78.1.1" "wheel>=0.46.2"

COPY requirements.txt .
RUN python -m pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: Runtime image
FROM python:3.11-slim@sha256:673719e071ecf27434fd2635bbd8d00f49f4fdd81f243143438a42ced8bedf60

RUN apt-get update && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python -m pip install --no-cache-dir --upgrade \
    "setuptools>=78.1.1" "wheel>=0.46.2"

COPY --from=builder /install /usr/local
COPY . .

RUN useradd --create-home appuser

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)" || exit 1

CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
