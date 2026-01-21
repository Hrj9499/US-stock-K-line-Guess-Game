FROM python:3.11-slim

WORKDIR /app

# System deps (optional but helps with some wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY . /app

RUN pip install -U pip && pip install -e .

EXPOSE 8000

# Default provider can be switched at runtime
ENV UKG_DATA_PROVIDER=stooq

CMD ["uvicorn", "us_kline_guess.api:app", "--host", "0.0.0.0", "--port", "8000"]
