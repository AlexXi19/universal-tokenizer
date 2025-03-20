FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install gunicorn

COPY . .

EXPOSE 8080

ENV FLASK_APP=run.py
ENV FLASK_ENV=production
ENV WORKERS=4
ENV PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus_multiproc

# Create directory for multiprocess metrics
RUN mkdir -p /tmp/prometheus_multiproc && chmod 777 /tmp/prometheus_multiproc

CMD ["gunicorn", "--config", "gunicorn_config.py", "run:app"]
