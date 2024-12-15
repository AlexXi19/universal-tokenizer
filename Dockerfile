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

CMD ["sh", "-c", "gunicorn --preload -w $WORKERS -b 0.0.0.0:8080 run:app"]
