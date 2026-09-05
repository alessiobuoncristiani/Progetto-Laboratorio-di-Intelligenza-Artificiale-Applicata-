FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt
COPY . .

EXPOSE 5000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/api/health', timeout=3)" || exit 1
CMD ["sh", "-c", "if [ ! -f models/diabetes_model.joblib ]; then python -m src.train; fi; exec gunicorn --worker-class gthread --threads 4 --timeout 60 --keep-alive 2 --bind 0.0.0.0:5000 app.app:app"]
