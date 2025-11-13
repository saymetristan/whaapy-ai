FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

# Railway provee PORT automáticamente, default 8000 para local
EXPOSE 8000

# Usar shell form para que la variable se expanda correctamente
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
