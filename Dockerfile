FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY start.sh .

# Dar permisos de ejecución al script
RUN chmod +x start.sh

EXPOSE 8000

# Usar el script como punto de entrada
CMD ["./start.sh"]
