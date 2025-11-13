# Whaapy AI Service

Servicio de IA independiente para Whaapy - Python + FastAPI + LangChain

## Stack

- **Runtime**: Python 3.11+
- **Framework**: FastAPI
- **Database**: PostgreSQL (schema `ai`)
- **LLM Provider**: OpenAI (gpt-5-mini default)
- **Deploy**: Railway

## Setup Local

```bash
# Crear virtual environment
python -m venv venv
source venv/bin/activate  # Mac/Linux

# Instalar dependencias
pip install -r requirements.txt

# Configurar variables de entorno
cp .env.example .env
# Editar .env con tus credenciales

# Correr servidor
uvicorn app.main:app --reload --port 8000
```

## Endpoints

- `GET /health` - Health check
- `GET /ai/config/{business_id}` - Obtener configuración de agente
- `PUT /ai/config/{business_id}` - Actualizar configuración de agente

## Deploy a Railway

El deploy es automático al hacer push a `main`.

Variables de entorno requeridas:
- `DATABASE_URL`
- `AI_SERVICE_TOKEN`
- `BACKEND_URL`
- `OPENAI_API_KEY`

## URLs de Producción

- **Health check**: https://ai.whaapy.com/health
- **API**: https://ai.whaapy.com/ai/*
