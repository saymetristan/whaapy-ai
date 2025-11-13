#!/bin/bash
# Script de inicio para Railway que expande variables de entorno
exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
