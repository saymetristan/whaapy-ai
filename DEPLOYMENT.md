# Deployment Instructions - whaapy-ai

## Railway Deployment

El servicio whaapy-ai está diseñado para deployarse en Railway con deploy automático desde GitHub.

### Configuración Manual Requerida

1. **Crear Servicio en Railway Dashboard**
   - Ir a https://railway.app
   - Crear nuevo proyecto o agregar servicio al proyecto existente
   - Conectar al repo de GitHub: `saymetristan/whaapy-ai`
   - Railway detectará automáticamente el `railway.toml` y `requirements.txt`

2. **Configurar Variables de Entorno**

   Agregar las siguientes variables en Railway dashboard:

   ```
   DATABASE_URL=postgresql://...  (mismo que whaapy-backend)
   AI_SERVICE_TOKEN=92551104-a8e8-403f-a37c-e874bf2189ed
   BACKEND_URL=https://api.whaapy.com
   OPENAI_API_KEY=sk-proj-...  (mismo que whaapy-backend)
   ```

3. **Generar Dominio Custom**
   - En Railway dashboard, ir a Settings del servicio
   - Agregar dominio custom: `ai.whaapy.com`
   - Configurar DNS record en el proveedor de dominio

4. **Verificar Deploy**
   - El deploy se ejecuta automáticamente al hacer push a `main`
   - Verificar health check: `https://ai.whaapy.com/health`
   - Debe retornar: `{"service": "whaapy-ai", "status": "healthy", "database": "healthy"}`

## Backend Configuration

El backend (`whaapy-backend`) ya está configurado con:
- `AI_SERVICE_URL=https://ai.whaapy.com`
- `AI_SERVICE_TOKEN=92551104-a8e8-403f-a37c-e874bf2189ed`

## Testing

### Test Health Check

```bash
curl https://ai.whaapy.com/health
```

### Test Config Endpoint (requiere token)

```bash
curl https://ai.whaapy.com/ai/config/{business_id} \
  -H "Authorization: Bearer 92551104-a8e8-403f-a37c-e874bf2189ed"
```

## Troubleshooting

### Deploy Fails

Si el deploy falla en Railway:
1. Verificar logs en Railway dashboard
2. Asegurar que todas las variables de entorno estén configuradas
3. Verificar que `DATABASE_URL` sea accesible desde Railway

### Database Connection Error

Si hay errores de conexión a DB:
1. Verificar que `DATABASE_URL` incluya el pooler de Supabase
2. Verificar que el servicio tenga acceso a internet para conectarse a Supabase
3. Verificar que las policies RLS estén configuradas correctamente en schema `ai`

### Authentication Errors

Si el backend no puede conectarse al servicio AI:
1. Verificar que `AI_SERVICE_TOKEN` sea el mismo en ambos servicios
2. Verificar que `AI_SERVICE_URL` apunte al dominio correcto
3. Verificar logs del servicio AI en Railway

## Status

- ✅ Repo creado en GitHub
- ✅ Schema `ai` creado en Supabase
- ✅ Código del servicio implementado
- ⏳ Pendiente: Configuración manual de servicio en Railway dashboard
- ⏳ Pendiente: Configuración de dominio custom ai.whaapy.com
