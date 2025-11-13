from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import health, agent, embeddings, chat, ai_operations
from app.config import settings

app = FastAPI(
    title="Whaapy AI Service",
    version="1.0.0",
    description="Servicio de IA para Whaapy con LangChain + LangGraph"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.backend_url, "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes públicas
app.include_router(health.router, tags=["health"])

# Routes protegidas (requieren token)
app.include_router(
    agent.router, 
    prefix="/ai", 
    tags=["agent"]
)

app.include_router(
    embeddings.router,
    prefix="/ai",
    tags=["embeddings"]
)

app.include_router(
    chat.router,
    prefix="/ai",
    tags=["chat"]
)

app.include_router(
    ai_operations.router,
    prefix="/ai",
    tags=["ai-operations"]
)

@app.get("/")
async def root():
    return {
        "service": "whaapy-ai",
        "version": "1.0.0",
        "status": "healthy"
    }
