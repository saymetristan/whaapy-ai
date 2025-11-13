from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from app.middleware.auth import verify_token
from app.services.agent_engine.llm_factory import LLMFactory
import json
import re
import os
from openai import OpenAI

router = APIRouter()

# ============================================
# HELPERS
# ============================================

def is_gpt5_model(model: str) -> bool:
    """
    Verifica si un modelo soporta reasoning controls (GPT-5 family).
    
    Según la documentación de OpenAI:
    - gpt-5, gpt-5-mini, gpt-5-nano soportan reasoning.effort y text.verbosity
    - gpt-4o, gpt-4o-mini, gpt-4, gpt-3.5 NO soportan estos parámetros
    
    Los nombres de API son:
    - gpt-5 (system card: gpt-5-thinking)
    - gpt-5-mini (system card: gpt-5-thinking-mini)
    - gpt-5-nano (system card: gpt-5-thinking-nano)
    - gpt-5-chat-latest (system card: gpt-5-main)
    """
    gpt5_models = ['gpt-5', 'gpt-5-mini', 'gpt-5-nano', 'gpt-5-chat-latest']
    return any(model.startswith(m) for m in gpt5_models)

# ============================================
# SCHEMAS
# ============================================

class AnalyzePromptRequest(BaseModel):
    prompt: str = Field(..., max_length=5000)
    business_name: str

class AnalyzePromptResponse(BaseModel):
    score: int  # 0-100
    strengths: List[str]
    weaknesses: List[str]
    suggestions: List[str]
    tone: str  # formal, casual, mixed, unclear
    completeness: int  # 0-100

class GenerateSuggestionRequest(BaseModel):
    system_prompt: str
    conversation_history: List[Dict[str, str]]  # [{"role": "user", "content": "..."}, ...]
    model: Optional[str] = "gpt-5-mini"

class GenerateSuggestionResponse(BaseModel):
    suggestion: str
    confidence: int  # 0-100

class ExtractDocumentRequest(BaseModel):
    page_images: List[str]  # base64 images
    max_pages: Optional[int] = 10

class ExtractDocumentResponse(BaseModel):
    extracted_text: str
    pages_processed: int

# ============================================
# ENDPOINTS
# ============================================

@router.post("/analyze-prompt", response_model=AnalyzePromptResponse)
async def analyze_prompt(
    request: AnalyzePromptRequest,
    _: bool = Depends(verify_token)
):
    """
    Analiza un system prompt y retorna feedback detallado.
    Migrado desde whaapy-backend/src/services/openai-client.ts
    """
    client = LLMFactory.create_responses_client()
    
    analysis_input = f"""Eres un experto en diseño de system prompts para agentes de IA de atención al cliente.
Analiza el prompt del usuario y proporciona feedback constructivo y accionable.

Prompt del negocio "{request.business_name}":
{request.prompt}

Evalúa:
1. Claridad de instrucciones
2. Definición de tono de comunicación
3. Políticas y procedimientos mencionados
4. Manejo de casos edge y situaciones difíciles
5. Estructura general y organización

Responde en JSON con este formato exacto:
{{
  "score": <número 0-100>,
  "strengths": ["fortaleza 1", "fortaleza 2"],
  "weaknesses": ["debilidad 1", "debilidad 2"],
  "suggestions": ["sugerencia 1", "sugerencia 2"],
  "tone": "<formal|casual|mixed|unclear>",
  "completeness": <número 0-100>
}}"""
    
    try:
        # Responses API es SÍNCRONA, no usar await
        model = "gpt-5-mini"
        
        # Solo usar reasoning/text si el modelo soporta GPT-5 controls
        if is_gpt5_model(model):
            response = client.responses.create(
                model=model,
                input=analysis_input,
                reasoning={ "effort": "low" },
                text={ "verbosity": "low" }
            )
        else:
            # Fallback para modelos no-GPT5 (sin reasoning controls)
            response = client.responses.create(
                model=model,
                input=analysis_input
            )
        
        analysis = json.loads(response.output_text)
        
        # Validación básica
        required_keys = ['score', 'strengths', 'weaknesses', 'suggestions', 'tone', 'completeness']
        if not all(k in analysis for k in required_keys):
            raise ValueError("Invalid response format")
        
        print(f"✅ Prompt analizado: Score {analysis['score']}/100")
        
        return AnalyzePromptResponse(**analysis)
        
    except Exception as e:
        print(f"Error analyzing prompt: {e}")
        raise HTTPException(status_code=500, detail=f"Error analyzing prompt: {str(e)}")


@router.post("/generate-suggestion", response_model=GenerateSuggestionResponse)
async def generate_suggestion(
    request: GenerateSuggestionRequest,
    _: bool = Depends(verify_token)
):
    """
    Genera una sugerencia de respuesta para una conversación.
    Migrado desde whaapy-backend/src/services/openai-client.ts
    """
    client = LLMFactory.create_responses_client()
    
    # Construir input en formato conversacional
    conversation_text = f"System: {request.system_prompt}\n\n"
    
    for msg in request.conversation_history:
        role = msg.get('role', 'user').capitalize()
        content = msg.get('content', '')
        conversation_text += f"{role}: {content}\n"
    
    try:
        # Responses API es SÍNCRONA, no usar await
        # Solo usar reasoning/text si el modelo soporta GPT-5 controls
        if is_gpt5_model(request.model):
            response = client.responses.create(
                model=request.model,
                input=conversation_text,
                reasoning={ "effort": "medium" },
                text={ "verbosity": "low" }
            )
        else:
            # Fallback para modelos no-GPT5 (sin reasoning controls)
            response = client.responses.create(
                model=request.model,
                input=conversation_text
            )
        
        suggestion = response.output_text
        
        # Calcular confidence score
        confidence = calculate_confidence(suggestion)
        
        print(f"✅ Sugerencia generada: {len(suggestion)} caracteres, confianza {confidence}%")
        
        return GenerateSuggestionResponse(
            suggestion=suggestion,
            confidence=confidence
        )
        
    except Exception as e:
        print(f"Error generating suggestion: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating suggestion: {str(e)}")


@router.post("/extract-document", response_model=ExtractDocumentResponse)
async def extract_document(
    request: ExtractDocumentRequest,
    _: bool = Depends(verify_token)
):
    """
    Extrae texto de imágenes de documentos usando visión (OCR).
    Migrado desde whaapy-backend/src/services/document-extractor.ts
    
    IMPORTANTE: Responses API NO soporta visión aún.
    Este endpoint usa Chat Completions como excepción.
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    extracted_text = ""
    pages_processed = 0
    
    for idx, base64_image in enumerate(request.page_images[:request.max_pages]):
        try:
            print(f"📄 Procesando página {idx + 1}...")
            
            # Chat Completions con visión (gpt-5-mini soporta visión)
            response = client.chat.completions.create(
                model="gpt-5-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Extract all text from this document image. Return only the text content, no explanations. If no text, return 'NO_TEXT_FOUND'."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=4096
            )
            
            page_text = response.choices[0].message.content.strip()
            
            if page_text and page_text != 'NO_TEXT_FOUND':
                extracted_text += f"\n\n=== Página {idx + 1} ===\n\n{page_text}"
                pages_processed += 1
                print(f"✅ Página {idx + 1} procesada")
                
        except Exception as e:
            print(f"❌ Error procesando página {idx + 1}: {e}")
    
    if not extracted_text:
        raise HTTPException(status_code=400, detail="No se pudo extraer texto del documento")
    
    print(f"✅ Documento extraído: {pages_processed} páginas procesadas")
    
    return ExtractDocumentResponse(
        extracted_text=extracted_text.strip(),
        pages_processed=pages_processed
    )


# ============================================
# HELPERS
# ============================================

def calculate_confidence(suggestion: str) -> int:
    """Calcula un score de confianza basado en la calidad de la sugerencia"""
    if not suggestion or not suggestion.strip():
        return 0
    
    confidence = 50  # Base
    
    # Más contenido = más confianza
    if len(suggestion) > 100:
        confidence += 10
    if len(suggestion) > 200:
        confidence += 10
    
    # Tiene puntuación adecuada
    if re.search(r'[.!?]', suggestion):
        confidence += 10
    
    # No tiene marcadores de incertidumbre
    if not re.search(r'no estoy seguro|quizás|tal vez|podría ser', suggestion, re.IGNORECASE):
        confidence += 10
    
    # Tiene estructura (múltiples oraciones)
    sentences = re.split(r'[.!?]', suggestion)
    sentences = [s for s in sentences if s.strip()]
    if len(sentences) >= 2:
        confidence += 10
    
    return min(100, confidence)

