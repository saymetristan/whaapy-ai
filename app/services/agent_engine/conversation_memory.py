"""
Conversation Memory Service - Genera y gestiona resúmenes de conversaciones.

Sprint 3: Permite al orchestrator tener contexto de 30-50 mensajes previos
sin consumir 1000+ tokens, usando summarization con gpt-5-mini.
"""

import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from langchain_core.messages import BaseMessage
from app.db.database import get_db
from app.services.llm_tracker import LLMCallTracker
from app.services.agent_engine.llm_factory import LLMFactory


# JSON Schema para structured output del summary
SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "text": {
            "type": "string",
            "description": "Resumen de 2-3 párrafos de la conversación"
        },
        "topics": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Lista de temas principales discutidos"
        }
    },
    "required": ["text", "topics"],
    "additionalProperties": False
}


SUMMARIZATION_SYSTEM_PROMPT = """Eres un experto en resumir conversaciones entre clientes y agentes de IA.

Tu objetivo es crear un resumen CONCISO Y ÚTIL que capture:

1. **Contexto general**: ¿De qué trata la conversación?
2. **Necesidades del cliente**: ¿Qué busca o necesita?
3. **Temas discutidos**: ¿Qué tópicos se han tratado?
4. **Decisiones/Acuerdos**: ¿Qué se ha decidido o resuelto?
5. **Estado actual**: ¿En qué punto está la conversación?

**FORMATO**:
- 2-3 párrafos máximo (150-250 palabras)
- Lenguaje claro y directo
- Enfocado en información útil para continuar la conversación

**EVITAR**:
- Detalles irrelevantes
- Repeticiones
- Mensajes de saludo/despedida sin contenido

Resume la siguiente conversación en JSON estructurado."""


async def get_or_create_summary(
    conversation_id: str,
    messages: List[BaseMessage],
    business_id: str,
    execution_id: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Obtener summary existente o generar uno nuevo si es necesario.
    
    Criterios de refresh:
    - No existe summary previo → generar
    - Han pasado 10+ mensajes desde último summary → generar
    - Summary tiene >24h de antigüedad → generar
    - Caso contrario → retornar summary cached
    
    Args:
        conversation_id: ID de la conversación
        messages: Lista de mensajes actuales (para contar)
        business_id: ID del negocio (para tracking)
        execution_id: ID de la ejecución actual (opcional)
    
    Returns:
        Dict con {text, last_updated_at, message_count, topics} o None si error
    """
    try:
        # 1. Cargar summary existente de la BD
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT summary 
                FROM conversations 
                WHERE id = %s
            """, (conversation_id,))
            
            result = cursor.fetchone()
            cursor.close()
            
            existing_summary = result['summary'] if result and result['summary'] else None
        
        # 2. Determinar si necesitamos refresh
        current_message_count = len(messages)
        needs_refresh = False
        
        if existing_summary is None:
            # Caso 1: No hay summary previo
            needs_refresh = True
            print(f"📝 No hay summary previo para conversación {conversation_id[:8]}... → generar nuevo")
        
        elif current_message_count < 5:
            # Caso 2: Muy pocos mensajes para resumir
            print(f"⏭️ Solo {current_message_count} mensajes → no generar summary todavía")
            return None
        
        else:
            # Caso 3: Verificar criterios de refresh
            last_summary_count = existing_summary.get('message_count', 0)
            last_updated_str = existing_summary.get('last_updated_at')
            
            # Criterio A: 10+ mensajes nuevos
            messages_since_summary = current_message_count - last_summary_count
            if messages_since_summary >= 10:
                needs_refresh = True
                print(f"📝 {messages_since_summary} mensajes nuevos desde último summary → refresh")
            
            # Criterio B: Summary tiene >24h
            if last_updated_str:
                try:
                    last_updated = datetime.fromisoformat(last_updated_str)
                    age_hours = (datetime.now() - last_updated).total_seconds() / 3600
                    if age_hours > 24:
                        needs_refresh = True
                        print(f"⏰ Summary tiene {age_hours:.1f}h de antigüedad → refresh")
                except:
                    pass
        
        # 3. Si no necesita refresh, retornar existing
        if not needs_refresh and existing_summary:
            print(f"✅ Usando summary cached ({existing_summary.get('message_count', 0)} msgs)")
            return existing_summary
        
        # 4. Generar nuevo summary
        if needs_refresh:
            new_summary = await generate_summary(
                messages=messages,
                business_id=business_id,
                execution_id=execution_id
            )
            
            if new_summary:
                # 5. Guardar en BD
                await save_summary(
                    conversation_id=conversation_id,
                    summary=new_summary,
                    message_count=current_message_count
                )
                
                return new_summary
        
        return None
        
    except Exception as e:
        print(f"❌ Error en get_or_create_summary: {e}")
        # No crashear el agente, retornar None
        return None


async def generate_summary(
    messages: List[BaseMessage],
    business_id: str,
    execution_id: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Generar summary de conversación usando gpt-5-mini.
    
    Args:
        messages: Lista de mensajes a resumir (últimos 50 o todos)
        business_id: ID del negocio (para tracking)
        execution_id: ID de la ejecución (opcional)
    
    Returns:
        Dict con {text: str, topics: List[str]} o None si error
    """
    try:
        # 1. Tomar últimos 50 mensajes (o todos si son menos)
        messages_to_summarize = messages[-50:] if len(messages) > 50 else messages
        
        # 2. Formatear mensajes para el prompt
        conversation_text = "\n".join([
            f"{'Cliente' if msg.type == 'human' else 'Asistente'}: {msg.content}"
            for msg in messages_to_summarize
        ])
        
        # 3. Crear LLM client
        llm_factory = LLMFactory()
        openai_client = llm_factory.create_openai_client()
        
        # 4. Trackear llamada con LLMCallTracker
        async with LLMCallTracker(
            business_id=business_id,
            operation_type="summarization",
            provider="openai",
            model="gpt-5-mini",
            execution_id=execution_id,
            operation_context={"message_count": len(messages_to_summarize)},
            reasoning_effort="low"
        ) as tracker:
            
            # 5. Llamar a gpt-5-mini con structured output
            response = openai_client.responses.create(
                model="gpt-5-mini",
                reasoning={
                    "effort": "low"  # Balance costo/calidad
                },
                text={
                    "verbosity": "low",
                    "format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "conversation_summary",
                            "strict": True,
                            "schema": SUMMARY_SCHEMA
                        }
                    }
                },
                messages=[
                    {"role": "system", "content": SUMMARIZATION_SYSTEM_PROMPT},
                    {"role": "user", "content": f"CONVERSACIÓN:\n\n{conversation_text}"}
                ]
            )
            
            # 6. Registrar tokens
            tracker.record(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens
            )
            
            # 7. Parsear respuesta
            summary_json = json.loads(response.choices[0].message.content)
            
            print(f"✅ Summary generado: {len(summary_json['text'])} chars, {len(summary_json['topics'])} topics")
            print(f"   Topics: {', '.join(summary_json['topics'][:3])}...")
            
            return summary_json
            
    except Exception as e:
        print(f"❌ Error generando summary: {e}")
        return None


async def save_summary(
    conversation_id: str,
    summary: Dict[str, Any],
    message_count: int
) -> bool:
    """
    Guardar summary en la BD (columna conversations.summary).
    
    Args:
        conversation_id: ID de la conversación
        summary: Dict con {text, topics}
        message_count: Cantidad de mensajes al momento del summary
    
    Returns:
        True si guardó exitosamente, False si error
    """
    try:
        # Agregar metadata al summary
        full_summary = {
            **summary,
            "message_count": message_count,
            "last_updated_at": datetime.now().isoformat()
        }
        
        # Guardar en BD
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE conversations
                SET summary = %s
                WHERE id = %s
            """, (json.dumps(full_summary), conversation_id))
            
            conn.commit()
            cursor.close()
        
        print(f"✅ Summary guardado en BD para conversación {conversation_id[:8]}...")
        return True
        
    except Exception as e:
        print(f"❌ Error guardando summary: {e}")
        # No crashear el agente
        return False

