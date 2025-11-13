from typing import Optional, Dict, Any
from app.db.database import get_db
from datetime import datetime


class AgentConfigManager:
    """Manager para configuración de agentes en schema ai"""
    
    def get_config(self, business_id: str) -> Optional[Dict[str, Any]]:
        """Obtener configuración del agente desde schema ai"""
        with get_db() as conn:
            cursor = conn.cursor()
            
            # Nota: search_path ya está configurado como "ai,public"
            # por lo que agent_config se resuelve a ai.agent_config
            cursor.execute("""
                SELECT id, business_id, system_prompt, provider, model, 
                       temperature, max_tokens, enabled, created_at, updated_at
                FROM agent_config
                WHERE business_id = %s
            """, (business_id,))
            
            result = cursor.fetchone()
            cursor.close()
            
            if not result:
                return self.create_default_config(business_id)
            
            return dict(result)
    
    def update_config(self, business_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Actualizar configuración"""
        with get_db() as conn:
            cursor = conn.cursor()
            
            # Construir query dinámicamente
            set_clauses = []
            values = []
            
            for key, value in updates.items():
                set_clauses.append(f"{key} = %s")
                values.append(value)
            
            values.append(business_id)
            set_clause = ", ".join(set_clauses)
            
            cursor.execute(f"""
                UPDATE agent_config
                SET {set_clause}, updated_at = NOW()
                WHERE business_id = %s
                RETURNING id, business_id, system_prompt, provider, model,
                          temperature, max_tokens, enabled, created_at, updated_at
            """, values)
            
            result = cursor.fetchone()
            cursor.close()
            
            if not result:
                raise ValueError(f"Agent config not found for business {business_id}")
            
            return dict(result)
    
    def create_default_config(self, business_id: str) -> Dict[str, Any]:
        """Crear configuración por defecto"""
        with get_db() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO agent_config (
                    business_id, system_prompt, provider, model,
                    temperature, max_tokens, enabled
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id, business_id, system_prompt, provider, model,
                          temperature, max_tokens, enabled, created_at, updated_at
            """, (
                business_id,
                self.get_default_prompt(),
                'openai',
                'gpt-5-mini',
                0.2,
                2000,
                True
            ))
            
            result = cursor.fetchone()
            cursor.close()
            
            return dict(result)
    
    def get_default_prompt(self) -> str:
        """Prompt por defecto para agentes nuevos"""
        return """Eres un asistente virtual de atención al cliente profesional y amable.

Tu objetivo es:
- Responder preguntas de los clientes de forma clara y precisa
- Usar la información de la base de conocimiento cuando esté disponible
- Ser cortés y mantener un tono profesional
- Si no sabes algo, admítelo y ofrece transferir con un humano

Reglas:
- Nunca inventes información
- Sé breve y conciso
- Mantén la conversación enfocada en ayudar al cliente"""
