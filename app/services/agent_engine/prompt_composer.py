"""
PromptComposer: Sistema multi-layer enterprise para composición de prompts.

Arquitectura:
- Layer 1: System Prompt (contexto general, personalidad)
- Layer 2: Agent Prompt (instrucciones específicas)
- Layer 3: Specialized Prompts (greet, handoff, fallback)
- Layer 4: Dynamic Context (KB, sentiment, disclaimers, memory)

Variables soportadas:
- Sistema: {business_name}, {customer_name}, {current_time}, {current_date}, 
           {day_of_week}, {conversation_summary}, {sentiment}, {complexity}
- Custom: Definidas por usuario en config['custom_variables']
"""

from typing import Dict, Any, List
from datetime import datetime
import locale

# Configurar locale para español (día de la semana, mes)
try:
    locale.setlocale(locale.LC_TIME, 'es_ES.UTF-8')
except:
    try:
        locale.setlocale(locale.LC_TIME, 'es_MX.UTF-8')
    except:
        pass  # Fallback a locale por defecto


class PromptComposer:
    """
    Compositor de prompts multi-layer para el agente de IA.
    Permite máxima flexibilidad con separación de concerns.
    """
    
    # Variables del sistema disponibles para inyección
    SYSTEM_VARIABLES = {
        'business_name': lambda config, state: config.get('business_name', 'nuestro negocio'),
        'customer_name': lambda config, state: state.get('customer_name', 'Cliente'),
        'current_time': lambda config, state: datetime.now().strftime("%I:%M %p"),
        'current_date': lambda config, state: datetime.now().strftime("%d de %B, %Y"),
        'day_of_week': lambda config, state: datetime.now().strftime("%A"),
        'conversation_summary': lambda config, state: state.get('conversation_summary', {}).get('text', ''),
        'sentiment': lambda config, state: state.get('customer_sentiment', 'neutral'),
        'complexity': lambda config, state: state.get('complexity', 'medium'),
    }
    
    @classmethod
    def compose_system_prompt(
        cls,
        config: Dict[str, Any],
        state: Dict[str, Any],
        layer: str = 'system'
    ) -> str:
        """
        Compone un prompt específico según la capa solicitada.
        
        Args:
            config: Configuración del agente (de ai_configs)
            state: Estado actual de la conversación
            layer: Capa a componer ('system' | 'agent' | 'greet' | 'handoff' | 'fallback')
            
        Returns:
            str: Prompt compuesto con variables inyectadas
        """
        # 1. Obtener base prompt según layer
        if layer == 'system':
            base_prompt = config.get('system_prompt', cls._get_default_system_prompt())
        elif layer == 'agent':
            base_prompt = config.get('agent_prompt', '')
        elif layer == 'greet':
            base_prompt = config.get('greet_prompt', '')
        elif layer == 'handoff':
            base_prompt = config.get('handoff_prompt', cls._get_default_handoff_prompt())
        elif layer == 'fallback':
            base_prompt = config.get('fallback_prompt', cls._get_default_fallback_prompt())
        else:
            base_prompt = ''
        
        if not base_prompt:
            return ''
        
        # 2. Inyectar variables del sistema (si está habilitado)
        if config.get('enable_dynamic_variables', True):
            base_prompt = cls._inject_system_variables(base_prompt, config, state)
        
        # 3. Inyectar variables personalizadas del usuario
        custom_vars = config.get('custom_variables', {})
        if isinstance(custom_vars, dict):
            for var_name, var_value in custom_vars.items():
                placeholder = f'{{{var_name}}}'
                if placeholder in base_prompt:
                    base_prompt = base_prompt.replace(placeholder, str(var_value))
        
        # 4. Inyectar conversation memory si está habilitado (solo para system layer)
        if layer == 'system' and config.get('enable_conversation_memory', True):
            summary = state.get('conversation_summary', {})
            if summary and summary.get('text'):
                memory_text = summary['text'][:300]  # Limitar a 300 chars
                memory_context = f"\n\n--- Resumen de Conversación Previa ---\n{memory_text}\n"
                base_prompt += memory_context
                print(f"💭 [PROMPT_COMPOSER] Memory inyectada: {len(memory_text)} chars")
        
        return base_prompt
    
    @classmethod
    def compose_full_prompt(
        cls,
        config: Dict[str, Any],
        state: Dict[str, Any],
        include_kb_context: bool = True,
        include_disclaimers: bool = True
    ) -> str:
        """
        Compone el prompt completo con todas las capas.
        Usado para respond_node y preview en frontend.
        
        Args:
            config: Configuración del agente
            state: Estado actual de la conversación
            include_kb_context: Incluir contexto de knowledge base
            include_disclaimers: Incluir disclaimers según confidence
            
        Returns:
            str: Prompt final completo listo para el LLM
        """
        layers = []
        
        # Layer 1: System Prompt (siempre presente)
        system = cls.compose_system_prompt(config, state, 'system')
        if system:
            layers.append(system)
            print(f"📝 [PROMPT_COMPOSER] Layer 1 (System): {len(system)} chars")
        
        # Layer 2: Agent Prompt (instrucciones específicas - opcional)
        agent = cls.compose_system_prompt(config, state, 'agent')
        if agent:
            layers.append(f"\n--- Instrucciones Específicas ---\n{agent}")
            print(f"📝 [PROMPT_COMPOSER] Layer 2 (Agent): {len(agent)} chars")
        
        # Layer 3: Knowledge Base Context (si existe y está habilitado)
        if include_kb_context and state.get('retrieved_docs'):
            docs = state['retrieved_docs']
            kb_context = "\n\n".join(docs)
            layers.append(f"\n--- Base de Conocimiento ---\n{kb_context}")
            print(f"📝 [PROMPT_COMPOSER] Layer 3 (KB): {len(docs)} documentos, {len(kb_context)} chars")
        
        # Layer 4: Disclaimers según confidence (si está habilitado)
        if include_disclaimers:
            confidence = state.get('confidence', 1.0)
            suggest_handoff = state.get('suggest_handoff_in_response', False)
            
            disclaimer = cls._build_confidence_disclaimer(confidence, suggest_handoff)
            if disclaimer:
                layers.append(disclaimer)
                print(f"⚠️ [PROMPT_COMPOSER] Layer 4 (Disclaimer): confidence={confidence:.2f}")
        
        # Combinar todas las capas
        full_prompt = "\n".join(layers)
        print(f"✅ [PROMPT_COMPOSER] Prompt final: {len(full_prompt)} chars, {len(layers)} layers")
        
        return full_prompt
    
    @classmethod
    def compose_specialized_prompt(
        cls,
        config: Dict[str, Any],
        state: Dict[str, Any],
        prompt_type: str
    ) -> str:
        """
        Compone un prompt especializado (greet, handoff, fallback).
        
        Args:
            config: Configuración del agente
            state: Estado actual
            prompt_type: Tipo ('greet' | 'handoff' | 'fallback')
            
        Returns:
            str: Prompt especializado o default si no está configurado
        """
        return cls.compose_system_prompt(config, state, prompt_type)
    
    @classmethod
    def _inject_system_variables(
        cls,
        prompt: str,
        config: Dict[str, Any],
        state: Dict[str, Any]
    ) -> str:
        """
        Reemplaza variables del sistema como {customer_name}, {current_time}, etc.
        
        Args:
            prompt: Prompt con placeholders
            config: Configuración del agente
            state: Estado actual
            
        Returns:
            str: Prompt con variables reemplazadas
        """
        for var_name, var_func in cls.SYSTEM_VARIABLES.items():
            placeholder = f'{{{var_name}}}'
            if placeholder in prompt:
                try:
                    value = var_func(config, state)
                    prompt = prompt.replace(placeholder, str(value))
                except Exception as e:
                    print(f"⚠️ [PROMPT_COMPOSER] Error inyectando variable {var_name}: {e}")
                    # Mantener placeholder si hay error
        
        return prompt
    
    @classmethod
    def _build_confidence_disclaimer(cls, confidence: float, suggest_handoff: bool) -> str:
        """
        Construye disclaimer según nivel de confidence (Sprint 4).
        
        Args:
            confidence: Score de confidence (0.0-1.0)
            suggest_handoff: Flag de orchestrator
            
        Returns:
            str: Disclaimer o string vacío
        """
        if confidence < 0.4:
            # Very low confidence → force handoff directo
            return """

⚠️ CRÍTICO: Tu nivel de confianza sobre esta consulta es MUY BAJO (<40%).
No tienes información suficiente para responder con certeza.
DEBES ofrecer conectar al usuario con un asesor humano de forma directa y clara.
Ejemplo: "Para ayudarte mejor con esto, te recomiendo hablar con uno de nuestros asesores. ¿Te conecto?"
"""
        
        elif 0.4 <= confidence < 0.6:
            # Low-medium confidence → sugerir handoff naturalmente
            return """

💡 NOTA: Tu nivel de confianza sobre esta consulta es MEDIO (40-60%).
Responde lo mejor que puedas con la información disponible, pero al final
sugiere de forma natural que pueden contactar a un asesor si necesitan más ayuda.
Ejemplo: "Si necesitas más detalles específicos, puedo conectarte con un asesor 👤"
"""
        
        elif suggest_handoff:
            # Orchestrator detectó necesidad de handoff (independiente de confidence)
            return """

💡 SUGERENCIA: Aunque puedes responder, el usuario podría beneficiarse de atención humana.
Incluye sutilmente la opción de hablar con un asesor si lo prefiere.
"""
        
        return ''
    
    @classmethod
    def _get_default_system_prompt(cls) -> str:
        """Prompt por defecto si el usuario no ha configurado uno."""
        return """Eres un asistente de atención al cliente de WhatsApp.

REGLAS CRÍTICAS:
- Siempre sé respetuoso y profesional
- Si no tienes información en la base de conocimiento, di "No tengo esa información" y ofrece conectar con un humano
- Usa el contexto de conocimiento provisto para responder
- NO inventes información"""
    
    @classmethod
    def _get_default_handoff_prompt(cls) -> str:
        """Mensaje por defecto para handoff."""
        return "Te conecto con un asesor que te ayudará mejor 👤"
    
    @classmethod
    def _get_default_fallback_prompt(cls) -> str:
        """Mensaje por defecto cuando no tiene respuesta."""
        return "Lo siento, no tengo información específica sobre eso. ¿Te gustaría que te conecte con un asesor?"
    
    @classmethod
    def get_available_variables(cls) -> List[Dict[str, str]]:
        """
        Retorna lista de variables disponibles para documentación/frontend.
        
        Returns:
            List[Dict]: Lista de variables con metadata
        """
        return [
            {
                'name': '{business_name}',
                'description': 'Nombre del negocio',
                'example': 'Tienda El Sol',
                'category': 'sistema'
            },
            {
                'name': '{customer_name}',
                'description': 'Nombre del cliente actual',
                'example': 'Juan Pérez',
                'category': 'sistema'
            },
            {
                'name': '{current_time}',
                'description': 'Hora actual (formato 12h)',
                'example': '08:30 PM',
                'category': 'sistema'
            },
            {
                'name': '{current_date}',
                'description': 'Fecha actual',
                'example': '14 de noviembre, 2025',
                'category': 'sistema'
            },
            {
                'name': '{day_of_week}',
                'description': 'Día de la semana',
                'example': 'Jueves',
                'category': 'sistema'
            },
            {
                'name': '{conversation_summary}',
                'description': 'Resumen de conversación previa (si existe)',
                'example': 'El cliente preguntó sobre precios...',
                'category': 'sistema'
            },
            {
                'name': '{sentiment}',
                'description': 'Sentiment detectado del cliente',
                'example': 'positive, neutral, negative',
                'category': 'sistema'
            },
            {
                'name': '{complexity}',
                'description': 'Complejidad de la consulta',
                'example': 'simple, medium, complex',
                'category': 'sistema'
            },
        ]

