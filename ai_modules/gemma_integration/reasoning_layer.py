"""
Gemma Reasoning Layer
====================
High-level reasoning layer that combines prompt building and engine execution.
"""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
import json

from .prompt_builder import GemmaPromptBuilder
from .gemma3n_engine import Gemma3nEngine, get_gemma_engine

logger = logging.getLogger(__name__)

class GemmaReasoningLayer:
    """
    High-level reasoning layer that orchestrates the Vision Assistant's AI responses.
    Combines vision detection results with intelligent prompt building and safe execution.
    """
    
    def __init__(self, 
                 model_name: str = "gemma:2b",  # Use faster model by default
                 enable_gemma: bool = True,
                 log_interactions: bool = True):
        """
        Initialize the reasoning layer.
        
        Args:
            model_name: Name of the Gemma model to use
            enable_gemma: Whether to enable Gemma3n reasoning (True) or use fallbacks (False)
            log_interactions: Whether to log prompts and responses for debugging
        """
        self.enable_gemma = enable_gemma
        self.log_interactions = log_interactions
        self.model_name = model_name
        
        # Initialize components
        self.prompt_builder = GemmaPromptBuilder()
        
        if self.enable_gemma:
            self.engine = get_gemma_engine(model_name)
        else:
            self.engine = None
            
        # Interaction history for debugging
        self.interaction_history = []
        self.max_history_length = 50
        
        logger.info(f"GemmaReasoningLayer initialized - Gemma enabled: {enable_gemma}")
    
    def process_vision_input(self, 
                           vision_description: str,
                           user_question: str = "",
                           template_type: str = "vision_description",
                           additional_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process vision input and generate intelligent response.
        
        Args:
            vision_description: Description from vision detection (YOLO/BLIP/OCR)
            user_question: Optional user question or request
            template_type: Type of reasoning template to use
            additional_context: Additional context information
            
        Returns:
            Dictionary containing response and metadata
        """
        try:
            start_time = datetime.now()
            
            # Create interaction record
            interaction = {
                'timestamp': start_time.isoformat(),
                'input_type': 'vision',
                'vision_description': vision_description,
                'user_question': user_question,
                'template_type': template_type,
                'additional_context': additional_context,
                'gemma_enabled': self.enable_gemma
            }
            
            # Generate response
            if self.enable_gemma and self.engine:
                response = self._generate_gemma_response(
                    vision_description, user_question, template_type, additional_context
                )
                interaction['response_source'] = 'gemma3n'
            else:
                response = self._generate_fallback_response(
                    vision_description, user_question, template_type
                )
                interaction['response_source'] = 'fallback'
            
            # Finalize interaction record
            interaction['response'] = response
            interaction['processing_time'] = (datetime.now() - start_time).total_seconds()
            
            # Log interaction if enabled
            if self.log_interactions:
                self._log_interaction(interaction)
            
            # Return structured response
            return {
                'success': True,
                'response': response,
                'metadata': {
                    'source': interaction['response_source'],
                    'template_type': template_type,
                    'processing_time': interaction['processing_time'],
                    'gemma_enabled': self.enable_gemma
                }
            }
            
        except Exception as e:
            logger.error(f"Error in process_vision_input: {e}")
            return {
                'success': False,
                'response': self._get_error_response(str(e)),
                'metadata': {
                    'source': 'error_handler',
                    'error': str(e)
                }
            }
    
    def process_voice_input(self,
                          voice_text: str,
                          scene_context: str = "",
                          template_type: str = "general_assistance") -> Dict[str, Any]:
        """
        Process voice input with optional scene context.
        
        Args:
            voice_text: Recognized voice input text
            scene_context: Optional scene description for context
            template_type: Type of reasoning template to use
            
        Returns:
            Dictionary containing response and metadata
        """
        try:
            start_time = datetime.now()
            
            # Create interaction record
            interaction = {
                'timestamp': start_time.isoformat(),
                'input_type': 'voice',
                'voice_text': voice_text,
                'scene_context': scene_context,
                'template_type': template_type,
                'gemma_enabled': self.enable_gemma
            }
            
            # Generate response
            if self.enable_gemma and self.engine:
                response = self._generate_gemma_response(
                    scene_context, voice_text, template_type
                )
                interaction['response_source'] = 'gemma3n'
            else:
                response = self._generate_fallback_response(
                    scene_context, voice_text, template_type
                )
                interaction['response_source'] = 'fallback'
            
            # Finalize interaction record
            interaction['response'] = response
            interaction['processing_time'] = (datetime.now() - start_time).total_seconds()
            
            # Log interaction if enabled
            if self.log_interactions:
                self._log_interaction(interaction)
            
            return {
                'success': True,
                'response': response,
                'metadata': {
                    'source': interaction['response_source'],
                    'template_type': template_type,
                    'processing_time': interaction['processing_time']
                }
            }
            
        except Exception as e:
            logger.error(f"Error in process_voice_input: {e}")
            return {
                'success': False,
                'response': self._get_error_response(str(e)),
                'metadata': {
                    'source': 'error_handler',
                    'error': str(e)
                }
            }
    
    def _generate_gemma_response(self,
                               scene_description: str,
                               user_question: str,
                               template_type: str,
                               additional_context: Optional[Dict[str, Any]] = None) -> str:
        """Generate response using Gemma3n."""
        try:
            # Build the prompt
            prompt = self.prompt_builder.build_prompt(
                scene_description=scene_description,
                user_question=user_question,
                template_type=template_type,
                additional_context=additional_context
            )
            
            # Log prompt if debugging enabled
            if self.log_interactions:
                logger.debug(f"Generated prompt ({template_type}): {prompt[:200]}...")
            
            # Generate response
            response = self.engine.generate_response(prompt)
            
            # Validate response
            if not response or response.startswith("Error:") or response.startswith("Connection Error:"):
                logger.warning(f"Gemma3n returned error response: {response[:100]}...")
                return self._generate_fallback_response(scene_description, user_question, template_type, response)
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating Gemma response: {e}")
            return self._generate_fallback_response(scene_description, user_question, template_type)
    
    def _generate_fallback_response(self,
                                  scene_description: str,
                                  user_question: str,
                                  template_type: str,
                                  error_response: str = None) -> str:
        """Generate fallback response when Gemma3n is unavailable."""
        logger.info("Using fallback response system")
        
        # Basic response construction based on input
        response_parts = []
        
        if scene_description.strip():
            if template_type == "vision_description":
                response_parts.append(f"I can see: {scene_description}")
                
                # Add safety considerations
                if any(word in scene_description.lower() for word in ['stairs', 'step', 'edge', 'drop']):
                    response_parts.append("Please be careful of elevation changes.")
                
                if any(word in scene_description.lower() for word in ['glass', 'fragile', 'sharp']):
                    response_parts.append("I notice some fragile items - please move carefully.")
                    
            else:
                response_parts.append(f"Based on what I can observe: {scene_description}")
        
        if user_question.strip():
            question_lower = user_question.lower()
            
            if any(word in question_lower for word in ['help', 'what', 'how']):
                if template_type == "navigation_help":
                    response_parts.append("For safety, I recommend moving slowly and feeling ahead with your hands or mobility aid.")
                else:
                    response_parts.append("I'm here to help. Please let me know what specific assistance you need.")
            
            elif any(word in question_lower for word in ['safe', 'danger', 'careful']):
                response_parts.append("Your safety is the priority. Take your time and move carefully.")
            
            elif any(word in question_lower for word in ['where', 'location', 'find']):
                response_parts.append("Based on the current scene, I'll do my best to help you locate what you're looking for.")
        
        # Default response if nothing specific was detected
        if not response_parts:
            response_parts.append("I'm here to assist you. The AI reasoning system is currently using basic responses, but I can still help with your questions.")
        
        # Add specific note based on error type
        if error_response and "timed out" in error_response.lower():
            response_parts.append("(Note: AI is processing slowly - try again or use simpler requests)")
        elif error_response and "connection" in error_response.lower():
            response_parts.append("(Note: AI service temporarily unavailable - using basic responses)")
        else:
            response_parts.append("(Note: Using simplified response mode - advanced AI reasoning temporarily unavailable)")
        
        return " ".join(response_parts)
    
    def _get_error_response(self, error_msg: str) -> str:
        """Generate user-friendly error response."""
        return "I'm experiencing some technical difficulties right now. Please try again in a moment, or let me know if you need immediate assistance."
    
    def _log_interaction(self, interaction: Dict[str, Any]):
        """Log interaction for debugging purposes."""
        # Add to history
        self.interaction_history.append(interaction)
        
        # Trim history if too long
        if len(self.interaction_history) > self.max_history_length:
            self.interaction_history = self.interaction_history[-self.max_history_length:]
        
        # Log key details
        logger.info(f"Interaction logged - Type: {interaction['input_type']}, "
                   f"Source: {interaction.get('response_source', 'unknown')}, "
                   f"Time: {interaction.get('processing_time', 0):.2f}s")
    
    def get_interaction_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent interaction history."""
        return self.interaction_history[-limit:]
    
    def clear_interaction_history(self):
        """Clear interaction history."""
        self.interaction_history.clear()
        logger.info("Interaction history cleared")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        status = {
            'gemma_enabled': self.enable_gemma,
            'model_name': self.model_name,
            'total_interactions': len(self.interaction_history),
            'logging_enabled': self.log_interactions
        }
        
        if self.enable_gemma and self.engine:
            # Get engine stats
            engine_stats = self.engine.get_stats()
            status['engine_stats'] = engine_stats
            
            # Check connection
            is_connected, connection_msg = self.engine.verify_connection()
            status['connection_status'] = {
                'connected': is_connected,
                'message': connection_msg
            }
        else:
            status['connection_status'] = {
                'connected': False,
                'message': 'Gemma3n disabled - using fallback responses'
            }
        
        return status
    
    def toggle_gemma(self, enable: bool) -> bool:
        """Enable or disable Gemma3n reasoning."""
        old_status = self.enable_gemma
        self.enable_gemma = enable
        
        if enable and not self.engine:
            self.engine = get_gemma_engine(self.model_name)
        
        logger.info(f"Gemma3n toggled from {old_status} to {enable}")
        return True

# Global instance for easy access
_reasoning_layer = None

def get_reasoning_layer(enable_gemma: bool = True) -> GemmaReasoningLayer:
    """Get or create global reasoning layer instance."""
    global _reasoning_layer
    if _reasoning_layer is None:
        _reasoning_layer = GemmaReasoningLayer(enable_gemma=enable_gemma)
    return _reasoning_layer

# Test the reasoning layer if run directly
if __name__ == "__main__":
    # Test the reasoning layer
    reasoning = GemmaReasoningLayer(enable_gemma=True)
    
    # Test vision input
    vision_result = reasoning.process_vision_input(
        vision_description="A table with a bottle and a glass. Chair on the right.",
        user_question="What should I do?",
        template_type="vision_description"
    )
    
    print("Vision Processing Result:")
    print(f"Success: {vision_result['success']}")
    print(f"Response: {vision_result['response']}")
    print(f"Metadata: {vision_result['metadata']}")
    
    # Test voice input
    voice_result = reasoning.process_voice_input(
        voice_text="Tell me what's around me",
        scene_context="Kitchen with various appliances and utensils",
        template_type="general_assistance"
    )
    
    print("\\nVoice Processing Result:")
    print(f"Success: {voice_result['success']}")
    print(f"Response: {voice_result['response']}")
    print(f"Metadata: {voice_result['metadata']}")
    
    # Show system status
    status = reasoning.get_system_status()
    print(f"\\nSystem Status: {status}")
