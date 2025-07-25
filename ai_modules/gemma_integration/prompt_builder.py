"""
Gemma3n Prompt Builder
======================
Constructs intelligent prompts for the Gemma3n reasoning layer.
"""

import logging
from datetime import datetime
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class GemmaPromptBuilder:
    """
    Builds structured prompts for Gemma3n with context awareness.
    """
    
    def __init__(self):
        self.default_system_prompt = "You are a helpful AI assistant for a blind person."
        
        # Pre-defined prompt templates for different scenarios
        self.templates = {
            'vision_description': {
                'system': "You are a helpful AI assistant for a blind person. Focus on providing clear, detailed, and actionable descriptions.",
                'context_prefix': "Scene Description: ",
                'question_prefix': "User Question: "
            },
            'navigation_help': {
                'system': "You are a navigation assistant for a blind person. Provide clear, step-by-step guidance focusing on safety and accessibility.",
                'context_prefix': "Current Scene: ",
                'question_prefix': "Navigation Request: "
            },
            'object_identification': {
                'system': "You are an object identification assistant for a blind person. Describe objects clearly, including their location, size, and potential hazards or benefits.",
                'context_prefix': "Objects in Scene: ",
                'question_prefix': "User's Question: "
            },
            'general_assistance': {
                'system': "You are a helpful AI assistant for a blind person. Provide supportive, clear, and practical guidance.",
                'context_prefix': "Context: ",
                'question_prefix': "Question: "
            }
        }
    
    def build_prompt(self, 
                    scene_description: str = "", 
                    user_question: str = "", 
                    template_type: str = "general_assistance",
                    custom_system_prompt: Optional[str] = None,
                    additional_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Build a structured prompt for Gemma3n.
        
        Args:
            scene_description: Description of the current scene/image
            user_question: Optional user question or request
            template_type: Type of prompt template to use
            custom_system_prompt: Custom system prompt to override default
            additional_context: Additional context information
            
        Returns:
            Structured prompt string ready for Gemma3n
        """
        try:
            # Get template or use default
            template = self.templates.get(template_type, self.templates['general_assistance'])
            
            # Build system prompt
            system_prompt = custom_system_prompt or template['system']
            
            # Start building the prompt
            prompt_parts = [system_prompt]
            
            # Add timestamp for context
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            prompt_parts.append(f"\nCurrent Time: {timestamp}")
            
            # Add scene description if provided
            if scene_description.strip():
                context_prefix = template.get('context_prefix', 'Context: ')
                prompt_parts.append(f"\n{context_prefix}{scene_description.strip()}")
            
            # Add additional context if provided
            if additional_context:
                for key, value in additional_context.items():
                    if value:
                        prompt_parts.append(f"\n{key}: {value}")
            
            # Add user question if provided
            if user_question.strip():
                question_prefix = template.get('question_prefix', 'Question: ')
                prompt_parts.append(f"\n{question_prefix}{user_question.strip()}")
            
            # Add specific instructions based on template type
            if template_type == 'vision_description':
                prompt_parts.append(self._get_vision_instructions())
            elif template_type == 'navigation_help':
                prompt_parts.append(self._get_navigation_instructions())
            elif template_type == 'object_identification':
                prompt_parts.append(self._get_object_identification_instructions())
            
            # Join all parts
            full_prompt = "\n".join(prompt_parts)
            
            logger.info(f"Built prompt for template: {template_type}")
            return full_prompt
            
        except Exception as e:
            logger.error(f"Error building prompt: {e}")
            # Fallback to simple prompt
            return self._build_fallback_prompt(scene_description, user_question)
    
    def _get_vision_instructions(self) -> str:
        """Get specific instructions for vision description tasks."""
        return """
Instructions:
- Describe the scene clearly and systematically
- Start with the overall environment/setting
- Mention objects from left to right or top to bottom
- Include important details about colors, sizes, and positions
- Highlight any potential hazards or obstacles
- Mention anything that might be useful or important for a blind person
- Be specific about distances and locations when possible
- Use clear, descriptive language without complex jargon"""
    
    def _get_navigation_instructions(self) -> str:
        """Get specific instructions for navigation assistance."""
        return """
Instructions:
- Prioritize safety above all else
- Give step-by-step directions
- Mention obstacles, hazards, or changes in terrain
- Describe landmarks or reference points
- Be specific about distances and directions (left, right, forward, back)
- Suggest the safest path available
- Warn about stairs, curbs, or elevation changes
- Mention handrails, walls, or other guidance aids if available"""
    
    def _get_object_identification_instructions(self) -> str:
        """Get specific instructions for object identification."""
        return """
Instructions:
- Identify each object clearly and precisely
- Describe the location of each object relative to the person
- Mention the approximate size and shape
- Include color information when relevant
- Specify if objects are fragile, hot, sharp, or otherwise require caution
- Mention the purpose or function of objects when helpful
- Group similar objects together in your description
- Highlight anything that might be immediately useful or important"""
    
    def _build_fallback_prompt(self, scene_description: str, user_question: str) -> str:
        """Build a simple fallback prompt if the main builder fails."""
        parts = [self.default_system_prompt]
        
        if scene_description.strip():
            parts.append(f"Scene: {scene_description.strip()}")
        
        if user_question.strip():
            parts.append(f"Question: {user_question.strip()}")
        
        return "\n".join(parts)
    
    def get_template_types(self) -> list:
        """Get list of available template types."""
        return list(self.templates.keys())
    
    def add_custom_template(self, name: str, template: Dict[str, str]):
        """Add a custom template."""
        if 'system' not in template:
            template['system'] = self.default_system_prompt
        if 'context_prefix' not in template:
            template['context_prefix'] = 'Context: '
        if 'question_prefix' not in template:
            template['question_prefix'] = 'Question: '
            
        self.templates[name] = template
        logger.info(f"Added custom template: {name}")

# Example usage and testing
if __name__ == "__main__":
    # Test the prompt builder
    builder = GemmaPromptBuilder()
    
    # Test vision description
    scene = "A table with a bottle and a glass. Chair on the right."
    question = "What should I do?"
    
    prompt = builder.build_prompt(
        scene_description=scene,
        user_question=question,
        template_type="vision_description"
    )
    
    print("Generated Prompt:")
    print("-" * 50)
    print(prompt)
    print("-" * 50)
