"""
AI Modules Configuration
========================
Configuration settings for AI module integrations.
"""

import os
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIModulesConfig:
    """Configuration manager for AI modules."""
    
    def __init__(self):
        # Gemma3n Integration Settings
        self.ENABLE_GEMMA = self._get_bool_env('ENABLE_GEMMA', True)
        self.GEMMA_MODEL_NAME = os.getenv('GEMMA_MODEL_NAME', 'gemma:2b')  # Use faster model by default
        self.GEMMA_OLLAMA_URL = os.getenv('GEMMA_OLLAMA_URL', 'http://localhost:11434')
        self.GEMMA_MAX_RETRIES = int(os.getenv('GEMMA_MAX_RETRIES', '2'))
        self.GEMMA_TIMEOUT = int(os.getenv('GEMMA_TIMEOUT', '60'))  # Increased timeout
        
        # Performance options
        self.USE_LIGHTWEIGHT_MODEL = self._get_bool_env('USE_LIGHTWEIGHT_MODEL', False)
        self.LIGHTWEIGHT_MODEL_NAME = os.getenv('LIGHTWEIGHT_MODEL_NAME', 'gemma:2b')
        
        # Logging and Debugging
        self.LOG_INTERACTIONS = self._get_bool_env('LOG_INTERACTIONS', True)
        self.DEBUG_PROMPTS = self._get_bool_env('DEBUG_PROMPTS', False)
        
        # Safety and Performance
        self.MAX_PROMPT_LENGTH = int(os.getenv('MAX_PROMPT_LENGTH', '4000'))
        self.MAX_RESPONSE_LENGTH = int(os.getenv('MAX_RESPONSE_LENGTH', '1000'))
        self.ENABLE_FALLBACK_RESPONSES = self._get_bool_env('ENABLE_FALLBACK_RESPONSES', True)
        
        # Template Settings
        self.DEFAULT_TEMPLATE_TYPE = os.getenv('DEFAULT_TEMPLATE_TYPE', 'general_assistance')
        self.VISION_TEMPLATE_TYPE = os.getenv('VISION_TEMPLATE_TYPE', 'vision_description')
        self.VOICE_TEMPLATE_TYPE = os.getenv('VOICE_TEMPLATE_TYPE', 'general_assistance')
        
        logger.info(f"AI Modules Config initialized - Gemma enabled: {self.ENABLE_GEMMA}")
    
    def _get_bool_env(self, key: str, default: bool) -> bool:
        """Get boolean environment variable."""
        value = os.getenv(key, str(default)).lower()
        return value in ('true', '1', 'yes', 'on')
    
    def toggle_gemma(self, enable: bool) -> bool:
        """Toggle Gemma3n integration on/off."""
        old_status = self.ENABLE_GEMMA
        self.ENABLE_GEMMA = enable
        logger.info(f"Gemma3n toggled from {old_status} to {enable}")
        return True
    
    def get_gemma_config(self) -> Dict[str, Any]:
        """Get Gemma3n configuration dictionary."""
        return {
            'enabled': self.ENABLE_GEMMA,
            'model_name': self.GEMMA_MODEL_NAME,
            'ollama_url': self.GEMMA_OLLAMA_URL,
            'max_retries': self.GEMMA_MAX_RETRIES,
            'timeout': self.GEMMA_TIMEOUT,
            'log_interactions': self.LOG_INTERACTIONS,
            'debug_prompts': self.DEBUG_PROMPTS,
            'max_prompt_length': self.MAX_PROMPT_LENGTH,
            'max_response_length': self.MAX_RESPONSE_LENGTH,
            'enable_fallback': self.ENABLE_FALLBACK_RESPONSES
        }
    
    def get_template_config(self) -> Dict[str, str]:
        """Get template configuration."""
        return {
            'default': self.DEFAULT_TEMPLATE_TYPE,
            'vision': self.VISION_TEMPLATE_TYPE,
            'voice': self.VOICE_TEMPLATE_TYPE
        }
    
    def validate_config(self) -> Dict[str, bool]:
        """Validate configuration settings."""
        validation = {
            'gemma_model_name_valid': bool(self.GEMMA_MODEL_NAME),
            'ollama_url_valid': self.GEMMA_OLLAMA_URL.startswith('http'),
            'retries_valid': 1 <= self.GEMMA_MAX_RETRIES <= 10,
            'timeout_valid': 5 <= self.GEMMA_TIMEOUT <= 120,
            'prompt_length_valid': 100 <= self.MAX_PROMPT_LENGTH <= 10000,
            'response_length_valid': 50 <= self.MAX_RESPONSE_LENGTH <= 5000
        }
        
        all_valid = all(validation.values())
        if not all_valid:
            logger.warning(f"Configuration validation issues found: {validation}")
        
        return validation

# Global configuration instance
_config = None

def get_config() -> AIModulesConfig:
    """Get or create global configuration instance."""
    global _config
    if _config is None:
        _config = AIModulesConfig()
    return _config

# Convenience functions
def is_gemma_enabled() -> bool:
    """Check if Gemma3n is enabled."""
    return get_config().ENABLE_GEMMA

def toggle_gemma_integration(enable: bool) -> bool:
    """Toggle Gemma3n integration."""
    return get_config().toggle_gemma(enable)

def get_gemma_model_name() -> str:
    """Get Gemma model name."""
    return get_config().GEMMA_MODEL_NAME

# Test configuration if run directly
if __name__ == "__main__":
    config = get_config()
    print("AI Modules Configuration:")
    print("-" * 40)
    print(f"Gemma Enabled: {config.ENABLE_GEMMA}")
    print(f"Model Name: {config.GEMMA_MODEL_NAME}")
    print(f"Ollama URL: {config.GEMMA_OLLAMA_URL}")
    print(f"Max Retries: {config.GEMMA_MAX_RETRIES}")
    print(f"Timeout: {config.GEMMA_TIMEOUT}")
    print(f"Log Interactions: {config.LOG_INTERACTIONS}")
    print(f"Debug Prompts: {config.DEBUG_PROMPTS}")
    print("-" * 40)
    
    # Validate configuration
    validation = config.validate_config()
    print(f"Configuration Valid: {all(validation.values())}")
    if not all(validation.values()):
        print(f"Issues: {[k for k, v in validation.items() if not v]}")
    
    # Test toggle
    print(f"\nTesting toggle...")
    config.toggle_gemma(False)
    print(f"Gemma Enabled: {config.ENABLE_GEMMA}")
    config.toggle_gemma(True)
    print(f"Gemma Enabled: {config.ENABLE_GEMMA}")
