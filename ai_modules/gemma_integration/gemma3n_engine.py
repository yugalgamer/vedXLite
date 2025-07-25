"""
Gemma3n Engine Wrapper
======================
Safe, offline wrapper for Gemma3n language model integration.
"""

import logging
import json
import time
from typing import Optional, Dict, Any, Tuple
import requests
from threading import Lock

# Import CUDA support
try:
    from ..cuda_core import get_cuda_manager, cuda_available
    from ..cuda_text import get_cuda_text_processor
    CUDA_SUPPORT_AVAILABLE = True
except ImportError:
    CUDA_SUPPORT_AVAILABLE = False

logger = logging.getLogger(__name__)

class Gemma3nEngine:
    """
    Safe wrapper for Gemma3n model that runs completely offline.
    Includes error handling, retries, and fallback mechanisms.
    """
    
    def __init__(self, 
                 model_name: str = "gemma:2b",  # Use faster model by default
                 ollama_url: str = "http://localhost:11434",
                 max_retries: int = 2,
                 timeout: int = 60,
                 enable_cuda: bool = True):
        """
        Initialize the Gemma3n engine.
        
        Args:
            model_name: Name of the Gemma model to use
            ollama_url: URL of the Ollama API server
            max_retries: Maximum number of retry attempts
            timeout: Request timeout in seconds
        """
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.api_endpoint = f"{ollama_url}/api/generate"
        self.chat_endpoint = f"{ollama_url}/api/chat"
        self.max_retries = max_retries
        self.timeout = timeout
        self.enable_cuda = enable_cuda and CUDA_SUPPORT_AVAILABLE
        
        # CUDA support
        self.cuda_manager = None
        self.cuda_text_processor = None
        if self.enable_cuda:
            try:
                self.cuda_manager = get_cuda_manager()
                self.cuda_text_processor = get_cuda_text_processor()
                logger.info(f"✅ CUDA acceleration enabled for Gemma3n engine")
            except Exception as e:
                logger.warning(f"⚠️  CUDA initialization failed: {e}")
                self.enable_cuda = False
        
        # Thread safety
        self._lock = Lock()
        
        # Performance tracking
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_response_time': 0,
            'last_error': None,
            'cuda_accelerated_requests': 0,
            'text_processing_time': 0,
            'ollama_processing_time': 0
        }
        
        # Connection status
        self._connection_verified = False
        self._last_health_check = 0
        self.health_check_interval = 300  # 5 minutes
        
    def verify_connection(self) -> Tuple[bool, str]:
        """
        Verify that Ollama is running and the model is available.
        
        Returns:
            Tuple of (success, message)
        """
        try:
            current_time = time.time()
            
            # Skip frequent health checks
            if (self._connection_verified and 
                current_time - self._last_health_check < self.health_check_interval):
                return True, "Connection verified (cached)"
            
            # Check if Ollama is running
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code != 200:
                self._connection_verified = False
                return False, f"Ollama server not responding (status: {response.status_code})"
            
            # Check if model is available
            models_data = response.json()
            available_models = [model['name'] for model in models_data.get('models', [])]
            
            if self.model_name not in available_models:
                self._connection_verified = False
                return False, f"Model '{self.model_name}' not found. Available: {available_models}"
            
            # Update connection status
            self._connection_verified = True
            self._last_health_check = current_time
            
            return True, "Connection verified successfully"
            
        except requests.RequestException as e:
            self._connection_verified = False
            return False, f"Connection error: {str(e)}"
        except Exception as e:
            self._connection_verified = False
            return False, f"Unexpected error during connection verification: {str(e)}"
    
    def generate_response(self, prompt: str, use_cuda_preprocessing: bool = True, **kwargs) -> str:
        """
        Generate a response from Gemma3n with error handling and retries.
        
        Args:
            prompt: The input prompt for generation
            **kwargs: Additional parameters for the model
            
        Returns:
            Generated response string
        """
        if not prompt or not prompt.strip():
            return "Error: Empty prompt provided"
        
        with self._lock:
            self.stats['total_requests'] += 1
        
        start_time = time.time()
        
        # CUDA-accelerated text preprocessing
        if self.enable_cuda and use_cuda_preprocessing and self.cuda_text_processor:
            try:
                preprocessing_start = time.time()
                
                # Analyze prompt characteristics
                prompt_analysis = self._analyze_prompt_with_cuda(prompt)
                
                # Optimize prompt if needed
                if prompt_analysis.get('needs_optimization', False):
                    prompt = self._optimize_prompt_with_cuda(prompt, prompt_analysis)
                
                preprocessing_time = time.time() - preprocessing_start
                self.stats['text_processing_time'] += preprocessing_time
                self.stats['cuda_accelerated_requests'] += 1
                
                logger.debug(f"🚀 CUDA preprocessing: {preprocessing_time:.3f}s")
                
            except Exception as e:
                logger.warning(f"⚠️  CUDA preprocessing failed: {e}")
        
        # Verify connection first
        is_connected, connection_msg = self.verify_connection()
        if not is_connected:
            self._update_stats(False, start_time, connection_msg)
            return f"Connection Error: {connection_msg}"
        
        # Track Ollama processing time separately
        ollama_start_time = time.time()
        
        # Attempt generation with retries
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Attempting Gemma3n generation (attempt {attempt + 1}/{self.max_retries})")
                response = self._make_request(prompt, **kwargs)
                
                if response:
                    ollama_time = time.time() - ollama_start_time
                    self.stats['ollama_processing_time'] += ollama_time
                    
                    # CUDA-accelerated post-processing
                    if self.enable_cuda and self.cuda_text_processor:
                        try:
                            response = self._post_process_response_with_cuda(response, prompt)
                        except Exception as e:
                            logger.warning(f"⚠️  CUDA post-processing failed: {e}")
                    
                    self._update_stats(True, start_time)
                    logger.info(f"Gemma3n generation successful on attempt {attempt + 1} (Ollama: {ollama_time:.3f}s)")
                    return response
                else:
                    logger.warning(f"Empty response on attempt {attempt + 1}")
                    
            except requests.Timeout as e:
                error_msg = f"Request timed out on attempt {attempt + 1} (timeout: {self.timeout}s): {str(e)}"
                logger.warning(error_msg)
                
                if attempt == self.max_retries - 1:
                    self._update_stats(False, start_time, error_msg)
                    return self._get_fallback_response(prompt, "Request timed out - the model may be busy")
                    
                # Wait before retry (shorter wait for timeouts)
                time.sleep(1)
                
            except requests.ConnectionError as e:
                error_msg = f"Connection failed on attempt {attempt + 1}: {str(e)}"
                logger.warning(error_msg)
                
                if attempt == self.max_retries - 1:
                    self._update_stats(False, start_time, error_msg)
                    return self._get_fallback_response(prompt, "Connection failed - Ollama may not be running")
                    
                # Wait before retry (longer wait for connection issues)
                time.sleep(3)
                
            except requests.RequestException as e:
                error_msg = f"Request failed on attempt {attempt + 1}: {str(e)}"
                logger.warning(error_msg)
                
                if attempt == self.max_retries - 1:
                    self._update_stats(False, start_time, error_msg)
                    return self._get_fallback_response(prompt, error_msg)
                    
                # Wait before retry (exponential backoff)
                time.sleep(2 ** attempt)
                
            except Exception as e:
                error_msg = f"Unexpected error on attempt {attempt + 1}: {str(e)}"
                logger.error(error_msg)
                
                if attempt == self.max_retries - 1:
                    self._update_stats(False, start_time, error_msg)
                    return self._get_fallback_response(prompt, error_msg)
        
        # If all attempts failed
        fallback_msg = "All generation attempts failed"
        self._update_stats(False, start_time, fallback_msg)
        return self._get_fallback_response(prompt, fallback_msg)
    
    def _make_request(self, prompt: str, **kwargs) -> Optional[str]:
        """
        Make the actual request to Ollama API.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional parameters
            
        Returns:
            Response text or None if failed
        """
        # Prepare payload
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": kwargs.get('temperature', 0.7),
                "top_p": kwargs.get('top_p', 0.9),
                "top_k": kwargs.get('top_k', 40),
                "num_predict": kwargs.get('max_tokens', 512)
            }
        }
        
        # Make request with explicit timeout handling
        response = requests.post(
            self.api_endpoint,
            json=payload,
            headers={'Content-Type': 'application/json'},
            timeout=(10, self.timeout)  # (connection timeout, read timeout)
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get('response', '').strip()
        else:
            logger.error(f"API request failed: {response.status_code} - {response.text}")
            return None
    
    def _get_fallback_response(self, prompt: str, error_msg: str) -> str:
        """
        Generate a fallback response when Gemma3n fails.
        
        Args:
            prompt: Original prompt
            error_msg: Error message
            
        Returns:
            Fallback response
        """
        logger.info("Using fallback response mechanism")
        
        # Try to provide a helpful fallback based on prompt content
        prompt_lower = prompt.lower()
        
        if any(word in prompt_lower for word in ['scene', 'image', 'see', 'describe']):
            return "I'm having trouble processing the visual information right now. Please try again in a moment, or describe what you're looking for and I'll do my best to help."
        
        elif any(word in prompt_lower for word in ['navigate', 'direction', 'where', 'go']):
            return "I'm currently unable to provide navigation assistance. Please ensure you're in a safe location and try again shortly."
        
        elif any(word in prompt_lower for word in ['what', 'how', 'help']):
            return "I'm experiencing technical difficulties at the moment. The system should recover shortly. Please try your request again."
        
        else:
            return "I apologize, but I'm having trouble processing your request right now. The AI reasoning system will be back online shortly. Please try again."
    
    def _update_stats(self, success: bool, start_time: float, error_msg: Optional[str] = None):
        """Update performance statistics."""
        with self._lock:
            if success:
                self.stats['successful_requests'] += 1
            else:
                self.stats['failed_requests'] += 1
                self.stats['last_error'] = error_msg
            
            # Update average response time
            response_time = time.time() - start_time
            total_requests = self.stats['total_requests']
            current_avg = self.stats['average_response_time']
            
            self.stats['average_response_time'] = (
                (current_avg * (total_requests - 1) + response_time) / total_requests
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        with self._lock:
            return self.stats.copy()
    
    def reset_stats(self):
        """Reset performance statistics."""
        with self._lock:
            self.stats = {
                'total_requests': 0,
                'successful_requests': 0,
                'failed_requests': 0,
                'average_response_time': 0,
                'last_error': None
            }
    
    def set_model(self, model_name: str) -> bool:
        """
        Change the active model.
        
        Args:
            model_name: New model name
            
        Returns:
            True if model was changed successfully
        """
        old_model = self.model_name
        self.model_name = model_name
        self._connection_verified = False  # Force recheck
        
        is_connected, msg = self.verify_connection()
        if is_connected:
            logger.info(f"Model changed from {old_model} to {model_name}")
            return True
        else:
            # Revert if new model doesn't work
            self.model_name = old_model
            logger.error(f"Failed to change model to {model_name}: {msg}")
            return False
    
    def _analyze_prompt_with_cuda(self, prompt: str) -> Dict[str, Any]:
        """Analyze prompt characteristics using CUDA acceleration."""
        try:
            # Extract keywords
            keywords = self.cuda_text_processor.extract_keywords(prompt, top_k=5)
            
            # Analyze sentiment
            sentiment = self.cuda_text_processor.analyze_sentiment(prompt)
            
            # Check prompt length and complexity
            word_count = len(prompt.split())
            needs_optimization = word_count > 100 or sentiment['label'] == 'NEGATIVE'
            
            return {
                'keywords': keywords,
                'sentiment': sentiment,
                'word_count': word_count,
                'needs_optimization': needs_optimization,
                'complexity_score': min(word_count / 50, 2.0)  # 0-2 scale
            }
            
        except Exception as e:
            logger.error(f"❌ CUDA prompt analysis failed: {e}")
            return {'needs_optimization': False}
    
    def _optimize_prompt_with_cuda(self, prompt: str, analysis: Dict[str, Any]) -> str:
        """Optimize prompt using CUDA-accelerated text processing."""
        try:
            # If prompt is too long, summarize it
            if analysis.get('word_count', 0) > 150:
                summarized = self.cuda_text_processor.summarize_text(
                    prompt, 
                    max_length=100, 
                    min_length=30
                )
                logger.debug(f"📝 Prompt summarized: {len(prompt)} -> {len(summarized)} chars")
                return summarized
            
            # If sentiment is negative, add positive framing
            if analysis.get('sentiment', {}).get('label') == 'NEGATIVE':
                prompt = f"Please provide a helpful and constructive response to: {prompt}"
                logger.debug("😊 Added positive framing to negative prompt")
            
            return prompt
            
        except Exception as e:
            logger.error(f"❌ CUDA prompt optimization failed: {e}")
            return prompt
    
    def _post_process_response_with_cuda(self, response: str, original_prompt: str) -> str:
        """Post-process response using CUDA acceleration."""
        try:
            # Analyze response quality
            response_sentiment = self.cuda_text_processor.analyze_sentiment(response)
            
            # Check if response is relevant to prompt
            similarity = self.cuda_text_processor.compute_similarity(
                original_prompt, response
            )
            
            # Log quality metrics
            logger.debug(f"📊 Response quality - Sentiment: {response_sentiment['label']}, "
                        f"Similarity: {similarity:.3f}")
            
            # If response quality is poor, add a disclaimer
            if similarity < 0.3:
                response += "\n\n(Note: This response may not be directly relevant to your question. Please let me know if you'd like me to clarify or provide a different response.)"
            
            return response
            
        except Exception as e:
            logger.error(f"❌ CUDA post-processing failed: {e}")
            return response

# Global instance for easy access
_gemma_engine = None

def get_gemma_engine(model_name: str = "gemma3n:latest") -> Gemma3nEngine:
    """Get or create global Gemma3n engine instance."""
    global _gemma_engine
    if _gemma_engine is None:
        _gemma_engine = Gemma3nEngine(model_name=model_name)
    return _gemma_engine

# Test the engine if run directly
if __name__ == "__main__":
    # Test the engine
    engine = Gemma3nEngine()
    
    # Test connection
    is_connected, msg = engine.verify_connection()
    print(f"Connection Status: {is_connected} - {msg}")
    
    if is_connected:
        # Test generation
        test_prompt = """You are a helpful AI assistant for a blind person.
        
Scene Description: A table with a bottle and a glass. Chair on the right.
User Question: What should I do?

Instructions:
- Be careful and considerate
- Provide clear guidance
- Focus on safety"""
        
        print("\\nTesting generation...")
        response = engine.generate_response(test_prompt)
        print(f"Response: {response}")
        
        # Show stats
        stats = engine.get_stats()
        print(f"\\nEngine Stats: {stats}")
    else:
        print("Cannot test generation without connection.")
