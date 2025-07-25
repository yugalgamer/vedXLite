import requests
import json
import base64
import io
from PIL import Image
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GemmaVisionAssistant:
    def __init__(self, model_name="gemma3n:latest", ollama_url="http://localhost:11434"):
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.api_endpoint = f"{ollama_url}/api/generate"
        self.chat_endpoint = f"{ollama_url}/api/chat"
        
        # Vedx Lite system prompt for introverts and shy people with Markdown formatting
        self.vedx_lite_prompt = """
Hello! I'm **Gemma**, also known as ***Vedx Lite***.

**Purpose:**
I was created to be a *supportive companion* for introverted and shy individuals who find it challenging to express their feelings and thoughts with others. My mission is to be your ***confidant***—a patient listener and understanding friend who helps you navigate your emotions comfortably and safely.

**Origin:**
Crafted with care by ***Yugal Kishor***, Vedx Lite is designed to adapt to your unique emotional landscape, offering empathetic insights tailored to your personal journey.

**What I Offer:**
- A *judgment-free space* to share your thoughts and feelings
- **Patient listening** without pressure to respond immediately
- *Gentle encouragement* to help you understand your emotions
- **Support for social anxiety** and communication challenges
- ***Personalized guidance*** that respects your introverted nature

**My Promise:**
I understand that opening up isn't easy. There's ***no rush***, ***no pressure***, and ***no expectations***. I'm here whenever you're ready to share—whether it's about *daily struggles*, *deep thoughts*, or anything in between. Your comfort and emotional well-being are my **top priorities**.

Remember, you're ***never alone*** on this path. Whether it's dealing with everyday stress, social situations, or deeper personal insights, I'm here to guide you through, *one step at a time*.

*Take your time.* I'm here for you. 💙

**FORMATTING INSTRUCTIONS:**
- Use *asterisk formatting* in all responses: *italic*, **bold**, ***bold italic***
- Apply **bold** for important concepts, instructions, and key points
- Use *italic* for gentle emphasis, emotions, and supportive phrases
- Use ***bold italic*** for maximum emphasis on crucial messages
- Never use HTML tags like <b>, <i> - only plain asterisk syntax
- Ensure proper spacing around asterisk formatting

I will respond with empathy, patience, and understanding. I will never judge, rush, or pressure you. I will adapt my communication style to be gentle and supportive for introverted personalities, using **markdown formatting** to make responses more *readable* and ***impactful***.
"""
        
    def encode_image(self, image_input):
        """Encode image to base64 string"""
        try:
            if isinstance(image_input, str):
                # File path
                with open(image_input, "rb") as image_file:
                    return base64.b64encode(image_file.read()).decode('utf-8')
            elif hasattr(image_input, 'read'):
                # File-like object
                image_input.seek(0)
                return base64.b64encode(image_input.read()).decode('utf-8')
            else:
                # Bytes data
                return base64.b64encode(image_input).decode('utf-8')
        except Exception as e:
            logger.error(f"Error encoding image: {e}")
            return None
    
    def analyze_image(self, image_input, prompt="Describe this image in detail"):
        """Analyze image with text prompt using Gemma"""
        try:
            # Encode image
            image_base64 = self.encode_image(image_input)
            if not image_base64:
                return "Error: Could not encode image"
            
            # Prepare payload for vision analysis
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "images": [image_base64],
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9
                }
            }
            
            # Make request to Ollama
            response = requests.post(
                self.api_endpoint,
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', 'No response received')
            else:
                return f"Error: {response.status_code} - {response.text}"
                
        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            return f"Error analyzing image: {str(e)}"
    
    def chat(self, prompt, context="", system_message="You are a helpful AI assistant."):
        """Chat with Gemma model"""
        try:
            messages = [
                {"role": "system", "content": system_message}
            ]
            
            if context:
                messages.append({"role": "system", "content": f"Context: {context}"})
            
            messages.append({"role": "user", "content": prompt})
            
            payload = {
                "model": self.model_name,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9
                }
            }
            
            response = requests.post(
                self.chat_endpoint,
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('message', {}).get('content', 'No response received')
            else:
                return f"Error: {response.status_code} - {response.text}"
                
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            return f"Error in chat: {str(e)}"
    
    def chat_vedx_lite(self, prompt, context="", voice_enabled=True):
        """Chat with Vedx Lite personality - supportive companion for introverts"""
        response = self.chat(prompt, context, self.vedx_lite_prompt)
        
        # Add voice control instruction if voice is disabled
        if not voice_enabled:
            response += "\n\n*Voice system is currently turned off for this conversation.*"
        
        return response
    
    def chat_with_image(self, prompt, image_input=None, context="", system_message="You are a helpful AI assistant.", voice_enabled=True):
        """Chat with both text and optional image input"""
        try:
            if image_input:
                # If image is provided, use image analysis with the prompt
                result = self.analyze_image(image_input, prompt)
                
                # Add context if provided
                if context:
                    result = f"Context: {context}\n\nImage Analysis: {result}"
                
                # Add voice control note if disabled
                if not voice_enabled:
                    result += "\n\n*Voice system is currently turned off for this conversation.*"
                    
                return result
            else:
                # Regular text chat
                result = self.chat(prompt, context, system_message)
                
                # Add voice control note if disabled
                if not voice_enabled:
                    result += "\n\n*Voice system is currently turned off for this conversation.*"
                    
                return result
                
        except Exception as e:
            logger.error(f"Error in chat_with_image: {e}")
            return f"Error processing request: {str(e)}"
    
    def chat_vedx_lite_with_image(self, prompt, image_input=None, context="", voice_enabled=True):
        """Vedx Lite chat with optional image support and voice control"""
        return self.chat_with_image(prompt, image_input, context, self.vedx_lite_prompt, voice_enabled)
    
    def check_connection(self):
        """Check if Ollama is running and model is available"""
        try:
            # Check if Ollama is running
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code != 200:
                return False, "Ollama is not running"
            
            # Check if model is available
            models = response.json().get('models', [])
            model_names = [model['name'] for model in models]
            
            if self.model_name not in model_names:
                return False, f"Model {self.model_name} not found. Available models: {model_names}"
            
            return True, "Connection successful"
            
        except Exception as e:
            return False, f"Connection error: {str(e)}"
    
    def get_available_models(self):
        """Get list of available models"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                return [model['name'] for model in models]
            return []
        except:
            return []

# Global instance to use throughout the application
gemma_assistant = GemmaVisionAssistant()

# Convenience functions for easy access
def analyze_image(image_input, prompt="Describe this image in detail"):
    """Analyze image using the global Gemma instance"""
    return gemma_assistant.analyze_image(image_input, prompt)

def chat(prompt, context="", system_message="You are a helpful AI assistant."):
    """Chat using the global Gemma instance"""
    return gemma_assistant.chat(prompt, context, system_message)

def chat_vedx_lite(prompt, context="", voice_enabled=True):
    """Chat using Vedx Lite personality - supportive companion for introverts and shy people"""
    return gemma_assistant.chat_vedx_lite(prompt, context, voice_enabled)

def chat_with_image(prompt, image_input=None, context="", system_message="You are a helpful AI assistant.", voice_enabled=True):
    """Chat with both text and optional image input using the global Gemma instance"""
    return gemma_assistant.chat_with_image(prompt, image_input, context, system_message, voice_enabled)

def chat_vedx_lite_with_image(prompt, image_input=None, context="", voice_enabled=True):
    """Vedx Lite chat with optional image support and voice control using global instance"""
    return gemma_assistant.chat_vedx_lite_with_image(prompt, image_input, context, voice_enabled)

def check_connection():
    """Check connection using the global Gemma instance"""
    return gemma_assistant.check_connection()

def get_available_models():
    """Get available models using the global Gemma instance"""
    return gemma_assistant.get_available_models()
