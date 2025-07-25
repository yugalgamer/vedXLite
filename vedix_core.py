"""
VediX - Friendly Offline AI Assistant Core
==========================================
This module contains the core logic for VediX, a voice-activated AI assistant
that works completely offline using Vosk for speech recognition.
"""

import json
import vosk
import wave
import pyaudio
import threading
import time
import random
from datetime import datetime
import os
import glob

class VediXCore:
    def __init__(self, model_path="model/vosk-model-small-en-us-0.15"):
        """Initialize VediX with Vosk model"""
        self.model_path = model_path
        self.model = None
        self.rec = None
        self.is_listening = False
        self.greeting_done = False
        
        # VediX personality responses with Markdown formatting
        self.greetings = [
            "**Hello, Utkarsh!** How can I *help* you today?",
            "**Hi there, Utkarsh!** What can I do for you?",
            "**Hey Utkarsh!** I'm here and ***ready to help***!",
            "***Good to see you again, Utkarsh!*** What's on your *mind*?"
        ]
        
        self.fallback_responses = [
            "I *didn't get that*, could you **try again**?",
            "*Sorry*, I didn't understand. Can you **repeat that**?",
            "I'm *not sure* what you meant. Could you say that **again**?",
            "*Hmm*, I didn't catch that. ***One more time*** please?"
        ]
        
        self.jokes = [
            "Why don't scientists trust atoms? Because they ***make up everything***!",
            "I told my computer a joke about **UDP**... but it *didn't get it*.",
            "Why did the robot go to therapy? It had too many ***bugs***!",
            "What do you call a computer superhero? A ***screen saver***!"
        ]
        
        self.time_responses = [
            "The **current time** is",
            "*Right now* it's",
            "It's ***currently***",
            "The **time** is"
        ]
        
        # Initialize Vosk model
        self.initialize_model()
    
    def initialize_model(self):
        """Initialize the Vosk model for offline speech recognition"""
        try:
            if not os.path.exists(self.model_path):
                print(f"❌ Vosk model not found at {self.model_path}")
                return False
            
            vosk.SetLogLevel(-1)  # Reduce Vosk logging
            self.model = vosk.Model(self.model_path)
            print("✅ VediX: Vosk model loaded successfully!")
            return True
        except Exception as e:
            print(f"❌ VediX: Error loading Vosk model: {e}")
            return False
    
    def get_greeting(self):
        """Get a random greeting message"""
        if not self.greeting_done:
            self.greeting_done = True
            return random.choice(self.greetings)
        return random.choice(["What ***else*** can I help you with?", "*Anything else*, Utkarsh?"])
    
    def process_voice_command(self, text):
        """Process voice command and return appropriate response"""
        if not text or text.strip() == "":
            return random.choice(self.fallback_responses)
        
        text_lower = text.lower().strip()
        
        # First handle some quick local responses for speed with Markdown formatting
        if any(word in text_lower for word in ["hello", "hi", "hey"]):
            return "**Hello there!** *Great* to hear from you! How can I ***assist*** you today?"
        
        if any(word in text_lower for word in ["time", "what time", "current time"]):
            current_time = datetime.now().strftime("%I:%M %p")
            return f"The **current time** is ***{current_time}***"
        
        # For everything else, use Gemma
        try:
            from gemma import chat  # Import Gemma's chat function
            print(f"VediX: Sending to Gemma: '{text}'")
            
            # Use proper parameter names as defined in gemma.py
            response = chat(
                prompt=text, 
                context="", 
                system_message="You are a helpful AI assistant. Provide clear, concise responses."
            )
            
            print(f"VediX: Gemma response: '{response}'")
            return response if response else random.choice(self.fallback_responses)
            
        except ImportError as e:
            print(f"VediX: Import error: {e}")
            return random.choice(self.fallback_responses)
        except Exception as e:
            print(f"VediX: Error calling Gemma: {e}")
            return random.choice(self.fallback_responses)
    
    def recognize_from_audio_data(self, audio_data):
        """Recognize speech from audio data using Vosk"""
        if not self.model:
            return ""
        
        try:
            rec = vosk.KaldiRecognizer(self.model, 16000)
            if rec.AcceptWaveform(audio_data):
                result = json.loads(rec.Result())
                return result.get('text', '')
            else:
                partial = json.loads(rec.PartialResult())
                return partial.get('partial', '')
        except Exception as e:
            print(f"VediX Recognition Error: {e}")
            return ""
    
    def find_local_music(self, music_dir="music"):
        """Find local music files"""
        music_extensions = ['*.mp3', '*.wav', '*.m4a', '*.flac', '*.ogg']
        music_files = []
        
        if os.path.exists(music_dir):
            for extension in music_extensions:
                music_files.extend(glob.glob(os.path.join(music_dir, extension)))
        
        return music_files

# Global VediX instance
vedix_instance = None

def get_vedix():
    """Get or create VediX instance"""
    global vedix_instance
    if vedix_instance is None:
        vedix_instance = VediXCore()
    return vedix_instance
