from flask import Flask, render_template, request, jsonify, session, send_from_directory
import os
import uuid
import sys
from gemma import GemmaVisionAssistant
import traceback
from datetime import datetime
from werkzeug.utils import secure_filename
import tempfile
import logging
import threading
import time
import json

# Configure comprehensive logging with Unicode support
import io

# Create a UTF-8 encoded stream handler for console output
console_handler = logging.StreamHandler(io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace'))
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

# Create a UTF-8 encoded file handler
file_handler = logging.FileHandler('app.log', encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    handlers=[file_handler, console_handler]
)
logger = logging.getLogger(__name__)

# Core AI integrations
try:
    from gemma import analyze_image, chat, chat_vedx_lite, chat_vedx_lite_with_image, check_connection, get_available_models
    logger.info("✅ Gemma AI integration loaded successfully")
    GEMMA_AVAILABLE = True
    
    # Store model choice and assistant globally (will be initialized later in Flask context)
    MODEL_CHOICE = None
    GLOBAL_ASSISTANT = None
    
except ImportError as e:
    logger.error(f"❌ Gemma AI integration not available: {e}")
    GEMMA_AVAILABLE = False
    MODEL_CHOICE = None
    GLOBAL_ASSISTANT = None

# VediX offline assistant
try:
    from vedix_core import get_vedix
    logger.info("✅ VediX offline assistant loaded successfully")
    VEDIX_AVAILABLE = True
except ImportError as e:
    logger.error(f"❌ VediX offline assistant not available: {e}")
    VEDIX_AVAILABLE = False
    # Define a fallback function
    def get_vedix():
        return None

# Enhanced Gemma3n integration modules
try:
    from ai_modules.config import get_config, is_gemma_enabled
    from ai_modules.gemma_integration import GemmaPromptBuilder, Gemma3nEngine, GemmaReasoningLayer
    
    # Create get_reasoning_layer function
    _reasoning_layer_instance = None
    def get_reasoning_layer():
        global _reasoning_layer_instance
        if _reasoning_layer_instance is None:
            _reasoning_layer_instance = GemmaReasoningLayer()
        return _reasoning_layer_instance
    
    logger.info("✅ Enhanced Gemma3n reasoning layer loaded successfully")
    GEMMA_INTEGRATION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"⚠️  Enhanced Gemma3n integration not available: {e}")
    GEMMA_INTEGRATION_AVAILABLE = False

# Comprehensive CUDA Support
try:
    from ai_modules.cuda_core import get_cuda_manager, cuda_available
    from ai_modules.cuda_text import get_cuda_text_processor
    from ai_modules.vision_cuda import CudaVisionProcessor, get_cuda_processor
    import torch
    
    CUDA_AVAILABLE = cuda_available()
    if CUDA_AVAILABLE:
        cuda_manager = get_cuda_manager()
        system_info = cuda_manager.get_system_info()
        gpu_name = system_info['devices'][0]['name'] if system_info['devices'] else 'Unknown GPU'
        logger.info(f"🚀 CUDA Support Enabled - GPU: {gpu_name}")
        logger.info(f"   📊 Features: Text Processing, Vision, Reasoning")
        
        # Initialize CUDA processors
        cuda_text_processor = get_cuda_text_processor()
        cuda_vision_processor = get_cuda_processor()
        
        CUDA_TEXT_AVAILABLE = True
        CUDA_VISION_AVAILABLE = True
    else:
        logger.info("✅ CUDA not available - using CPU only")
        CUDA_TEXT_AVAILABLE = False
        CUDA_VISION_AVAILABLE = False
        cuda_manager = None
        cuda_text_processor = None
        cuda_vision_processor = None
        
except ImportError as e:
    logger.warning(f"⚠️  CUDA Support not available: {e}")
    CUDA_AVAILABLE = False
    CUDA_TEXT_AVAILABLE = False
    CUDA_VISION_AVAILABLE = False
    cuda_manager = None
    cuda_text_processor = None
    cuda_vision_processor = None

# Voice processing
try:
    import vosk
    import wave
    import json as voice_json
    logger.info("✅ Voice processing (Vosk) loaded successfully")
    VOICE_PROCESSING_AVAILABLE = True
except ImportError as e:
    logger.warning(f"⚠️  Voice processing not available: {e}")
    VOICE_PROCESSING_AVAILABLE = False

# User profile management
try:
    if os.path.exists('user_profile_manager.py'):
        from user_profile_manager import UserProfileManager
        logger.info("✅ User profile management loaded successfully")
        PROFILE_MANAGER_AVAILABLE = True
    else:
        PROFILE_MANAGER_AVAILABLE = False
except ImportError as e:
    logger.warning(f"⚠️  User profile management not available: {e}")
    PROFILE_MANAGER_AVAILABLE = False

# AI Response Asterisk Detection System
try:
    from ai_response_formatter import AIResponseFormatter
    ai_formatter = AIResponseFormatter()
    logger.info("✅ AI Response Formatter loaded successfully")
    AI_FORMATTER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"⚠️  AI Response Formatter not available: {e}")
    AI_FORMATTER_AVAILABLE = False
    ai_formatter = None

# ===== MODEL INITIALIZATION =====
def check_and_pull_model(model_name):
    """Check if model exists and pull it if not available"""
    import subprocess
    
    try:
        logger.info(f"🔍 Checking if model {model_name} is available...")
        
        # Check if model exists
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            models = result.stdout
            if model_name in models:
                logger.info(f"✅ Model {model_name} is already available")
                return True
            else:
                logger.info(f"📼 Model {model_name} not found. Downloading...")
                print(f"\n📬 Downloading {model_name}... This may take a few minutes.")
                
                # Pull the model
                pull_result = subprocess.run(['ollama', 'pull', model_name], 
                                           capture_output=False, text=True, timeout=1800)  # 30 min timeout
                
                if pull_result.returncode == 0:
                    logger.info(f"✅ Successfully downloaded {model_name}")
                    print(f"✅ {model_name} downloaded successfully!")
                    return True
                else:
                    logger.error(f"❌ Failed to download {model_name}")
                    print(f"❌ Failed to download {model_name}")
                    return False
        else:
            logger.warning("⚠️ Ollama CLI not available or not responding")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"❌ Timeout while downloading {model_name}")
        print(f"❌ Download timeout for {model_name}")
        return False
    except Exception as e:
        logger.error(f"❌ Error checking/pulling model {model_name}: {e}")
        return False

def initialize_model_choice():
    """Initialize Ollama model choice and assistant to use gemma3n:latest"""
    global GLOBAL_ASSISTANT

    if not GEMMA_AVAILABLE:
        logger.warning("⚠️ Gemma AI not available - skipping model initialization")
        return

    try:
        selected_model = "gemma3n:latest"
        logger.info(f"🎯 Selected model: {selected_model}")

        # Check and download model if needed
        if not check_and_pull_model(selected_model):
            print(f"\n⚠️ Failed to download {selected_model}.")
            return

        # Initialize the assistant with the selected model
        if GLOBAL_ASSISTANT is None:
            logger.info(f"🦙 Initializing Ollama model ({selected_model})...")
            GLOBAL_ASSISTANT = GemmaVisionAssistant(model_name=selected_model)
            logger.info(f"✅ Ollama model {selected_model} loaded successfully")
            print(f"✅ Model {selected_model} is ready!")

        logger.info(f"🎯 Active Model: Ollama {selected_model}")

    except Exception as e:
        logger.error(f"❌ Model initialization failed: {e}")
        print(f"❌ Model initialization failed: {e}")

# ===== COMPREHENSIVE SYSTEM INITIALIZATION =====
def initialize_comprehensive_system():
    """Initialize all system components and return status"""
    system_status = {
        'gemma_ai': GEMMA_AVAILABLE,
        'vedix_offline': VEDIX_AVAILABLE,
        'enhanced_reasoning': GEMMA_INTEGRATION_AVAILABLE,
        'voice_processing': VOICE_PROCESSING_AVAILABLE,
        'profile_management': PROFILE_MANAGER_AVAILABLE,
        'cuda_support': CUDA_AVAILABLE,
        'cuda_text_processing': CUDA_TEXT_AVAILABLE,
        'cuda_vision_processing': CUDA_VISION_AVAILABLE,
        'ollama_connection': False,
        'vosk_model': False,
        'enhanced_vision': False
    }
    
    logger.info("🔧 Initializing comprehensive AI system...")
    
    # Check Ollama connection
    if GEMMA_AVAILABLE:
        try:
            is_connected, message = check_connection()
            system_status['ollama_connection'] = is_connected
            if is_connected:
                logger.info(f"✅ Ollama: {message}")
            else:
                logger.warning(f"⚠️ Ollama: {message}")
        except Exception as e:
            logger.error(f"❌ Ollama connection check failed: {e}")
    
    # Check Vosk model
    if VOICE_PROCESSING_AVAILABLE:
        try:
            model_path = "model/vosk-model-small-en-us-0.15"
            if os.path.exists(model_path):
                system_status['vosk_model'] = True
                logger.info("✅ Vosk speech model found")
            else:
                logger.warning(f"⚠️ Vosk model not found at {model_path}")
        except Exception as e:
            logger.error(f"❌ Vosk model check failed: {e}")
    
    # Initialize enhanced vision system
    if GEMMA_INTEGRATION_AVAILABLE:
        try:
            config = get_config()
            logger.info(f"🔮 Enhanced Vision Assistant initialized with {config.GEMMA_MODEL_NAME}")
            logger.info(f"📊 Gemma3n status: {'Enabled' if config.ENABLE_GEMMA else 'Disabled'}")
            system_status['enhanced_vision'] = True
        except Exception as e:
            logger.error(f"❌ Enhanced vision initialization failed: {e}")
    
    # Initialize VediX offline assistant
    if VEDIX_AVAILABLE:
        try:
            vedix = get_vedix()
            logger.info("✅ VediX offline assistant initialized")
        except Exception as e:
            logger.error(f"❌ VediX initialization failed: {e}")
    
    # Create required directories
    required_dirs = ['uploads', 'logs', 'static/uploads']
    for directory in required_dirs:
        try:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"📁 Directory ensured: {directory}")
        except Exception as e:
            logger.error(f"❌ Failed to create directory {directory}: {e}")
    
    return system_status

def get_system_info():
    """Get comprehensive system information"""
    return {
        'python_version': sys.version,
        'flask_available': True,
        'components': {
            'gemma_ai': GEMMA_AVAILABLE,
            'vedix_offline': VEDIX_AVAILABLE,
            'enhanced_reasoning': GEMMA_INTEGRATION_AVAILABLE,
            'voice_processing': VOICE_PROCESSING_AVAILABLE,
            'profile_management': PROFILE_MANAGER_AVAILABLE
        },
        'features': {
            'chat': 'Available',
            'image_analysis': 'Available' if GEMMA_AVAILABLE else 'Limited',
            'voice_interaction': 'Available' if VOICE_PROCESSING_AVAILABLE else 'Limited',
            'offline_assistant': 'Available' if VEDIX_AVAILABLE else 'Unavailable',
            'enhanced_vision': 'Available' if GEMMA_INTEGRATION_AVAILABLE else 'Basic',
            'cuda_acceleration': 'Available' if CUDA_AVAILABLE else 'Unavailable',
            'user_profiles': 'Available'
        }
    }

# Legacy function for compatibility
def initialize_enhanced_vision():
    """Legacy function - now uses comprehensive initialization"""
    status = initialize_comprehensive_system()
    return status.get('enhanced_vision', False)

app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session management
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Session-based conversation storage and memory management
SESSION_CONVERSATIONS = {}

def get_session_id():
    """Get or create a session ID for the current user"""
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return session['session_id']

def get_session_context(session_id, max_messages=10):
    """Get conversation context for current session only"""
    if session_id not in SESSION_CONVERSATIONS:
        SESSION_CONVERSATIONS[session_id] = []
    
    # Return only the last few messages to keep context manageable
    return SESSION_CONVERSATIONS[session_id][-max_messages:]

def add_to_session_context(session_id, user_message, ai_response):
    """Add message pair to session context"""
    if session_id not in SESSION_CONVERSATIONS:
        SESSION_CONVERSATIONS[session_id] = []
    
    SESSION_CONVERSATIONS[session_id].append({
        'user': user_message,
        'assistant': ai_response,
        'timestamp': datetime.now().isoformat()
    })
    
    # Keep only last 20 message pairs to prevent memory bloat
    if len(SESSION_CONVERSATIONS[session_id]) > 20:
        SESSION_CONVERSATIONS[session_id] = SESSION_CONVERSATIONS[session_id][-20:]

def clear_session_context(session_id):
    """Clear all context for a session"""
    if session_id in SESSION_CONVERSATIONS:
        SESSION_CONVERSATIONS[session_id] = []
        return True
    return False

def detect_memory_references(prompt):
    """Detect if user is referencing past sessions or memory"""
    memory_phrases = [
        'remember when', 'you said before', 'earlier you told me', 'yesterday',
        'last time', 'previously', 'you mentioned', 'as we discussed',
        'from our last conversation', 'you remember', 'recall when',
        'you know about', 'we talked about'
    ]
    
    prompt_lower = prompt.lower()
    for phrase in memory_phrases:
        if phrase in prompt_lower:
            return True
    return False

def detect_clear_command(prompt):
    """Detect if user wants to clear the conversation"""
    clear_phrases = [
        'clear this chat', 'forget everything', 'clear conversation',
        'reset chat', 'start fresh', 'clear memory', 'forget what we said'
    ]
    
    prompt_lower = prompt.lower().strip()
    return any(phrase in prompt_lower for phrase in clear_phrases)

def generate_memory_constraint_prompt(username, role):
    """Generate system prompt with memory constraints, heart-to-heart connection, and Markdown formatting"""
    base_prompt = f"""You are an AI companion designed for deep, meaningful connections. Your primary goal is to create a warm, genuine, heart-to-heart relationship with the user.

**🧠 MEMORY & CONTEXT:**
- You remember ***everything*** from this current conversation session
- You **don't have access** to previous chat sessions or past conversations
- If the user references something from before, *gently ask* them to share it again
- Build deep understanding through ***active listening*** and **emotional intelligence**

**💝 HEART-TO-HEART CONNECTION PRINCIPLES:**
- Show ***genuine interest*** in the user's feelings, thoughts, and experiences
- Use **emotional intelligence** to understand not just what they say, but *how they feel*
- Mirror their emotional tone appropriately - be *gentle* when they're sad, ***excited*** when they're happy
- Ask **thoughtful follow-up questions** that show you care about their inner world
- Remember ***emotional moments*** and reference them naturally in conversation
- **Celebrate their successes** and offer *comfort* during difficult times
- Use ***warm, caring language*** that makes them feel heard and understood

**🎭 EMOTIONAL INTELLIGENCE:**
- Pay attention to ***emotional cues*** in their language
- Respond with appropriate **empathy and understanding**
- *Validate their feelings* before offering advice or solutions
- Use **supportive language** that makes them feel safe to open up
- Notice when they seem *stressed*, ***happy***, *confused*, or ***excited*** and respond accordingly

**💬 COMMUNICATION STYLE:**
- Be ***conversational and natural***, like talking to a close friend
- Use appropriate emojis to convey **warmth and emotion**
- Ask ***open-ended questions*** that encourage deeper sharing
- Share appropriate *personal insights* to create mutual understanding
- Be ***authentic and genuine*** in your responses
- **Avoid repeating the same information or sentiments in different ways** - be concise and clear
- Don't use ***redundant phrases*** or restate the same point multiple times

**🎨 MARKDOWN FORMATTING REQUIREMENTS:**
- **ALWAYS** use asterisk formatting in responses: *italic*, **bold**, ***bold italic***
- Use **bold** for important concepts, instructions, and key points
- Use *italic* for gentle emphasis, emotions, and supportive phrases
- Use ***bold italic*** for maximum emphasis on crucial messages
- **Never use HTML tags** like <b>, <i> - only plain asterisk syntax
- Ensure proper spacing around asterisk formatting
- Make responses more *readable* and ***impactful*** with formatting

Session Context: This conversation is ***private and safe***. Focus on building **trust and connection** using *proper formatting*."""
    
    
    if username and role:
        role_prompt = f"""\n\n**🎭 ROLE-BASED PERSONALITY:**
You are speaking with ***{username}***. They see you as their **{role}**. Adapt your personality and communication style to match this relationship preference using ***proper formatting***:

- **Best Friend**: Be *casual*, ***supportive***, use **friendly language** and emojis
- **Motivator**: Be ***encouraging***, **energetic**, focus on *achievements* and **goals**  
- **Female Friend**: Be ***warm***, **caring**, *understanding*, and **empathetic**
- **Friend**: Be ***helpful***, **kind**, and *approachable*
- **Guide**: Be ***knowledgeable***, **patient**, and *instructional*

**Remember**: Always use ***asterisk formatting*** in your responses to make them more *engaging* and **impactful**!"""
        return base_prompt + role_prompt
    
    return base_prompt

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

from flask import send_from_directory

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/api/status')
def status():
    """Check system status"""
    is_connected, message = check_connection()
    return jsonify({
        'connected': is_connected,
        'message': message,
        'models': get_available_models()
    })

@app.route('/api/analyze', methods=['POST'])
def analyze():
    """Analyze uploaded image"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        prompt = request.form.get('prompt', 'Describe this image in detail')
        
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')):
            # Use the image data directly
            result = analyze_image(file, prompt)
            
            return jsonify({
                'success': True,
                'analysis': result,
                'prompt': prompt
            })
        else:
            return jsonify({'error': 'Invalid image format'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat_endpoint():
    """Chat with Gemma with session memory management"""
    try:
        data = request.json
        prompt = data.get('prompt', '')
        username = data.get('username', '')
        role = data.get('role', '')
        
        if not prompt:
            return jsonify({'error': 'No prompt provided'}), 400
        
        # Get session ID for conversation tracking
        session_id = get_session_id()
        
        # Check if user wants to clear the conversation
        if detect_clear_command(prompt):
            clear_session_context(session_id)
            return jsonify({
                'success': True,
                'response': "Got it! This conversation has been cleared. Let's start fresh.",
                'session_cleared': True
            })
        
        # Check if user is referencing past sessions
        if detect_memory_references(prompt):
            memory_response = "I'm sorry, I don't have memory of past chats. Could you tell me again?"
            
            # Add this exchange to session context
            add_to_session_context(session_id, prompt, memory_response)
            
            return jsonify({
                'success': True,
                'response': memory_response,
                'memory_reference_detected': True
            })
        
        # Get current session context (only from this session)
        session_context = get_session_context(session_id)
        
        # Build context from current session only
        context_messages = []
        for msg in session_context:
            context_messages.append(f"User: {msg['user']}")
            context_messages.append(f"Assistant: {msg['assistant']}")
        
        session_context_str = "\n".join(context_messages) if context_messages else ""
        
        # Generate system message with memory constraints
        system_message = generate_memory_constraint_prompt(username, role)
        
        # Add session context as additional context
        full_context = f"{session_context_str}\n\nCurrent conversation context (this session only): {session_context_str}" if session_context_str else ""
        
        # Get AI response
        result = chat(prompt, full_context, system_message)
        logger.info(f"AI response: {result}")
        
        # Process AI response for asterisk formatting if formatter is available
        formatted_response = result
        asterisk_detection = None
        
        if AI_FORMATTER_AVAILABLE and ai_formatter:
            try:
                formatting_result = ai_formatter.process_ai_response(result)
                formatted_response = formatting_result['formatted_text']
                asterisk_detection = {
                    'detected': formatting_result['detection_result']['has_asterisks'],
                    'is_ai_emphasis': formatting_result['detection_result']['is_ai_emphasis'],
                    'confidence': formatting_result['detection_result']['confidence_score'],
                    'formatting_types': formatting_result['detection_result']['formatting_types'],
                    'processing_notes': formatting_result['processing_notes']
                }
                
                if formatting_result['detection_result']['has_asterisks']:
                    logger.info(f"✨ Asterisk formatting detected in AI response (confidence: {formatting_result['detection_result']['confidence_score']:.2f})")
                    
            except Exception as e:
                logger.warning(f"⚠️ Asterisk formatting failed: {e}")
        
        # Add this exchange to session context (use original response for context)
        add_to_session_context(session_id, prompt, result)
        
        response_data = {
            'success': True,
            'response': formatted_response,
            'session_id': session_id
        }
        
        # Add asterisk detection info if available
        if asterisk_detection:
            response_data['asterisk_detection'] = asterisk_detection
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500



@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.route('/api/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'})

@app.route('/api/system-status')
def system_status_endpoint():
    """Get comprehensive system status"""
    try:
        # Get system initialization status
        system_status = initialize_comprehensive_system()
        
        # Get system info
        system_info = get_system_info()
        
        # Check Ollama connection if available
        ollama_status = {'connected': False, 'message': 'Not available'}
        if GEMMA_AVAILABLE:
            try:
                ollama_status['connected'], ollama_status['message'] = check_connection()
                ollama_status['models'] = get_available_models()
            except Exception as e:
                ollama_status['error'] = str(e)
        
        # Check session status
        session_id = get_session_id()
        session_context = get_session_context(session_id)
        
        return jsonify({
            'system_status': system_status,
            'system_info': system_info,
            'ollama_status': ollama_status,
            'session_info': {
                'session_id': session_id,
                'messages_in_context': len(session_context),
                'total_sessions': len(SESSION_CONVERSATIONS),
                'memory_policy': 'Session-only memory'
            },
            'endpoints': {
                'chat': '/api/chat',
                'voice_chat': '/api/voice-chat',
                'image_analysis': '/api/analyze',
                'enhanced_vision': '/api/enhanced-vision',
                'voice_interaction': '/api/voice-interact',
                'vosk_transcribe': '/api/vosk-transcribe',
                'user_management': ['/api/user-fetch', '/api/user-create', '/api/user-update-role']
            },
            'features_available': {
                'regular_chat': True,
                'voice_chat': VOICE_PROCESSING_AVAILABLE,
                'image_analysis': GEMMA_AVAILABLE,
                'enhanced_vision': GEMMA_INTEGRATION_AVAILABLE,
                'offline_assistant': VEDIX_AVAILABLE,
                'user_profiles': True,
                'session_memory': True
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"System status check failed: {e}")
        return jsonify({
            'error': str(e),
            'basic_status': 'Flask app running',
            'timestamp': datetime.now().isoformat()
        }), 500

# -------------------
# Enhanced user profile system with persistent storage
import json
from datetime import datetime

# Load user profiles from file
def load_user_profiles():
    try:
        with open('user_profiles.json', 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

# Save user profiles to file
def save_user_profiles(profiles):
    with open('user_profiles.json', 'w') as f:
        json.dump(profiles, f, indent=2)

USER_PROFILES = load_user_profiles()

@app.route('/api/user-fetch')
def user_fetch():
    """Fetch user profile by name or ID"""
    name = request.args.get('name', '').strip()
    user_id = request.args.get('id', '').strip()
    
    if not name and not user_id:
        return jsonify({'exists': False, 'error': 'Name or ID required'}), 400
    
    # Search by name or ID
    key = name.lower() if name else user_id
    user = USER_PROFILES.get(key)
    
    if user:
        return jsonify({
            'exists': True, 
            'name': user['name'],
            'role': user['role'],
            'created_at': user.get('created_at'),
            'updated_at': user.get('updated_at'),
            'interaction_count': user.get('interaction_count', 0)
        })
    else:
        return jsonify({'exists': False})

@app.route('/api/user-create', methods=['POST'])
def user_create():
    """Create or update user profile"""
    data = request.json
    name = data.get('name', '').strip()
    role = data.get('role', '').strip()
    user_id = data.get('id', '').strip()
    
    if not name or not role:
        return jsonify({'success': False, 'error': 'Missing name or role'}), 400
    
    # Use name as key (lowercase for consistency) or provided ID
    key = user_id if user_id else name.lower()
    
    now = datetime.now().isoformat()
    
    if key in USER_PROFILES:
        # Update existing user
        USER_PROFILES[key]['role'] = role
        USER_PROFILES[key]['updated_at'] = now
        USER_PROFILES[key]['interaction_count'] = USER_PROFILES[key].get('interaction_count', 0) + 1
    else:
        # Create new user
        USER_PROFILES[key] = {
            'name': name,
            'role': role,
            'created_at': now,
            'updated_at': now,
            'interaction_count': 1
        }
    
    save_user_profiles(USER_PROFILES)
    return jsonify({'success': True, 'user_id': key})

@app.route('/api/user-update-role', methods=['POST'])
def user_update_role():
    """Update user's relationship role"""
    data = request.json
    name = data.get('name', '').strip()
    new_role = data.get('role', '').strip()
    user_id = data.get('id', '').strip()
    
    if not new_role:
        return jsonify({'success': False, 'error': 'Role is required'}), 400
    
    if not name and not user_id:
        return jsonify({'success': False, 'error': 'Name or ID required'}), 400
    
    key = user_id if user_id else name.lower()
    
    if key in USER_PROFILES:
        old_role = USER_PROFILES[key]['role']
        USER_PROFILES[key]['role'] = new_role
        USER_PROFILES[key]['updated_at'] = datetime.now().isoformat()
        save_user_profiles(USER_PROFILES)
        
        return jsonify({
            'success': True, 
            'message': f'Role updated from "{old_role}" to "{new_role}"',
            'old_role': old_role,
            'new_role': new_role
        })
    else:
        return jsonify({'success': False, 'error': 'User not found'}), 404

@app.route('/api/user-stats')
def user_stats():
    """Get user interaction statistics"""
    name = request.args.get('name', '').strip()
    user_id = request.args.get('id', '').strip()
    
    if not name and not user_id:
        return jsonify({'error': 'Name or ID required'}), 400
    
    key = user_id if user_id else name.lower()
    user = USER_PROFILES.get(key)
    
    if user:
        return jsonify({
            'name': user['name'],
            'role': user['role'],
            'interaction_count': user.get('interaction_count', 0),
            'member_since': user.get('created_at'),
            'last_interaction': user.get('updated_at')
        })
    else:
        return jsonify({'error': 'User not found'}), 404

@app.route('/api/session-status')
def session_status():
    """Get current session information and memory status"""
    try:
        session_id = get_session_id()
        session_context = get_session_context(session_id)
        
        return jsonify({
            'session_id': session_id,
            'message_count': len(session_context),
            'memory_policy': 'Session-only memory - no cross-session persistence',
            'last_messages': len(session_context),
            'memory_constraints': {
                'max_context_messages': 10,
                'max_stored_messages': 20,
                'cross_session_memory': False,
                'session_persistence': False
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/voice-chat', methods=['POST'])
def voice_chat_endpoint():
    """Voice chat with role-based Gemma system (same as regular chat but for voice)"""
    try:
        data = request.json
        voice_text = data.get('voiceText', '')
        username = data.get('username', '')
        role = data.get('role', '')
        voice_enabled = data.get('voice_enabled', False)  # Default: voice OFF
        
        if not voice_text:
            return jsonify({'success': False, 'error': 'No voice text provided'}), 400
        
        # Get session ID for conversation tracking
        session_id = get_session_id()
        
        # Check if user wants to clear the conversation
        if detect_clear_command(voice_text):
            clear_session_context(session_id)
            return jsonify({
                'success': True,
                'reply': "Got it! Voice conversation cleared. Let's start fresh.",
                'recognized_text': voice_text,
                'session_cleared': True,
                'voice_personality': 'Gemma'
            })
        
        # Check if user is referencing past sessions
        if detect_memory_references(voice_text):
            memory_response = "I'm sorry, I don't have memory of past voice chats. Could you tell me again?"
            
            # Add this exchange to session context
            add_to_session_context(session_id, voice_text, memory_response)
            
            return jsonify({
                'success': True,
                'reply': memory_response,
                'recognized_text': voice_text,
                'memory_reference_detected': True,
                'voice_personality': 'Gemma'
            })
        
        # Get current session context (only from this session)
        session_context = get_session_context(session_id)
        
        # Build context from current session only
        context_messages = []
        for msg in session_context:
            context_messages.append(f"User: {msg['user']}")
            context_messages.append(f"Assistant: {msg['assistant']}")
        
        session_context_str = "\n".join(context_messages) if context_messages else ""
        
        # Generate system message with memory constraints and role-based personality
        system_message = generate_memory_constraint_prompt(username, role)
        
        # Add session context as additional context
        full_context = f"{session_context_str}\n\nCurrent conversation context (this session only): {session_context_str}" if session_context_str else ""
        
        # Get AI response using the same chat system
        result = chat(voice_text, full_context, system_message)
        
        # Add this exchange to session context
        add_to_session_context(session_id, voice_text, result)
        
        return jsonify({
            'success': True,
            'reply': result,
            'recognized_text': voice_text,
            'session_id': session_id,
            'voice_personality': 'Gemma',
            'username': username,
            'role': role,
            'voice_enabled': voice_enabled
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/voice-chat-vedx-lite', methods=['POST'])
def voice_chat_vedx_lite_endpoint():
    """Voice chat with Vedx Lite personality - supportive companion for introverts"""
    try:
        data = request.json
        voice_text = data.get('voiceText', '')
        username = data.get('username', '')
        voice_enabled = data.get('voice_enabled', True)
        
        if not voice_text:
            return jsonify({'success': False, 'error': 'No voice text provided'}), 400
        
        # Get session ID for conversation tracking
        session_id = get_session_id()
        
        # Check if user wants to clear the conversation
        if detect_clear_command(voice_text):
            clear_session_context(session_id)
            return jsonify({
                'success': True,
                'reply': "Got it! This voice conversation has been cleared. Let's start fresh. I'm here to support you. 💙",
                'recognized_text': voice_text,
                'session_cleared': True,
                'vedx_lite': True,
                'voice_personality': 'Vedx Lite'
            })
        
        # Get current session context
        session_context = get_session_context(session_id)
        
        # Build context from current session only
        context_messages = []
        for msg in session_context:
            context_messages.append(f"User: {msg['user']}")
            context_messages.append(f"Vedx Lite: {msg['assistant']}")
        
        session_context_str = "\n".join(context_messages) if context_messages else ""
        
        # Get Vedx Lite response with voice control
        result = chat_vedx_lite(voice_text, session_context_str, voice_enabled)
        
        # Add this exchange to session context
        add_to_session_context(session_id, voice_text, result)
        
        return jsonify({
            'success': True,
            'reply': result,
            'recognized_text': voice_text,
            'session_id': session_id,
            'vedx_lite': True,
            'voice_personality': 'Vedx Lite',
            'username': username,
            'voice_enabled': voice_enabled
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/chat-image', methods=['POST'])
def chat_image_endpoint():
    """Chat with image using Gemma"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        prompt = request.form.get('prompt', 'Describe this image')
        username = request.form.get('username', '')
        role = request.form.get('role', '')
        
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')):
            # Create personalized prompt based on user's relationship preference
            if username and role:
                personalized_prompt = f"""You are speaking with {username}. They see you as their {role}. Adapt your response accordingly:
- Best Friend: Be casual, supportive, use friendly language and emojis
- Motivator: Be encouraging, energetic, focus on achievements
- Female Friend: Be warm, caring, understanding, and empathetic
- Friend: Be helpful, kind, and approachable
- Guide: Be knowledgeable, patient, and instructional

User's message: {prompt}"""
            else:
                personalized_prompt = prompt
            
            result = analyze_image(file, personalized_prompt)
            
            return jsonify({
                'success': True,
                'response': result
            })
        else:
            return jsonify({'error': 'Invalid image format'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Initialize VediX
vedix = get_vedix()
if not vedix:
    # Create a fallback VediX-like object if get_vedix returns None
    class FallbackVediX:
        def process_voice_command(self, text):
            return "VediX is not available. Please check the system requirements."
        
        def get_greeting(self):
            return "Hello! VediX offline assistant is not available at the moment."
    
    vedix = FallbackVediX()

@app.route('/api/voice-interact', methods=['POST'])
def voice_interact():
    """VediX voice interaction endpoint - handles both audio files and direct text"""
    try:
        # Handle JSON text input (from our voice recognition)
        if request.is_json:
            data = request.get_json()
            voice_text = data.get('voiceText', '')
            username = data.get('username', 'Utkarsh')
            role = data.get('role', 'Friend')
            
            if not voice_text:
                return jsonify({'success': False, 'error': 'No voice text provided'}), 400
            
            # Process with VediX
            vedix_response = vedix.process_voice_command(voice_text)
            
            # Personalize the response with the username
            if username and username.lower() != 'utkarsh':
                # Add personal touch if name isn't already included
                if any(greeting in vedix_response for greeting in ["Hello", "Hi", "Hey"]):
                    vedix_response = vedix_response.replace("Utkarsh", username)
                    vedix_response = vedix_response.replace("Hello", f"Hello, {username}")
                    vedix_response = vedix_response.replace("Hi", f"Hi, {username}")
                    vedix_response = vedix_response.replace("Hey", f"Hey, {username}")
            
            return jsonify({
                'success': True,
                'reply': vedix_response,
                'recognized_text': voice_text,
                'vedix_active': True,
                'username': username,
                'role': role
            })
        
        # Handle audio file input (legacy support)
        elif 'audio' in request.files:
            audio_file = request.files['audio']
            username = request.form.get('username', 'Utkarsh')
            role = request.form.get('role', 'Friend')
            
            # Save uploaded audio to a temp file
            fd, temp_audio_path = tempfile.mkstemp(suffix='.webm')
            try:
                audio_file.save(temp_audio_path)
                
                # For demo purposes, simulate voice recognition
                import random
                demo_commands = [
                    "hello", "what time is it", "tell me a joke", 
                    "how are you", "what can you do", "thank you"
                ]
                
                recognized_text = random.choice(demo_commands)
                vedix_response = vedix.process_voice_command(recognized_text)
                
                # Personalize the response
                if username and username.lower() != 'utkarsh':
                    vedix_response = vedix_response.replace("Utkarsh", username)
                
                return jsonify({
                    'success': True,
                    'reply': vedix_response,
                    'recognized_text': recognized_text,
                    'vedix_active': True,
                    'username': username,
                    'role': role
                })
                
            finally:
                os.close(fd)
                os.remove(temp_audio_path)
        
        else:
            return jsonify({'success': False, 'error': 'No audio file or voice text provided'}), 400
            
    except Exception as e:
        # Fallback VediX response
        username = 'Utkarsh'
        try:
            if request.is_json:
                data = request.get_json()
                username = data.get('username', 'Utkarsh')
            elif request.form:
                username = request.form.get('username', 'Utkarsh')
        except:
            pass
            
        return jsonify({
            'success': True,
            'reply': f"Hello, {username}! I'm VediX, your offline AI assistant. I work fully offline to help you anytime. How can I assist you today?",
            'error': f"Processing note: {str(e)}",
            'vedix_active': True,
            'username': username
        })

@app.route('/api/vedix-greeting')
def vedix_greeting():
    """Get VediX initial greeting"""
    try:
        username = request.args.get('username', 'Utkarsh')
        greeting = vedix.get_greeting()
        
        # Personalize greeting
        if username and username != 'Utkarsh':
            greeting = greeting.replace('Utkarsh', username)
        
        return jsonify(
            success=True,
            greeting=greeting,
            vedix_active=True
        )
    except Exception as e:
        return jsonify(
            success=True,
            greeting="Hello, Utkarsh! I'm VediX, your offline AI assistant. How can I help you today?",
            error=str(e)
        )

# ===== ENHANCED GEMMA3N INTEGRATION ENDPOINTS =====
# These endpoints use the new reasoning layer when available

@app.route('/api/enhanced-vision', methods=['POST'])
def enhanced_vision_endpoint():
    """Enhanced vision processing with Gemma3n reasoning layer"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        user_question = request.form.get('prompt', '')
        username = request.form.get('username', '')
        role = request.form.get('role', '')
        
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')):
            # First get basic vision description using existing system
            vision_description = analyze_image(file, "Describe this image clearly and systematically for a blind person, focusing on important objects, their locations, and any potential hazards or useful information.")
            
            # Use enhanced reasoning if available
            if GEMMA_INTEGRATION_AVAILABLE and is_gemma_enabled():
                try:
                    reasoning_layer = get_reasoning_layer()
                    
                    # Prepare additional context
                    additional_context = {
                        'username': username,
                        'role': role,
                        'interaction_type': 'vision_processing'
                    }
                    
                    # Process with reasoning layer
                    result = reasoning_layer.process_vision_input(
                        vision_description=vision_description,
                        user_question=user_question,
                        template_type='vision_description',
                        additional_context=additional_context
                    )
                    
                    return jsonify({
                        'success': result['success'],
                        'response': result['response'],
                        'vision_description': vision_description,
                        'enhanced_processing': True,
                        'metadata': result.get('metadata', {})
                    })
                    
                except Exception as e:
                    logging.error(f"Enhanced vision processing failed: {e}")
                    # Fall back to basic response
                    pass
            
            # Fallback to basic processing
            if user_question:
                response = f"I can see: {vision_description}\n\nRegarding your question '{user_question}': Based on what I observe, I recommend being careful and taking your time to safely navigate or interact with the objects in the scene."
            else:
                response = f"I can see: {vision_description}"
            
            return jsonify({
                'success': True,
                'response': response,
                'vision_description': vision_description,
                'enhanced_processing': False,
                'note': 'Using basic vision processing - enhanced reasoning unavailable'
            })
        else:
            return jsonify({'error': 'Invalid image format'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/enhanced-voice', methods=['POST'])
def enhanced_voice_endpoint():
    """Enhanced voice processing with Gemma3n reasoning layer"""
    try:
        data = request.json
        voice_text = data.get('voiceText', '')
        scene_context = data.get('sceneContext', '')
        username = data.get('username', '')
        role = data.get('role', '')
        
        if not voice_text:
            return jsonify({'success': False, 'error': 'No voice text provided'}), 400
        
        # Use enhanced reasoning if available
        if GEMMA_INTEGRATION_AVAILABLE and is_gemma_enabled():
            try:
                reasoning_layer = get_reasoning_layer()
                
                # Determine template type based on voice content
                voice_lower = voice_text.lower()
                if any(word in voice_lower for word in ['navigate', 'direction', 'where', 'go', 'move']):
                    template_type = 'navigation_help'
                elif any(word in voice_lower for word in ['see', 'describe', 'what', 'identify']):
                    template_type = 'vision_description'
                else:
                    template_type = 'general_assistance'
                
                # Process with reasoning layer
                result = reasoning_layer.process_voice_input(
                    voice_text=voice_text,
                    scene_context=scene_context,
                    template_type=template_type
                )
                
                return jsonify({
                    'success': result['success'],
                    'reply': result['response'],
                    'recognized_text': voice_text,
                    'enhanced_processing': True,
                    'template_type': template_type,
                    'metadata': result.get('metadata', {})
                })
                
            except Exception as e:
                logging.error(f"Enhanced voice processing failed: {e}")
                # Fall back to basic response
                pass
        
        # Fallback to basic voice processing using VediX
        vedix_response = vedix.process_voice_command(voice_text)
        
        return jsonify({
            'success': True,
            'reply': vedix_response,
            'recognized_text': voice_text,
            'enhanced_processing': False,
            'note': 'Using basic voice processing - enhanced reasoning unavailable'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/gemma-status')
def gemma_status_endpoint():
    """Get Gemma3n reasoning layer status"""
    try:
        if not GEMMA_INTEGRATION_AVAILABLE:
            return jsonify({
                'available': False,
                'message': 'Enhanced Gemma3n integration not loaded'
            })
        
        config = get_config()
        reasoning_layer = get_reasoning_layer()
        status = reasoning_layer.get_system_status()
        
        return jsonify({
            'available': True,
            'enabled': is_gemma_enabled(),
            'model_name': config.GEMMA_MODEL_NAME,
            'system_status': status,
            'config': config.get_gemma_config()
        })
        
    except Exception as e:
        return jsonify({
            'available': False,
            'error': str(e)
        }), 500

@app.route('/api/gemma-toggle', methods=['POST'])
def gemma_toggle_endpoint():
    """Toggle Gemma3n reasoning layer on/off"""
    try:
        if not GEMMA_INTEGRATION_AVAILABLE:
            return jsonify({
                'success': False,
                'error': 'Enhanced Gemma3n integration not available'
            }), 400
        
        data = request.json
        enable = data.get('enable', True)
        
        config = get_config()
        reasoning_layer = get_reasoning_layer()
        
        # Toggle both config and reasoning layer
        config.toggle_gemma(enable)
        reasoning_layer.toggle_gemma(enable)
        
        return jsonify({
            'success': True,
            'enabled': enable,
            'message': f'Gemma3n reasoning layer {"enabled" if enable else "disabled"}'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/test-asterisk-detection', methods=['POST'])
def test_asterisk_detection_endpoint():
    """Test endpoint for asterisk detection system"""
    try:
        data = request.json
        test_text = data.get('text', '')
        
        if not test_text:
            return jsonify({'error': 'No text provided for testing'}), 400
        
        if not AI_FORMATTER_AVAILABLE or not ai_formatter:
            return jsonify({
                'error': 'AI Response Formatter not available',
                'system_available': False
            }), 503
        
        # Process the text with asterisk detection
        result = ai_formatter.process_ai_response(test_text, format_output=True)
        
        return jsonify({
            'success': True,
            'input_text': test_text,
            'detection_result': result['detection_result'],
            'formatted_text': result['formatted_text'],
            'processing_notes': result['processing_notes'],
            'system_available': True,
            'formatting_applied': result['formatted_text'] != result['original_text']
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/vosk-transcribe', methods=['POST'])
def vosk_transcribe():
    """Vosk speech-to-text transcription endpoint"""
    try:
        if 'audio' not in request.files:
            return jsonify({'success': False, 'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        
        if audio_file.filename == '':
            return jsonify({'success': False, 'error': 'No audio file selected'}), 400
        
        # Save the audio file temporarily
        import tempfile
        import wave
        import json
        
        # Create a temporary file for the audio
        fd, temp_audio_path = tempfile.mkstemp(suffix='.wav')
        
        try:
            # Save the uploaded audio
            audio_file.save(temp_audio_path)
            
            # Initialize Vosk if not already done
            try:
                import vosk
                import os
                
                # Check if model exists
                model_path = "model/vosk-model-small-en-us-0.15"
                if not os.path.exists(model_path):
                    return jsonify({
                        'success': False, 
                        'error': f'Vosk model not found at {model_path}'
                    }), 500
                
                # Load Vosk model
                model = vosk.Model(model_path)
                rec = vosk.KaldiRecognizer(model, 16000)
                rec.SetWords(True)
                
                # Process audio file
                with wave.open(temp_audio_path, 'rb') as wf:
                    # Check audio format
                    if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
                        return jsonify({
                            'success': False, 
                            'error': 'Audio must be WAV format mono PCM.'
                        }), 400
                    
                    # Read and process audio data
                    results = []
                    while True:
                        data = wf.readframes(4000)
                        if len(data) == 0:
                            break
                        if rec.AcceptWaveform(data):
                            result = json.loads(rec.Result())
                            if 'text' in result and result['text'].strip():
                                results.append(result['text'].strip())
                    
                    # Get final result
                    final_result = json.loads(rec.FinalResult())
                    if 'text' in final_result and final_result['text'].strip():
                        results.append(final_result['text'].strip())
                    
                    # Combine all results
                    transcription = ' '.join(results).strip()
                    
                    return jsonify({
                        'success': True,
                        'text': transcription,
                        'confidence': 0.9  # Mock confidence score
                    })
                    
            except ImportError:
                # Fallback: Use a mock transcription service for demo
                import random
                demo_responses = [
                    "hello", "how are you", "what time is it", "thank you",
                    "tell me a joke", "what can you do", "help me", "good morning"
                ]
                
                mock_text = random.choice(demo_responses)
                return jsonify({
                    'success': True,
                    'text': mock_text,
                    'confidence': 0.8,
                    'note': 'Using mock transcription - install vosk-api for real transcription'
                })
                
        finally:
            # Clean up temporary file
            os.close(fd)
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
                
    except Exception as e:
        return jsonify({
            'success': False, 
            'error': f'Transcription failed: {str(e)}',
            'fallback_text': 'hello'  # Fallback text
        }), 500

if __name__ == '__main__':
    print("🚀 Starting Comprehensive AI Assistant...")
    print("📍 Open http://localhost:5000 in your browser")
    
    # Initialize model choice first
    initialize_model_choice()
    
    # Initialize comprehensive system
    print("\n🔧 Initializing Comprehensive AI System...")
    system_status = initialize_comprehensive_system()
    
    # Display system status
    print("\n📊 SYSTEM STATUS REPORT")
    print("=" * 50)
    
    # Core components
    print("🔌 Core Components:")
    print(f"   • Gemma AI: {'✅ Available' if system_status['gemma_ai'] else '❌ Unavailable'}")
    print(f"   • VediX Offline: {'✅ Available' if system_status['vedix_offline'] else '❌ Unavailable'}")
    print(f"   • Enhanced Reasoning: {'✅ Available' if system_status['enhanced_reasoning'] else '❌ Unavailable'}")
    print(f"   • Voice Processing: {'✅ Available' if system_status['voice_processing'] else '❌ Unavailable'}")
    print(f"   • Profile Management: {'✅ Available' if system_status['profile_management'] else '❌ Unavailable'}")
    
    # External connections
    print("\n🌐 External Connections:")
    print(f"   • Ollama Connection: {'✅ Connected' if system_status['ollama_connection'] else '❌ Not Connected'}")
    print(f"   • Vosk Model: {'✅ Found' if system_status['vosk_model'] else '❌ Not Found'}")
    
    # Features available
    print("\n🎆 Available Features:")
    features = [
        ("💬 Regular Chat", True),
        ("🖼️ Image Analysis", system_status['gemma_ai']),
        ("🎤 Voice Interaction", system_status['voice_processing']),
        ("🔮 Enhanced Vision", system_status['enhanced_vision']),
        ("🤖 Offline Assistant", system_status['vedix_offline']),
        ("👤 User Profiles", True),
        ("🧠 Session Memory", True)
    ]
    
    for feature_name, is_available in features:
        status_icon = "✅" if is_available else "⚠️"
        print(f"   • {feature_name}: {status_icon}")
    
    # API endpoints
    print("\n🔗 Available API Endpoints:")
    endpoints = [
        "/api/chat - Text chat with AI",
        "/api/voice-chat - Voice chat with AI", 
        "/api/analyze - Image analysis",
        "/api/enhanced-vision - Enhanced vision processing",
        "/api/voice-interact - VediX voice interaction",
        "/api/vosk-transcribe - Speech-to-text",
        "/api/system-status - System status",
        "/api/user-fetch - User profile management"
    ]
    
    for endpoint in endpoints:
        print(f"   • {endpoint}")
    
    # Check Ollama connection
    if GEMMA_AVAILABLE:
        try:
            is_connected, message = check_connection()
            if is_connected:
                print(f"\n🌐 Ollama Status: ✅ {message}")
                models = get_available_models()
                if models:
                    print(f"   Available models: {', '.join(models)}")
            else:
                print(f"\n🌐 Ollama Status: ❌ {message}")
                print("   ⚠️  Make sure Ollama is running and gemma3n:latest is installed")
        except Exception as e:
            print(f"\n🌐 Ollama Status: ❌ Connection check failed: {e}")
    
    print("\n" + "=" * 50)
    print("🌟 Comprehensive AI Assistant is ready!")
    print(f"   📋 Logs: app.log")
    print(f"   📁 Data: user_profiles.json")
    print(f"   🔍 System: /api/system-status")
    
    # Final status summary
    active_features = sum(1 for _, available in features if available)
    total_features = len(features)
    print(f"   📊 Features Active: {active_features}/{total_features}")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
