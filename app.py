#!/usr/bin/env python3
"""
Simple Flask Backend for AI Assistant
=====================================
Handles API endpoints for chat, vision analysis, and other features.
"""

from flask import Flask, request, jsonify, send_from_directory
import os
import base64
import json
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static', static_url_path='/static')

# Simple in-memory storage (replace with database in production)
chat_history = []
gemma_status = {
    'available': True,  # Make available by default for demo
    'enabled': True,   # Enable by default
    'model_name': 'gemma3n:latest',
    'system_status': {'status': 'running', 'memory_usage': '2.1GB'}
}

@app.route('/')
def index():
    """Serve the main index.html file"""
    return send_from_directory('.', 'index.html')

@app.route('/<path:filename>')
def serve_files(filename):
    """Serve static files from the root directory"""
    return send_from_directory('.', filename)

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        prompt = data.get('prompt', '')
        username = data.get('username', 'User')
        role = data.get('role', 'Friend')
        
        logger.info(f"Chat request from {username}: {prompt[:50]}...")
        
        # Store message in history
        chat_history.append({
            'timestamp': datetime.now().isoformat(),
            'username': username,
            'role': role,
            'prompt': prompt
        })
        
        # Simple AI response (replace with actual AI integration)
        if 'hello' in prompt.lower():
            response = f"Hello {username}! How can I help you today?"
        elif 'how are you' in prompt.lower():
            response = "I'm doing great, thank you for asking! How are you feeling today?"
        elif 'weather' in prompt.lower():
            response = "I don't have access to current weather data, but you can check your local weather service for accurate information."
        elif 'time' in prompt.lower():
            response = f"The current server time is {datetime.now().strftime('%H:%M:%S')}."
        else:
            response = f"I understand you said: '{prompt}'. I'm a simple demo AI assistant. In a full implementation, I would provide more helpful responses!"
        
        return jsonify({
            'success': True,
            'response': response,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    """Handle basic image analysis"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        image_file = request.files['image']
        prompt = request.form.get('prompt', 'Describe this image')
        
        logger.info(f"Image analysis request: {image_file.filename}")
        
        # In a real implementation, you would process the image here
        # For demo purposes, return a simulated response
        response = """I can see an image has been uploaded for analysis. In a full implementation, this would use:

**Object Detection**: Identifying people, objects, vehicles, and other elements
**Scene Analysis**: Understanding the environment and context  
**Safety Assessment**: Highlighting potential hazards or obstacles
**Spatial Relationships**: Describing where objects are positioned relative to each other

For blind users, the analysis would focus on:
- Navigation safety and obstacles
- Important landmarks and reference points  
- People and their activities
- Text that might be visible
- Overall scene context for orientation"""
        
        return jsonify({
            'success': True,
            'analysis': response,
            'response': response,
            'metadata': {
                'processing_time': 0.5,
                'source': 'basic_vision',
                'template_type': 'accessibility_focused'
            }
        })
        
    except Exception as e:
        logger.error(f"Image analysis error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/enhanced-vision', methods=['POST'])
def enhanced_vision():
    """Handle enhanced vision analysis with Gemma3n"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        image_file = request.files['image']
        prompt = request.form.get('prompt', 'Provide detailed visual analysis for a blind user')
        
        logger.info(f"Enhanced vision analysis: {image_file.filename}")
        
        # Simulated enhanced analysis
        response = """**Enhanced AI Vision Analysis** 🧠

**Scene Overview**: This appears to be an indoor/outdoor environment with multiple elements present.

**Safety Analysis**: 
- No immediate hazards detected in the foreground
- Clear pathways visible
- Stable ground surface

**Object Detection**:
- Various objects and structures identified
- People may be present in the scene
- Furniture or architectural elements visible

**Navigation Guidance**:
- Primary path appears clear ahead
- Reference points available for orientation
- No obstacles in immediate walking path

**Additional Context**:
This enhanced analysis would normally provide much more detailed information using advanced AI models like Gemma3n for deeper scene understanding and more accurate object recognition."""
        
        return jsonify({
            'success': True,
            'response': response,
            'enhanced_processing': True,
            'vision_description': 'Basic computer vision detected: objects, people, structures',
            'metadata': {
                'processing_time': 1.2,
                'source': 'enhanced_gemma3n',
                'template_type': 'accessibility_enhanced'
            }
        })
        
    except Exception as e:
        logger.error(f"Enhanced vision error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/gemma-status', methods=['GET'])
def get_gemma_status():
    """Get Gemma3n model status"""
    return jsonify(gemma_status)

@app.route('/api/gemma-toggle', methods=['POST'])
def toggle_gemma():
    """Toggle Gemma3n enhanced mode"""
    try:
        data = request.get_json()
        enable = data.get('enable', False)
        
        # In real implementation, this would check if Gemma3n is actually available
        gemma_status['enabled'] = enable
        gemma_status['available'] = True  # Simulate availability for demo
        
        message = f"Enhanced mode {'enabled' if enable else 'disabled'}"
        logger.info(f"Gemma toggle: {message}")
        
        return jsonify({
            'success': True,
            'enabled': enable,
            'message': message
        })
        
    except Exception as e:
        logger.error(f"Gemma toggle error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat-history', methods=['GET'])
def get_chat_history():
    """Get chat history"""
    return jsonify({
        'success': True,
        'history': chat_history[-20:]  # Return last 20 messages
    })

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get server status"""
    return jsonify({
        'success': True,
        'status': 'running',
        'timestamp': datetime.now().isoformat(),
        'endpoints_available': True
    })

@app.route('/api/user-fetch', methods=['GET'])
def user_fetch():
    """Fetch user information"""
    name = request.args.get('name', 'Unknown')
    return jsonify({
        'success': True,
        'user': {
            'name': name,
            'role': 'Friend',
            'first_visit': False
        }
    })

@app.route('/api/user-create', methods=['POST', 'GET'])
def user_create():
    """Create or update user"""
    if request.method == 'POST':
        data = request.get_json() or {}
    else:
        data = {'name': request.args.get('name', 'Unknown')}
    
    name = data.get('name', 'Unknown')
    role = data.get('role', 'Friend')
    
    return jsonify({
        'success': True,
        'user': {
            'name': name,
            'role': role,
            'created': True
        }
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("🚀 Starting AI Assistant Backend Server...")
    print("📡 Server will be available at: http://localhost:8000")
    print("🔧 Available endpoints:")
    print("   - POST /api/chat - Chat with AI")
    print("   - POST /api/analyze - Basic image analysis")
    print("   - POST /api/enhanced-vision - Enhanced vision analysis")
    print("   - GET /api/gemma-status - Check Gemma3n status")
    print("   - POST /api/gemma-toggle - Toggle enhanced mode")
    print("   - GET /api/chat-history - Get chat history")
    print("📁 Static files served from current directory")
    print("✨ Ready for connections!")
    
    app.run(host='0.0.0.0', port=8000, debug=True)
