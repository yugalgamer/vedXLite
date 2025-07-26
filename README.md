# 🤖 VedXlite - Advanced Multi-Modal AI Assistant

**A sophisticated, accessibility-focused AI assistant with advanced chat, voice, and vision capabilities powered by Gemma3n, CUDA acceleration, and offline processing.**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![Flask](https://img.shields.io/badge/Flask-2.3%2B-green) ![CUDA](https://img.shields.io/badge/CUDA-Accelerated-orange) ![Accessibility](https://img.shields.io/badge/Accessibility-First-purple) ![Ollama](https://img.shields.io/badge/Ollama-Powered-red)

---

## 🌟 Overview

VedXlite is a comprehensive AI assistant designed for accessibility, emotional intelligence, and multi-modal interaction. Built with a focus on supporting blind users and introverted individuals, it provides advanced chat, voice interaction, and vision analysis capabilities through a modern web interface.

---

## ✨ Core Features

### 💬 **Advanced Chat System**
- **Multi-Modal Conversations**: Text, voice, and image inputs
- **Role-Based Personalities**: Best Friend, Motivator, Guide, Female Friend
- **Session Memory**: Maintains context within current session only
- **Markdown Formatting**: Enhanced responses with *italic*, **bold**, and ***bold italic*** formatting
- **Heart-to-Heart Connection**: Emotional intelligence and empathetic responses
- **VedXlite Mode**: Specialized support for introverts and shy individuals

### 🎤 **Voice Interaction System**
- **Web Speech API Integration**: Real-time speech recognition
- **Vosk Offline Recognition**: Privacy-focused offline speech processing
- **Voice Visualizer**: Real-time audio frequency visualization
- **Continuous Listening**: Passive voice activation with visual feedback
- **Voice Settings Toggle**: User-controlled voice enable/disable
- **Multiple Voice Modes**: 
  - Standard voice chat with role-based personalities
  - VedXlite voice mode for introverted users
  - VediX offline assistant mode

### 👁️ **Enhanced Vision Processing**
- **Gemma3n Vision Analysis**: Advanced AI-powered image understanding
- **Accessibility Focus**: Detailed descriptions optimized for blind users
- **Safety Analysis**: Hazard identification and navigation guidance
- **Object Detection**: Precise location and spatial relationship descriptions
- **Drag & Drop Interface**: Easy image upload with preview
- **Live Camera Support**: Real-time camera capture and analysis
- **Quick Actions**: Speak results, copy text, re-analyze options

### 🚀 **CUDA Acceleration**
- **GPU-Accelerated Processing**: Up to 10x performance improvement
- **CUDA Text Processing**: Optimized natural language processing
- **CUDA Vision Processing**: Hardware-accelerated image analysis
- **Memory Management**: Intelligent GPU resource allocation
- **Fallback Support**: Automatic CPU fallback when CUDA unavailable

### 🔒 **Privacy & Security**
- **Session-Only Memory**: No persistent conversation storage
- **VediX Offline Mode**: Complete offline operation for sensitive interactions
- **Local Processing**: Optional offline mode for privacy-critical tasks
- **Secure File Handling**: Safe image upload and processing
- **No Cross-Session Memory**: Each session is completely isolated

---

## **TECHNICAL WRJTEUP**

**[TechWriteup](TECHNICAL_WRITEUP.md)**

## 🏗️ Architecture & Technology Stack

### **Backend Components**
- **Flask Application** (`main.py`): Core server with comprehensive endpoint management
- **Gemma Integration** (`gemma.py`): Ollama-powered language model interface
- **Enhanced AI Modules** (`ai_modules/`):
  - `gemma_integration/`: Advanced reasoning and prompt building
  - `cuda_core/`: GPU acceleration and memory management
  - `cuda_text/`: CUDA-accelerated text processing
  - `vision_cuda/`: GPU-accelerated computer vision
- **VediX Core** (`vedix_core.py`): Offline AI assistant with Vosk integration
- **User Profile System**: Persistent user preferences and role management

### **Frontend Components**
- **Modern Web Interface** (`index.html`): Responsive, accessible design
- **Modular JavaScript Architecture**:
  - `main.js`: Application initialization and module loading
  - `feature_voice.js`: Voice recognition and interaction system
  - `enhanced_vision.js`: Vision processing interface
  - `voice_visualizer.js`: Real-time audio visualization
  - `advanced_memory.js`: Session memory management
  - `request_queue.js`: API request management
  - `voice_settings.js`: Voice control and preferences

### **AI Processing Pipeline**
1. **Input Processing**: Text/Voice/Image preprocessing with CUDA acceleration
2. **Context Building**: Session-aware context construction
3. **AI Inference**: Gemma3n model processing via Ollama
4. **Response Enhancement**: Markdown formatting and accessibility optimization
5. **Output Delivery**: Multi-modal response delivery (text/voice/visual)

---

## 🛠️ Installation & Setup

### **Prerequisites**
1. **Python 3.8+** with pip
2. **Ollama** - Install from [ollama.ai](https://ollama.ai/)
3. **Gemma Model**: 
   ```bash
   ollama pull gemma3n:latest
   # or for faster performance:
   ollama pull gemma:2b
   ```
4. **CUDA Toolkit** (Optional): For GPU acceleration
5. **Vosk Model** (Optional): For offline voice recognition

### **Installation Steps**

1. **Clone/Download the Project**
   ```bash
   git clone <repository-url>
   cd VedXlite
   ```

2. **Install Python Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download Vosk Model** (Optional for offline voice)
   ```bash
   # Download and extract to model/ directory
   wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
   unzip vosk-model-small-en-us-0.15.zip -d model/
   ```

4. **Start Ollama Service**
   ```bash
   ollama serve
   ```

5. **Launch VedXlite**
   ```bash
   python main.py
   ```

6. **Access the Interface**
   - Open browser to `http://localhost:5000`
   - Complete initial user setup (name and relationship preference)

---

## 🎯 Usage Guide

### **Chat Interface**
1. **Text Chat**: Type messages in the main chat area
2. **Image Chat**: Click image icon to attach photos for analysis
3. **Role Selection**: Choose your preferred AI personality
4. **Memory Management**: Use "clear chat" commands to reset session

### **Voice Features**
1. **Enable Voice**: Click the voice toggle in the header
2. **Start Listening**: Click "Start Voice" in the Voice Interact tab
3. **Speak Naturally**: The system will continuously listen and respond
4. **Visual Feedback**: Watch the audio visualizer for activity indication

### **Vision Analysis**
1. **Upload Image**: Drag & drop or click to select images
2. **Ask Questions**: Add specific questions about the image
3. **Enhanced Mode**: Toggle Gemma3n reasoning for detailed analysis
4. **Quick Actions**: Use speak, copy, or re-analyze buttons

### **Accessibility Features**
- **Screen Reader Compatible**: Full ARIA support
- **High Contrast Mode**: Available in settings
- **Voice-Optimized Responses**: Specially formatted for speech synthesis
- **Safety-First Descriptions**: Focus on navigation and hazard identification

---

## 🔌 API Endpoints

### **Chat & Communication**
- `POST /api/chat` - Text chat with role-based AI
- `POST /api/voice-chat` - Voice interaction with session memory
- `POST /api/voice-chat-vedx-lite` - Specialized chat for introverts
- `POST /api/chat-image` - Chat with image analysis

### **Vision & Analysis**
- `POST /api/analyze` - Basic image analysis
- `POST /api/enhanced-vision` - Advanced Gemma3n vision processing
- `POST /api/voice-interact` - VediX offline voice interaction

### **System & Management**
- `GET /api/system-status` - Comprehensive system information
- `GET /api/gemma-status` - Gemma3n reasoning layer status
- `POST /api/gemma-toggle` - Enable/disable enhanced reasoning
- `GET /api/session-status` - Current session memory information

### **User Management**
- `GET /api/user-fetch` - Retrieve user profile
- `POST /api/user-create` - Create/update user profile
- `POST /api/user-update-role` - Update relationship preference
- `GET /api/user-stats` - User interaction statistics

### **Voice Processing**
- `POST /api/vosk-transcribe` - Offline speech-to-text
- `POST /api/enhanced-voice` - CUDA-accelerated voice processing

---

## ⚙️ Configuration

### **Environment Variables**
```bash
# Gemma Configuration
ENABLE_GEMMA=true
GEMMA_MODEL_NAME=gemma3n:latest
GEMMA_OLLAMA_URL=http://localhost:11434
GEMMA_TIMEOUT=60

# Performance Options
USE_LIGHTWEIGHT_MODEL=false
LIGHTWEIGHT_MODEL_NAME=gemma:2b
MAX_PROMPT_LENGTH=4000
MAX_RESPONSE_LENGTH=1000

# CUDA Settings
CUDA_ENABLED=true
CUDA_DEVICE=0
CUDA_MEMORY_FRACTION=0.8

# Privacy Settings
ENABLE_OFFLINE_MODE=false
LOG_INTERACTIONS=true
ENABLE_FALLBACK_RESPONSES=true
```

### **Model Selection**
- **High Performance**: `gemma3n:latest` (Requires 8GB+ GPU memory)
- **Balanced**: `gemma:7b` (Requires 4GB+ GPU memory)
- **Fast/Lightweight**: `gemma:2b` (Requires 2GB+ GPU memory)

---

## 🧠 AI Capabilities

### **Natural Language Understanding**
- **Context Awareness**: Maintains conversation flow within sessions
- **Emotional Intelligence**: Recognizes and responds to emotional cues
- **Role Adaptation**: Adjusts personality based on user preference
- **Multi-Turn Reasoning**: Complex conversation handling

### **Vision Analysis**
- **Object Detection**: Identifies and locates objects in images
- **Scene Understanding**: Provides contextual environmental descriptions
- **Safety Assessment**: Identifies potential hazards and obstacles
- **Accessibility Focus**: Optimized descriptions for navigation assistance

### **Voice Processing**
- **Continuous Recognition**: Real-time speech-to-text conversion
- **Natural Interaction**: Conversational voice responses
- **Offline Capability**: Privacy-focused local processing
- **Multi-Language Support**: Expandable language model support

---

## 🔧 Troubleshooting

### **Common Issues**

**Ollama Connection Failed**
- Ensure Ollama is running: `ollama serve`
- Check model availability: `ollama list`
- Install required model: `ollama pull gemma3n:latest`

**CUDA Not Available**
- Install NVIDIA CUDA Toolkit
- Verify GPU compatibility: `nvidia-smi`
- Check PyTorch CUDA support: `python -c "import torch; print(torch.cuda.is_available())"`

**Voice Recognition Not Working**
- Enable microphone permissions in browser
- Check Vosk model installation in `model/` directory
- Try the Web Speech API fallback mode

**Image Analysis Failing**
- Verify image file format (PNG, JPG, JPEG, GIF, BMP, WebP)
- Check file size limit (16MB maximum)
- Ensure Gemma model supports vision tasks

### **Performance Optimization**

**Memory Management**
- Use lighter models for low-memory systems
- Adjust `CUDA_MEMORY_FRACTION` for shared GPU usage
- Enable automatic garbage collection

**Response Speed**
- Use `gemma:2b` for faster responses
- Enable CUDA acceleration
- Reduce context window size

**Offline Mode**
- Download Vosk models for local speech recognition
- Use VediX mode for complete offline operation
- Cache frequently used responses

---

## 📊 System Requirements

### **Minimum Requirements**
- **CPU**: 4-core processor (2.5GHz+)
- **RAM**: 8GB system memory
- **Storage**: 5GB free space
- **Network**: Internet connection for Ollama models
- **Browser**: Modern browser with Web Speech API support

### **Recommended Requirements**
- **CPU**: 8-core processor (3.0GHz+)
- **RAM**: 16GB system memory
- **GPU**: NVIDIA GPU with 4GB+ VRAM (CUDA support)
- **Storage**: 20GB free space (for models and data)
- **Network**: High-speed internet for model downloads

### **Optimal Performance**
- **CPU**: High-performance CPU (Intel i7/AMD Ryzen 7+)
- **RAM**: 32GB system memory
- **GPU**: NVIDIA RTX series with 8GB+ VRAM
- **Storage**: NVMe SSD with 50GB+ free space

---

## 🤝 Contributing

VedXlite is designed to be accessible and inclusive. When contributing:

1. **Accessibility First**: Ensure all features work with screen readers
2. **Privacy Focused**: Maintain the session-only memory model
3. **Performance Aware**: Optimize for both CPU and CUDA execution
4. **User-Centric**: Consider the needs of introverted and visually impaired users

---

## 📄 License

This project is open source and available under the MIT License.

---

## 🙏 Acknowledgments

- **Ollama Team**: For the excellent local LLM serving platform
- **Google**: For the Gemma model architecture
- **Vosk**: For offline speech recognition capabilities
- **Community**: For accessibility feedback and inclusive design principles

---

## 📞 Support

For issues, questions, or feature requests, please:
1. Check the troubleshooting section above
2. Review the system status at `/api/system-status`
3. Check the application logs in `app.log`
4. Create an issue with detailed system information

**VedXlite** - *Empowering accessible AI interaction for everyone* 🚀
