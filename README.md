# 🤖 VedXlite - Advanced Multi-Modal AI Assistant
**An accessibility-focused AI assistant with advanced chat, voice, and vision capabilities powered by Gemma3n, CUDA acceleration, and offline processing.**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![Flask](https://img.shields.io/badge/Flask-2.3%2B-green) ![CUDA](https://img.shields.io/badge/CUDA-Accelerated-orange) ![Accessibility](https://img.shields.io/badge/Accessibility-First-purple) ![Ollama](https://img.shields.io/badge/Ollama-Powered-red)

---

## 🌟 Overview

VedXlite is a comprehensive AI assistant designed for accessibility, emotional intelligence, and multi-modal interaction. Built with a focus on supporting blind users and introverted individuals, it provides advanced chat, voice interaction, and vision analysis capabilities through a modern web interface.

---

## ✨ Core Features

### 💬 **Advanced Chat System**
- **Multi-Modal Conversations**: Text, voice, and image inputs
- **Role-Based Personalities**: Best Friend, Motivator, Guide, Female Friend
- **Session Memory**: Maintains context within the current session only and includes structured prompt-based responses for greetings
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
  - Enhanced voice interactions with prompt-based dynamic greeting responses
  - Improved conversational flow using AI-driven prompt systems
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

## Technical Writeup
**[Technical writeup](TECHNICAL_WRITEUP.md)**

## 🏗️ Architecture & Technology Stack

### **Backend Components**
- **Flask Application**: Core server and API logic.
- **Gemma Integration**: Interfaces with language models using Ollama.
- **AI Modules**:
  - Reasoning and prompt systems.
  - Core GPU acceleration and vision processing.
  - User profile handling and persistent storage.

### **Frontend Components**
- **Dynamic Web Interface**: Responsive design with advanced JS integration.

### **AI Processing Pipeline**
1. **Input Handling**: Text, Voice, Image input processed with CUDA.
2. **Context Management**: Context-aware response generation.
3. **Inference & Interaction**: AI-driven query handling using Gemma3n.
4. **Output & Delivery**: Interface for multi-modal responses (text/voice/visual).

---

## 🛠️ Installation & Setup
### **Prerequisites**
1. **Python 3.8+** with pip
2. **Ollama** - Install from [ollama.ai](https://ollama.ai/)
3. **Gemma Model**: 
   ```bash
   ollama pull gemma3n:latest
   # or for improved performance:
   ollama pull gemma3n:e2b
   ```
4. **CUDA Toolkit** (Optional): Enables GPU acceleration. Install with: ``` pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 ```
   
6. **Vosk Model** for offline voice processing.

### **Installation Steps**
1. **Clone the Project**
   ```bash
   git clone https://github.com/yugalgamer/vedXLite.git
   cd VedXlite
   ```
2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Vosk Model Setup**
   *download it if inside model file vosk folder are not excited*
   ```bash
   wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
   unzip vosk-model-small-en-us-0.15.zip -d model/
   ```
5. **Start Ollama Service**
   ```bash
   ollama serve
   ```
6. **Launch Application**
   ```bash
   python main.py
   ```
7. **User Interface Access**
   - Open browser: `http://localhost:5000`
   - Complete initial setup.

### **Linux Installation with Virtual Environment**

Follow these steps to set up VedXlite on a Linux system using a Python virtual environment:

1. **Ensure Python 3.8+ and Virtualenv are Installed**:
   ```bash
   sudo apt update
   sudo apt install python3 python3-venv
   ```

2. **Clone the Project**:
   ```bash
   git clone https://github.com/yugalgamer/vedXLite.git
   cd vedXlite
   ```

3. **Create and Activate a Virtual Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

4. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Setup Vosk Model**:
   *Download it if not already present in the `model` directory*:
   ```bash
   wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
   unzip vosk-model-small-en-us-0.15.zip -d model/
   ```

6. **Start the Ollama Service**:
   ```bash
   ollama serve
   ```

7. **Launch the Application**:
   ```bash
   python main.py
   ```

8. **Access the User Interface**:
   - Open a browser and go to: `http://localhost:5000`
   - Complete the initial setup.

---

## 🎯 Usage Guide

### **Chat Interface**
- **Text Entry**: Type messages and send.
- **Image Upload**: Attach photos for analysis.
- **Select Role**: Adjust AI personality.
- **Session Clear**: Commands to reset interactions.

### **Voice Features**
- **Voice Activation**: Enable in header.
- **Real-Time Listening**: Continuous listening.
- **Audio Visualization**: See microphone input activity.

### **Vision Analysis**
- **Upload and Analyze**: Use images for AI-driven analysis.
- **Detailed Queries**: Ask questions about images.

### **Accessibility Features**
- **Screen Reader**: Compatible interface.
- **High Contrast**: Interface settings for visibility.

### 🔌 API Endpoints
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
LIGHTWEIGHT_MODEL_NAME=gemma3n:e2b
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
- **Balanced**: `gemma3n:e4b` (Requires 4GB+ GPU memory)
- **Fast/Lightweight**: `gemma3n:e2b` (Requires 2GB+ GPU memory)

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
- Use `gemma3n:e2b` for faster responses
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
