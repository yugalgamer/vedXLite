# 🤖 VedXLite AI - Advanced Multi-Modal AI Assistant

![VedXLite AI](https://img.shields.io/badge/VedXLite%20AI-Multi--Modal%20Assistant-brightgreen)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Platform](https://img.shields.io/badge/Platform-Windows-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)
![AI Powered](https://img.shields.io/badge/AI-Offline%20%2B%20Local-red)

> **VedXLite AI** is a comprehensive, offline-first artificial intelligence assistant that combines vision, voice, wellness, and specialized capabilities into a single, privacy-focused application.

## 🌟 Key Features

### 🎯 **Core AI Capabilities**
- **🧠 Offline AI Brain** - Complete AI functionality without internet dependency
- **💬 ChatGPT-like Interface** - Modern, intuitive web-based chat interface
- **🎤 Voice Interaction** - Natural voice input/output with emotion modulation
- **📷 Computer Vision** - Real-time camera processing and analysis
- **👤 Face Recognition** - Advanced biometric authentication and user identification
- **😊 Emotion Detection** - Real-time facial emotion analysis and response

### 🚀 **Specialized Assistants**

#### 👁️ **Vision Assistant** (Accessibility Support)
- **Scene Description** - Detailed, AI-powered description of camera view
- **Navigation Guidance** - Spatial awareness and obstacle detection
- **Object Recognition** - Intelligent identification of objects and people
- **OCR & Text Reading** - Convert images to speech for document reading
- **Perfect for visually impaired users**

#### 💖 **Wellness Assistant** (Mental Health Support)
- **Mood Assessment** - Multi-modal emotion detection (voice + face)
- **Stress Management** - Guided breathing exercises and relaxation techniques
- **Crisis Intervention** - Emergency support protocols and resources
- **Mental Health Tracking** - Long-term emotional wellbeing monitoring
- **Empathetic Conversations** - AI-powered supportive dialogue

#### 🌱 **Plant Disease Detection** (Agriculture)
- **Disease Identification** - AI-powered plant health analysis
- **Treatment Recommendations** - Specific care instructions and remedies
- **Agricultural Guidance** - Professional farming advice and best practices
- **Preventive Care** - Proactive plant health monitoring

#### 🧮 **AI Tutor** (Education)
- **Math Problem Solving** - Step-by-step equation solutions
- **Educational Support** - Adaptive learning assistance
- **Homework Help** - Patient, detailed explanations
- **Knowledge Base** - Comprehensive subject matter expertise

### 🔧 **Advanced Features**
- **🔐 Production-Grade Authentication** - Multi-factor biometric security
- **💬 Conversation History** - ChatGPT-like session management
- **🎵 Human-like Voice** - Natural TTS with emotional modulation
- **⏰ Smart Reminders** - Natural language reminder system
- **🧠 Memory System** - Persistent user learning and preferences
- **🚨 Emergency Support** - Crisis detection and response protocols

## 📦 Installation & Setup

### 🚀 **Quick Start** (Recommended)

#### Option 1: Direct Executable (Coming Soon)
```bash
# Download the standalone executable
VedXLiteAI.exe
# No installation required - all dependencies embedded!
```

#### Option 2: Web Interface
```bash
# Start the web server
python web_server.py
# Open browser to http://localhost:5000
```

### 🔧 **Development Setup**

#### Prerequisites
- **Python 3.8+**
- **Windows 10/11** (Primary platform)
- **Webcam** (For vision features)
- **Microphone** (For voice features)
- **4GB+ RAM** (8GB recommended)

#### Installation Steps

1. **Clone Repository**
   ```bash
   git clone <repository-url>
   cd Merathon
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   # or
   source venv/bin/activate  # Linux/Mac
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download AI Models** (Automatic on first run)
   ```bash
   # Models will be downloaded to ./models/ directory
   # Approximately 2GB download on first run
   ```

5. **Run Application**
   ```bash
   # Console Interface
   python main.py
   
   # Web Interface
   python web_server.py
   ```

## 🎮 Usage Guide

### 🖥️ **Web Interface**

1. **Launch Web Server**
   ```bash
   python web_server.py
   ```

2. **Open Browser**
   - Navigate to `http://localhost:5000`
   - Modern, responsive design works on all devices

3. **User Authentication**
   - **Face Recognition**: Automatic user identification
   - **Manual Login**: Username-based authentication
   - **New User Registration**: Complete setup wizard

4. **Feature Access**
   - **Chat Interface**: ChatGPT-like conversation
   - **Feature Buttons**: Quick access to specialized functions
   - **Camera Integration**: Real-time vision processing
   - **Voice Controls**: Speech input/output

### 🖥️ **Console Interface**

```bash
python main.py
```

#### Voice Commands
```
"describe what you see"      → Vision assistance
"I feel stressed"           → Wellness support  
"analyze my plant"          → Plant health analysis
"remind me to call mom"     → Smart reminders
"solve 2x + 3 = 7"         → Math problem solving
"emergency help"           → Crisis support
```

## 🎯 **Feature Deep Dive**

### 👁️ **Vision Assistance Features**
```python
# Example commands
"What do you see in front of me?"
"Help me navigate this room"
"Read the text in this image"
"Identify objects on the table"
"Describe the scene for navigation"
```

**Capabilities:**
- Real-time scene understanding
- Obstacle detection and warnings
- Text extraction and reading (OCR)
- Object and person identification
- Spatial relationship analysis

### 💖 **Wellness Support Features**
```python
# Example interactions
"I'm feeling anxious today"
"Help me with stress management"
"I need someone to talk to"
"Guide me through breathing exercises"
"Emergency - I need immediate help"
```

**Capabilities:**
- Mood detection via voice/face analysis
- Guided meditation and breathing exercises
- Crisis intervention protocols
- Long-term emotional tracking
- Emergency contact integration

### 🌱 **Plant Disease Detection**
```python
# Example usage
"Check if my plant is healthy"
"What's wrong with these leaves?"
"How do I treat plant disease?"
"Provide agricultural advice"
```

**Capabilities:**
- AI-powered disease identification
- Treatment recommendation engine
- Growth stage monitoring
- Agricultural best practices
- Preventive care guidance

## 🔧 **Configuration**

### 📁 **Project Structure**
```
Merathon/
├── main.py                 # Console application entry point
├── web_server.py          # Web interface server
├── index.html             # Web UI
├── style.css             # UI styling
├── script.js             # Frontend JavaScript
├── requirements.txt       # Python dependencies
├── assistants/           # Specialized AI assistants
│   ├── vision_assistant.py
│   ├── wellness_assistant.py
│   ├── plant_disease_detection.py
│   └── ai_tutor.py
├── auth/                 # Authentication system
├── conversation/         # Chat history management
├── core/                # Core AI engine
├── memory/              # User memory system
├── voice/               # Voice processing
├── vision/              # Computer vision
└── tools/               # Utility tools
```

### ⚙️ **Feature Configuration**
```python
# main.py - Feature toggles
FEATURES = {
    "ENABLE_GEMMA_AI": True,
    "USE_FACE_RECOGNITION": True,
    "USE_EMOTION_DETECTION": True,
    "VISION_ASSISTANCE": True,
    "WELLNESS_ASSISTANT": True,
    "PLANT_DISEASE_DETECTION": True,
    "CONVERSATION_HISTORY": True,
    "ENHANCED_USER_AUTH": True,
    "HUMAN_LIKE_VOICE": True,
    "REMINDER_SYSTEM": True,
    "MEMORY_SYSTEM": True,
    "VOICE_INTERACTION": True,
}
```

## 🔒 **Privacy & Security**

### 🛡️ **Data Protection**
- **100% Offline Processing** - No data sent to external servers
- **Local AI Models** - All AI computation happens on your device
- **Encrypted Storage** - User data secured with industry-standard encryption
- **Biometric Security** - Advanced face recognition authentication
- **Privacy by Design** - No telemetry or data collection

### 🔐 **Security Features**
- **Multi-factor Authentication** - Face + username verification
- **Secure Memory Storage** - Encrypted user profiles and memories
- **Session Management** - Secure conversation history
- **Access Controls** - Feature-level permission system

## 🚨 **Emergency Features**

### 🆘 **Crisis Support**
- **Automatic Crisis Detection** - AI identifies emergency situations
- **Emergency Protocols** - Immediate support and guidance
- **Resource Database** - Crisis helplines and emergency contacts
- **Offline Operation** - Works without internet during emergencies

## 📊 **System Requirements**

### 💻 **Minimum Requirements**
- **OS**: Windows 10/11, Linux (Ubuntu 18.04+), macOS 10.15+
- **CPU**: Intel i3 / AMD Ryzen 3 (or equivalent)
- **RAM**: 4GB (8GB recommended)
- **Storage**: 4GB free space (for AI models)
- **GPU**: Integrated graphics (Dedicated GPU recommended)
- **Camera**: USB/Built-in webcam (for vision features)
- **Microphone**: Any microphone (for voice features)

### 🚀 **Recommended Requirements**
- **CPU**: Intel i5 / AMD Ryzen 5 (or better)
- **RAM**: 8GB+ 
- **GPU**: NVIDIA GTX 1060 / AMD RX 580 (for faster AI processing)
- **Storage**: SSD with 8GB+ free space

### 🔧 **Hardware Acceleration**
- **CUDA Support** - NVIDIA GPU acceleration
- **CPU Optimization** - Multi-core processing
- **Memory Management** - Intelligent model loading

## 🔧 **Troubleshooting**

### 🐛 **Common Issues**

#### Camera Not Working
```bash
✅ Check camera permissions in Windows Settings
✅ Ensure no other applications are using the camera
✅ Restart the application
✅ Update camera drivers
```

#### Voice Recognition Issues
```bash
✅ Check microphone permissions
✅ Test microphone in Windows Sound settings
✅ Verify audio device selection
✅ Check for background noise
```

#### AI Model Loading Errors
```bash
✅ Ensure stable internet connection (first run only)
✅ Check available disk space (4GB+ required)
✅ Verify Python version (3.8+ required)
✅ Reinstall dependencies: pip install -r requirements.txt
```

#### Performance Issues
```bash
✅ Close resource-intensive applications
✅ Enable hardware acceleration (CUDA if available)
✅ Reduce concurrent features
✅ Restart the application
```

## 🧪 **Development**

### 🛠️ **Contributing**
1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit changes**: `git commit -m 'Add amazing feature'`
4. **Push to branch**: `git push origin feature/amazing-feature`
5. **Open Pull Request**

### 🧪 **Testing**
```bash
# Run all tests
python -m pytest

# Run specific test modules
python -m pytest tests/test_vision.py
python -m pytest tests/test_wellness.py
```

### 📚 **API Documentation**
```python
# Core AI Brain usage
from core.ai_brain import AIBrain
brain = AIBrain()
response = brain.process_request(user_context)

# Vision Assistant
from assistants.vision_assistant import VisionAssistant
vision = VisionAssistant()
description = vision.capture_and_describe_scene()

# Wellness Assistant
from assistants.wellness_assistant import WellnessAssistant
wellness = WellnessAssistant()
mood = wellness.assess_current_mood(user_input)
```

## 🗺️ **Roadmap**

### 🎯 **Version 2.0 (Planned)**
- [ ] **Mobile App** - Android/iOS native applications
- [ ] **Multi-language Support** - 20+ languages
- [ ] **Cloud Sync** - Optional cloud backup (privacy-preserving)
- [ ] **Plugin System** - Third-party extensions
- [ ] **Advanced AI Models** - Latest transformer architectures
- [ ] **Collaborative Features** - Multi-user sessions

### 🎯 **Version 1.5 (In Development)**
- [ ] **Enhanced Web UI** - Progressive Web App (PWA)
- [ ] **Voice Customization** - Multiple voice profiles
- [ ] **Advanced Plant Detection** - 1000+ plant species
- [ ] **Wellness Analytics** - Long-term mood tracking
- [ ] **Smart Home Integration** - IoT device control

## 📄 **License**

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## 🙏 **Acknowledgments**

### 🤖 **AI & ML Libraries**
- **[Transformers](https://huggingface.co/transformers/)** - Hugging Face transformer models
- **[PyTorch](https://pytorch.org/)** - Deep learning framework
- **[OpenCV](https://opencv.org/)** - Computer vision library
- **[MediaPipe](https://mediapipe.dev/)** - Real-time perception pipeline

### 🎤 **Audio & Voice**
- **[SpeechRecognition](https://pypi.org/project/SpeechRecognition/)** - Voice input processing
- **[pyttsx3](https://pypi.org/project/pyttsx3/)** - Text-to-speech synthesis
- **[Vosk](https://alphacephei.com/vosk/)** - Offline speech recognition

### 🖼️ **Computer Vision**
- **[face-recognition](https://pypi.org/project/face-recognition/)** - Face detection and recognition
- **[dlib](http://dlib.net/)** - Advanced computer vision algorithms

## 📞 **Support & Community**

### 🆘 **Getting Help**
- **📖 Documentation**: Check this README and inline code comments
- **🐛 Bug Reports**: [GitHub Issues](issues)
- **💡 Feature Requests**: [GitHub Discussions](discussions)
- **💬 Community**: Join our Discord server

### 📧 **Contact**
- **Developer**: Yugal Kishor
- **Project**: VedXLite AI
- **Repository**: [GitHub Repository]

---

## 🎯 **Quick Start Commands**

```bash
# Clone and setup
git clone <repo-url> && cd Merathon
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt

# Run console version
python main.py

# Run web interface
python web_server.py
# Then open http://localhost:5000

# Enable all features
# Edit main.py FEATURES dict to enable/disable capabilities
```

---

**🤖 VedXLite AI - Your Complete AI Companion**

*Built with ❤️ for accessibility, wellness, education, and productivity*

> **"Empowering users with AI that works offline, respects privacy, and enhances daily life."**
