# VedXLite AI - All-in-One Assistant

🤖 **Your Complete AI Companion in a Single Executable**

![VedXLite AI](https://img.shields.io/badge/VedXLite%20AI-All--in--One-brightgreen)
![Platform](https://img.shields.io/badge/Platform-Windows-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)

## 🌟 Features

VedXLite AI is a comprehensive artificial intelligence assistant that combines multiple AI capabilities into a single, easy-to-use application with a ChatGPT-like interface.

### 🎯 Core Features

- **💬 ChatGPT-like UI** - Modern, intuitive interface for seamless interaction
- **🎤 Voice Input/Output** - Speak to the AI and hear responses
- **📷 Camera Integration** - Real-time camera feed with visual processing
- **👤 Face Recognition** - Recognize and remember users
- **😊 Emotion Detection** - Detect and respond to facial emotions
- **🧠 Offline AI** - Complete AI functionality without internet

### 🚀 Specialized Assistants

#### 👁️ Vision Assistant (For Blind Users)
- **Scene Description** - Detailed description of camera view
- **Navigation Guidance** - Help with spatial awareness
- **Object Recognition** - Identify objects and obstacles
- **Text Reading (OCR)** - Read text from images aloud

#### 💖 Wellness Assistant
- **Mood Assessment** - Analyze emotional state from voice and face
- **Stress Management** - Guided breathing and relaxation exercises
- **Mental Health Support** - Empathetic conversations and advice
- **Crisis Intervention** - Emergency support and resources

#### 🌱 Plant Disease Detection
- **Health Analysis** - Identify plant diseases from camera
- **Treatment Recommendations** - Specific care instructions
- **Agricultural Advice** - Professional farming guidance
- **Preventive Care** - Tips for healthy plant growth

#### 🧮 Advanced Capabilities
- **Math Solver** - Step-by-step mathematical problem solving
- **Smart Reminders** - Natural language reminder setting
- **Memory System** - Learns and remembers user preferences
- **Emergency Support** - Quick access to crisis resources

## 📦 Installation & Setup

### 🎯 Quick Start (Recommended)

1. **Download the Executable** (Once built)
   ```
   VedXLiteAI_GUI.exe    - For graphical interface
   VedXLiteAI_Console.exe - For command-line interface
   ```

2. **Run the Application**
   - Double-click the executable
   - No installation required!
   - All dependencies are embedded

### 🔧 Building from Source

#### Prerequisites
- Windows 10/11
- Python 3.8 or higher
- Chocolatey (for make installation)

#### Option 1: Using the Build Script (Easiest)
```bash
# Run the automated build script
build.bat
```

#### Option 2: Using Make
```bash
# Install dependencies and build
make all

# Or step by step
make setup    # Install dependencies
make build    # Build executables
make package  # Create portable package
```

#### Option 3: Manual Build
```bash
# Install dependencies
pip install -r requirements.txt
pip install pyinstaller

# Build GUI version
pyinstaller --onefile --windowed gui_main.py

# Build console version
pyinstaller --onefile --console main.py
```

## 🎮 Usage Guide

### 🖥️ GUI Version

1. **Launch the Application**
   ```bash
   VedXLiteAI_GUI.exe
   ```

2. **Main Interface**
   - **Chat Area**: ChatGPT-like conversation interface
   - **Camera Panel**: Live camera feed with controls
   - **Feature Buttons**: Quick access to specialized functions
   - **Voice Controls**: Start/stop voice input and speech output

3. **Key Controls**
   - **🎤 Voice Button**: Toggle voice input
   - **🔊 Speak Button**: Toggle text-to-speech
   - **📷 Camera Button**: Start/stop camera
   - **👤 Face Detection**: Enable face recognition
   - **Send Button**: Send text messages

### 🖥️ Console Version

1. **Launch the Application**
   ```bash
   VedXLiteAI_Console.exe
   ```

2. **Available Commands**
   ```
   describe what you see     - Vision assistance
   I feel stressed          - Wellness support
   analyze my plant         - Plant health check
   remind me to call mom    - Set reminders
   solve 2x + 3 = 7        - Math problems
   emergency help          - Crisis support
   ```

## 🌟 Feature Details

### 👁️ Vision Assistance
Perfect for visually impaired users or anyone needing visual help:

- **"Describe what you see"** - Get detailed scene descriptions
- **"Help me navigate"** - Spatial awareness and guidance
- **"Read this text"** - OCR for reading printed text
- **"What objects are here?"** - Object identification

### 💖 Wellness Support
Comprehensive mental health and wellness features:

- **Mood Detection** - Automatic emotion recognition
- **Stress Relief** - Guided meditation and breathing
- **Supportive Conversations** - Empathetic AI responses
- **Crisis Support** - Emergency resources and guidance

### 🌱 Plant Care
Advanced plant health monitoring:

- **Disease Detection** - AI-powered plant diagnosis
- **Treatment Plans** - Specific care recommendations
- **Preventive Care** - Health maintenance tips
- **Agricultural Guidance** - Professional farming advice

### 🧮 Math & Logic
Powerful problem-solving capabilities:

- **Equation Solving** - Step-by-step solutions
- **Word Problems** - Natural language math
- **Graph Analysis** - Visual mathematics
- **Logic Puzzles** - Reasoning and deduction

## 🎯 Voice Commands

### Basic Interaction
```
"Hello VedX"              - Greeting
"How are you?"            - Casual conversation
"What can you do?"        - Feature overview
"Help me with..."         - General assistance
```

### Vision Commands
```
"Describe the scene"      - Scene description
"What do you see?"        - Visual analysis
"Help me navigate"        - Navigation assistance
"Read the text"           - OCR functionality
```

### Wellness Commands
```
"I feel sad"             - Emotional support
"I'm stressed"           - Stress management
"Help me relax"          - Relaxation techniques
"I need someone to talk" - Supportive conversation
```

### Plant Care Commands
```
"Check my plant"         - Plant health analysis
"Plant disease"          - Disease detection
"How to care for..."     - Care instructions
"Plant treatment"        - Treatment recommendations
```

### Productivity Commands
```
"Remind me to..."        - Set reminders
"Solve this equation"    - Math problems
"Calculate..."           - Calculations
"Schedule..."            - Time management
```

## 🔧 Configuration

### Audio Settings
- **Microphone**: Automatic detection
- **Speakers**: System default
- **Voice Speed**: Adjustable in interface
- **Volume**: System controlled

### Camera Settings
- **Resolution**: Auto-optimized
- **Frame Rate**: 30 FPS
- **Face Detection**: Optional
- **Privacy**: Local processing only

### AI Settings
- **Offline Mode**: Always enabled
- **Memory**: Persistent user learning
- **Response Style**: Empathetic and helpful
- **Languages**: English (primary)

## 🔒 Privacy & Security

### Data Protection
- **Local Processing**: All AI runs on your computer
- **No Internet Required**: Complete offline functionality
- **Private Conversations**: Nothing sent to external servers
- **Secure Storage**: Local SQLite database encryption

### Camera & Microphone
- **User Controlled**: Manual on/off controls
- **Local Analysis**: No cloud processing
- **Privacy Indicators**: Clear on/off status
- **Data Retention**: User controlled

## 🎭 Use Cases

### 👥 Accessibility
- **Visual Impairment**: Complete vision assistance
- **Mobility Issues**: Voice-controlled interface
- **Learning Disabilities**: Patient, adaptive responses
- **Elderly Users**: Simple, intuitive design

### 🏥 Healthcare Support
- **Mental Wellness**: Daily emotional support
- **Stress Management**: Professional techniques
- **Crisis Prevention**: Early intervention
- **Medication Reminders**: Smart scheduling

### 🌾 Agriculture
- **Crop Monitoring**: Disease detection
- **Treatment Planning**: Professional advice
- **Yield Optimization**: Growth strategies
- **Sustainable Farming**: Eco-friendly practices

### 📚 Education
- **Math Tutoring**: Step-by-step learning
- **Problem Solving**: Logical thinking
- **Study Support**: Personalized assistance
- **Homework Help**: Patient explanations

## 🔧 Troubleshooting

### Common Issues

#### Camera Not Working
```
✅ Check camera permissions
✅ Ensure camera is not used by other apps
✅ Restart the application
✅ Check camera drivers
```

#### Audio Issues
```
✅ Check microphone permissions
✅ Verify audio device settings
✅ Test system audio
✅ Restart audio services
```

#### Performance Issues
```
✅ Close other resource-intensive apps
✅ Check available RAM (minimum 4GB recommended)
✅ Ensure sufficient disk space
✅ Update graphics drivers
```

#### AI Not Responding
```
✅ Wait for model loading (first run takes longer)
✅ Check system resources
✅ Restart the application
✅ Verify all files are present
```

### System Requirements

#### Minimum Requirements
- **OS**: Windows 10/11
- **RAM**: 4 GB
- **Storage**: 2 GB free space
- **CPU**: Intel i3 or AMD equivalent
- **GPU**: Integrated graphics

#### Recommended Requirements
- **OS**: Windows 11
- **RAM**: 8 GB or more
- **Storage**: 4 GB free space
- **CPU**: Intel i5 or AMD equivalent
- **GPU**: Dedicated graphics card

## 🚀 Building & Distribution

### Build Commands
```bash
make help       # Show all available commands
make setup      # Install dependencies
make build      # Build executables
make gui        # Build GUI version only
make console    # Build console version only
make clean      # Clean build artifacts
make test       # Run tests
make release    # Complete release build
```

### Distribution Package
The build process creates:
- `VedXLiteAI_GUI.exe` - Main GUI application
- `VedXLiteAI_Console.exe` - Console version
- `VedXLiteAI_Portable/` - Portable package folder

## 🤝 Contributing

We welcome contributions! Please see our contribution guidelines:

1. **Code Style**: Follow PEP 8
2. **Testing**: Add tests for new features
3. **Documentation**: Update docs for changes
4. **Pull Requests**: Use descriptive titles

### Development Setup
```bash
# Clone repository
git clone [repository-url]

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run tests
python -m pytest
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Transformers Library** - Hugging Face AI models
- **OpenCV** - Computer vision capabilities
- **SpeechRecognition** - Voice input processing
- **PyTTS3** - Text-to-speech functionality
- **Tkinter** - GUI framework

## 📞 Support

For support and questions:

- **Issues**: Use GitHub issues for bug reports
- **Feature Requests**: Submit via GitHub discussions
- **Documentation**: Check this README and code comments
- **Community**: Join our Discord server [link]

## 🗺️ Roadmap

### Upcoming Features
- [ ] **Web Interface** - Browser-based access
- [ ] **Mobile App** - Android/iOS versions
- [ ] **Plugin System** - Extensible architecture
- [ ] **Multi-language** - Support for more languages
- [ ] **Cloud Sync** - Optional cloud features
- [ ] **Team Features** - Multi-user support

### Version History
- **v1.0.0** - Initial release with core features
- **v1.1.0** - Enhanced UI and stability improvements
- **v1.2.0** - Advanced AI models and new features

---

**🤖 VedXLite AI - Your Complete AI Companion**

*Built with ❤️ for accessibility, wellness, and productivity*
