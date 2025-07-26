# 🤖 Comprehensive AI Assistant

**A sophisticated, multi-modal AI assistant designed for accessibility, empathy, and intelligent interaction**

![Project Banner](https://img.shields.io/badge/AI%20Assistant-Multi%20Modal-blue) ![Python](https://img.shields.io/badge/Python-3.8%2B-green) ![CUDA](https://img.shields.io/badge/CUDA-Accelerated-orange) ![Accessibility](https://img.shields.io/badge/Accessibility-First-purple)



## ✨ Key Features

### 🧠 **Advanced AI Capabilities**
- **Multi-Modal Processing**: Text, voice, and vision AI integration
- **CUDA Acceleration**: Up to 10x performance improvement with GPU support
- **Gemma 3n Integration**: State-of-the-art language model via Ollama
- **Enhanced Reasoning**: Advanced prompt building and context management

### ♿ **Accessibility Focus**
- **Vision Assistance**: Detailed scene descriptions for blind users
- **Voice Interaction**: Continuous speech recognition and processing
- **Safety Analysis**: Hazard identification and navigation guidance
- **Screen Reader Friendly**: Optimized for assistive technologies

### 💝 **Emotional Intelligence**
- **Role-Based Personalities**: Best Friend, Motivator, Guide, Female Friend
- **Vedx Lite Mode**: Specialized support for introverts and shy individuals
- **Empathetic Responses**: Emotion detection and supportive communication
- **Markdown Formatting**: Enhanced response readability with proper emphasis

### 🔒 **Privacy & Security**
- **Session-Only Memory**: No persistent conversation storage
- **Offline Capabilities**: VediX offline assistant for privacy
- **Local Processing**: Optional offline mode for sensitive interactions
- **Data Protection**: Comprehensive input validation and secure handling

## Tech Stack
Check out the **[Tech Stack](TECHNICAL_WRITEUP.md)** used in this project.



## Prerequisites

1. **Ollama**: Install Ollama from https://ollama.ai/
2. **Gemma 3n Model**: Pull the model using:
   ```bash
   ollama pull gemma3n:latest
   ```

## Installation

1. Clone or download this project
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Start Ollama** (if not already running):
   ```bash
   ollama serve
   ```

2. **Run the application**:
   ```bash
   python main.py
   ```

3. **Open your browser** and navigate to:
   ```
   http://localhost:5000
   ```

## File Structure

```
├── app.py           # Main Flask backend
├── mai├── vedix_core.py    # Offline assistant engine
├── ai_modules/       # Advanced AI integrations
├── templates/
│   └── index.html    # Main web interface
│   ├── css/
│   │   └── style.css # Styling
│   └── js/
│       └── script.js # JavaScript functionality
├── requirements.txt  # Dependencies
└── README.md        # This file
```

## Detailed Functionality
1. **app.py**: Serves as the main interaction backend, providing API endpoints for:
   - Chat communications
   - Image analysis through AI models
   - Vision enhancement via the Gemma3n engine
   - User management and role-based personalities
   - Comprehensive error handling and logging
2. **main.py**: Acts as the system's central orchestrator with:
   - Comprehensive system initialization
   - Advanced session and memory management
   - Robust logging and monitoring mechanisms
   - Dynamic component loading and availability checks

3. **Web Interface**: Modern and responsive design featuring:
   - Glassmorphism elements for a sleek look
   - Accessibility-focused features
   - Voice interaction capabilities
   - Detailed system status displays

## Available API Endpoints
- `GET /`: Main web interface
- `GET /api/status`: Check connection status
- `POST /api/analyze`: Analyze uploaded image
- `POST /api/chat`: Chat with Gemma
- `GET /api/health`: Health check

## Configuration

The application uses `gemma3n:latest` by default. You can modify the model in `gemma.py`:

```python
gemma_assistant = GemmaVisionAssistant(model_name="Gemma:3n")
```

## Troubleshooting

- **Connection Issues**: Ensure Ollama is running on port 11434
- **Model Not Found**: Make sure you've pulled the gemma3n:latest model
- **Image Upload Issues**: Check file size (max 16MB) and format (PNG, JPG, etc.)

## License

This project is open source and available under the MIT License.
