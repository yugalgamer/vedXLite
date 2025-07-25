// --- Voice Interaction System with Toggle Support ---
import { VoiceVisualizer } from './voice_visualizer.js';
import { voiceSettings } from './voice_settings.js';

let visualizer, recognition, isListening = false, isSpeaking = false, isRestarting = false, errorCount = 0;
const MAX_ERROR_COUNT = 3;
const ERROR_RESET_TIME = 30000; // 30 seconds

// Export the main initialization function
export async function initVoiceRecorder() {
    console.log('Initializing voice system with toggle support...');
    
    // Initialize visualizer
    visualizer = new VoiceVisualizer('voice-visualizer');
    
    // Initialize Web Speech API
    initSpeechRecognition();
    
    // Set up UI event listeners
    setupVoiceUI();
    
    // Listen for voice settings changes
    window.addEventListener('voiceSettingsChanged', (e) => {
        handleVoiceSettingsChange(e.detail);
    });
    
    // Don't auto-start voice anymore - user must enable it
    console.log('Voice system initialized! Voice is OFF by default.');
    console.log('Click the Voice toggle button to enable voice features.');
}

function initSpeechRecognition() {
    // Check for Web Speech API support
    if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
        console.error('Web Speech API not supported');
        document.getElementById('voice-model-response').textContent = 'Voice recognition not supported in this browser';
        return;
    }
    
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    recognition = new SpeechRecognition();
    
    // Configure speech recognition
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = 'en-US';
    
    // Event handlers
    recognition.onstart = function() {
        console.log('Voice recognition started');
        isListening = true;
        document.getElementById('voice-model-response').textContent = 'Listening... Say something!';
    };
    
    recognition.onresult = function(event) {
        let finalTranscript = '';
        let interimTranscript = '';
        
        for (let i = event.resultIndex; i < event.results.length; i++) {
            const transcript = event.results[i][0].transcript;
            if (event.results[i].isFinal) {
                finalTranscript += transcript;
            } else {
                interimTranscript += transcript;
            }
        }
        
        if (finalTranscript) {
            handleVoiceResult(finalTranscript.trim());
        } else if (interimTranscript) {
            document.getElementById('voice-model-response').textContent = `Listening: ${interimTranscript}`;
        }
    };
    
    recognition.onerror = function(event) {
        console.error('Speech recognition error:', event.error);
        handleVoiceError(event.error);
    };
    
    recognition.onend = function() {
        console.log('Voice recognition ended');
        if (isListening && !isSpeaking) {
            // Restart listening if we're still supposed to be listening
            setTimeout(() => {
                if (isListening && !isSpeaking) {
                    recognition.start();
                }
            }, 1000);
        }
    };
}

function setupVoiceUI() {
    // Update button functionality
    const startBtn = document.getElementById('voice-rec-btn');
    const stopBtn = document.getElementById('voice-stop-btn');
    
    if (startBtn) {
        startBtn.textContent = 'Start Voice';
        startBtn.onclick = () => startPassiveListening();
    }
    
    if (stopBtn) {
        stopBtn.textContent = 'Stop Voice';
        stopBtn.onclick = () => stopVoiceSystem();
    }
}

async function startPassiveListening() {
    if (isListening || !recognition) return;
    
    console.log('Starting passive voice listening...');
    
    // Start the glowing animation
    startGlowingAnimation();
    
    try {
        recognition.start();
        
        // Update UI
        const startBtn = document.getElementById('voice-rec-btn');
        if (startBtn) {
            startBtn.textContent = 'Voice Active';
            startBtn.disabled = true;
            startBtn.style.opacity = '0.7';
        }
        
    } catch (error) {
        console.error('Error starting voice listening:', error);
        handleVoiceError(error.message);
    }
}

function startGlowingAnimation() {
    const neonRing = document.getElementById('voice-neon-ring');
    const animPulse = document.getElementById('voice-anim-pulse');
    
    if (neonRing) {
        neonRing.classList.add('listening-active');
    }
    
    if (animPulse) {
        animPulse.classList.add('pulsing');
    }
    
    // Start visualizer if available
    if (visualizer) {
        visualizer.start().catch(console.error);
    }
}

function stopGlowingAnimation() {
    const neonRing = document.getElementById('voice-neon-ring');
    const animPulse = document.getElementById('voice-anim-pulse');
    
    if (neonRing) {
        neonRing.classList.remove('listening-active');
    }
    
    if (animPulse) {
        animPulse.classList.remove('pulsing');
    }
    
    // Stop visualizer
    if (visualizer) {
        visualizer.stop();
    }
}

function handleVoiceResult(transcription) {
    if (!transcription || transcription.trim() === '') return;
    
    console.log('Voice transcription:', transcription);
    
    // Show what user said
    document.getElementById('voice-model-response').textContent = `You said: "${transcription}"`;
    
    // Send to backend AI for processing
    sendToBackendAI(transcription);
}

async function sendToBackendAI(text) {
    try {
        document.getElementById('voice-model-response').textContent = 'Processing...';
        
        const response = await fetch('/api/voice-interact', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ 
                voiceText: text,
                username: window.currentUserName || '',
                role: window.currentUserRole || ''
            })
        });
        
        const result = await response.json();
        
        if (result.success) {
            // Display AI response
            document.getElementById('voice-model-response').textContent = result.reply;
            
            // Convert AI response to speech and play it
            await speakText(result.reply);
            
        } else {
            const errorMsg = 'Error: ' + (result.error || 'Unknown error');
            document.getElementById('voice-model-response').textContent = errorMsg;
            await speakText("Sorry, I encountered an error processing your request.");
        }
        
    } catch (error) {
        console.error('Error sending to backend:', error);
        const errorMsg = 'Connection error. Please check your network.';
        document.getElementById('voice-model-response').textContent = errorMsg;
        await speakText("Sorry, I'm having trouble connecting. Please try again.");
    }
}

function handleVoiceError(error) {
    console.error('Voice recognition error:', error);
    document.getElementById('voice-model-response').textContent = `Voice error: ${error}`;
    
    // Try to restart listening after a short delay
    setTimeout(() => {
        if (!isListening) {
            startPassiveListening();
        }
    }, 3000);
}

// Handle voice settings changes
function handleVoiceSettingsChange(settings) {
    console.log('Voice settings changed:', settings);
    
    if (!settings.voiceEnabled || !settings.sttEnabled) {
        // Stop voice recognition if disabled
        if (isListening) {
            stopVoiceSystem();
        }
    }
    
    // Update UI elements
    updateVoiceButtons(settings);
}

function updateVoiceButtons(settings) {
    // Update microphone button in chat
    const micBtn = document.getElementById('mic-btn');
    if (micBtn) {
        const isSTTEnabled = settings.voiceEnabled && settings.sttEnabled;
        micBtn.disabled = !isSTTEnabled;
        micBtn.style.opacity = isSTTEnabled ? '1' : '0.5';
        micBtn.title = isSTTEnabled ? 'Click to speak' : 'Voice input disabled';
    }
    
    // Update voice tab buttons
    const startBtn = document.getElementById('voice-rec-btn');
    if (startBtn) {
        const canStart = settings.voiceEnabled && settings.sttEnabled && !isListening;
        startBtn.disabled = !canStart;
        startBtn.style.opacity = canStart ? '1' : '0.5';
    }
}

export function speakText(text) {
    return new Promise((resolve) => {
        // Check if TTS is enabled in settings
        if (!voiceSettings.isTTSEnabled()) {
            console.log('Text-to-speech disabled by user settings');
            resolve();
            return;
        }
        
        if ('speechSynthesis' in window) {
            const utter = new SpeechSynthesisUtterance(text);
            
            // Get settings from voice manager
            const settings = voiceSettings.getVoiceSettings();
            
            // Configure voice settings from user preferences
            utter.rate = settings.rate || 0.9;
            utter.pitch = settings.pitch || 1.0;
            utter.volume = settings.volume || 0.8;
            
            // Use female voice if available
            const voices = speechSynthesis.getVoices();
            const femaleVoice = voices.find(voice => 
                voice.name.toLowerCase().includes('female') || 
                voice.name.toLowerCase().includes('karen') ||
                voice.name.toLowerCase().includes('samantha')
            );
            
            if (femaleVoice) {
                utter.voice = femaleVoice;
            }
            
            utter.onend = () => resolve();
            utter.onerror = () => resolve();
            
            window.speechSynthesis.speak(utter);
        } else {
            console.warn('Text-to-speech not supported');
            resolve();
        }
    });
}

function stopVoiceSystem() {
    console.log('Stopping voice system...');
    
    isListening = false;
    
    // Stop speech recognition
    if (recognition) {
        recognition.stop();
    }
    
    // Stop speech synthesis if speaking
    if (window.speechSynthesis && window.speechSynthesis.speaking) {
        window.speechSynthesis.cancel();
    }
    
    // Stop animations
    stopGlowingAnimation();
    
    // Update UI
    document.getElementById('voice-model-response').textContent = 'Voice system stopped.';
    
    // Re-enable start button
    const startBtn = document.getElementById('voice-rec-btn');
    if (startBtn) {
        startBtn.textContent = 'Start Voice';
        startBtn.disabled = false;
        startBtn.style.opacity = '1';
        startBtn.onclick = () => startPassiveListening();
    }
}

// Legacy support for existing code
export function startVoiceRecognition() {
    console.log('Starting voice recognition (legacy support)');
    if (!isListening) {
        startPassiveListening();
    }
}

// Make functions available globally for HTML onclick handlers
window.initVoiceRecorder = initVoiceRecorder;
window.speakText = speakText;
window.startVoiceRecognition = startVoiceRecognition;
