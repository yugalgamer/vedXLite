// Custom Vosk Speech Recognizer for offline voice recognition
class VoskSpeechRecognizer {
    constructor() {
        this.isListening = false;
        this.mediaRecorder = null;
        this.audioStream = null;
        this.onResultCallback = null;
        this.onErrorCallback = null;
        this.audioChunks = [];
        this.silenceTimer = null;
        this.voiceStarted = false;
        this.silenceThreshold = 2000; // 2 seconds of silence to stop recording
    }

    // Start passive voice listening
    async startListening(onResult, onError) {
        if (this.isListening) return;
        
        this.onResultCallback = onResult;
        this.onErrorCallback = onError;
        
        try {
            // Get microphone access
            this.audioStream = await navigator.mediaDevices.getUserMedia({ 
                audio: {
                    sampleRate: 16000,
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true
                } 
            });
            
            // Create audio context for voice activity detection
            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const source = audioContext.createMediaStreamSource(this.audioStream);
            const analyser = audioContext.createAnalyser();
            analyser.fftSize = 512;
            source.connect(analyser);
            
            // Voice activity detection
            const dataArray = new Uint8Array(analyser.frequencyBinCount);
            let silenceStart = Date.now();
            
            const checkForVoice = () => {
                if (!this.isListening) return;
                
                analyser.getByteFrequencyData(dataArray);
                const average = dataArray.reduce((a, b) => a + b) / dataArray.length;
                
                if (average > 10) { // Voice detected
                    if (!this.voiceStarted) {
                        this.startRecording();
                        this.voiceStarted = true;
                    }
                    silenceStart = Date.now();
                } else { // Silence
                    if (this.voiceStarted && Date.now() - silenceStart > this.silenceThreshold) {
                        this.stopRecording();
                        this.voiceStarted = false;
                    }
                }
                
                requestAnimationFrame(checkForVoice);
            };
            
            this.isListening = true;
            checkForVoice();
            
        } catch (error) {
            if (this.onErrorCallback) {
                this.onErrorCallback('Microphone access denied: ' + error.message);
            }
        }
    }

    startRecording() {
        if (this.mediaRecorder) return;
        
        this.audioChunks = [];
        this.mediaRecorder = new MediaRecorder(this.audioStream, {
            mimeType: 'audio/webm;codecs=opus'
        });
        
        this.mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                this.audioChunks.push(event.data);
            }
        };
        
        this.mediaRecorder.onstop = () => {
            const audioBlob = new Blob(this.audioChunks, { type: 'audio/webm' });
            this.processAudio(audioBlob);
            this.mediaRecorder = null;
        };
        
        this.mediaRecorder.start();
    }

    stopRecording() {
        if (this.mediaRecorder && this.mediaRecorder.state === 'recording') {
            this.mediaRecorder.stop();
        }
    }

    async processAudio(audioBlob) {
        try {
            // Convert audio to format suitable for Vosk
            const audioBuffer = await audioBlob.arrayBuffer();
            const audioContext = new AudioContext();
            const audioData = await audioContext.decodeAudioData(audioBuffer);
            
            // Convert to PCM data
            const pcmData = this.audioBuferToPCM(audioData);
            
            // Send to Vosk for transcription (we'll use a simulated version)
            const transcription = await this.transcribeWithVosk(pcmData);
            
            if (transcription && this.onResultCallback) {
                this.onResultCallback(transcription);
            }
        } catch (error) {
            if (this.onErrorCallback) {
                this.onErrorCallback('Audio processing error: ' + error.message);
            }
        }
    }

    audioBuferToPCM(audioBuffer) {
        const length = audioBuffer.length;
        const pcmData = new Int16Array(length);
        const inputData = audioBuffer.getChannelData(0);
        
        for (let i = 0; i < length; i++) {
            pcmData[i] = Math.max(-1, Math.min(1, inputData[i])) * 32767;
        }
        
        return pcmData;
    }

    async transcribeWithVosk(pcmData) {
        // Since we can't directly use Vosk WASM in this environment,
        // we'll send the audio to the backend for transcription
        try {
            const blob = new Blob([pcmData], { type: 'application/octet-stream' });
            const formData = new FormData();
            formData.append('audio', blob, 'audio.raw');
            
            const response = await fetch('/api/vosk-transcribe', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            return result.text || '';
        } catch (error) {
            console.error('Vosk transcription error:', error);
            return '';
        }
    }

    stop() {
        this.isListening = false;
        this.voiceStarted = false;
        
        if (this.mediaRecorder && this.mediaRecorder.state === 'recording') {
            this.mediaRecorder.stop();
        }
        
        if (this.audioStream) {
            this.audioStream.getTracks().forEach(track => track.stop());
        }
        
        if (this.silenceTimer) {
            clearTimeout(this.silenceTimer);
        }
    }
}

export { VoskSpeechRecognizer };
