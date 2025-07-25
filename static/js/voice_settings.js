// Voice Settings Manager - Controls voice functionality state
export class VoiceSettingsManager {
    constructor() {
        this.settings = {
            voiceEnabled: false, // Default: voice is OFF
            ttsEnabled: false,   // Text-to-speech OFF by default
            sttEnabled: false,   // Speech-to-text OFF by default
            autoStart: false,    // Don't auto-start voice on page load
            volume: 0.8,
            rate: 0.9,
            pitch: 1.0
        };
        
        this.loadSettings();
        this.initUI();
        
        console.log('Voice Settings initialized:', this.settings);
    }
    
    loadSettings() {
        try {
            const saved = localStorage.getItem('voiceSettings');
            if (saved) {
                this.settings = { ...this.settings, ...JSON.parse(saved) };
            }
        } catch (error) {
            console.warn('Could not load voice settings:', error);
        }
    }
    
    saveSettings() {
        try {
            localStorage.setItem('voiceSettings', JSON.stringify(this.settings));
        } catch (error) {
            console.warn('Could not save voice settings:', error);
        }
    }
    
    initUI() {
        this.createSettingsPanel();
        this.updateUIState();
    }
    
    createSettingsPanel() {
        // Create voice settings toggle in header
        const header = document.querySelector('.app-header .header-left');
        if (!header) return;
        
        const settingsContainer = document.createElement('div');
        settingsContainer.className = 'voice-settings-container';
        settingsContainer.innerHTML = `
            <div class="voice-toggle-group">
                <button id="voice-toggle-btn" class="voice-toggle-btn ${this.settings.voiceEnabled ? 'enabled' : 'disabled'}" 
                        title="Toggle Voice Features">
                    <i class="fas ${this.settings.voiceEnabled ? 'fa-volume-up' : 'fa-volume-mute'}"></i>
                    <span class="toggle-text">${this.settings.voiceEnabled ? 'Voice ON' : 'Voice OFF'}</span>
                </button>
                <div class="voice-settings-dropdown" id="voice-settings-dropdown">
                    <div class="setting-item">
                        <label>
                            <input type="checkbox" id="tts-toggle" ${this.settings.ttsEnabled ? 'checked' : ''}>
                            <span>AI Voice Response</span>
                        </label>
                    </div>
                    <div class="setting-item">
                        <label>
                            <input type="checkbox" id="stt-toggle" ${this.settings.sttEnabled ? 'checked' : ''}>
                            <span>Voice Input</span>
                        </label>
                    </div>
                    <div class="setting-item">
                        <label>
                            <input type="range" id="volume-slider" min="0" max="1" step="0.1" value="${this.settings.volume}">
                            <span>Volume: <span id="volume-display">${Math.round(this.settings.volume * 100)}%</span></span>
                        </label>
                    </div>
                    <div class="setting-item">
                        <label>
                            <input type="range" id="rate-slider" min="0.5" max="2" step="0.1" value="${this.settings.rate}">
                            <span>Speed: <span id="rate-display">${this.settings.rate}x</span></span>
                        </label>
                    </div>
                </div>
            </div>
        `;
        
        header.appendChild(settingsContainer);
        this.attachEventListeners();
        this.addStyles();
    }
    
    addStyles() {
        const style = document.createElement('style');
        style.textContent = `
            .voice-settings-container {
                position: relative;
                margin-left: 15px;
                display: inline-block;
            }
            
            .voice-toggle-btn {
                background: linear-gradient(135deg, #64748b 0%, #475569 100%);
                color: white;
                border: none;
                padding: 8px 12px;
                border-radius: 8px;
                cursor: pointer;
                font-size: 14px;
                font-weight: 500;
                transition: all 0.3s ease;
                display: flex;
                align-items: center;
                gap: 6px;
                min-width: 100px;
            }
            
            .voice-toggle-btn:hover {
                transform: translateY(-1px);
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
            }
            
            .voice-toggle-btn.enabled {
                background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            }
            
            .voice-toggle-btn.disabled {
                background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
            }
            
            .voice-settings-dropdown {
                position: absolute;
                top: 100%;
                left: 0;
                background: white;
                border: 2px solid #e5e7eb;
                border-radius: 8px;
                padding: 15px;
                min-width: 250px;
                box-shadow: 0 10px 25px rgba(0, 0, 0, 0.15);
                z-index: 1000;
                display: none;
                margin-top: 5px;
            }
            
            .voice-settings-dropdown.show {
                display: block;
                animation: dropdownSlide 0.2s ease-out;
            }
            
            @keyframes dropdownSlide {
                from {
                    opacity: 0;
                    transform: translateY(-10px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
            
            .setting-item {
                margin-bottom: 12px;
                padding: 8px 0;
                border-bottom: 1px solid #f3f4f6;
            }
            
            .setting-item:last-child {
                border-bottom: none;
                margin-bottom: 0;
            }
            
            .setting-item label {
                display: flex;
                align-items: center;
                justify-content: space-between;
                cursor: pointer;
                font-size: 14px;
                color: #374151;
            }
            
            .setting-item input[type="checkbox"] {
                margin-right: 8px;
                transform: scale(1.2);
            }
            
            .setting-item input[type="range"] {
                width: 120px;
                margin-right: 10px;
            }
            
            .voice-status-indicator {
                position: fixed;
                top: 70px;
                right: 20px;
                background: rgba(0, 0, 0, 0.8);
                color: white;
                padding: 8px 12px;
                border-radius: 6px;
                font-size: 12px;
                z-index: 1001;
                display: none;
            }
            
            .voice-status-indicator.show {
                display: block;
                animation: fadeInOut 2s ease-in-out;
            }
            
            @keyframes fadeInOut {
                0%, 100% { opacity: 0; }
                50% { opacity: 1; }
            }
        `;
        document.head.appendChild(style);
    }
    
    attachEventListeners() {
        const toggleBtn = document.getElementById('voice-toggle-btn');
        const dropdown = document.getElementById('voice-settings-dropdown');
        const ttsToggle = document.getElementById('tts-toggle');
        const sttToggle = document.getElementById('stt-toggle');
        const volumeSlider = document.getElementById('volume-slider');
        const rateSlider = document.getElementById('rate-slider');
        
        // Main toggle button
        toggleBtn?.addEventListener('click', (e) => {
            e.stopPropagation();
            this.toggleVoice();
            dropdown?.classList.toggle('show');
        });
        
        // Close dropdown when clicking outside
        document.addEventListener('click', (e) => {
            if (!toggleBtn?.contains(e.target)) {
                dropdown?.classList.remove('show');
            }
        });
        
        // TTS toggle
        ttsToggle?.addEventListener('change', (e) => {
            this.settings.ttsEnabled = e.target.checked;
            this.saveSettings();
            this.showStatusMessage(`AI Voice Response ${e.target.checked ? 'ON' : 'OFF'}`);
        });
        
        // STT toggle
        sttToggle?.addEventListener('change', (e) => {
            this.settings.sttEnabled = e.target.checked;
            this.saveSettings();
            this.showStatusMessage(`Voice Input ${e.target.checked ? 'ON' : 'OFF'}`);
            
            // Enable/disable microphone button
            this.updateMicrophoneButton();
        });
        
        // Volume slider
        volumeSlider?.addEventListener('input', (e) => {
            this.settings.volume = parseFloat(e.target.value);
            document.getElementById('volume-display').textContent = `${Math.round(this.settings.volume * 100)}%`;
            this.saveSettings();
        });
        
        // Rate slider
        rateSlider?.addEventListener('input', (e) => {
            this.settings.rate = parseFloat(e.target.value);
            document.getElementById('rate-display').textContent = `${this.settings.rate}x`;
            this.saveSettings();
        });
    }
    
    toggleVoice() {
        this.settings.voiceEnabled = !this.settings.voiceEnabled;
        
        // If turning voice on, enable both TTS and STT by default
        if (this.settings.voiceEnabled) {
            this.settings.ttsEnabled = true;
            this.settings.sttEnabled = true;
        } else {
            this.settings.ttsEnabled = false;
            this.settings.sttEnabled = false;
        }
        
        this.saveSettings();
        this.updateUIState();
        this.showStatusMessage(`Voice Features ${this.settings.voiceEnabled ? 'ENABLED' : 'DISABLED'}`);
        
        // Update checkboxes
        const ttsToggle = document.getElementById('tts-toggle');
        const sttToggle = document.getElementById('stt-toggle');
        if (ttsToggle) ttsToggle.checked = this.settings.ttsEnabled;
        if (sttToggle) sttToggle.checked = this.settings.sttEnabled;
        
        // Update microphone button
        this.updateMicrophoneButton();
        
        // Trigger custom event for other components
        window.dispatchEvent(new CustomEvent('voiceSettingsChanged', {
            detail: this.settings
        }));
    }
    
    updateUIState() {
        const toggleBtn = document.getElementById('voice-toggle-btn');
        if (!toggleBtn) return;
        
        const icon = toggleBtn.querySelector('i');
        const text = toggleBtn.querySelector('.toggle-text');
        
        if (this.settings.voiceEnabled) {
            toggleBtn.className = 'voice-toggle-btn enabled';
            if (icon) icon.className = 'fas fa-volume-up';
            if (text) text.textContent = 'Voice ON';
        } else {
            toggleBtn.className = 'voice-toggle-btn disabled';
            if (icon) icon.className = 'fas fa-volume-mute';
            if (text) text.textContent = 'Voice OFF';
        }
        
        this.updateMicrophoneButton();
    }
    
    updateMicrophoneButton() {
        const micBtn = document.getElementById('mic-btn');
        if (!micBtn) return;
        
        const isEnabled = this.settings.voiceEnabled && this.settings.sttEnabled;
        
        micBtn.disabled = !isEnabled;
        micBtn.style.opacity = isEnabled ? '1' : '0.5';
        micBtn.title = isEnabled ? 'Click to speak' : 'Voice input disabled';
        
        if (isEnabled) {
            micBtn.classList.remove('disabled');
        } else {
            micBtn.classList.add('disabled');
        }
    }
    
    showStatusMessage(message) {
        let indicator = document.querySelector('.voice-status-indicator');
        if (!indicator) {
            indicator = document.createElement('div');
            indicator.className = 'voice-status-indicator';
            document.body.appendChild(indicator);
        }
        
        indicator.textContent = message;
        indicator.classList.add('show');
        
        setTimeout(() => {
            indicator.classList.remove('show');
        }, 2000);
    }
    
    // Public methods for other components
    isVoiceEnabled() {
        return this.settings.voiceEnabled;
    }
    
    isTTSEnabled() {
        return this.settings.voiceEnabled && this.settings.ttsEnabled;
    }
    
    isSTTEnabled() {
        return this.settings.voiceEnabled && this.settings.sttEnabled;
    }
    
    getVoiceSettings() {
        return { ...this.settings };
    }
    
    updateSetting(key, value) {
        if (key in this.settings) {
            this.settings[key] = value;
            this.saveSettings();
            this.updateUIState();
        }
    }
}

// Create global instance
export const voiceSettings = new VoiceSettingsManager();

// Make available globally
window.voiceSettings = voiceSettings;
