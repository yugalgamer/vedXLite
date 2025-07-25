// Audio visualizer for live voice interaction
export class VoiceVisualizer {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.analyser = null;
        this.animationFrame = null;
        this.audioStream = null;
    }

    async start() {
        try {
            // Check if we're on localhost or HTTPS
            const isSecureContext = window.location.protocol === 'https:' || 
                                   window.location.hostname === 'localhost' ||
                                   window.location.hostname === '127.0.0.1';
            
            if (!isSecureContext) {
                console.warn('Microphone requires HTTPS or localhost');
                // Create a visual simulation instead
                this.simulateAudioVisualization();
                return;
            }
            
            if (typeof navigator === 'undefined' || !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                console.warn('Audio input not supported by this browser');
                this.simulateAudioVisualization();
                return;
            }
            
            this.audioStream = await navigator.mediaDevices.getUserMedia({ 
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true
                } 
            });
            
            const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            const source = audioCtx.createMediaStreamSource(this.audioStream);
            this.analyser = audioCtx.createAnalyser();
            source.connect(this.analyser);
            this.analyser.fftSize = 128;
            this.dataArray = new Uint8Array(this.analyser.frequencyBinCount);
            this.draw();
            
        } catch (error) {
            console.warn('Microphone access denied or failed:', error);
            this.simulateAudioVisualization();
        }
    }
    
    simulateAudioVisualization() {
        // Create a simulated audio visualization for when mic is not available
        this.dataArray = new Uint8Array(64);
        this.drawSimulated();
    }
    
    drawSimulated() {
        this.animationFrame = requestAnimationFrame(() => this.drawSimulated());
        
        // Generate random audio-like data
        for (let i = 0; i < this.dataArray.length; i++) {
            this.dataArray[i] = Math.random() * 100 + Math.sin(Date.now() * 0.01 + i * 0.1) * 50;
        }
        
        const { width, height } = this.canvas;
        this.ctx.clearRect(0, 0, width, height);
        const barWidth = (width / this.dataArray.length) * 2.5;
        let x = 0;
        
        for (let i = 0; i < this.dataArray.length; i++) {
            const barHeight = this.dataArray[i] * 0.8;
            this.ctx.fillStyle = 'rgba(0,123,255, 0.8)';
            this.ctx.fillRect(x, height - barHeight, barWidth, barHeight);
            x += barWidth + 1;
        }
    }

    stop() {
        if (this.animationFrame) {
            cancelAnimationFrame(this.animationFrame);
        }
        if (this.audioStream) {
            this.audioStream.getTracks().forEach(t => t.stop());
        }
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }

    draw() {
        this.animationFrame = requestAnimationFrame(() => this.draw());
        this.analyser.getByteFrequencyData(this.dataArray);
        const { width, height } = this.canvas;
        this.ctx.clearRect(0, 0, width, height);
        const barWidth = (width / this.dataArray.length) * 2.5;
        let x = 0;
        for (let i = 0; i < this.dataArray.length; i++) {
            const barHeight = this.dataArray[i] * 0.8;
            this.ctx.fillStyle = 'rgba(0,123,255, 0.8)';
            this.ctx.fillRect(x, height - barHeight, barWidth, barHeight);
            x += barWidth + 1;
        }
    }
}

