// Request Queue System - Only one request at a time
class RequestQueue {
    constructor() {
        this.isProcessing = false;
        this.queue = [];
        this.currentRequestController = null;
    }

    async addRequest(requestConfig) {
        return new Promise((resolve, reject) => {
            const requestItem = {
                config: requestConfig,
                resolve,
                reject,
                timestamp: Date.now()
            };

            // If already processing, queue this request
            if (this.isProcessing) {
                this.queue.push(requestItem);
                this.showQueueMessage(this.queue.length);
                return;
            }

            // Process immediately if not busy
            this.processRequest(requestItem);
        });
    }

    async processRequest(requestItem) {
        this.isProcessing = true;
        this.showProcessingIndicator();

        try {
            // Create AbortController for this request
            this.currentRequestController = new AbortController();
            
            // Add abort signal to fetch config
            const config = {
                ...requestItem.config,
                signal: this.currentRequestController.signal
            };

            const response = await fetch(config.url, config);
            const data = await response.json();
            
            requestItem.resolve(data);
        } catch (error) {
            if (error.name === 'AbortError') {
                requestItem.reject(new Error('Request was cancelled'));
            } else {
                requestItem.reject(error);
            }
        } finally {
            this.isProcessing = false;
            this.currentRequestController = null;
            this.hideProcessingIndicator();
            
            // Process next item in queue
            this.processNextInQueue();
        }
    }

    processNextInQueue() {
        if (this.queue.length > 0) {
            const nextRequest = this.queue.shift();
            this.updateQueueMessage();
            this.processRequest(nextRequest);
        }
    }

    cancelAllRequests() {
        // Cancel current request
        if (this.currentRequestController) {
            this.currentRequestController.abort();
        }

        // Reject all queued requests
        while (this.queue.length > 0) {
            const request = this.queue.shift();
            request.reject(new Error('Request queue cleared'));
        }

        this.isProcessing = false;
        this.hideProcessingIndicator();
    }

    showProcessingIndicator() {
        // Show typing animation for AI thinking
        this.showTypingAnimation('AI is thinking...');
    }

    hideProcessingIndicator() {
        this.hideTypingAnimation();
    }

    showQueueMessage(queueLength) {
        const queueMsg = document.getElementById('queue-message');
        if (queueMsg) {
            queueMsg.textContent = `Request queued (${queueLength} in queue)`;
            queueMsg.style.display = 'block';
        } else {
            // Create queue message element
            const msgDiv = document.createElement('div');
            msgDiv.id = 'queue-message';
            msgDiv.className = 'queue-message';
            msgDiv.textContent = `Request queued (${queueLength} in queue)`;
            msgDiv.style.cssText = `
                position: fixed;
                top: 70px;
                right: 20px;
                background: #ff9800;
                color: white;
                padding: 8px 16px;
                border-radius: 20px;
                font-size: 12px;
                z-index: 1000;
                animation: slideInRight 0.3s ease-out;
            `;
            document.body.appendChild(msgDiv);
        }
    }

    updateQueueMessage() {
        const queueMsg = document.getElementById('queue-message');
        if (queueMsg && this.queue.length > 0) {
            queueMsg.textContent = `Request queued (${this.queue.length} in queue)`;
        } else if (queueMsg) {
            queueMsg.style.display = 'none';
        }
    }

    showTypingAnimation(message = 'AI is thinking...') {
        // Remove existing typing indicator
        this.hideTypingAnimation();

        // Create typing animation
        const typingDiv = document.createElement('div');
        typingDiv.id = 'typing-indicator';
        typingDiv.className = 'message assistant-message typing-message';
        typingDiv.innerHTML = `
            <div class="message-avatar">
                <i class="fas fa-robot"></i>
            </div>
            <div class="message-content">
                <div class="typing-text">${message}</div>
                <div class="typing-dots">
                    <span class="typing-dot"></span>
                    <span class="typing-dot"></span>
                    <span class="typing-dot"></span>
                </div>
            </div>
        `;

        // Add to chat messages
        const chatMessages = document.getElementById('chat-messages');
        if (chatMessages) {
            chatMessages.appendChild(typingDiv);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }

        // Also show in voice tab if active
        const voiceResponse = document.getElementById('voice-model-response');
        if (voiceResponse && document.getElementById('voice-interact-tab').classList.contains('active')) {
            voiceResponse.innerHTML = `
                <div class="voice-typing">
                    <div>${message}</div>
                    <div class="typing-dots">
                        <span class="typing-dot"></span>
                        <span class="typing-dot"></span>
                        <span class="typing-dot"></span>
                    </div>
                </div>
            `;
        }
    }

    hideTypingAnimation() {
        const typingIndicator = document.getElementById('typing-indicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }

        const voiceResponse = document.getElementById('voice-model-response');
        if (voiceResponse && voiceResponse.querySelector('.voice-typing')) {
            voiceResponse.innerHTML = '';
        }
    }

    // Simulate typewriter effect for responses
    async simulateTypingResponse(element, text, speed = 50) {
        element.innerHTML = '';
        
        for (let i = 0; i < text.length; i++) {
            element.innerHTML += text.charAt(i);
            await new Promise(resolve => setTimeout(resolve, speed));
            
            // Scroll to keep text visible
            if (element.closest('#chat-messages')) {
                element.closest('#chat-messages').scrollTop = element.closest('#chat-messages').scrollHeight;
            }
        }
    }

    getQueueStatus() {
        return {
            isProcessing: this.isProcessing,
            queueLength: this.queue.length,
            hasActiveRequest: this.currentRequestController !== null
        };
    }
}

// Create global instance
window.requestQueue = new RequestQueue();

export { RequestQueue };
export default window.requestQueue;
