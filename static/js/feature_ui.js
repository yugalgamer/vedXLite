import { startVoiceRecognition, speakText } from './feature_voice.js';

// -------- Chat UI functions (adapted from original script.js) ----------
function switchTab(tabName) {
    document.querySelectorAll('.tab-content').forEach(tabContent => {
        tabContent.classList.remove('active');
    });
    document.getElementById(`${tabName}-tab`).classList.add('active');

    document.querySelectorAll('.tab-button').forEach(tabButton => {
        tabButton.classList.remove('active');
    });
    document.querySelector(`button[data-tab="${tabName}"]`).classList.add('active');
    
    // Special handling for enhanced vision tab
    if (tabName === 'enhanced-vision') {
        // Initialize enhanced vision if not already done
        if (window.initEnhancedVision && typeof window.initEnhancedVision === 'function') {
            console.log('📸 Switching to Enhanced Vision Assistant');
        }
    }
};

// ------------- Chat Image Upload Integration -------------
window.setupChatImageUpload = function() {
    const imageInput = document.getElementById('chat-image-input');
    const imagePreviewContainer = document.getElementById('chat-image-preview');
    const previewImg = document.getElementById('chat-preview-img');
    const removeBtn = document.getElementById('chat-remove-img-btn');

    imageInput.addEventListener('change', () => {
        const file = imageInput.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                previewImg.src = e.target.result;
                imagePreviewContainer.style.display = 'flex';
            };
            reader.readAsDataURL(file);
        }
    });
    removeBtn.addEventListener('click', () => {
        imageInput.value = null;
        imagePreviewContainer.style.display = 'none';
        previewImg.src = '';
    });
};

window.sendMessage = function() {
    const chatInput = document.getElementById('chat-input');
    const userMessage = chatInput.value.trim();
    const imageInput = document.getElementById('chat-image-input');
    const file = imageInput.files[0];
    if (!userMessage && !file) return;

    // Check for role change commands
    if (checkForRoleChangeCommand(userMessage)) {
        chatInput.value = '';
        return;
    }
    
    // Check for clear/reset commands
    if (checkForClearCommand(userMessage)) {
        clearCurrentChat();
        chatInput.value = '';
        return;
    }

    // Show user message (with image if any)
    appendMessage('user', userMessage, file ? URL.createObjectURL(file) : null);
    chatInput.value = '';
    imageInput.value = null;
    document.getElementById('chat-image-preview').style.display = 'none';

    displayLoading(true);

    let request;
    if (file) {
        // Send image + message as a multipart form
        const formData = new FormData();
        formData.append('prompt', userMessage);
        formData.append('image', file);
        if (window.currentUserName) formData.append('username', window.currentUserName);
        if (window.currentUserRole) formData.append('role', window.currentUserRole);
        request = window.requestQueue.addRequest({
            url: '/api/chat-image',
            method: 'POST',
            body: formData
        });
    } else {
        // Send as JSON
        request = window.requestQueue.addRequest({
            url: '/api/chat',
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ 
                prompt: userMessage, 
                username: window.currentUserName || '',
                role: window.currentUserRole || ''
            })
        });
    }
    request.then(data => {
        displayLoading(false);
        if (data.success) {
            appendMessage('assistant', data.response, data.imageUrl || null);
        } else {
            alert(`Error: ${data.error}`);
        }
      })
      .catch(error => {
        displayLoading(false);
        alert(`Error: ${error}`);
      });
};

async function appendMessage(role, text, imageUrl = null) {
    const chatMessages = document.getElementById('chat-messages');
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message');

    const avatarDiv = document.createElement('div');
    avatarDiv.classList.add('message-avatar');
    if (role === 'user') {
        avatarDiv.innerHTML = '<i class="fas fa-user"></i>';
    } else {
        avatarDiv.innerHTML = '<i class="fas fa-robot"></i>';
    }

    const contentDiv = document.createElement('div');
    contentDiv.classList.add('message-content');
    
    if (imageUrl) {
        const img = document.createElement('img');
        img.className = 'message-img';
        img.src = imageUrl;
        img.alt = 'Image sent';
        img.style.maxWidth = '180px';
        img.style.display = 'block';
        img.style.marginBottom = '8px';
        contentDiv.appendChild(img);
    }
    
    if (text) {
        const textP = document.createElement('div');
        contentDiv.appendChild(textP);
        
        // Add to advanced memory system
        if (window.advancedMemory && role === 'user') {
            // Store user message in advanced memory
            const context = {
                timestamp: new Date(),
                username: window.currentUserName,
                role: window.currentUserRole
            };
            window.advancedMemory.addConversationMemory(text, '', context);
        }
        
        // Apply typewriter effect for AI responses
        if (role === 'assistant' && text.length > 0) {
            await simulateTyping(textP, text);
        } else {
            textP.innerHTML = text;
        }
    }

    messageDiv.appendChild(avatarDiv);
    messageDiv.appendChild(contentDiv);

    if (role === 'assistant') messageDiv.classList.add('assistant-message');
    if (role === 'user') messageDiv.classList.add('user-message');

    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// **NEW: Markdown Asterisk Formatter** 🎨
function formatMarkdownAsterisks(text) {
    if (!text || typeof text !== 'string') return text;
    
    // Escape HTML first to prevent XSS
    text = text.replace(/\u0026/g, '\u0026amp;')
              .replace(/\u003c/g, '\u0026lt;')
              .replace(/\u003e/g, '\u0026gt;');
    
    // Convert line breaks to HTML
    text = text.replace(/\\n/g, '\u003cbr\u003e');
    
    // **Process nested formatting (order matters!)**
    // First handle triple asterisks (bold + italic)
    text = text.replace(/\\*\\*\\*(.*?)\\*\\*\\*/g, '\u003cstrong\u003e\u003cem\u003e$1\u003c/em\u003e\u003c/strong\u003e');
    
    // Then handle double asterisks (bold)
    text = text.replace(/\\*\\*(.*?)\\*\\*/g, '\u003cstrong\u003e$1\u003c/strong\u003e');
    
    // Finally handle single asterisks (italic)
    text = text.replace(/\\*(.*?)\\*/g, '\u003cem\u003e$1\u003c/em\u003e');
    
    return text;
}

// Enhanced typewriter effect with Markdown formatting
async function simulateTyping(element, text, speed = 30) {
    element.innerHTML = '';
    
    // **Apply Markdown formatting to the text first**
    const formattedText = formatMarkdownAsterisks(text);
    
    // Create a temporary div to parse HTML
    const tempDiv = document.createElement('div');
    tempDiv.innerHTML = formattedText;
    
    // Type character by character, preserving HTML formatting
    await typeHTMLContent(element, tempDiv, speed);
    
    // Final scroll to keep text visible
    const chatMessages = document.getElementById('chat-messages');
    if (chatMessages) {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
}

// **Helper function to type HTML content with formatting preserved**
async function typeHTMLContent(targetElement, sourceElement, speed) {
    const walker = document.createTreeWalker(
        sourceElement,
        NodeFilter.SHOW_TEXT,
        null,
        false
    );
    
    const textNodes = [];
    let node;
    while (node = walker.nextNode()) {
        textNodes.push({
            node: node,
            parent: node.parentNode,
            text: node.textContent
        });
    }
    
    // Clone the structure but with empty text
    targetElement.innerHTML = sourceElement.innerHTML.replace(/\u003e[^\u003c]*\u003c/g, '\u003e\u003c');
    
    // Find corresponding text nodes in target
    const targetWalker = document.createTreeWalker(
        targetElement,
        NodeFilter.SHOW_TEXT,
        null,
        false
    );
    
    const targetTextNodes = [];
    while (node = targetWalker.nextNode()) {
        targetTextNodes.push(node);
    }
    
    // Type each text node character by character
    for (let i = 0; i < textNodes.length && i < targetTextNodes.length; i++) {
        const sourceText = textNodes[i].text;
        const targetNode = targetTextNodes[i];
        
        for (let j = 0; j < sourceText.length; j++) {
            targetNode.textContent = sourceText.substring(0, j + 1);
            await new Promise(resolve => setTimeout(resolve, speed));
            
            // Scroll periodically during typing
            if (j % 10 === 0) {
                const chatMessages = document.getElementById('chat-messages');
                if (chatMessages) {
                    chatMessages.scrollTop = chatMessages.scrollHeight;
                }
            }
        }
    }
}

window.handleChatKeyPress = function(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        window.sendMessage();
    }
};

function displayLoading(show) {
    const loadingOverlay = document.getElementById('loading-overlay');
    loadingOverlay.style.display = show ? 'flex' : 'none';
}

window.checkConnection = function() {
    fetch('/api/status')
    .then(response => response.json())
    .then(data => {
const statusIndicator = document.getElementById('status-indicator');
        const statusText = document.getElementById('status-text');
        if (statusIndicator && statusText) {
            if (data.connected) {
                statusIndicator.classList.add('connected');
                statusIndicator.classList.remove('disconnected');
                statusText.innerText = `Connected: ${data.message}`;
            } else {
                statusIndicator.classList.add('disconnected');
                statusIndicator.classList.remove('connected');
                statusText.innerText = `Disconnected: ${data.message}`;
            }
        }
    })
    .catch(error => {
        console.error('Error checking connection:', error);
    });
};

document.addEventListener('DOMContentLoaded', function() {
    window.checkConnection();
    window.setupChatImageUpload();
    showPersonaModalIfNeeded();
    if (window.initVoiceRecorder) window.initVoiceRecorder(); // Voice interaction (animated recorder)
    
    // Add event listeners for all buttons to replace inline HTML handlers
    setupEventListeners();
});

// -------------------------
// Persona modal logic
function showPersonaModalIfNeeded() {
    const modal = document.getElementById('persona-modal');
    let savedName = localStorage.getItem('user_name');
    let savedRole = localStorage.getItem('user_role');
    if (savedName && savedRole) {
        window.currentUserName = savedName;
        window.currentUserRole = savedRole;
        updatePersonalizedGreeting(savedName, savedRole);
        return;
    }
modal.style.display = 'flex';
    document.getElementById('persona-modal-title').innerText = "Hey there! 👋 What’s your name?";
    document.getElementById('persona-role-section').style.display = 'none';
    document.getElementById('persona-next-btn').onclick = async function() {
        const name = document.getElementById('user-name-input').value.trim();
        if (!name) { alert('Please enter your name!'); return; }
        // Try to fetch user data
        let resp = await fetch('/api/user-fetch?name='+encodeURIComponent(name));
        let data = await resp.json();
        if (data.exists) {
            window.currentUserName = name;
            window.currentUserRole = data.role || 'Friend';
            localStorage.setItem('user_name', name);
            localStorage.setItem('user_role', window.currentUserRole);
            modal.style.display = 'none';
            updatePersonalizedGreeting(name, window.currentUserRole);
        } else {
            // Show persona choices
            document.getElementById('persona-modal-title').innerText = 'How do you see the AI?';
            document.getElementById('persona-role-section').style.display = 'block';
            document.getElementById('persona-next-btn').style.display = 'none';
            document.querySelectorAll('.role-btn').forEach(btn => {
                btn.onclick = function() {
                    window.currentUserName = name;
                    window.currentUserRole = btn.dataset.role;
                    localStorage.setItem('user_name', name);
                    localStorage.setItem('user_role', btn.dataset.role);
                    // Save user
                    fetch('/api/user-create', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ name, role: btn.dataset.role })
                    });
                    modal.style.display = 'none';
                    updatePersonalizedGreeting(name, btn.dataset.role);
                };
            });
        }
    };
}

// Update greeting based on user profile
function updatePersonalizedGreeting(name, role) {
    const greetingElement = document.getElementById('initial-greeting');
    let greeting = '';
    
    switch(role) {
        case 'Best Friend':
            greeting = `Hey ${name}! 👋 Your bestie here! What's up? Ready to chat about anything and everything? 😊`;
            break;
        case 'Motivator':
            greeting = `Hello ${name}! 💪 I'm here to help you crush your goals today! What amazing things are we working on? 🚀`;
            break;
        case 'Female Friend':
            greeting = `Hi ${name}! 💫 I'm so happy to see you! How are you feeling today? I'm here to listen and chat about whatever's on your mind 💕`;
            break;
        case 'Friend':
            greeting = `Hello ${name}! 🙂 Great to see you again! What can I help you with today?`;
            break;
        case 'Guide':
            greeting = `Greetings ${name}! 🧠 I'm ready to guide you through any questions or challenges you have. What would you like to learn about today?`;
            break;
        default:
            greeting = `Hello ${name}! I'm your Gemma Vision Assistant. How can I help you today?`;
    }
    
    if (greetingElement) {
        greetingElement.innerHTML = `<p>${greeting}</p>`;
    }
    
    // Update voice greeting too
    const voiceGreeting = document.getElementById('voice-greeting');
    if (voiceGreeting) {
        voiceGreeting.innerHTML = `HELLO, <span class="voice-username">${name.toUpperCase()}!</span> <span class="wave">👋</span>`;
    }
}

// Check for role change commands
function checkForRoleChangeCommand(message) {
    const lowerMessage = message.toLowerCase().trim();
    
    // Commands that trigger role change
    const roleChangeCommands = [
        'change how i see you',
        'update my role',
        'change your role',
        'i want to see you as',
        'update how i see you',
        'change our relationship',
        'update our relationship'
    ];
    
    const isRoleChangeCommand = roleChangeCommands.some(cmd => 
        lowerMessage.includes(cmd)
    );
    
    if (isRoleChangeCommand) {
        showRoleUpdateModal();
        return true;
    }
    
    return false;
}

// Show role update modal
function showRoleUpdateModal() {
    const modal = document.getElementById('persona-modal');
    if (!modal) return;
    
    // Configure modal for role update
    document.getElementById('persona-modal-title').innerText = `Change How You See Me`;
    document.getElementById('user-name-input').style.display = 'none';
    document.getElementById('persona-role-section').style.display = 'block';
    document.getElementById('persona-next-btn').style.display = 'none';
    
    // Show current role info
    if (window.currentUserRole) {
        const currentRoleInfo = document.createElement('p');
        currentRoleInfo.id = 'current-role-info';
        currentRoleInfo.innerHTML = `Currently you see me as your <strong>${window.currentUserRole}</strong>. Choose a new role:`;
        currentRoleInfo.style.textAlign = 'center';
        currentRoleInfo.style.marginBottom = '15px';
        currentRoleInfo.style.color = '#666';
        
        const roleSection = document.getElementById('persona-role-section');
        const existingInfo = document.getElementById('current-role-info');
        if (existingInfo) existingInfo.remove();
        roleSection.insertBefore(currentRoleInfo, roleSection.firstChild);
    }
    
    // Setup role buttons for updating
    document.querySelectorAll('.role-btn').forEach(btn => {
        // Highlight current role
        if (btn.dataset.role === window.currentUserRole) {
            btn.style.backgroundColor = '#4CAF50';
            btn.style.color = 'white';
        } else {
            btn.style.backgroundColor = '';
            btn.style.color = '';
        }
        
        btn.onclick = function() {
            updateUserRole(btn.dataset.role);
            modal.style.display = 'none';
        };
    });
    
    modal.style.display = 'flex';
}

// Update user role
async function updateUserRole(newRole) {
    try {
        const response = await fetch('/api/user-update-role', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                name: window.currentUserName,
                role: newRole 
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            const oldRole = window.currentUserRole;
            window.currentUserRole = newRole;
            localStorage.setItem('user_role', newRole);
            
            // Update greeting
            updatePersonalizedGreeting(window.currentUserName, newRole);
            
            // Show confirmation message
            let confirmationMessage;
            switch(newRole) {
                case 'Best Friend':
                    confirmationMessage = `Awesome! 🎉 Now I'm your best friend! I'll be more casual and supportive. Let's have some fun chats! 😊`;
                    break;
                case 'Motivator':
                    confirmationMessage = `Perfect! 💪 I'm now your motivator! I'll be energetic and help you crush your goals! Let's get pumped! 🚀`;
                    break;
                case 'Female Friend':
                    confirmationMessage = `Sweet! 💕 I'm now your female friend! I'll be warm, caring, and understanding. Ready to chat about anything! ✨`;
                    break;
                case 'Friend':
                    confirmationMessage = `Great! 🙂 I'm your friendly AI now! I'll be helpful and kind. What can I help you with?`;
                    break;
                case 'Guide':
                    confirmationMessage = `Excellent! 🧠 I'm now your guide! I'll be knowledgeable and help you learn. What shall we explore together?`;
                    break;
                default:
                    confirmationMessage = `Role updated successfully! I'm now your ${newRole}.`;
            }
            
            appendMessage('assistant', confirmationMessage);
            
        } else {
            appendMessage('assistant', `Sorry, I couldn't update your role: ${data.error}`);
        }
    } catch (error) {
        console.error('Error updating role:', error);
        appendMessage('assistant', 'Sorry, there was an error updating your role. Please try again.');
    }
}

// Check for clear/reset commands
function checkForClearCommand(message) {
    const lowerMessage = message.toLowerCase().trim();
    
    // Commands that trigger chat clearing
    const clearCommands = [
        'clear this chat',
        'forget everything', 
        'clear conversation',
        'reset chat',
        'start fresh',
        'clear memory',
        'forget what we said',
        'new conversation'
    ];
    
    return clearCommands.some(cmd => lowerMessage.includes(cmd));
}

// Clear current chat session
function clearCurrentChat() {
    // Make API call to clear server-side session
    fetch('/api/chat', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ 
            prompt: 'clear this chat',
            username: window.currentUserName || '',
            role: window.currentUserRole || ''
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success && data.session_cleared) {
            // Clear the chat UI
            const chatMessages = document.getElementById('chat-messages');
            if (chatMessages) {
                // Keep the initial greeting but clear other messages
                const initialGreeting = chatMessages.querySelector('.assistant-message');
                chatMessages.innerHTML = '';
                if (initialGreeting) {
                    chatMessages.appendChild(initialGreeting);
                }
            }
            
            // Show confirmation message
            appendMessage('assistant', data.response);
        }
    })
    .catch(error => {
        console.error('Error clearing chat:', error);
        appendMessage('assistant', 'Sorry, there was an error clearing the conversation.');
    });
}

// Add session management information display
function showSessionInfo() {
    appendMessage('assistant', '💡 **Session Memory Info:**\n\n' +
        '• I only remember our current conversation\n' +
        '• I don\'t have access to past chats or sessions\n' +
        '• Say "clear this chat" to start fresh\n' +
        '• Say "change how I see you" to update your role');
}

// Add command to show session info
// Clear chat function called from HTML
function clearChat() {
    clearCurrentChat();
}

// Setup event listeners for all interactive elements
function setupEventListeners() {
    // Delete chat button
    const deleteChatBtn = document.getElementById('delete-chat-btn');
    if (deleteChatBtn) {
        deleteChatBtn.addEventListener('click', clearChat);
    }
    
    // Chat input keypress (Enter to send)
    const chatInput = document.getElementById('chat-input');
    if (chatInput) {
        chatInput.addEventListener('keypress', window.handleChatKeyPress);
    }
    
    // Microphone button
    const micBtn = document.getElementById('mic-btn');
    if (micBtn) {
        micBtn.addEventListener('click', startVoiceRecognition);
    }
    
    // Send message button
    const sendBtn = document.getElementById('send-btn');
    if (sendBtn) {
        sendBtn.addEventListener('click', window.sendMessage);
    }
    
    // Gemma toggle button
    const gemmaToggleBtn = document.getElementById('gemma-toggle-btn');
    if (gemmaToggleBtn) {
        gemmaToggleBtn.addEventListener('click', toggleGemmaReasoning);
    }
    
    // Enhanced Vision buttons
    const visionUploadBtn = document.getElementById('vision-upload-btn');
    if (visionUploadBtn) {
        visionUploadBtn.addEventListener('click', triggerVisionUpload);
    }
    
    const analyzeVisionBtn = document.getElementById('analyze-vision-btn');
    if (analyzeVisionBtn) {
        analyzeVisionBtn.addEventListener('click', analyzeVisionImage);
    }
    
    const removeVisionBtn = document.getElementById('remove-vision-btn');
    if (removeVisionBtn) {
        removeVisionBtn.addEventListener('click', removeVisionImage);
    }
    
    const speakResultsBtn = document.getElementById('speak-results-btn');
    if (speakResultsBtn) {
        speakResultsBtn.addEventListener('click', speakResults);
    }
    
    const copyResultsBtn = document.getElementById('copy-results-btn');
    if (copyResultsBtn) {
        copyResultsBtn.addEventListener('click', copyResults);
    }
    
    const analyzeAgainBtn = document.getElementById('analyze-again-btn');
    if (analyzeAgainBtn) {
        analyzeAgainBtn.addEventListener('click', analyzeAgain);
    }
}

// Enhanced Vision functions - these need to be defined or imported
function toggleGemmaReasoning() {
    // This function should be defined based on your gemma reasoning functionality
    if (window.toggleGemmaReasoning) {
        window.toggleGemmaReasoning();
    } else {
        console.log('Gemma reasoning toggle functionality not yet implemented');
    }
}

function triggerVisionUpload() {
    // This function should trigger the vision upload functionality
    if (window.triggerVisionUpload) {
        window.triggerVisionUpload();
    } else {
        console.log('Vision upload functionality not yet implemented');
    }
}

function analyzeVisionImage() {
    // This function should analyze the uploaded vision image
    if (window.analyzeVisionImage) {
        window.analyzeVisionImage();
    } else {
        console.log('Vision image analysis functionality not yet implemented');
    }
}

function removeVisionImage() {
    // This function should remove the uploaded vision image
    if (window.removeVisionImage) {
        window.removeVisionImage();
    } else {
        console.log('Remove vision image functionality not yet implemented');
    }
}

function speakResults() {
    // This function should speak the results using text-to-speech
    if (window.speakResults) {
        window.speakResults();
    } else {
        console.log('Speak results functionality not yet implemented');
    }
}

function copyResults() {
    // This function should copy results to clipboard
    if (window.copyResults) {
        window.copyResults();
    } else {
        console.log('Copy results functionality not yet implemented');
    }
}

function analyzeAgain() {
    // This function should trigger re-analysis
    if (window.analyzeAgain) {
        window.analyzeAgain();
    } else {
        console.log('Analyze again functionality not yet implemented');
    }
}

// Attach functions to window to allow HTML use - moved here after all functions are defined
function defineGlobal() {
  window.startVoiceRecognition = startVoiceRecognition;
  window.speakText = speakText;
  window.switchTab = switchTab;
  window.clearChat = clearChat;
  window.toggleGemmaReasoning = toggleGemmaReasoning;
  window.triggerVisionUpload = triggerVisionUpload;
  window.analyzeVisionImage = analyzeVisionImage;
  window.removeVisionImage = removeVisionImage;
  window.speakResults = speakResults;
  window.copyResults = copyResults;
  window.analyzeAgain = analyzeAgain;
}

// Call defineGlobal after all functions are defined
defineGlobal();

window.addEventListener('DOMContentLoaded', function() {
    // Add session info command detection to sendMessage
    const originalSendMessage = window.sendMessage;
    window.sendMessage = function() {
        const chatInput = document.getElementById('chat-input');
        const userMessage = chatInput.value.trim().toLowerCase();
        
        // Check for session info command
        if (userMessage === 'session info' || userMessage === 'memory info' || userMessage === 'help') {
            showSessionInfo();
            chatInput.value = '';
            return;
        }
        
        // Call original function
        originalSendMessage();
    };
    
    // Add event listeners for tab switching
    const tabContainer = document.querySelector('.tabs');
    if (tabContainer) {
        tabContainer.addEventListener('click', function(event) {
            if (event.target.classList.contains('tab-button')) {
                const tabName = event.target.getAttribute('data-tab');
                if (tabName && window.switchTab) {
                    window.switchTab(tabName);
                }
            }
        });
    }
});
