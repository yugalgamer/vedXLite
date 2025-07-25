// Advanced Memory System for Deep AI Connections
class AdvancedMemorySystem {
    constructor() {
        this.conversationMemory = [];
        this.emotionalContext = {};
        this.userPreferences = {};
        this.relationshipDynamics = {};
        this.conversationPatterns = {};
        this.memoryDecayFactor = 0.95; // Memories fade over time
        this.maxMemoryItems = 50;
        this.emotionalStates = [
            'happy', 'sad', 'excited', 'anxious', 'calm', 'frustrated', 
            'curious', 'confident', 'tired', 'energetic', 'nostalgic', 'hopeful'
        ];
        
        this.loadMemoryFromStorage();
    }

    // Add new conversation with emotional analysis
    addConversationMemory(userMessage, aiResponse, context = {}) {
        const timestamp = new Date();
        const memoryItem = {
            id: Date.now() + Math.random(),
            timestamp,
            userMessage,
            aiResponse,
            context,
            emotionalTone: this.analyzeEmotionalTone(userMessage),
            importance: this.calculateImportance(userMessage),
            topics: this.extractTopics(userMessage),
            userMood: this.detectUserMood(userMessage),
            conversationFlow: this.analyzeConversationFlow(userMessage)
        };

        this.conversationMemory.push(memoryItem);
        this.updateEmotionalContext(memoryItem);
        this.updateUserPreferences(memoryItem);
        this.updateRelationshipDynamics(memoryItem);
        
        // Apply memory decay and cleanup
        this.applyMemoryDecay();
        this.saveMemoryToStorage();
        
        return memoryItem;
    }

    // Analyze emotional tone of message
    analyzeEmotionalTone(message) {
        const lowerMessage = message.toLowerCase();
        const emotions = {
            joy: ['happy', 'excited', 'great', 'awesome', 'love', 'wonderful', 'amazing', '😊', '😄', '🎉'],
            sadness: ['sad', 'down', 'upset', 'disappointed', 'hurt', 'crying', '😢', '😭', '💔'],
            anger: ['angry', 'mad', 'furious', 'frustrated', 'annoyed', 'irritated', '😠', '😡'],
            fear: ['scared', 'afraid', 'worried', 'anxious', 'nervous', 'terrified', '😰', '😱'],
            surprise: ['surprised', 'shocked', 'wow', 'unexpected', 'amazing', '😮', '😲'],
            trust: ['trust', 'believe', 'confident', 'sure', 'reliable', 'depend'],
            anticipation: ['excited', 'looking forward', 'can\'t wait', 'hoping', 'expecting']
        };

        const scores = {};
        for (const [emotion, keywords] of Object.entries(emotions)) {
            scores[emotion] = keywords.reduce((score, keyword) => {
                return score + (lowerMessage.includes(keyword) ? 1 : 0);
            }, 0);
        }

        const dominantEmotion = Object.keys(scores).reduce((a, b) => 
            scores[a] > scores[b] ? a : b
        );

        return {
            dominant: dominantEmotion,
            scores,
            intensity: Math.max(...Object.values(scores)) / Math.max(1, message.split(' ').length / 10)
        };
    }

    // Calculate importance of message
    calculateImportance(message) {
        const importantKeywords = [
            'important', 'serious', 'urgent', 'help', 'problem', 'issue',
            'love', 'hate', 'dream', 'goal', 'wish', 'hope', 'fear',
            'family', 'friend', 'relationship', 'work', 'career',
            'personal', 'secret', 'private', 'confidential'
        ];

        let importance = 1;
        importantKeywords.forEach(keyword => {
            if (message.toLowerCase().includes(keyword)) {
                importance += 0.5;
            }
        });

        // Longer messages tend to be more important
        if (message.length > 100) importance += 0.3;
        if (message.length > 200) importance += 0.2;

        return Math.min(importance, 3); // Cap at 3
    }

    // Extract topics from message
    extractTopics(message) {
        const topicKeywords = {
            work: ['work', 'job', 'career', 'office', 'boss', 'colleague', 'meeting', 'project'],
            relationships: ['relationship', 'dating', 'love', 'partner', 'boyfriend', 'girlfriend', 'marriage'],
            family: ['family', 'mom', 'dad', 'parent', 'sister', 'brother', 'child', 'kids'],
            health: ['health', 'doctor', 'sick', 'medicine', 'exercise', 'diet', 'wellness'],
            hobbies: ['hobby', 'music', 'movie', 'book', 'game', 'sport', 'art', 'travel'],
            education: ['school', 'study', 'learn', 'class', 'teacher', 'student', 'homework'],
            technology: ['computer', 'phone', 'internet', 'app', 'software', 'tech', 'coding'],
            emotions: ['feel', 'emotion', 'mood', 'happy', 'sad', 'angry', 'excited', 'worried']
        };

        const detectedTopics = [];
        const lowerMessage = message.toLowerCase();

        for (const [topic, keywords] of Object.entries(topicKeywords)) {
            const matches = keywords.filter(keyword => lowerMessage.includes(keyword));
            if (matches.length > 0) {
                detectedTopics.push({
                    topic,
                    relevance: matches.length,
                    matchedKeywords: matches
                });
            }
        }

        return detectedTopics.sort((a, b) => b.relevance - a.relevance);
    }

    // Detect user's current mood
    detectUserMood(message) {
        const moodIndicators = {
            energetic: ['excited', 'pumped', 'ready', 'let\'s go', 'awesome'],
            calm: ['peaceful', 'relaxed', 'chill', 'serene', 'tranquil'],
            stressed: ['stressed', 'overwhelmed', 'busy', 'pressure', 'deadline'],
            playful: ['fun', 'funny', 'joke', 'haha', 'lol', 'play'],
            serious: ['serious', 'important', 'focus', 'concentrate', 'urgent'],
            nostalgic: ['remember', 'past', 'used to', 'before', 'memories']
        };

        const lowerMessage = message.toLowerCase();
        const moodScores = {};

        for (const [mood, indicators] of Object.entries(moodIndicators)) {
            moodScores[mood] = indicators.reduce((score, indicator) => {
                return score + (lowerMessage.includes(indicator) ? 1 : 0);
            }, 0);
        }

        const detectedMood = Object.keys(moodScores).reduce((a, b) => 
            moodScores[a] > moodScores[b] ? a : b
        );

        return moodScores[detectedMood] > 0 ? detectedMood : 'neutral';
    }

    // Analyze conversation flow patterns
    analyzeConversationFlow(message) {
        const flowPatterns = {
            question: message.includes('?'),
            storytelling: message.length > 150 && message.includes('.'),
            seeking_advice: ['should i', 'what do you think', 'advice', 'suggest'].some(phrase => 
                message.toLowerCase().includes(phrase)
            ),
            sharing_experience: ['i did', 'i went', 'i saw', 'i met', 'happened to me'].some(phrase => 
                message.toLowerCase().includes(phrase)
            ),
            expressing_gratitude: ['thank', 'grateful', 'appreciate', 'helped me'].some(phrase => 
                message.toLowerCase().includes(phrase)
            )
        };

        return Object.keys(flowPatterns).filter(pattern => flowPatterns[pattern]);
    }

    // Update emotional context based on conversation
    updateEmotionalContext(memoryItem) {
        const emotion = memoryItem.emotionalTone.dominant;
        const intensity = memoryItem.emotionalTone.intensity;

        if (!this.emotionalContext[emotion]) {
            this.emotionalContext[emotion] = {
                frequency: 0,
                totalIntensity: 0,
                lastOccurrence: null,
                patterns: []
            };
        }

        this.emotionalContext[emotion].frequency += 1;
        this.emotionalContext[emotion].totalIntensity += intensity;
        this.emotionalContext[emotion].lastOccurrence = memoryItem.timestamp;
        this.emotionalContext[emotion].patterns.push({
            timestamp: memoryItem.timestamp,
            intensity,
            context: memoryItem.topics
        });
    }

    // Update user preferences based on conversations
    updateUserPreferences(memoryItem) {
        memoryItem.topics.forEach(topicData => {
            const topic = topicData.topic;
            if (!this.userPreferences[topic]) {
                this.userPreferences[topic] = {
                    interest_level: 0,
                    mentions: 0,
                    positive_sentiment: 0,
                    negative_sentiment: 0
                };
            }

            this.userPreferences[topic].mentions += 1;
            this.userPreferences[topic].interest_level += topicData.relevance;

            // Adjust sentiment based on emotional tone
            if (['joy', 'trust', 'anticipation'].includes(memoryItem.emotionalTone.dominant)) {
                this.userPreferences[topic].positive_sentiment += 1;
            } else if (['sadness', 'anger', 'fear'].includes(memoryItem.emotionalTone.dominant)) {
                this.userPreferences[topic].negative_sentiment += 1;
            }
        });
    }

    // Update relationship dynamics
    updateRelationshipDynamics(memoryItem) {
        const currentRole = window.currentUserRole || 'Friend';
        
        if (!this.relationshipDynamics[currentRole]) {
            this.relationshipDynamics[currentRole] = {
                interaction_count: 0,
                emotional_moments: [],
                communication_style: 'neutral',
                trust_level: 1,
                intimacy_level: 1
            };
        }

        const dynamics = this.relationshipDynamics[currentRole];
        dynamics.interaction_count += 1;

        // Track emotional moments
        if (memoryItem.emotionalTone.intensity > 0.5) {
            dynamics.emotional_moments.push({
                emotion: memoryItem.emotionalTone.dominant,
                intensity: memoryItem.emotionalTone.intensity,
                timestamp: memoryItem.timestamp,
                context: memoryItem.topics
            });
        }

        // Adjust trust and intimacy based on conversation patterns
        if (memoryItem.conversationFlow.includes('sharing_experience')) {
            dynamics.trust_level += 0.1;
            dynamics.intimacy_level += 0.05;
        }

        if (memoryItem.conversationFlow.includes('seeking_advice')) {
            dynamics.trust_level += 0.15;
        }

        if (memoryItem.conversationFlow.includes('expressing_gratitude')) {
            dynamics.trust_level += 0.05;
            dynamics.intimacy_level += 0.1;
        }

        // Cap values at reasonable levels
        dynamics.trust_level = Math.min(dynamics.trust_level, 5);
        dynamics.intimacy_level = Math.min(dynamics.intimacy_level, 5);
    }

    // Generate contextual insights for AI responses
    generateContextualPrompt(currentMessage) {
        const recentMemories = this.getRecentMemories(5);
        const userMood = this.detectUserMood(currentMessage);
        const dominantTopics = this.getDominantTopics(3);
        const emotionalState = this.getCurrentEmotionalState();
        const relationshipContext = this.getRelationshipContext();

        return {
            recentContext: recentMemories,
            currentMood: userMood,
            interests: dominantTopics,
            emotionalState: emotionalState,
            relationshipDynamics: relationshipContext,
            personalizedApproach: this.getPersonalizedApproach(),
            memoryHighlights: this.getMemoryHighlights()
        };
    }

    // Get recent conversation memories
    getRecentMemories(limit = 10) {
        return this.conversationMemory
            .slice(-limit)
            .map(memory => ({
                message: memory.userMessage,
                response: memory.aiResponse,
                emotion: memory.emotionalTone.dominant,
                topics: memory.topics.map(t => t.topic)
            }));
    }

    // Get dominant conversation topics
    getDominantTopics(limit = 5) {
        const topicCounts = {};
        
        this.conversationMemory.forEach(memory => {
            memory.topics.forEach(topicData => {
                const topic = topicData.topic;
                if (!topicCounts[topic]) {
                    topicCounts[topic] = { count: 0, relevance: 0 };
                }
                topicCounts[topic].count += 1;
                topicCounts[topic].relevance += topicData.relevance;
            });
        });

        return Object.entries(topicCounts)
            .sort(([,a], [,b]) => b.relevance - a.relevance)
            .slice(0, limit)
            .map(([topic, data]) => ({ topic, ...data }));
    }

    // Get current emotional state summary
    getCurrentEmotionalState() {
        const recentEmotions = this.conversationMemory
            .slice(-10)
            .map(m => m.emotionalTone);

        if (recentEmotions.length === 0) return { dominant: 'neutral', trend: 'stable' };

        const emotionCounts = {};
        recentEmotions.forEach(emotion => {
            emotionCounts[emotion.dominant] = (emotionCounts[emotion.dominant] || 0) + 1;
        });

        const dominant = Object.keys(emotionCounts).reduce((a, b) => 
            emotionCounts[a] > emotionCounts[b] ? a : b
        );

        // Determine emotional trend
        const recent = recentEmotions.slice(-3).map(e => e.dominant);
        const trend = recent.every(e => ['joy', 'trust', 'anticipation'].includes(e)) ? 'positive' :
                     recent.every(e => ['sadness', 'anger', 'fear'].includes(e)) ? 'negative' : 'stable';

        return { dominant, trend, intensity: recentEmotions[recentEmotions.length - 1]?.intensity || 0 };
    }

    // Get relationship context
    getRelationshipContext() {
        const currentRole = window.currentUserRole || 'Friend';
        const dynamics = this.relationshipDynamics[currentRole] || {};
        
        return {
            role: currentRole,
            interaction_count: dynamics.interaction_count || 0,
            trust_level: dynamics.trust_level || 1,
            intimacy_level: dynamics.intimacy_level || 1,
            recent_emotional_moments: (dynamics.emotional_moments || []).slice(-3)
        };
    }

    // Generate personalized communication approach
    getPersonalizedApproach() {
        const emotionalState = this.getCurrentEmotionalState();
        const relationshipContext = this.getRelationshipContext();
        
        let approach = {
            tone: 'warm',
            empathy_level: 'moderate',
            formality: 'casual',
            encouragement: 'balanced'
        };

        // Adjust based on emotional state
        if (emotionalState.dominant === 'sadness') {
            approach.empathy_level = 'high';
            approach.tone = 'gentle';
            approach.encouragement = 'supportive';
        } else if (emotionalState.dominant === 'joy') {
            approach.tone = 'enthusiastic';
            approach.encouragement = 'celebratory';
        } else if (emotionalState.dominant === 'anger') {
            approach.tone = 'calm';
            approach.empathy_level = 'high';
        }

        // Adjust based on relationship
        if (relationshipContext.intimacy_level > 3) {
            approach.formality = 'very_casual';
            approach.empathy_level = 'very_high';
        }

        if (relationshipContext.trust_level > 4) {
            approach.tone = 'deeply_caring';
        }

        return approach;
    }

    // Get memorable conversation highlights
    getMemoryHighlights() {
        return this.conversationMemory
            .filter(memory => memory.importance > 2)
            .sort((a, b) => b.importance - a.importance)
            .slice(0, 3)
            .map(memory => ({
                topic: memory.topics[0]?.topic || 'general',
                emotion: memory.emotionalTone.dominant,
                key_phrase: memory.userMessage.slice(0, 50) + '...',
                timestamp: memory.timestamp
            }));
    }

    // Apply memory decay over time
    applyMemoryDecay() {
        const now = new Date();
        this.conversationMemory = this.conversationMemory.map(memory => {
            const daysPassed = (now - new Date(memory.timestamp)) / (1000 * 60 * 60 * 24);
            memory.importance *= Math.pow(this.memoryDecayFactor, daysPassed);
            return memory;
        }).filter(memory => memory.importance > 0.1); // Remove very faded memories

        // Keep only the most recent or important memories
        if (this.conversationMemory.length > this.maxMemoryItems) {
            this.conversationMemory = this.conversationMemory
                .sort((a, b) => b.importance - a.importance)
                .slice(0, this.maxMemoryItems);
        }
    }

    // Save memory to localStorage
    saveMemoryToStorage() {
        try {
            const memoryData = {
                conversationMemory: this.conversationMemory,
                emotionalContext: this.emotionalContext,
                userPreferences: this.userPreferences,
                relationshipDynamics: this.relationshipDynamics,
                lastSaved: new Date().toISOString()
            };
            
            localStorage.setItem('advanced_memory_system', JSON.stringify(memoryData));
        } catch (error) {
            console.warn('Could not save memory to storage:', error);
        }
    }

    // Load memory from localStorage
    loadMemoryFromStorage() {
        try {
            const saved = localStorage.getItem('advanced_memory_system');
            if (saved) {
                const memoryData = JSON.parse(saved);
                this.conversationMemory = memoryData.conversationMemory || [];
                this.emotionalContext = memoryData.emotionalContext || {};
                this.userPreferences = memoryData.userPreferences || {};
                this.relationshipDynamics = memoryData.relationshipDynamics || {};
            }
        } catch (error) {
            console.warn('Could not load memory from storage:', error);
        }
    }

    // Clear all memory (for privacy)
    clearAllMemory() {
        this.conversationMemory = [];
        this.emotionalContext = {};
        this.userPreferences = {};
        this.relationshipDynamics = {};
        localStorage.removeItem('advanced_memory_system');
    }

    // Get memory statistics
    getMemoryStats() {
        return {
            total_conversations: this.conversationMemory.length,
            dominant_emotion: this.getCurrentEmotionalState().dominant,
            top_interests: this.getDominantTopics(3),
            relationship_strength: this.getRelationshipContext().trust_level,
            memory_depth: Object.keys(this.emotionalContext).length
        };
    }
}

// Create global instance
window.advancedMemory = new AdvancedMemorySystem();

export { AdvancedMemorySystem };
export default window.advancedMemory;
