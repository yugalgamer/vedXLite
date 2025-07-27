// Entry point for app, loads features
import './request_queue.js'; // handles single request management
import './advanced_memory.js'; // handles advanced memory system
import './voice_settings.js'; // handles voice toggle and settings
import './feature_voice.js'; // handles speech-to-text and TTS
import './feature_ui.js'; // handles UI switching, chat, image etc.
import './voice_visualizer.js'; // handles beautiful audio frequency animation
import './enhanced_vision.js'; // handles enhanced Gemma3n vision processing
// Add future features here as imports

// Initialize VedXlite AI assistant
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 VedXlite AI Assistant Loading...');
    console.log('✅ Request Queue System Active');
    console.log('✅ Advanced Memory System Active');
    console.log('🔊 Voice Settings Manager Active (Voice OFF by default)');
    console.log('✅ Typing Animations Enabled');
    console.log('✅ Heart-to-Heart Connection Mode Ready');
    console.log('💡 Tip: Click the Voice toggle button in header to enable voice features');
});

