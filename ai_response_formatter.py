import re
import logging
from typing import Dict, Tuple, List

logger = logging.getLogger(__name__)

class AIResponseFormatter:
    """
    Detects and formats AI responses containing asterisk-based emphasis and actions.
    Handles cases where AI might respond with asterisks indicating emphasis, actions, or emotions.
    """
    
    def __init__(self):
        # Pattern to detect asterisk formatting (various types)
        self.asterisk_patterns = {
            'bold_italic': re.compile(r'\*\*\*(.+?)\*\*\*'),  # ***text***
            'bold': re.compile(r'\*\*(.+?)\*\*'),            # **text**
            'italic': re.compile(r'\*([^*]+?)\*'),           # *text*
            'action': re.compile(r'\*([^*]*?)\*'),           # *action* (same as italic but for actions)
            'emoji_text': re.compile(r'\*(.+?)\* (\S+)'),    # *text* emoji
        }
        
        # Common patterns that indicate AI is using asterisks for emphasis/actions
        self.ai_asterisk_indicators = [
            'Hello there! 😊',
            'I\'m here for you',
            'looking forward to',
            'glad you reached out',
            'listening ear',
            'warm presence',
            'without judgment',
            'all ears',
            'supportive and caring'
        ]
    
    def detect_asterisk_formatting(self, text: str) -> Dict[str, any]:
        """
        Detect if the text contains asterisk-based formatting typically used by AI
        
        Args:
            text (str): The AI response text to analyze
            
        Returns:
            Dict containing detection results and metadata
        """
        detection_result = {
            'has_asterisks': False,
            'asterisk_count': 0,
            'formatting_types': [],
            'detected_patterns': [],
            'is_ai_emphasis': False,
            'confidence_score': 0.0
        }
        
        # Count total asterisks
        asterisk_count = text.count('*')
        detection_result['asterisk_count'] = asterisk_count
        
        if asterisk_count == 0:
            return detection_result
        
        detection_result['has_asterisks'] = True
        
        # Check for each formatting pattern
        for pattern_name, pattern in self.asterisk_patterns.items():
            matches = pattern.findall(text)
            if matches:
                detection_result['formatting_types'].append(pattern_name)
                detection_result['detected_patterns'].extend(matches)
        
        # Check if this looks like AI emphasis/action formatting
        ai_indicators_found = sum(1 for indicator in self.ai_asterisk_indicators 
                                if indicator.lower() in text.lower())
        
        # Calculate confidence score
        confidence_factors = []
        
        # Factor 1: Presence of AI-typical phrases
        if ai_indicators_found > 0:
            confidence_factors.append(0.4)
        
        # Factor 2: Multiple formatting types (suggests intentional emphasis)
        if len(detection_result['formatting_types']) > 1:
            confidence_factors.append(0.3)
        
        # Factor 3: High asterisk density
        if asterisk_count > 4:
            confidence_factors.append(0.2)
        
        # Factor 4: Emojis present (common in AI emotional responses)
        emoji_pattern = re.compile(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002700-\U000027BF\U0001F900-\U0001F9FF\U0001F018-\U0001F270]')
        if emoji_pattern.search(text):
            confidence_factors.append(0.1)
        
        detection_result['confidence_score'] = sum(confidence_factors)
        detection_result['is_ai_emphasis'] = detection_result['confidence_score'] > 0.3
        
        return detection_result
    
    def format_asterisk_response(self, text: str, output_format: str = 'html') -> str:
        """
        Convert asterisk-formatted text to proper formatting
        
        Args:
            text (str): Text with asterisk formatting
            output_format (str): 'html', 'markdown', or 'plain'
            
        Returns:
            str: Formatted text
        """
        formatted_text = text
        
        if output_format == 'html':
            # Convert in order: bold-italic first, then bold, then italic
            formatted_text = self.asterisk_patterns['bold_italic'].sub(r'<strong><em>\1</em></strong>', formatted_text)
            formatted_text = self.asterisk_patterns['bold'].sub(r'<strong>\1</strong>', formatted_text)
            formatted_text = self.asterisk_patterns['italic'].sub(r'<em>\1</em>', formatted_text)
            
        elif output_format == 'markdown':
            # Markdown already uses asterisks, so we keep them but ensure proper spacing
            formatted_text = re.sub(r'\*\*\*(.+?)\*\*\*', r'***\1***', formatted_text)
            formatted_text = re.sub(r'\*\*(.+?)\*\*', r'**\1**', formatted_text)
            formatted_text = re.sub(r'\*([^*]+?)\*', r'*\1*', formatted_text)
            
        elif output_format == 'plain':
            # Remove asterisks but preserve the emphasized text
            formatted_text = self.asterisk_patterns['bold_italic'].sub(r'\1', formatted_text)
            formatted_text = self.asterisk_patterns['bold'].sub(r'\1', formatted_text)
            formatted_text = self.asterisk_patterns['italic'].sub(r'\1', formatted_text)
            
        return formatted_text
    
    def process_ai_response(self, response_text: str, format_output: bool = True) -> Dict[str, any]:
        """
        Complete processing of AI response - detect and format asterisk usage
        
        Args:
            response_text (str): The raw AI response
            format_output (bool): Whether to format the output
            
        Returns:
            Dict containing processed response and metadata
        """
        # Detect asterisk formatting
        detection_result = self.detect_asterisk_formatting(response_text)
        
        # Format if requested and asterisks detected
        formatted_text = response_text
        if format_output and detection_result['has_asterisks']:
            formatted_text = self.format_asterisk_response(response_text, 'html')
        
        # Create processing result
        result = {
            'original_text': response_text,
            'formatted_text': formatted_text,
            'detection_result': detection_result,
            'processing_notes': []
        }
        
        # Add processing notes
        if detection_result['is_ai_emphasis']:
            result['processing_notes'].append('AI emphasis/action formatting detected')
        
        if detection_result['asterisk_count'] > 0:
            result['processing_notes'].append(f'Found {detection_result["asterisk_count"]} asterisks')
        
        if len(detection_result['formatting_types']) > 0:
            result['processing_notes'].append(f'Formatting types: {", ".join(detection_result["formatting_types"])}')
        
        return result

# Example usage and testing
def test_asterisk_detection():
    """Test the asterisk detection system with sample AI responses"""
    formatter = AIResponseFormatter()
    
    test_responses = [
        "*Hello there! 😊 It's so lovely to hear from you.* ✨ I'm here for you, as a friend. *I'm really glad you reached out.* 🤗",
        
        "I'm designed to be a **listening ear** and a *warm presence*. ***Is there anything on your mind you'd like to talk about?***",
        
        "Regular response without any special formatting.",
        
        "*I'll do my best* to understand and respond in a way that feels **supportive and caring**. ❤️",
        
        "***I'm really looking forward to getting to know you better!*** 💖"
    ]
    
    print("🔍 Testing AI Response Asterisk Detection System")
    print("=" * 60)
    
    for i, response in enumerate(test_responses, 1):
        print(f"\n📝 Test {i}:")
        print(f"Input: {response}")
        
        result = formatter.process_ai_response(response)
        
        print(f"✨ Detection Result:")
        print(f"   - Has asterisks: {result['detection_result']['has_asterisks']}")
        print(f"   - Is AI emphasis: {result['detection_result']['is_ai_emphasis']}")
        print(f"   - Confidence: {result['detection_result']['confidence_score']:.2f}")
        print(f"   - Formatting types: {result['detection_result']['formatting_types']}")
        
        if result['formatted_text'] != result['original_text']:
            print(f"📄 Formatted: {result['formatted_text']}")
        
        if result['processing_notes']:
            print(f"📋 Notes: {', '.join(result['processing_notes'])}")
        
        print("-" * 40)

if __name__ == "__main__":
    test_asterisk_detection()
