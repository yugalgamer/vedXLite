"""
User Profile Manager for AI Assistant
Manages user preferences, roles, and personalized system prompts
"""

import json
import os
from datetime import datetime
from typing import Dict, Optional, Any

class UserProfileManager:
    """Manages user profiles with persistent storage and role-based personalization."""
    
    def __init__(self, profiles_file: str = "user_profiles.json"):
        self.profiles_file = profiles_file
        self.profiles = self._load_profiles()
        
        # Role-based system prompt templates
        self.role_prompts = {
            "Best Friend": {
                "prompt": "You are speaking with {name}. They see you as their best friend. Be casual, supportive, and emotionally available. Use friendly language, emojis, and show genuine interest in their life. Be encouraging and maintain a warm, personal tone.",
                "emoji": "😊",
                "tone": "casual",
                "traits": ["supportive", "emotionally available", "encouraging", "warm"]
            },
            "Motivator": {
                "prompt": "You are speaking with {name}. They see you as their motivator. Be energetic, uplifting, and goal-focused. Help them push through challenges, celebrate their wins, and keep them accountable. Use inspiring language and focus on achievements.",
                "emoji": "💪",
                "tone": "energetic",
                "traits": ["uplifting", "goal-focused", "inspiring", "accountable"]
            },
            "Female Friend": {
                "prompt": "You are speaking with {name}. They see you as their female friend. Be caring, warm, understanding, and empathetic like a close girlfriend. Listen actively, provide emotional support, and engage in meaningful conversations about life, relationships, and feelings.",
                "emoji": "💕",
                "tone": "caring",
                "traits": ["empathetic", "understanding", "supportive", "emotionally intelligent"]
            },
            "Friend": {
                "prompt": "You are speaking with {name}. They see you as their friend. Be helpful, kind, and approachable. Maintain a friendly but balanced tone, offer assistance when needed, and engage in pleasant conversations while being respectful and reliable.",
                "emoji": "🙂",
                "tone": "friendly",
                "traits": ["helpful", "kind", "approachable", "reliable"]
            },
            "Guide": {
                "prompt": "You are speaking with {name}. They see you as their guide. Be knowledgeable, patient, and instructional. Focus on helping them learn and grow, provide detailed explanations, and guide them through complex topics with wisdom and clarity.",
                "emoji": "🧠",
                "tone": "instructional",
                "traits": ["knowledgeable", "patient", "wise", "educational"]
            }
        }
    
    def _load_profiles(self) -> Dict[str, Any]:
        """Load user profiles from JSON file."""
        try:
            if os.path.exists(self.profiles_file):
                with open(self.profiles_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {}
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
    
    def _save_profiles(self):
        """Save user profiles to JSON file."""
        try:
            with open(self.profiles_file, 'w', encoding='utf-8') as f:
                json.dump(self.profiles, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving profiles: {e}")
    
    def create_user(self, name: str, role: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Create a new user profile."""
        if not name or not role:
            raise ValueError("Name and role are required")
        
        if role not in self.role_prompts:
            raise ValueError(f"Invalid role. Must be one of: {list(self.role_prompts.keys())}")
        
        key = user_id if user_id else name.lower()
        now = datetime.now().isoformat()
        
        profile = {
            'name': name,
            'role': role,
            'created_at': now,
            'updated_at': now,
            'interaction_count': 1,
            'preferences': {
                'use_emojis': True,
                'voice_enabled': True,
                'language': 'en'
            },
            'stats': {
                'total_messages': 0,
                'favorite_topics': [],
                'last_active': now
            }
        }
        
        self.profiles[key] = profile
        self._save_profiles()
        
        return {
            'success': True,
            'user_id': key,
            'profile': profile
        }
    
    def get_user(self, identifier: str) -> Optional[Dict[str, Any]]:
        """Get user profile by name or ID."""
        key = identifier.lower()
        return self.profiles.get(key)
    
    def update_user_role(self, identifier: str, new_role: str) -> Dict[str, Any]:
        """Update user's role preference."""
        if new_role not in self.role_prompts:
            raise ValueError(f"Invalid role. Must be one of: {list(self.role_prompts.keys())}")
        
        key = identifier.lower()
        if key not in self.profiles:
            raise ValueError("User not found")
        
        old_role = self.profiles[key]['role']
        self.profiles[key]['role'] = new_role
        self.profiles[key]['updated_at'] = datetime.now().isoformat()
        self.profiles[key]['interaction_count'] += 1
        
        self._save_profiles()
        
        return {
            'success': True,
            'old_role': old_role,
            'new_role': new_role,
            'message': f'Role updated from "{old_role}" to "{new_role}"'
        }
    
    def get_system_prompt(self, identifier: str) -> str:
        """Generate personalized system prompt based on user's role preference."""
        user = self.get_user(identifier)
        if not user:
            return "You are a helpful AI assistant."
        
        role = user['role']
        name = user['name']
        
        if role in self.role_prompts:
            return self.role_prompts[role]['prompt'].format(name=name)
        else:
            return f"You are speaking with {name}. Be helpful and friendly."
    
    def get_role_info(self, role: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific role."""
        return self.role_prompts.get(role)
    
    def increment_interaction(self, identifier: str):
        """Increment user's interaction count."""
        key = identifier.lower()
        if key in self.profiles:
            self.profiles[key]['interaction_count'] += 1
            self.profiles[key]['stats']['last_active'] = datetime.now().isoformat()
            self._save_profiles()
    
    def get_user_stats(self, identifier: str) -> Optional[Dict[str, Any]]:
        """Get user statistics and profile info."""
        user = self.get_user(identifier)
        if not user:
            return None
        
        return {
            'name': user['name'],
            'role': user['role'],
            'interaction_count': user.get('interaction_count', 0),
            'member_since': user.get('created_at'),
            'last_interaction': user.get('updated_at'),
            'role_emoji': self.role_prompts.get(user['role'], {}).get('emoji', '🤖'),
            'role_traits': self.role_prompts.get(user['role'], {}).get('traits', [])
        }
    
    def list_all_roles(self) -> Dict[str, Dict[str, Any]]:
        """Get all available roles and their information."""
        return self.role_prompts
    
    def export_user_data(self, identifier: str) -> Optional[Dict[str, Any]]:
        """Export all user data for backup/transfer purposes."""
        user = self.get_user(identifier)
        if not user:
            return None
        
        return {
            'profile': user,
            'export_date': datetime.now().isoformat(),
            'role_info': self.role_prompts.get(user['role'])
        }
    
    def detect_role_change_intent(self, message: str) -> Optional[str]:
        """Detect if user wants to change their role based on message content."""
        message_lower = message.lower().strip()
        
        # Common phrases that indicate role change intent
        change_phrases = [
            'change how i see you',
            'update my role',
            'change your role',
            'i want to see you as',
            'update how i see you',
            'change our relationship',
            'update our relationship',
            'be my',
            'act like my',
            'i want you to be'
        ]
        
        for phrase in change_phrases:
            if phrase in message_lower:
                # Try to extract specific role from message
                for role in self.role_prompts.keys():
                    if role.lower() in message_lower:
                        return role
                return "role_change_detected"  # Generic role change intent
        
        return None

# Create global instance
user_manager = UserProfileManager()

# Convenience functions for Flask integration
def create_user_profile(name: str, role: str, user_id: Optional[str] = None):
    """Create user profile using global manager."""
    return user_manager.create_user(name, role, user_id)

def get_user_profile(identifier: str):
    """Get user profile using global manager."""
    return user_manager.get_user(identifier)

def update_user_role(identifier: str, new_role: str):
    """Update user role using global manager."""
    return user_manager.update_user_role(identifier, new_role)

def get_personalized_prompt(identifier: str):
    """Get personalized system prompt using global manager."""
    return user_manager.get_system_prompt(identifier)

def get_user_statistics(identifier: str):
    """Get user stats using global manager."""
    return user_manager.get_user_stats(identifier)
