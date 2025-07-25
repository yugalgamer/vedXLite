#!/usr/bin/env python3
"""
Ollama Optimization and Performance Checker
===========================================
This script helps optimize Ollama performance and provides model recommendations.
"""

import requests
import json
import time
import subprocess
import sys
from typing import List, Dict, Any, Tuple

class OllamaOptimizer:
    def __init__(self, ollama_url: str = "http://localhost:11434"):
        self.ollama_url = ollama_url
        self.api_endpoint = f"{ollama_url}/api"
        
    def check_ollama_status(self) -> Tuple[bool, str]:
        """Check if Ollama is running and responsive."""
        try:
            response = requests.get(f"{self.api_endpoint}/tags", timeout=10)
            if response.status_code == 200:
                models = response.json().get('models', [])
                return True, f"Ollama running with {len(models)} models"
            else:
                return False, f"Ollama responded with status {response.status_code}"
        except requests.RequestException as e:
            return False, f"Ollama not accessible: {str(e)}"
    
    def list_available_models(self) -> List[Dict[str, Any]]:
        """List all available models with their sizes."""
        try:
            response = requests.get(f"{self.api_endpoint}/tags", timeout=10)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_info = []
                for model in models:
                    size_gb = model.get('size', 0) / (1024**3)  # Convert to GB
                    model_info.append({
                        'name': model.get('name', 'Unknown'),
                        'size_gb': round(size_gb, 2),
                        'modified': model.get('modified_at', 'Unknown')
                    })
                return sorted(model_info, key=lambda x: x['size_gb'])
            return []
        except Exception as e:
            print(f"Error listing models: {e}")
            return []
    
    def test_model_performance(self, model_name: str, prompt: str = "Hello, how are you?") -> Dict[str, Any]:
        """Test model performance with a simple prompt."""
        print(f"Testing {model_name} performance...")
        
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 100  # Short response for testing
            }
        }
        
        start_time = time.time()
        try:
            response = requests.post(
                f"{self.api_endpoint}/generate",
                json=payload,
                timeout=120  # 2 minute timeout for testing
            )
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get('response', '')
                return {
                    'success': True,
                    'response_time': round(end_time - start_time, 2),
                    'response_length': len(response_text),
                    'response_preview': response_text[:100] + '...' if len(response_text) > 100 else response_text
                }
            else:
                return {
                    'success': False,
                    'error': f"HTTP {response.status_code}: {response.text}",
                    'response_time': round(end_time - start_time, 2)
                }
        except requests.Timeout:
            return {
                'success': False,
                'error': "Request timed out (2 minutes)",
                'response_time': 120
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'response_time': round(time.time() - start_time, 2)
            }
    
    def recommend_best_model(self) -> Dict[str, Any]:
        """Test all models and recommend the best one for vision processing."""
        models = self.list_available_models()
        if not models:
            return {'error': 'No models available'}
        
        print(f"\nTesting {len(models)} models for performance...")
        test_prompt = "You are a helpful AI assistant. Describe what you see in this scene: A table with objects on it."
        
        results = []
        for model in models:
            model_name = model['name']
            performance = self.test_model_performance(model_name, test_prompt)
            
            # Calculate score based on success, speed, and size
            if performance['success']:
                # Prefer faster models, but not too small
                speed_score = max(0, 60 - performance['response_time'])  # Prefer < 60s
                size_score = min(20, model['size_gb'])  # Prefer reasonable size
                total_score = speed_score + (size_score / 2)
            else:
                total_score = 0
            
            results.append({
                'model': model_name,
                'size_gb': model['size_gb'],
                'performance': performance,
                'score': total_score
            })
        
        # Sort by score (highest first)
        results.sort(key=lambda x: x['score'], reverse=True)
        
        return {
            'recommended': results[0] if results else None,
            'all_results': results
        }
    
    def install_lightweight_model(self) -> bool:
        """Install a lightweight model for better performance."""
        lightweight_models = [
            "gemma:2b",      # 2 billion parameters
            "llama3.2:1b",   # 1 billion parameters  
            "llama3.2:3b",   # 3 billion parameters
        ]
        
        print("Installing lightweight model for better performance...")
        
        for model in lightweight_models:
            try:
                print(f"Trying to install {model}...")
                result = subprocess.run(
                    ["ollama", "pull", model],
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minute timeout
                )
                
                if result.returncode == 0:
                    print(f"✅ Successfully installed {model}")
                    return True
                else:
                    print(f"❌ Failed to install {model}: {result.stderr}")
            except subprocess.TimeoutExpired:
                print(f"⏰ Timeout installing {model}")
            except Exception as e:
                print(f"❌ Error installing {model}: {e}")
        
        print("❌ Failed to install any lightweight models")
        return False

def main():
    print("🔧 Ollama Optimization and Performance Checker")
    print("=" * 50)
    
    optimizer = OllamaOptimizer()
    
    # Check Ollama status
    is_running, status_msg = optimizer.check_ollama_status()
    print(f"📡 Ollama Status: {status_msg}")
    
    if not is_running:
        print("❌ Ollama is not running. Please start Ollama and try again.")
        print("   Run: ollama serve")
        return
    
    # List available models
    models = optimizer.list_available_models()
    if not models:
        print("❌ No models found. Installing a lightweight model...")
        if optimizer.install_lightweight_model():
            models = optimizer.list_available_models()
    
    if models:
        print(f"\n📋 Available Models ({len(models)}):")
        for model in models:
            print(f"   • {model['name']} ({model['size_gb']} GB)")
    
    # Test performance and get recommendations
    if len(models) > 0:
        print("\n🧪 Running Performance Tests...")
        recommendation = optimizer.recommend_best_model()
        
        if 'error' in recommendation:
            print(f"❌ Error: {recommendation['error']}")
            return
        
        best_model = recommendation['recommended']
        if best_model and best_model['performance']['success']:
            print(f"\n🏆 Recommended Model: {best_model['model']}")
            print(f"   • Size: {best_model['size_gb']} GB")
            print(f"   • Response Time: {best_model['performance']['response_time']}s")
            print(f"   • Score: {best_model['score']:.1f}")
            
            # Show configuration recommendation
            print(f"\n⚙️  Configuration Recommendation:")
            print(f"   Set GEMMA_MODEL_NAME='{best_model['model']}'")
            if best_model['performance']['response_time'] > 30:
                print(f"   Set GEMMA_TIMEOUT=120  # Increase timeout")
            
            # Create a simple config update
            config_update = f"""
# Add to your environment or config:
export GEMMA_MODEL_NAME='{best_model['model']}'
export GEMMA_TIMEOUT={max(60, int(best_model['performance']['response_time']) + 30)}
export GEMMA_MAX_RETRIES=2
"""
            
            with open('ollama_config_recommendation.txt', 'w') as f:
                f.write(config_update)
            print(f"   📄 Config saved to: ollama_config_recommendation.txt")
        
        else:
            print("\n❌ No models performed successfully")
            print("   Consider installing a lighter model:")
            print("   • ollama pull gemma:2b")
            print("   • ollama pull llama3.2:1b")
    
    # Performance summary
    print(f"\n📊 Performance Summary:")
    all_results = recommendation.get('all_results', [])
    for result in all_results[:3]:  # Show top 3
        status = "✅" if result['performance']['success'] else "❌"
        print(f"   {status} {result['model']}: {result['performance']['response_time']}s")
    
    print(f"\n✨ Optimization complete!")
    print(f"   • Restart your application to use the optimized settings")
    print(f"   • Monitor performance through the /api/system-status endpoint")

if __name__ == "__main__":
    main()
