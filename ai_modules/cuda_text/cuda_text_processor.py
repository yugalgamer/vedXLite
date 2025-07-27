"""
CUDA-Accelerated Text Processing
===============================
GPU-accelerated text processing using Ollama for Gemma3n model.
"""

import torch
import torch.nn.functional as F
import numpy as np
import requests
import json
import logging
import time
import re
import math
from typing import Dict, Any, List, Optional, Tuple, Union
from collections import Counter

logger = logging.getLogger(__name__)

# Transformers imports with error handling
try:
    from transformers import AutoTokenizer, AutoModel, pipeline
    from sentence_transformers import SentenceTransformer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("⚠️  Transformers library not available. Some features will be limited.")

try:
    from ..cuda_core import get_cuda_manager
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False

class CudaTextProcessor:
    """
    CUDA-accelerated text processing using Ollama for Gemma3n model.
    Provides text analysis capabilities using only Ollama API.
    """
    
    def __init__(self, 
                 model_name: str = "gemma3n:latest",
                 ollama_url: str = "http://localhost:11434",
                 device: str = "auto"):
        """
        Initialize the CUDA text processor.
        
        Args:
            model_name: Ollama model to use for text processing
            ollama_url: URL of Ollama API server
            device: Device to use ('auto', 'cuda', 'cpu')
        """
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.api_endpoint = f"{ollama_url}/api/generate"
        self.chat_endpoint = f"{ollama_url}/api/chat"
        
        if CUDA_AVAILABLE:
            try:
                self.cuda_manager = get_cuda_manager()
                self.device = self.cuda_manager.get_optimal_device()
            except:
                self.cuda_manager = None
                self.device = torch.device("cpu")
        else:
            self.cuda_manager = None
            self.device = torch.device("cpu")
        
        # Initialize storage for models and pipelines
        self.models = {}
        self.pipelines = {}
        self.tokenizers = {}
        
        # Performance tracking
        self.stats = {
            'texts_processed': 0,
            'average_processing_time': 0,
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'embeddings_generated': 0
        }
        
        logger.info(f"CudaTextProcessor initialized with {self.model_name} on {self.device}")
        self._verify_ollama_connection()
        
        # Initialize models and pipelines
        self._initialize_models()
    
    def _verify_ollama_connection(self):
        """Verify Ollama connection and model availability."""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [model['name'] for model in models]
                if self.model_name in model_names:
                    logger.info(f"✅ Ollama connection verified, {self.model_name} available")
                else:
                    logger.warning(f"⚠️  Model {self.model_name} not found. Available: {model_names}")
            else:
                logger.warning(f"⚠️  Ollama server responded with status {response.status_code}")
        except Exception as e:
            logger.warning(f"⚠️  Could not verify Ollama connection: {e}")
    
    def _make_ollama_request(self, prompt: str, **kwargs) -> str:
        """Make request to Ollama API."""
        try:
            self.stats['total_requests'] += 1
            
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": kwargs.get('temperature', 0.7),
                    "top_p": kwargs.get('top_p', 0.9),
                    "num_predict": kwargs.get('max_tokens', 512)
                }
            }
            
            response = requests.post(
                self.api_endpoint,
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=120  # Increased timeout to match Gemma3n engine
            )
            
            if response.status_code == 200:
                result = response.json()
                self.stats['successful_requests'] += 1
                return result.get('response', '').strip()
            else:
                self.stats['failed_requests'] += 1
                logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                return ""
                
        except Exception as e:
            self.stats['failed_requests'] += 1
            logger.error(f"Error making Ollama request: {e}")
            return ""
    
    def _initialize_models(self):
        """Initialize text processing models on GPU."""
        logger.info("🔄 Loading text processing models...")
        
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("⚠️  Transformers not available - using fallback implementations")
            return
        
        try:
            # 1. Custom CUDA text embeddings using transformers directly
            if TRANSFORMERS_AVAILABLE:
                self.tokenizers['embeddings'] = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
                self.models['embeddings'] = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
                
                if self.cuda_manager and self.cuda_manager.cuda_available:
                    self.models['embeddings'] = self.cuda_manager.optimize_model_for_cuda(
                        self.models['embeddings'], 
                        use_half_precision=True
                    )
                logger.info("✅ Custom CUDA embeddings model loaded")
            
        except Exception as e:
            logger.warning(f"⚠️  Failed to load embeddings model: {e}")
            # Create a simple embedding layer as fallback
            try:
                self.models['embeddings'] = self._create_simple_embedding_model()
                logger.info("✅ Fallback embedding model created")
            except Exception as fallback_e:
                logger.error(f"❌ Failed to create fallback model: {fallback_e}")
        
        if TRANSFORMERS_AVAILABLE:
            try:
                # 2. Sentiment analysis pipeline
                self.pipelines['sentiment'] = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    device=0 if (self.cuda_manager and self.cuda_manager.cuda_available) else -1
                )
                logger.info("✅ Sentiment analysis pipeline loaded")
                
            except Exception as e:
                logger.warning(f"⚠️  Failed to load sentiment model: {e}")
            
            try:
                # 3. Text summarization pipeline
                self.pipelines['summarization'] = pipeline(
                    "summarization",
                    model="facebook/bart-large-cnn",
                    device=0 if (self.cuda_manager and self.cuda_manager.cuda_available) else -1
                )
                logger.info("✅ Text summarization pipeline loaded")
                
            except Exception as e:
                logger.warning(f"⚠️  Failed to load summarization model: {e}")
            
            try:
                # 4. Question answering pipeline
                self.pipelines['qa'] = pipeline(
                    "question-answering",
                    model="distilbert-base-cased-distilled-squad",
                    device=0 if (self.cuda_manager and self.cuda_manager.cuda_available) else -1
                )
                logger.info("✅ Question answering pipeline loaded")
                
            except Exception as e:
                logger.warning(f"⚠️  Failed to load QA model: {e}")
        
        # Clear cache after loading
        if self.cuda_manager and self.cuda_manager.cuda_available:
            self.cuda_manager.clear_cache()
    
    def _create_simple_embedding_model(self):
        """Create a simple embedding model fallback."""
        class SimpleEmbedding:
            def encode(self, texts, convert_to_tensor=False, device=None):
                # Simple hash-based embedding fallback
                if isinstance(texts, str):
                    texts = [texts]
                
                embeddings = []
                for text in texts:
                    # Create a simple vector based on text characteristics
                    vec = torch.zeros(384)  # Standard sentence transformer size
                    if text:
                        for i, char in enumerate(text[:384]):
                            vec[i % 384] += ord(char) / 1000.0
                    embeddings.append(vec)
                
                result = torch.stack(embeddings)
                if device and device != torch.device('cpu'):
                    result = result.to(device)
                return result
        
        return SimpleEmbedding()
    
    def _profile_if_available(self, operation_name: str):
        """Decorator helper that profiles operations if CUDA manager is available."""
        def decorator(func):
            def wrapper(*args, **kwargs):
                if self.cuda_manager and hasattr(self.cuda_manager, 'profile_operation'):
                    return self.cuda_manager.profile_operation(operation_name)(func)(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
            return wrapper
        return decorator
    
    def generate_embeddings(self, texts: Union[str, List[str]], 
                          normalize: bool = True) -> torch.Tensor:
        """
        Generate embeddings for text(s) using GPU acceleration.
        
        Args:
            texts: Text or list of texts to embed
            normalize: Whether to normalize embeddings
            
        Returns:
            Tensor of embeddings
        """
        if 'embeddings' not in self.models:
            raise ValueError("Embeddings model not available")
        
        start_time = time.time()
        
        try:
            if self.cuda_manager:
                with self.cuda_manager.cuda_context():
                    # Convert single text to list
                    if isinstance(texts, str):
                        texts = [texts]
                    
                    # Generate embeddings
                    embeddings = self.models['embeddings'].encode(
                        texts,
                        convert_to_tensor=True,
                        device=self.device
                    )
                    
                    # Normalize if requested
                    if normalize:
                        embeddings = F.normalize(embeddings, p=2, dim=1)
                    
                    # Update stats
                    self.stats['embeddings_generated'] += len(texts)
                    self.stats['texts_processed'] += len(texts)
                    
                    processing_time = time.time() - start_time
                    self._update_processing_time(processing_time)
                    
                    return embeddings
            else:
                # CPU fallback
                if isinstance(texts, str):
                    texts = [texts]
                
                embeddings = self.models['embeddings'].encode(
                    texts,
                    convert_to_tensor=True,
                    device=self.device
                )
                
                if normalize:
                    embeddings = F.normalize(embeddings, p=2, dim=1)
                
                self.stats['embeddings_generated'] += len(texts)
                self.stats['texts_processed'] += len(texts)
                
                processing_time = time.time() - start_time
                self._update_processing_time(processing_time)
                
                return embeddings
                
        except Exception as e:
            logger.error(f"❌ Embedding generation failed: {e}")
            raise
    
    def compute_similarity(self, text1: Union[str, torch.Tensor], 
                          text2: Union[str, torch.Tensor],
                          metric: str = "cosine") -> float:
        """
        Compute similarity between two texts or embeddings.
        
        Args:
            text1: First text or embedding
            text2: Second text or embedding
            metric: Similarity metric ('cosine', 'euclidean', 'dot')
            
        Returns:
            Similarity score
        """
        try:
            # Generate embeddings if needed
            if isinstance(text1, str):
                emb1 = self.generate_embeddings(text1)
            else:
                emb1 = text1
            
            if isinstance(text2, str):
                emb2 = self.generate_embeddings(text2)
            else:
                emb2 = text2
            
            # Ensure tensors are on the same device
            emb1 = emb1.to(self.device)
            emb2 = emb2.to(self.device)
            
            # Compute similarity
            if metric == "cosine":
                similarity = F.cosine_similarity(emb1, emb2, dim=-1)
            elif metric == "euclidean":
                similarity = -torch.norm(emb1 - emb2, dim=-1)  # Negative for similarity
            elif metric == "dot":
                similarity = torch.sum(emb1 * emb2, dim=-1)
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            return float(similarity.cpu().item())
            
        except Exception as e:
            logger.error(f"❌ Similarity computation failed: {e}")
            raise
    
    def semantic_search(self, query: str, documents: List[str], 
                       top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform semantic search using GPU-accelerated embeddings.
        
        Args:
            query: Search query
            documents: List of documents to search
            top_k: Number of top results to return
            
        Returns:
            List of search results with scores
        """
        try:
            # Generate embeddings
            query_emb = self.generate_embeddings(query)
            doc_embeddings = self.generate_embeddings(documents)
            
            # Compute similarities
            similarities = F.cosine_similarity(
                query_emb.unsqueeze(0), 
                doc_embeddings, 
                dim=-1
            )
            
            # Get top-k results
            top_scores, top_indices = torch.topk(similarities, min(top_k, len(documents)))
            
            results = []
            for score, idx in zip(top_scores.cpu().numpy(), top_indices.cpu().numpy()):
                results.append({
                    'document': documents[idx],
                    'score': float(score),
                    'index': int(idx)
                })
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Semantic search failed: {e}")
            raise
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of text using GPU acceleration.
        
        Args:
            text: Text to analyze
            
        Returns:
            Sentiment analysis results
        """
        if 'sentiment' not in self.pipelines:
            # Fallback to simple rule-based sentiment
            return self._simple_sentiment(text)
        
        try:
            result = self.pipelines['sentiment'](text)
            
            return {
                'label': result[0]['label'],
                'score': result[0]['score'],
                'confidence': result[0]['score'],
                'processing_device': str(self.device)
            }
            
        except Exception as e:
            logger.error(f"❌ Sentiment analysis failed: {e}")
            return self._simple_sentiment(text)
    
    def _simple_sentiment(self, text: str) -> Dict[str, Any]:
        """Simple rule-based sentiment analysis fallback."""
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'disappointing', 'sad']
        
        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count > neg_count:
            return {'label': 'POSITIVE', 'score': 0.7, 'confidence': 0.7}
        elif neg_count > pos_count:
            return {'label': 'NEGATIVE', 'score': 0.7, 'confidence': 0.7}
        else:
            return {'label': 'NEUTRAL', 'score': 0.5, 'confidence': 0.5}
    
    def summarize_text(self, text: str, max_length: int = 150, 
                      min_length: int = 30) -> str:
        """
        Summarize text using GPU acceleration.
        
        Args:
            text: Text to summarize
            max_length: Maximum summary length
            min_length: Minimum summary length
            
        Returns:
            Summarized text
        """
        if 'summarization' not in self.pipelines:
            # Fallback to simple truncation
            sentences = text.split('. ')
            return '. '.join(sentences[:3]) + '...' if len(sentences) > 3 else text
        
        try:
            # Truncate if text is too long
            if len(text) > 1000:
                text = text[:1000] + "..."
            
            result = self.pipelines['summarization'](
                text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False
            )
            
            return result[0]['summary_text']
            
        except Exception as e:
            logger.error(f"❌ Text summarization failed: {e}")
            return text[:200] + "..." if len(text) > 200 else text
    
    def answer_question(self, question: str, context: str) -> Dict[str, Any]:
        """
        Answer a question based on context using GPU acceleration.
        
        Args:
            question: Question to answer
            context: Context containing the answer
            
        Returns:
            Answer with confidence score
        """
        if 'qa' not in self.pipelines:
            return {
                'answer': "Question answering model not available",
                'score': 0.0,
                'start': 0,
                'end': 0
            }
        
        try:
            result = self.pipelines['qa'](question=question, context=context)
            
            return {
                'answer': result['answer'],
                'score': result['score'],
                'start': result['start'],
                'end': result['end'],
                'processing_device': str(self.device)
            }
            
        except Exception as e:
            logger.error(f"❌ Question answering failed: {e}")
            return {
                'answer': "Unable to process question",
                'score': 0.0,
                'start': 0,
                'end': 0
            }
    
    def extract_keywords(self, text: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Extract keywords from text using simple frequency analysis.
        
        Args:
            text: Text to extract keywords from
            top_k: Number of top keywords to return
            
        Returns:
            List of keywords with scores
        """
        import re
        from collections import Counter
        
        # Simple keyword extraction
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Remove common stop words
        stop_words = {'the', 'and', 'but', 'for', 'are', 'with', 'this', 'that', 'have', 'will'}
        words = [word for word in words if word not in stop_words]
        
        # Count frequencies
        word_counts = Counter(words)
        total_words = len(words)
        
        keywords = []
        for word, count in word_counts.most_common(top_k):
            keywords.append({
                'keyword': word,
                'frequency': count,
                'score': count / total_words
            })
        
        return keywords
    
    def batch_process_texts(self, texts: List[str], 
                          operations: List[str] = ['embeddings', 'sentiment'],
                          batch_size: int = 32) -> List[Dict[str, Any]]:
        """
        Process multiple texts in batches for efficiency.
        
        Args:
            texts: List of texts to process
            operations: Operations to perform on each text
            batch_size: Size of processing batches
            
        Returns:
            List of processing results
        """
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = []
            
            for text in batch:
                text_result = {'text': text}
                
                try:
                    if 'embeddings' in operations:
                        text_result['embedding'] = self.generate_embeddings(text)
                    
                    if 'sentiment' in operations:
                        text_result['sentiment'] = self.analyze_sentiment(text)
                    
                    if 'keywords' in operations:
                        text_result['keywords'] = self.extract_keywords(text)
                    
                    if 'summary' in operations and len(text) > 100:
                        text_result['summary'] = self.summarize_text(text)
                    
                except Exception as e:
                    logger.error(f"❌ Failed to process text: {e}")
                    text_result['error'] = str(e)
                
                batch_results.append(text_result)
            
            results.extend(batch_results)
            
            # Clear cache between batches
            if self.cuda_manager.cuda_available:
                self.cuda_manager.clear_cache()
        
        return results
    
    def _update_processing_time(self, processing_time: float):
        """Update average processing time statistics."""
        total_ops = self.stats['texts_processed']
        current_avg = self.stats['average_processing_time']
        
        if total_ops > 0:
            self.stats['average_processing_time'] = (
                (current_avg * (total_ops - 1) + processing_time) / total_ops
            )
        else:
            self.stats['average_processing_time'] = processing_time
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        stats = self.stats.copy()
        stats['device'] = str(self.device)
        stats['models_loaded'] = list(self.models.keys()) + list(self.pipelines.keys())
        stats['cuda_available'] = self.cuda_manager.cuda_available
        
        if self.cuda_manager.cuda_available:
            memory_info = self.cuda_manager.get_memory_info()
            stats['memory_usage'] = memory_info.__dict__
        
        return stats
    
    def clear_cache(self):
        """Clear GPU cache and reset memory."""
        self.cuda_manager.clear_cache()
    
    def benchmark(self, sample_texts: List[str] = None) -> Dict[str, float]:
        """Benchmark text processing performance."""
        if not sample_texts:
            sample_texts = [
                "This is a sample text for benchmarking.",
                "GPU acceleration makes text processing much faster.",
                "CUDA enables parallel processing of multiple texts simultaneously."
            ]
        
        results = {}
        
        # Benchmark embeddings
        start_time = time.time()
        for text in sample_texts:
            self.generate_embeddings(text)
        results['embeddings_per_second'] = len(sample_texts) / (time.time() - start_time)
        
        # Benchmark sentiment analysis
        start_time = time.time()
        for text in sample_texts:
            self.analyze_sentiment(text)
        results['sentiment_per_second'] = len(sample_texts) / (time.time() - start_time)
        
        # Benchmark similarity
        start_time = time.time()
        for i in range(len(sample_texts) - 1):
            self.compute_similarity(sample_texts[i], sample_texts[i + 1])
        results['similarity_per_second'] = (len(sample_texts) - 1) / (time.time() - start_time)
        
        return results

# Global instance for easy access
_cuda_text_processor = None

def get_cuda_text_processor(device: str = "auto") -> CudaTextProcessor:
    """Get or create global CUDA text processor instance."""
    global _cuda_text_processor
    if _cuda_text_processor is None:
        _cuda_text_processor = CudaTextProcessor(device=device)
    return _cuda_text_processor

# Test the processor if run directly
if __name__ == "__main__":
    print("🧪 Testing CUDA Text Processor...")
    
    processor = CudaTextProcessor()
    
    # Test embeddings
    text = "Hello, this is a test for GPU-accelerated text processing!"
    embedding = processor.generate_embeddings(text)
    print(f"📊 Generated embedding shape: {embedding.shape}")
    
    # Test sentiment
    sentiment = processor.analyze_sentiment(text)
    print(f"😊 Sentiment: {sentiment}")
    
    # Test similarity
    text2 = "Hi, this is another test for GPU text processing!"
    similarity = processor.compute_similarity(text, text2)
    print(f"🔗 Similarity: {similarity:.3f}")
    
    # Get stats
    stats = processor.get_stats()
    print(f"📈 Stats: {stats}")
    
    print("✅ CUDA Text Processor test complete!")
