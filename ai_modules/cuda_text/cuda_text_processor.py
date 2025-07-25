"""
CUDA-Accelerated Text Processing
===============================
GPU-accelerated text generation, embeddings, and language processing.
"""

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import logging
import time
from typing import Dict, Any, List, Optional, Tuple, Union
import requests
import json
from transformers import AutoTokenizer, AutoModel, pipeline
from sentence_transformers import SentenceTransformer

from ..cuda_core import get_cuda_manager

logger = logging.getLogger(__name__)

class CudaTextProcessor:
    """
    CUDA-accelerated text processing for all language tasks.
    Includes embeddings, similarity, sentiment analysis, and more.
    """
    
    def __init__(self, device: str = "auto"):
        """
        Initialize the CUDA text processor.
        
        Args:
            device: Device to use ('auto', 'cuda', 'cpu')
        """
        self.cuda_manager = get_cuda_manager()
        self.device = self.cuda_manager.get_optimal_device()
        
        # Models
        self.models = {}
        self.tokenizers = {}
        self.pipelines = {}
        
        # Performance tracking
        self.stats = {
            'embeddings_generated': 0,
            'texts_processed': 0,
            'average_processing_time': 0,
            'total_tokens_processed': 0
        }
        
        logger.info(f"CudaTextProcessor initialized on {self.device}")
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize text processing models on GPU."""
        logger.info("🔄 Loading text processing models...")
        
        try:
            # 1. Sentence embeddings model
            self.models['embeddings'] = SentenceTransformer('all-MiniLM-L6-v2')
            if self.cuda_manager.cuda_available:
                self.models['embeddings'] = self.models['embeddings'].to(self.device)
            logger.info("✅ Sentence embeddings model loaded")
            
        except Exception as e:
            logger.warning(f"⚠️  Failed to load embeddings model: {e}")
        
        try:
            # 2. Sentiment analysis pipeline
            self.pipelines['sentiment'] = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=0 if self.cuda_manager.cuda_available else -1
            )
            logger.info("✅ Sentiment analysis pipeline loaded")
            
        except Exception as e:
            logger.warning(f"⚠️  Failed to load sentiment model: {e}")
        
        try:
            # 3. Text summarization pipeline
            self.pipelines['summarization'] = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                device=0 if self.cuda_manager.cuda_available else -1
            )
            logger.info("✅ Text summarization pipeline loaded")
            
        except Exception as e:
            logger.warning(f"⚠️  Failed to load summarization model: {e}")
        
        try:
            # 4. Question answering pipeline
            self.pipelines['qa'] = pipeline(
                "question-answering",
                model="distilbert-base-cased-distilled-squad",
                device=0 if self.cuda_manager.cuda_available else -1
            )
            logger.info("✅ Question answering pipeline loaded")
            
        except Exception as e:
            logger.warning(f"⚠️  Failed to load QA model: {e}")
        
        # Clear cache after loading
        if self.cuda_manager.cuda_available:
            self.cuda_manager.clear_cache()
    
    @get_cuda_manager().profile_operation("text_embedding")
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
                
        except Exception as e:
            logger.error(f"❌ Embedding generation failed: {e}")
            raise
    
    @get_cuda_manager().profile_operation("text_similarity")
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
    
    @get_cuda_manager().profile_operation("semantic_search")
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
    
    @get_cuda_manager().profile_operation("sentiment_analysis")
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
    
    @get_cuda_manager().profile_operation("text_summarization")
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
    
    @get_cuda_manager().profile_operation("question_answering")
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
