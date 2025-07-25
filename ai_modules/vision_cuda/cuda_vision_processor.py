"""
CUDA-Accelerated Vision Processor
=================================
GPU-accelerated vision processing using PyTorch and transformers.
Provides fast image analysis for blind users with safety-focused descriptions.
"""

import torch
import torch.nn.functional as F
from torch import nn
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
import numpy as np
import logging
import time
from typing import Dict, Any, List, Optional, Tuple
import json
import cv2

# Try to import additional vision models
try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    from transformers import CLIPProcessor, CLIPModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("⚠️  transformers not available - install with: pip install transformers")
    TRANSFORMERS_AVAILABLE = False

try:
    import ultralytics
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("⚠️  ultralytics not available - install with: pip install ultralytics")
    YOLO_AVAILABLE = False

logger = logging.getLogger(__name__)

class CudaVisionProcessor:
    """
    CUDA-accelerated vision processor for blind assistance.
    Combines multiple vision models for comprehensive scene analysis.
    """
    
    def __init__(self, device: str = "auto", model_cache_dir: str = "models/"):
        """
        Initialize the CUDA vision processor.
        
        Args:
            device: Device to use ('auto', 'cuda', 'cpu')
            model_cache_dir: Directory to cache downloaded models
        """
        self.device = self._setup_device(device)
        self.model_cache_dir = model_cache_dir
        
        # Performance tracking
        self.stats = {
            'total_processed': 0,
            'average_processing_time': 0,
            'cuda_memory_used': 0,
            'last_processing_time': 0
        }
        
        # Initialize models
        self.models = {}
        self.processors = {}
        self.transforms = self._setup_transforms()
        
        logger.info(f"CudaVisionProcessor initialized on {self.device}")
        self._initialize_models()
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup the best available device."""
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                logger.info(f"🚀 CUDA detected: {torch.cuda.get_device_name(0)}")
                logger.info(f"🔋 CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            else:
                device = "cpu"
                logger.warning("⚠️  CUDA not available, using CPU")
        
        device = torch.device(device)
        
        # Set optimal settings for CUDA
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        
        return device
    
    def _setup_transforms(self) -> Dict[str, transforms.Compose]:
        """Setup image transformations for different models."""
        return {
            'efficient_net': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ]),
            'clip': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                   std=[0.26862954, 0.26130258, 0.27577711])
            ])
        }
    
    def _initialize_models(self):
        """Initialize all vision models on GPU."""
        logger.info("🔄 Loading vision models...")
        
        # 1. EfficientNet for general object detection
        try:
            self.models['efficientnet'] = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
            self.models['efficientnet'].eval()
            self.models['efficientnet'] = self.models['efficientnet'].to(self.device)
            logger.info("✅ EfficientNet loaded")
        except Exception as e:
            logger.error(f"❌ Failed to load EfficientNet: {e}")
        
        # 2. BLIP for image captioning (if available)
        if TRANSFORMERS_AVAILABLE:
            try:
                self.processors['blip'] = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
                self.models['blip'] = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
                self.models['blip'] = self.models['blip'].to(self.device)
                logger.info("✅ BLIP image captioning loaded")
            except Exception as e:
                logger.warning(f"⚠️  Failed to load BLIP: {e}")
        
        # 3. CLIP for scene understanding (if available)
        if TRANSFORMERS_AVAILABLE:
            try:
                self.processors['clip'] = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                self.models['clip'] = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                self.models['clip'] = self.models['clip'].to(self.device)
                logger.info("✅ CLIP model loaded")
            except Exception as e:
                logger.warning(f"⚠️  Failed to load CLIP: {e}")
        
        # 4. YOLO for object detection (if available)
        if YOLO_AVAILABLE:
            try:
                self.models['yolo'] = YOLO('yolov8n.pt')  # Nano version for speed
                if self.device.type == 'cuda':
                    self.models['yolo'].to(self.device)
                logger.info("✅ YOLOv8 object detection loaded")
            except Exception as e:
                logger.warning(f"⚠️  Failed to load YOLO: {e}")
        
        # Clear GPU cache after loading
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
    
    def process_image(self, image_input, analysis_type: str = "comprehensive") -> Dict[str, Any]:
        """
        Process an image with GPU acceleration.
        
        Args:
            image_input: PIL Image, numpy array, or file path
            analysis_type: Type of analysis ('comprehensive', 'safety', 'navigation', 'objects')
        
        Returns:
            Dictionary with analysis results
        """
        start_time = time.time()
        
        try:
            # Convert input to PIL Image
            if isinstance(image_input, str):
                image = Image.open(image_input).convert('RGB')
            elif isinstance(image_input, np.ndarray):
                image = Image.fromarray(image_input).convert('RGB')
            elif hasattr(image_input, 'read'):  # File-like object
                image = Image.open(image_input).convert('RGB')
            else:
                image = image_input.convert('RGB')
            
            # Run analysis based on type
            results = {}
            
            if analysis_type in ["comprehensive", "all"]:
                results.update(self._comprehensive_analysis(image))
            elif analysis_type == "safety":
                results.update(self._safety_analysis(image))
            elif analysis_type == "navigation":
                results.update(self._navigation_analysis(image))
            elif analysis_type == "objects":
                results.update(self._object_detection(image))
            else:
                results.update(self._basic_analysis(image))
            
            # Update stats
            processing_time = time.time() - start_time
            self._update_stats(processing_time)
            
            results['processing_info'] = {
                'processing_time': round(processing_time, 3),
                'device_used': str(self.device),
                'cuda_available': torch.cuda.is_available(),
                'memory_used': self._get_memory_usage()
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return {
                'error': str(e),
                'fallback_description': "Unable to process image with GPU acceleration. Please try again."
            }
    
    def _comprehensive_analysis(self, image: Image.Image) -> Dict[str, Any]:
        """Run comprehensive analysis using all available models."""
        results = {
            'description': "",
            'objects': [],
            'safety_assessment': "",
            'navigation_guidance': "",
            'scene_context': ""
        }
        
        # 1. Image captioning with BLIP
        if 'blip' in self.models:
            caption = self._generate_caption(image)
            results['description'] = caption
        
        # 2. Object detection with YOLO
        if 'yolo' in self.models:
            objects = self._detect_objects_yolo(image)
            results['objects'] = objects
        
        # 3. Scene classification with EfficientNet
        if 'efficientnet' in self.models:
            scene_info = self._classify_scene(image)
            results['scene_context'] = scene_info
        
        # 4. Safety and navigation analysis
        results['safety_assessment'] = self._analyze_safety(results['objects'])
        results['navigation_guidance'] = self._generate_navigation_guidance(results['objects'])
        
        # 5. Generate comprehensive description
        results['comprehensive_description'] = self._generate_comprehensive_description(results)
        
        return results
    
    def _generate_caption(self, image: Image.Image) -> str:
        """Generate image caption using BLIP."""
        try:
            if 'blip' not in self.models:
                return "Image captioning not available"
            
            with torch.no_grad():
                inputs = self.processors['blip'](image, return_tensors="pt").to(self.device)
                out = self.models['blip'].generate(**inputs, max_new_tokens=100, num_beams=5)
                caption = self.processors['blip'].decode(out[0], skip_special_tokens=True)
                return caption
        except Exception as e:
            logger.error(f"Caption generation error: {e}")
            return "Unable to generate caption"
    
    def _detect_objects_yolo(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Detect objects using YOLOv8."""
        try:
            if 'yolo' not in self.models:
                return []
            
            # Convert PIL to numpy for YOLO
            img_array = np.array(image)
            
            results = self.models['yolo'](img_array, verbose=False)
            
            objects = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        obj = {
                            'class': result.names[int(box.cls)],
                            'confidence': float(box.conf),
                            'bbox': box.xyxy[0].cpu().numpy().tolist(),
                            'position': self._describe_position(box.xyxy[0].cpu().numpy(), image.size)
                        }
                        objects.append(obj)
            
            return objects
        except Exception as e:
            logger.error(f"YOLO detection error: {e}")
            return []
    
    def _classify_scene(self, image: Image.Image) -> str:
        """Classify scene using EfficientNet."""
        try:
            if 'efficientnet' not in self.models:
                return "Scene classification not available"
            
            with torch.no_grad():
                input_tensor = self.transforms['efficient_net'](image).unsqueeze(0).to(self.device)
                outputs = self.models['efficientnet'](input_tensor)
                probabilities = F.softmax(outputs, dim=1)
                top_prob, top_class = torch.topk(probabilities, 5)
                
                # Load ImageNet class names (simplified)
                scene_descriptions = []
                for i in range(5):
                    prob = top_prob[0][i].item()
                    if prob > 0.1:  # Only include confident predictions
                        scene_descriptions.append(f"Scene type {i+1} (confidence: {prob:.2f})")
                
                return "; ".join(scene_descriptions) if scene_descriptions else "Indoor/outdoor scene"
        except Exception as e:
            logger.error(f"Scene classification error: {e}")
            return "Scene analysis unavailable"
    
    def _describe_position(self, bbox: np.ndarray, image_size: Tuple[int, int]) -> str:
        """Describe object position in accessible terms."""
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        width, height = image_size
        
        # Horizontal position
        if center_x < width * 0.33:
            h_pos = "left"
        elif center_x > width * 0.67:
            h_pos = "right"
        else:
            h_pos = "center"
        
        # Vertical position
        if center_y < height * 0.33:
            v_pos = "top"
        elif center_y > height * 0.67:
            v_pos = "bottom"
        else:
            v_pos = "middle"
        
        return f"{v_pos} {h_pos}"
    
    def _analyze_safety(self, objects: List[Dict[str, Any]]) -> str:
        """Analyze potential safety hazards."""
        hazards = []
        safe_objects = []
        
        for obj in objects:
            obj_class = obj['class'].lower()
            confidence = obj['confidence']
            
            if confidence > 0.5:  # Only consider confident detections
                if any(hazard in obj_class for hazard in ['knife', 'scissors', 'fire', 'stove', 'car', 'truck', 'motorcycle']):
                    hazards.append(f"{obj['class']} in {obj['position']}")
                elif any(furniture in obj_class for furniture in ['chair', 'table', 'sofa', 'bed']):
                    safe_objects.append(f"{obj['class']} in {obj['position']}")
        
        safety_msg = []
        if hazards:
            safety_msg.append(f"⚠️ Potential hazards detected: {', '.join(hazards)}")
        if safe_objects:
            safety_msg.append(f"✅ Safe objects available: {', '.join(safe_objects)}")
        
        return "; ".join(safety_msg) if safety_msg else "No specific safety concerns detected"
    
    def _generate_navigation_guidance(self, objects: List[Dict[str, Any]]) -> str:
        """Generate navigation guidance based on detected objects."""
        obstacles = []
        landmarks = []
        
        for obj in objects:
            obj_class = obj['class'].lower()
            position = obj['position']
            
            if obj['confidence'] > 0.5:
                if any(obstacle in obj_class for obstacle in ['chair', 'table', 'person', 'dog', 'cat']):
                    obstacles.append(f"{obj['class']} in {position}")
                elif any(landmark in obj_class for landmark in ['door', 'window', 'stairs', 'elevator']):
                    landmarks.append(f"{obj['class']} in {position}")
        
        guidance = []
        if obstacles:
            guidance.append(f"🚧 Navigate around: {', '.join(obstacles)}")
        if landmarks:
            guidance.append(f"🗺️ Reference points: {', '.join(landmarks)}")
        
        return "; ".join(guidance) if guidance else "Clear path, proceed with normal caution"
    
    def _generate_comprehensive_description(self, results: Dict[str, Any]) -> str:
        """Generate a comprehensive, accessibility-focused description."""
        description_parts = []
        
        # Start with main description
        if results.get('description'):
            description_parts.append(f"Scene overview: {results['description']}")
        
        # Add object information
        if results.get('objects'):
            obj_count = len(results['objects'])
            if obj_count > 0:
                main_objects = [obj['class'] for obj in results['objects'][:5]]  # Top 5 objects
                description_parts.append(f"I can identify {obj_count} objects including: {', '.join(main_objects)}")
        
        # Add safety information
        if results.get('safety_assessment'):
            description_parts.append(results['safety_assessment'])
        
        # Add navigation guidance
        if results.get('navigation_guidance'):
            description_parts.append(results['navigation_guidance'])
        
        return " | ".join(description_parts) if description_parts else "Image processed successfully"
    
    def _safety_analysis(self, image: Image.Image) -> Dict[str, Any]:
        """Focus on safety hazards and concerns."""
        return self._comprehensive_analysis(image)  # For now, use comprehensive analysis
    
    def _navigation_analysis(self, image: Image.Image) -> Dict[str, Any]:
        """Focus on navigation and mobility."""
        return self._comprehensive_analysis(image)  # For now, use comprehensive analysis
    
    def _object_detection(self, image: Image.Image) -> Dict[str, Any]:
        """Focus on object detection and identification."""
        results = {}
        if 'yolo' in self.models:
            results['objects'] = self._detect_objects_yolo(image)
        return results
    
    def _basic_analysis(self, image: Image.Image) -> Dict[str, Any]:
        """Basic analysis when specific type not specified."""
        return self._comprehensive_analysis(image)
    
    def _update_stats(self, processing_time: float):
        """Update processing statistics."""
        self.stats['total_processed'] += 1
        self.stats['last_processing_time'] = processing_time
        
        # Update average
        total = self.stats['total_processed']
        current_avg = self.stats['average_processing_time']
        self.stats['average_processing_time'] = (current_avg * (total - 1) + processing_time) / total
        
        # Update memory usage
        if self.device.type == 'cuda':
            self.stats['cuda_memory_used'] = torch.cuda.memory_allocated() / 1e9  # GB
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage."""
        if self.device.type == 'cuda':
            return {
                'allocated_gb': torch.cuda.memory_allocated() / 1e9,
                'cached_gb': torch.cuda.memory_reserved() / 1e9,
                'max_allocated_gb': torch.cuda.max_memory_allocated() / 1e9
            }
        return {'cpu_memory': 'Not tracked'}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        stats = self.stats.copy()
        stats['device'] = str(self.device)
        stats['models_loaded'] = list(self.models.keys())
        stats['cuda_available'] = torch.cuda.is_available()
        
        if self.device.type == 'cuda':
            stats['gpu_info'] = {
                'name': torch.cuda.get_device_name(0),
                'memory_total_gb': torch.cuda.get_device_properties(0).total_memory / 1e9,
                'memory_usage': self._get_memory_usage()
            }
        
        return stats
    
    def clear_cache(self):
        """Clear GPU cache to free memory."""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            logger.info("🔄 GPU cache cleared")
    
    def set_device(self, device: str):
        """Change the processing device."""
        new_device = torch.device(device)
        if new_device != self.device:
            logger.info(f"🔄 Switching from {self.device} to {new_device}")
            self.device = new_device
            # Move models to new device
            for model_name, model in self.models.items():
                try:
                    self.models[model_name] = model.to(self.device)
                except Exception as e:
                    logger.warning(f"⚠️  Failed to move {model_name} to {new_device}: {e}")

# Global instance for easy access
_cuda_processor = None

def get_cuda_processor(device: str = "auto") -> CudaVisionProcessor:
    """Get or create global CUDA processor instance."""
    global _cuda_processor
    if _cuda_processor is None:
        _cuda_processor = CudaVisionProcessor(device=device)
    return _cuda_processor

# Test the processor if run directly
if __name__ == "__main__":
    print("🧪 Testing CUDA Vision Processor...")
    
    processor = CudaVisionProcessor()
    stats = processor.get_stats()
    
    print(f"📊 Processor Stats:")
    print(f"   Device: {stats['device']}")
    print(f"   Models loaded: {stats['models_loaded']}")
    print(f"   CUDA available: {stats['cuda_available']}")
    
    if stats['cuda_available']:
        print(f"   GPU: {stats['gpu_info']['name']}")
        print(f"   GPU Memory: {stats['gpu_info']['memory_total_gb']:.1f} GB")
    
    print("✅ CUDA Vision Processor ready for use!")
