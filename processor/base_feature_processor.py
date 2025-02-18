from abc import ABC, abstractmethod
import os
import pickle
import hashlib
import faiss
import json
import numpy as np
from PIL import Image
import torch

class BaseFeatureProcessor(ABC):
    """
    Abstract base class for feature processors.
    Defines the interface that all feature processors must implement.
    """
    def __init__(self, features_json_path: str, cache_dir: str):
        """
        Initialize the feature processor.
        
        Args:
            features_json_path: Path to features definition JSON
            cache_dir: Directory for caching features
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Load feature definitions
        with open(features_json_path, 'r') as f:
            self.features_dict = json.load(f)
            
        # Initialize device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
 

    # @abstractmethod
    # def process_features(self, image=None, text=None):
    #     """
    #     Process image and/or text to extract features.
        
    #     Args:
    #         image: PIL Image, numpy array, or path (optional)
    #         text: Text string or list of strings (optional)
            
    #     Returns:
    #         dict: Dictionary containing extracted features
    #     """
    #     pass

    # @abstractmethod
    # def store_features(self, identifier, features, metadata=None):
    #     """
    #     Store features for future use.
        
    #     Args:
    #         identifier: Unique identifier for the features
    #         features: Features to store
    #         metadata: Additional metadata (optional)
            
    #     Returns:
    #         bool: Success status
    #     """
    #     pass

    # @abstractmethod
    # def search_similar(self, query_features, k=5):
    #     """
    #     Search for similar features.
        
    #     Args:
    #         query_features: Features to search for
    #         k: Number of results to return
            
    #     Returns:
    #         list: List of similar items with scores
    #     """
    #     pass

    # @abstractmethod
    # def analyze_segment(self, segment_image, segment_type):
    #     """
    #     Analyze an image segment.
        
    #     Args:
    #         segment_image: Image segment to analyze
    #         segment_type: Type of segment (e.g., 'face', 'body')
            
    #     Returns:
    #         dict: Analysis results
    #     """
    #     pass

    # def _preprocess_image(self, image_input):
    #     """
    #     Preprocess image input to standard format.
        
    #     Args:
    #         image_input: PIL Image, numpy array, or path
            
    #     Returns:
    #         PIL.Image: Preprocessed image
    #     """
    #     if isinstance(image_input, str):
    #         image = Image.open(image_input)
    #     elif isinstance(image_input, np.ndarray):
    #         image = Image.fromarray(image_input)
    #     elif isinstance(image_input, Image.Image):
    #         image = image_input
    #     else:
    #         raise ValueError("Input must be PIL Image, numpy array, or path")
            
    #     return image.convert('RGB')

    # def _normalize_features(self, features):
    #     """
    #     Normalize feature vectors.
        
    #     Args:
    #         features: Feature vectors to normalize
            
    #     Returns:
    #         numpy.ndarray: Normalized features
    #     """
    #     if isinstance(features, torch.Tensor):
    #         features = features.detach().cpu().numpy()
    #     return features / np.linalg.norm(features, axis=1, keepdims=True)

    # def _calculate_similarity(self, features1, features2):
    #     """
    #     Calculate similarity between feature vectors.
        
    #     Args:
    #         features1: First feature vector
    #         features2: Second feature vector
            
    #     Returns:
    #         float: Similarity score
    #     """
    #     features1 = self._normalize_features(features1)
    #     features2 = self._normalize_features(features2)
    #     return np.dot(features1, features2.T)

    # def _organize_results(self, results, confidence_threshold=30):
    #     """
    #     Organize analysis results.
        
    #     Args:
    #         results: Raw analysis results
    #         confidence_threshold: Minimum confidence score
            
    #     Returns:
    #         dict: Organized results
    #     """
    #     if not results:
    #         return None

    #     filtered_results = [r for r in results if r['confidence'] > confidence_threshold]
    #     if not filtered_results:
    #         return None

    #     return {
    #         'top_matches': sorted(filtered_results, key=lambda x: x['confidence'], reverse=True)[:5],
    #         'confidence': max(r['confidence'] for r in filtered_results)
    #     }

    # def cleanup(self):
    #     """Clean up resources"""
    #     pass