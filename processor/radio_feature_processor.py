
import torch
from torch.nn import functional as F
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
import os
import pickle
import hashlib
import faiss
import json
import numpy as np
import cv2
import time
from collections import defaultdict
import traceback
import model_defs.models
import h5py

class RADIOFeatureProcessor:
    def __init__(self, cache_dir: str = "static/images/face_features", model_version="radio_v2.5-b"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cache_dir = cache_dir
        self.model_manager = model_defs.models.model_manager
        
        # Create separate indices for different feature types
        self.indices = {
            'face': None,
            'body': None
        }
        self.feature_mappings = {
            'face': [],
            'body': []
        }
        
        # Initialize RADIO model
        print("Loading RADIO model...")
        self.model = torch.hub.load('NVlabs/RADIO', 'radio_model', 
                                  version=model_version, 
                                  progress=True, 
                                  skip_validation=True)
        self.model.to(self.device).eval()
        
        # Calculate model hash
        self.model_hash = self._calculate_model_hash()
        print(f"Model hash: {self.model_hash}")
        
        # Initialize indices
        self._initialize_indices()
        
        # Load cached features
        self._load_cached_features()



    def _get_paths(self, feature_type):
        """Get paths for index and mapping files"""
        base_path = os.path.join(self.cache_dir, f"{feature_type}_features_{self.model_hash}")
        return f"{base_path}.index", f"{base_path}.h5"

    def _save_feature_cache(self, feature_type):
        """Save feature cache for specific feature type"""
        try:
            # Get file paths
            index_path, mapping_path = self._get_paths(feature_type)
            
            # Create directory if needed
            os.makedirs(os.path.dirname(index_path), exist_ok=True)
            
            # Save FAISS index directly
            faiss.write_index(self.indices[feature_type], index_path)
            
            # Save feature mapping to HDF5
            with h5py.File(mapping_path, 'w') as f:
                f.attrs['model_hash'] = self.model_hash
                f.attrs['feature_type'] = feature_type
                
                # Store the mapping data
                mapping_data = self.feature_mappings[feature_type]
                for i, item in enumerate(mapping_data):
                    group = f.create_group(str(i))
                    for key, value in item.items():
                        group.attrs[key] = value
                
            print(f"Successfully cached {self.indices[feature_type].ntotal} {feature_type} features")
            
        except Exception as e:
            print(f"Error caching {feature_type} features: {e}")
            traceback.print_exc()

    def _load_cached_features(self):
        """Load cached features for all feature types"""
        try:
            if not hasattr(self, 'feature_dim'):
                self._initialize_indices()

            any_loaded = False
            
            for feature_type in ['face', 'body']:
                index_path, mapping_path = self._get_paths(feature_type)
                
                if not os.path.exists(index_path) or not os.path.exists(mapping_path):
                    print(f"No cached features found for {feature_type}")
                    continue
                    
                try:
                    # Verify mapping and load metadata
                    with h5py.File(mapping_path, 'r') as f:
                        stored_hash = f.attrs.get('model_hash')
                        stored_type = f.attrs.get('feature_type')
                        
                        if stored_hash != self.model_hash:
                            print(f"Model hash mismatch for {feature_type}")
                            continue
                            
                        if stored_type != feature_type:
                            print(f"Feature type mismatch for {feature_type}")
                            continue
                        
                        # Load mapping data
                        mapping_data = []
                        for i in range(len(f.keys())):
                            group = f[str(i)]
                            item = {key: value for key, value in group.attrs.items()}
                            mapping_data.append(item)
                    
                    # Load FAISS index
                    self.indices[feature_type] = faiss.read_index(index_path)
                    self.feature_mappings[feature_type] = mapping_data
                    
                    print(f"Loaded {self.indices[feature_type].ntotal} cached {feature_type} features")
                    any_loaded = True
                    
                except Exception as e:
                    print(f"Error loading {feature_type} features: {e}")
                    traceback.print_exc()
                    continue
                    
            return any_loaded
                    
        except Exception as e:
            print(f"Error loading cached features: {e}")
            traceback.print_exc()
            self._initialize_indices()
            return False

    def save_index(self, index_path):
        """Save all indices to specified path"""
        try:
            os.makedirs(os.path.dirname(index_path), exist_ok=True)
            
            for feature_type in ['face', 'body']:
                curr_index_path = f"{index_path}_{feature_type}.index"
                curr_mapping_path = f"{index_path}_{feature_type}.h5"
                
                # Save FAISS index
                faiss.write_index(self.indices[feature_type], curr_index_path)
                
                # Save mapping to HDF5
                with h5py.File(curr_mapping_path, 'w') as f:
                    f.attrs['model_hash'] = self.model_hash
                    f.attrs['feature_type'] = feature_type
                    
                    mapping_data = self.feature_mappings[feature_type]
                    for i, item in enumerate(mapping_data):
                        group = f.create_group(str(i))
                        for key, value in item.items():
                            group.attrs[key] = value
                    
                print(f"Saved {feature_type} index to {curr_index_path}")
            return True
            
        except Exception as e:
            print(f"Error saving indices: {e}")
            traceback.print_exc()
            return False

    def load_index(self, index_path):
        """Load all indices from specified path"""
        try:
            if not hasattr(self, 'feature_dim'):
                self._initialize_indices()

            any_loaded = False
            for feature_type in ['face', 'body']:
                curr_index_path = f"{index_path}_{feature_type}.index"
                curr_mapping_path = f"{index_path}_{feature_type}.h5"
                
                if not os.path.exists(curr_index_path) or not os.path.exists(curr_mapping_path):
                    print(f"Index files not found for {feature_type}")
                    continue
                
                try:
                    # Load and verify mapping from HDF5
                    with h5py.File(curr_mapping_path, 'r') as f:
                        stored_hash = f.attrs.get('model_hash')
                        stored_type = f.attrs.get('feature_type')
                        
                        if stored_hash != self.model_hash:
                            print(f"Model hash mismatch for {feature_type}")
                            print("Warning: Loading index from different model version")
                            
                        if stored_type != feature_type:
                            print(f"Feature type mismatch for {feature_type}")
                            continue
                        
                        # Load mapping data
                        mapping_data = []
                        for i in range(len(f.keys())):
                            group = f[str(i)]
                            item = {key: value for key, value in group.attrs.items()}
                            mapping_data.append(item)
                    
                    # Load FAISS index
                    self.indices[feature_type] = faiss.read_index(curr_index_path)
                    self.feature_mappings[feature_type] = mapping_data
                    
                    print(f"Loaded {feature_type} index with {self.indices[feature_type].ntotal} vectors")
                    any_loaded = True
                    
                except Exception as e:
                    print(f"Error loading {feature_type} index: {e}")
                    traceback.print_exc()
                    continue
                    
            return any_loaded
                
        except Exception as e:
            print(f"Error loading indices: {e}")
            traceback.print_exc()
            return False
        
    def _calculate_model_hash(self):
        """Calculate hash of model parameters for cache validation"""
        try:
            model_state = self.model.state_dict()
            model_bytes = pickle.dumps([(k, v.shape) for k, v in model_state.items()])
            return hashlib.md5(model_bytes).hexdigest()
        except Exception as e:
            print(f"Error calculating model hash: {e}")
            traceback.print_exc()
            return "default_hash"

  

    

    def _preprocess_image(self, image_input):
        """Preprocess image for RADIO model"""
        try:
            # Handle different input types
            if isinstance(image_input, Image.Image):
                image = image_input
            elif isinstance(image_input, str):
                image = Image.open(image_input).convert('RGB')
            elif isinstance(image_input, np.ndarray):
                image = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
            else:
                raise ValueError("Input must be PIL Image, path, or numpy array")
            
            # Convert to tensor and normalize
            x = pil_to_tensor(image).to(dtype=torch.float32, device=self.device)
            x.div_(255.0)  # Normalize to [0, 1]
            x = x.unsqueeze(0)  # Add batch dimension
            
            
            # Get nearest supported resolution
            nearest_res = self.model.get_nearest_supported_resolution(*x.shape[-2:])
            x = F.interpolate(x, nearest_res, mode='bilinear', align_corners=False)
            
            return x
            
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            traceback.print_exc()
            return None

    def _initialize_indices(self):
        """Initialize FAISS indices for different feature types"""
        try:
            # Get feature dimension using dummy input
            dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
            with torch.no_grad():
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    outputs = self.model(dummy_input)
                    
                    # Debug output
                    print("Model output type:", type(outputs))
                    if isinstance(outputs, dict):
                        print("Output keys:", outputs.keys())
                        
                    if isinstance(outputs, dict):
                        if 'backbone' in outputs:
                            summary = outputs['backbone'].summary
                        elif 'siglip' in outputs:
                            summary = outputs['siglip'].summary
                    elif isinstance(outputs, tuple):
                        summary = outputs[0]
                    else:
                        raise ValueError(f"Unexpected output format: {type(outputs)}")
                    
                    print("Summary type:", type(summary))
                    print("Summary shape:", summary.shape)

            # Store feature dimension
            self.feature_dim = summary.shape[1]
            
            # Initialize indices for each feature type
            for feature_type in ['face', 'body']:
                self.indices[feature_type] = faiss.IndexFlatIP(self.feature_dim)
                self.feature_mappings[feature_type] = []
                print(f"Created new FAISS index for {feature_type} with dimension {self.feature_dim}")
                
            return True
                
        except Exception as e:
            print(f"Error initializing indices: {e}")
            traceback.print_exc()
            return False


    def extract_features(self, image_input, feature_type='face'):
        """Extract features using RADIO model
        
        Args:
            image_input: PIL Image, path, or numpy array
            feature_type: Type of features to extract ('face' or 'body')
        """
        try:
            # Handle different input types
            if isinstance(image_input, Image.Image):
                image = image_input
            elif isinstance(image_input, str):
                image = Image.open(image_input).convert('RGB')
            elif isinstance(image_input, np.ndarray):
                image = Image.fromarray(image_input).convert('RGB')
            else:
                raise ValueError("Input must be PIL Image, path, or numpy array")

            # Apply GFPGAN enhancement for face features
            if feature_type == 'face':
                enhanced = self._enhance_face_with_gfpgan(np.array(image))
                image = Image.fromarray(enhanced)
                
            # Prepare input tensor
            x = pil_to_tensor(image).to(dtype=torch.float32, device=self.device)
            x.div_(255.0)
            x = x.unsqueeze(0)
            
            nearest_res = self.model.get_nearest_supported_resolution(*x.shape[-2:])
            x = F.interpolate(x, nearest_res, mode='bilinear', align_corners=False)
            
            # Extract features
            with torch.no_grad():
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    outputs = self.model(x)
                    
                    if isinstance(outputs, dict):
                        if 'backbone' in outputs:
                            features = outputs['backbone'].summary
                        elif 'siglip' in outputs:
                            features = outputs['siglip'].summary
                    elif isinstance(outputs, tuple):
                        features = outputs[0]
                    else:
                        raise ValueError(f"Unexpected output format: {type(outputs)}")
                    
                    features = F.normalize(features, p=2, dim=1)
                    
            return features.cpu().numpy()
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            traceback.print_exc()
            return None

    def add_reference_features(self, image_path, feature_type, identifier=None):
        """Add reference features to the appropriate index"""
        try:
            if feature_type not in self.indices:
                raise ValueError(f"Unsupported feature type: {feature_type}")
                
            if identifier is None:
                identifier = os.path.basename(image_path).split('.')[0]
                
            print(f"Processing {feature_type} features for {identifier}")
            
            # Extract features
            features = self.extract_features(image_path, feature_type)
            if features is None:
                return False
                
            # Add to appropriate index
            self.indices[feature_type].add(features)
            
            # Store metadata
            metadata = {
                'image_path': image_path if isinstance(image_path, str) else "embedded_image",
                'identifier': identifier,
                'feature_type': feature_type,
                'timestamp': int(time.time())
            }
            
            self.feature_mappings[feature_type].append(metadata)
            self._save_feature_cache(feature_type)
            
            print(f"Added {feature_type} features for {identifier} "
                  f"(total vectors: {self.indices[feature_type].ntotal})")
            return True
            
        except Exception as e:
            print(f"Error adding reference features: {e}")
            traceback.print_exc()
            return False

    def search_similar_features(self, image=None, feature_type='face', k=5):
        """Search for similar features of specific type"""
        try:
            if image is None:
                print("No image provided")
                return []
                
            if feature_type not in self.indices:
                raise ValueError(f"Unsupported feature type: {feature_type}")
                
            print(f"Starting {feature_type} search with index size: {self.indices[feature_type].ntotal}")
                
            # Check image dimensions
            if isinstance(image, np.ndarray):
                height, width = image.shape[:2]
                img_array = image
            elif isinstance(image, Image.Image):
                width, height = image.size
                img_array = np.array(image)
            elif isinstance(image, str):
                img = Image.open(image)
                width, height = img.size
                img_array = np.array(img)
            else:
                print("Unsupported image type")
                return []
                
            # Skip if image is too small
            pixel_count = width * height
            if pixel_count < 480:
                print(f"Image too small: {pixel_count} pixels (minimum 480)")
                return []

            # Extract features
            print(f"Extracting {feature_type} features...")
            query_features = self.extract_features(img_array, feature_type)
            if query_features is None:
                return []
                
            # Get appropriate index and mapping
            index = self.indices[feature_type]
            feature_mapping = self.feature_mappings[feature_type]
            
            if index.ntotal == 0:
                print(f"Empty {feature_type} index")
                return []
                
            # Search
            D, I = index.search(query_features, min(k, index.ntotal))
            
            # Format results
            results = []
            for distance, idx in zip(D[0], I[0]):
                if idx < len(feature_mapping):
                    metadata = feature_mapping[idx]
                    similarity_score = float((1 + distance) / 2) * 100
                    
                    result = {
                        'metadata': metadata,
                        'distance': float(distance),
                        'similarity_score': similarity_score,
                        'feature_type': feature_type
                    }
                    results.append(result)
                    
                    print(f"Found {feature_type} match: {metadata['identifier']} "
                          f"- {similarity_score:.1f}%")
                        
            return sorted(results, key=lambda x: x['similarity_score'], reverse=True)
            
        except Exception as e:
            print(f"Error searching features: {e}")
            traceback.print_exc()
            return []


   


    def _enhance_face_with_gfpgan(self, face_img):
        """Enhance face image using GFPGAN"""
        try:
            
            
            if len(face_img.shape) == 3 and face_img.shape[2] == 3:
                face_bgr = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)
            else:
                face_bgr = face_img

            _, _, enhanced = self.model_manager.get_gfpgan().enhance(
                face_bgr,
                has_aligned=False,
                only_center_face=True,
                paste_back=True
            )

            if len(face_img.shape) == 3 and face_img.shape[2] == 3:
                enhanced = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)

            return enhanced

        except Exception as e:
            print(f"Face enhancement failed: {e}")
            return face_img
        
    def cleanup(self):
        """Clean up resources"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()