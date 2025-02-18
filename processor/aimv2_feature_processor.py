import torch
import faiss
import json
import os
import pickle
import hashlib
import numpy as np
from PIL import Image
import cv2
import time
from collections import defaultdict
import model_defs.models
import traceback

import h5py
import torch.nn.functional as F

class AIMv2FeatureProcessor():
    """
    AIMv2-based implementation of feature processor.
    """
    def __init__(self, features_json_path: str, cache_dir: str = "face_feature_cache/aimv2"):
        # super().__init__(features_json_path, cache_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Initialize AIMv2 components
        self.model_manager = model_defs.models.model_manager
        # Then initialize model-dependent components
        self.model = self.model_manager.get_aimv2_model()
        self.processor = self.model_manager.get_aimv2_processor()
        self.cache_dir = cache_dir
        
        # Initialize feature dimensions
        self.feature_dim = self.model.config.projection_dim
        
        # Load or create default image
        self.default_image = self._load_or_create_default_image()
        self.model_hash = self._calculate_model_hash()
        self._h5file = None 
        
        # 建立快取目錄
        os.makedirs(cache_dir, exist_ok=True)
        
        # HDF5 檔案路徑
        self.hdf5_path = os.path.join(cache_dir, 'features.h5')
  
        self._initialize_index()
       
        self._load_cached_features()
     
        
        
    def _save_feature_to_cache(self):
        try:
            cache_path = self._get_cache_paths()
            
            # 確保先關閉任何可能開啟的檔案句柄
            if hasattr(self, '_h5file') and self._h5file is not None:
                self._h5file.close()
                
            # 使用新的檔案句柄寫入
            with h5py.File(cache_path, 'w') as f:
                # 保存 FAISS index
                index_bytes = faiss.serialize_index(self.index)
                index_array = np.frombuffer(index_bytes, dtype=np.uint8)
                f.create_dataset('index_data', data=index_array, compression='gzip')

                # 將 numpy arrays 轉換為列表
                serializable_mapping = []
                for entry in self.feature_mapping:
                    serializable_entry = {}
                    for key, value in entry.items():
                        if isinstance(value, np.ndarray):
                            serializable_entry[key] = value.tolist()
                        else:
                            serializable_entry[key] = value
                    serializable_mapping.append(serializable_entry)

                mapping_str = json.dumps(serializable_mapping)
                f.create_dataset('feature_mapping', data=mapping_str.encode('utf-8'))

        except Exception as e:
            print(f"Error saving features to HDF5: {e}")
            traceback.print_exc()

    def _load_cached_features(self):
        try:
            if not os.path.exists(self.hdf5_path):
                print("No cached features found.")
                return False
                
            with h5py.File(self.hdf5_path, 'r') as f:
                if 'index_data' in f:
                    index_array = f['index_data'][:]  # This should already be a byte buffer
                    self.index = faiss.deserialize_index(index_array)


                    
                if 'feature_mapping' in f:
                    mapping_bytes = f['feature_mapping'][()].tobytes()
                    self.feature_mapping = json.loads(mapping_bytes.decode('utf-8'))
                    
                print(f"Loaded cached features: {self.index.ntotal} vectors")
                return True
                
       
        except Exception as e:
            print(f"Error loading features from HDF5: {e}")
            traceback.print_exc()
            self._initialize_index()
            return False
        
    def _load_or_create_default_image(self):
        """Load or create a default image for text-only processing"""
        default_path = os.path.join(self.cache_dir, 'default.jpg')
        if not os.path.exists(default_path):
            # Create a blank gray image
            default_img = Image.new('RGB', (224, 224), (128, 128, 128))
            default_img.save(default_path)
        return Image.open(default_path)
    
    def _initialize_index(self):
        """Initialize FAISS index with proper configuration"""
        try:
            # 直接使用 IndexFlatIP
            self.index = faiss.IndexFlatIP(self.feature_dim)
            
            # 初始化特徵映射列表，使用列表而不是字典
            # 因為 IndexFlatIP 的索引位置自然對應到列表索引
            self.feature_mapping = []
            print(f"Created new FAISS index with dimension {self.feature_dim}")
        except Exception as e:
            print(f"Error initializing index: {e}")
            traceback.print_exc()
    
    def _get_cache_paths(self):
        """Get cache file paths"""
        base_name = f"face_features_{self.model_hash}"
        hdf5_path = os.path.join(self.cache_dir, f"{base_name}.h5")
        return hdf5_path
    
    
            

    def extract_features(self, image_input):
        """Extract features using AIMv2"""
        try:
            # Handle different input types
            if isinstance(image_input, Image.Image):
                image = image_input
            elif isinstance(image_input, str):
                image = Image.open(image_input)
            elif isinstance(image_input, np.ndarray):
                image = Image.fromarray(image_input)
            else:
                raise ValueError("Input must be PIL Image, path, or numpy array")
                
            # Ensure RGB mode
            image = image.convert('RGB')
            
            # Process input with AIMv2
            inputs = self.processor(
                images=image,
                return_tensors="pt"
            ).to(self.device)
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                features = outputs.image_features
                features = features / features.norm(dim=-1, keepdim=True)
                
            return features.cpu().numpy()
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            traceback.print_exc()
            return None
    def process_features(self, image=None, text=None):
        """
        Process image and/or text using AIMv2.
        """
        try:
            # Handle default cases
            if image is None:
                image = self.default_image
            if text is None:
                text = ["default"]

            # Preprocess image
            processed_image = self._preprocess_image(image)
            
            # Process inputs
            inputs = self.processor(
                images=processed_image,
                text=text,
                add_special_tokens=True,
                truncation=True,
                padding=True,
                return_tensors="pt"
            )
            
            # Move to device
            inputs = {k: v.to(self.device) if hasattr(v, 'to') else v 
                     for k, v in inputs.items()}
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                return {
                    'image_features': outputs['image_features'].cpu().detach().numpy(),
                    'text_features': outputs['text_features'].cpu().detach().numpy()
                }

        except Exception as e:
            print(f"Error processing features: {e}")
            traceback.print_exc()
            return None
        
    def add_reference_features(self, image_path, feature_type, identifier=None, text_descriptions=None):
        """添加參考特徵到索引中"""
        try:
            if identifier is None:
                identifier = os.path.basename(image_path).split('.')[0]
                
            # print(f"Processing image: {image_path}")
            
            features = self.process_image_and_text(
                image=image_path,
                text_descriptions=text_descriptions or ["default"]
            )
            
            if features is not None and 'image_features' in features:
                image_features = features['image_features'].astype('float32')
                self.index.add(image_features)
                
                metadata = {
                    'image_path': image_path,
                    'identifier': identifier,
                    'feature_type': feature_type,
                    'text_descriptions': text_descriptions,
                    'timestamp': int(time.time())
                }
                
                self.feature_mapping.append(metadata)
                self._save_feature_to_cache()
                
                print(f"Added features for {identifier}")
                return True
                
            return False
            
        except Exception as e:
            print(f"Error adding reference features: {e}")
            traceback.print_exc()
            return False
            

    def process_face_features(self, image=None, text=None, frame_index= None ,track_id=None, k=1):
        """
        Search for face features with cache support, size validation, and GFPGAN enhancement
        
        Args:
            image: PIL Image or path to image
            text: Optional text descriptions
            k: Number of nearest neighbors to return
            
        Returns:
            List of matching results
        """
        try:
            # Ensure cache is loaded
            if self.index.ntotal == 0:
                if not self._load_cached_features():
                    return []
            
            # Convert PIL Image to numpy array if needed
            if isinstance(image, Image.Image):
                image_array = np.array(image)
            else:
                image_array = image
                
            # Check image size
            height, width = image_array.shape[:2]
            pixels = height*width
            print(f"Processing face image: {width}x{height} ({pixels} pixels)")
            if pixels < 480:
                print(f"Face image too small ({width}x{height}), skipping processing")
                return []
                
            # Apply GFPGAN enhancement
            try:
                enhanced_image = self._enhance_face_with_gfpgan(image_array)
                # Convert back to PIL Image for further processing
                enhanced_pil = Image.fromarray(enhanced_image)
                height, width = image_array.shape[:2]
              
            except Exception as e:
                print(f"Face enhancement failed: {e}")
                enhanced_pil = Image.fromarray(image_array)
            
            # Process enhanced image and text
            features = self.process_image_and_text(
                image=enhanced_pil,
                text_descriptions=text or ["default"],
            )
            
            if features is None:
                return []
                
            # Prepare search features
            search_features = features['image_features'].astype(np.float32)
            
            # Normalize search features
            faiss.normalize_L2(search_features)
            
            # Search nearest neighbors
            D, I = self.index.search(search_features, min(k, self.index.ntotal))
            
            # Format results
            results = []
            for distance, idx in zip(D[0], I[0]):
                if idx >= 0 and idx < len(self.feature_mapping):
                    metadata = self.feature_mapping[idx]
                    similarity_score = float((1 + distance) / 2) * 100
                    
                    # Ensure metadata has person_name
                    if 'person_name' not in metadata and 'identifier' in metadata:
                        metadata['person_name'] = metadata['identifier']
                    
                    result = {
                        'metadata': metadata,
                        'distance': float(distance),
                        'similarity_score': similarity_score
                    }
                    results.append(result)
            
            return sorted(results, key=lambda x: x['similarity_score'], reverse=True)
            
        except Exception as e:
            print(f"Error in process_face_features: {e}")
            traceback.print_exc()
            return []
        
    def save_index(self, index_path):
        try:
            with h5py.File(f"{index_path}.h5", 'w') as f:
                index_bytes = faiss.serialize_index(self.index)
                index_array = np.frombuffer(index_bytes, dtype=np.uint8)
                f.create_dataset('index_data', data=index_array, compression='gzip')
                mapping_str = json.dumps(self.feature_mapping)
                f.create_dataset('feature_mapping', data=mapping_str.encode('utf-8'))
            return True
        except Exception as e:
            print(f"Error saving index: {e}")
            traceback.print_exc()
            return False

    def load_index(self, index_path):
        try:
            with h5py.File(f"{index_path}.h5", 'r') as f:
                index_array = f['index_data'][:]
                self.index = faiss.deserialize_index(bytes(index_array))
                mapping_bytes = f['feature_mapping'][()].tobytes()
                self.feature_mapping = json.loads(mapping_bytes.decode('utf-8'))
            return True
        except Exception as e:
            print(f"Error loading index: {e}")
            traceback.print_exc()
            return False



        
    
    def process_image_and_text(self, image=None, text_descriptions=None):
            """
            Process image and/or text to extract features using AIMv2
            Based on aimv2 demo implementation
            
            Args:
                image: PIL Image, path, or numpy array (optional)
                text: String or list of strings (optional)
                
            Returns:
                dict: Contains image_features, text_features, and probabilities if both inputs provided
            """
            try:
                if image is None and text_descriptions is None:
                    raise ValueError("Either image or text must be provided")

                if image is not None:
                    # Handle different image input types
                    if isinstance(image, Image.Image):
                        processed_image = image
                    elif isinstance(image, str):
                        processed_image = Image.open(image)
                    elif isinstance(image, np.ndarray):
                        processed_image = Image.fromarray(image)
                    else:
                        raise ValueError("Image must be PIL Image, path, or numpy array")

                    processed_image = processed_image.convert('RGB')
                    
                # Create default image if needed
                default_image = Image.new('RGB', (224, 224), color='white')  
                processed_image = processed_image if image is not None else default_image

                # Process text if provided, otherwise use default
                description = text_descriptions if text_descriptions is not None else "default_description"
                if isinstance(description, str):
                    description = [description]
    
                # Prepare processor inputs
                inputs = self.processor(
                    images=processed_image,
                    text=description,
                    add_special_tokens=True,
                    truncation=True,
                    padding=True,
                    return_tensors="pt"
                ).to(self.device)

                # Get embeddings from model
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    results = {}
                    
                    if image is not None:
                        image_features = outputs['image_features'].cpu().detach()
                        image_features = F.normalize(image_features, p=2, dim=1).numpy()
                        results['image_features'] = image_features

                    if text_descriptions is not None:
                        text_features = outputs['text_features'].cpu().detach()
                        text_features = F.normalize(text_features, p=2, dim=1).numpy()
                        results['text_features'] = text_features

                    # Calculate probabilities if both image and text are provided
                    if image is not None and text_descriptions is not None:
                        # Calculate logits and probabilities
                        logits = torch.matmul(
                            outputs['image_features'], 
                            outputs['text_features'].transpose(0, 1)
                        )
                        probabilities = F.softmax(logits, dim=-1).cpu().numpy()
                        
                        results.update({
                            'image_features': image_features,
                            'text_features': text_features,
                            'probabilities': probabilities
                        })

                return results

            except Exception as e:
                print(f"Error processing inputs: {e}")
                traceback.print_exc()
                return None


    
 
    def _generate_segment_descriptions(self, segment_type):
        """Generate text descriptions for segment analysis."""
        descriptions = []
        features = self.features_dict.get(segment_type, {})

        # Add basic description
        descriptions.append(f"a {segment_type.lower()}")

        # Add specific features
        for feature_type in ['types', 'colors', 'patterns']:
            if feature_type in features:
                for value in features[feature_type]:
                    descriptions.append(f"a {value} {segment_type.lower()}")

        return descriptions if descriptions else ["default"]
    def _calculate_model_hash(self):
        """Calculate hash of AIMv2 model parameters for cache validation"""
        model_state = self.model.state_dict()
        model_bytes = pickle.dumps([(k, v.shape) for k, v in model_state.items()])
        return hashlib.md5(model_bytes).hexdigest()

    
    def analyze_segment(self, segment_image, segment_type):
        """
        Analyze an image segment using AIMv2.
        """
        try:
            # Generate descriptions for this segment type
            descriptions = self._generate_segment_descriptions(segment_type)
            
            # Process features
            features = self.process_features(
                image=segment_image,
                text=descriptions
            )
            
            if not features:
                return None

            # Calculate similarities
            similarities = np.dot(
                features['image_features'], 
                features['text_features'].T
            )
            
            # Format results
            results = []
            for idx, similarity in enumerate(similarities[0]):
                if similarity > 0.3:  # 30% threshold
                    results.append({
                        'description': descriptions[idx],
                        'confidence': float(similarity * 100),
                        'category': 'visual_match'
                    })

            return self._organize_results(results)

        except Exception as e:
            print(f"Error analyzing segment: {e}")
            return None
    def search_similar(self, query_features, k=1):
        """
        Search for similar features in the index.
        """
        try:
            if self.index.ntotal == 0:
                return []

            # Search using inner product similarity
            D, I = self.index.search(query_features, min(k, self.index.ntotal))
            
            # Format results
            results = []
            for distance, idx in zip(D[0], I[0]):
                if idx < len(self.feature_mapping):
                    result = {
                        'metadata': self.feature_mapping[idx],
                        'distance': float(distance),
                        'similarity_score': float((1 + distance) / 2) * 100
                    }
                    results.append(result)

            return sorted(results, key=lambda x: x['similarity_score'], reverse=True)

        except Exception as e:
            print(f"Error searching similar features: {e}")
            return []

    def search_similar_features(self, image=None, k=1):
        try:
            # 檢查索引狀態
            if self.index.ntotal == 0:
                print("Index is empty, attempting to load cached features...")
                if not self._load_cached_features():
                    print("No cached features available")
                    return []
                    
            if image is None:
                print("No image provided")
                return []
                
            # 圖像前處理和類型檢查
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
                
            # 載入文字特徵
            try:
                with h5py.File('text_features.h5', 'r') as f:
                    text_features = {}
                    for category in f.keys():
                        text_features[category] = {
                            'features': f[category]['features'][:],
                            'descriptions': json.loads(f[category]['descriptions'][()])
                        }
            except Exception as e:
                print(f"Error loading text features: {e}")
                text_features = None

            # 處理圖像特徵
            features = self.process_image_and_text(
                image=image,
                text_descriptions=["default"]
            )
                
            if features is None:
                return []
                
            # 整合文字特徵分析
            results = []
            if text_features:
                for category, data in text_features.items():
                    if category in ['Customer', 'Shopping_Behavior', 'Products']:
                        similarities = np.dot(features['image_features'], data['features'].T)
                        category_matches = list(zip(data['descriptions'], similarities[0] * 100))
                        top_matches = sorted(category_matches, key=lambda x: x[1], reverse=True)[:3]
                        
                        for desc, score in top_matches:
                            if score > 10:  # 設定閾值
                                results.append({
                                    'category': category,
                                    'description': desc,
                                    'similarity_score': float(score)
                                })

            
            # 搜索相似特徵
            D, I = self.index.search(features['image_features'], min(k, self.index.ntotal))
                
            # 組織結果
            final_results = []
            for distance, idx in zip(D[0], I[0]):
                if idx < len(self.feature_mapping):
                    result = {
                        'metadata': self.feature_mapping[idx],
                        'distance': float(distance),
                        'similarity_score': float((1 + distance) / 2) * 100,
                        'text_features': sorted(results, key=lambda x: x['similarity_score'], reverse=True)
                    }
                    final_results.append(result)
            sorted_results = sorted(final_results, key=lambda x: x['similarity_score'], reverse=True)
            
            
            # print(f"Found sorted_results : {sorted_results}")
            return sorted_results

        except Exception as e:
            print(f"Error searching similar features: {e}")
            traceback.print_exc()
            return []

    def analyze_with_text(self, image, text_descriptions):
        """Analyze image with specific text descriptions"""
        try:
            features = self.process_image_and_text(image, text_descriptions)
            if not features:
                return None

            # Calculate similarities
            similarities = 100.0 * np.dot(features['image_features'], features['text_features'].T)
            
            # Format results
            results = []
            for idx, similarity in enumerate(similarities[0]):
                if similarity > 30:  # Threshold for meaningful matches
                    results.append({
                        'description': text_descriptions[idx],
                        'confidence': float(similarity),
                        'category': 'visual_match'
                    })

            return self._organize_results(results)

        except Exception as e:
            print(f"Error analyzing with text: {e}")
            return None

    def store_features(self, identifier, features, metadata=None):
        """
        Store features in the index.
        """
        try:
            if 'image_features' not in features:
                return False
                
            self.index.add(features['image_features'])
            
            feature_info = {
                'identifier': identifier,
                'timestamp': int(time.time()),
                'index': len(self.feature_mapping)
            }
            
            if metadata:
                feature_info.update(metadata)
                
            self.feature_mapping.append(feature_info)
            return True
            
        except Exception as e:
            print(f"Error storing features: {e}")
            return False

    def _organize_results(self, results):
        """Organize analysis results"""
        if not results:
            return None

        organized = {
            'top_matches': sorted(results, key=lambda x: x['confidence'], reverse=True)[:5],
            'confidence': max(r['confidence'] for r in results),
            'details': defaultdict(list)
        }

        for result in results:
            organized['details'][result['category']].append(result)

        return organized

    
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

    def _generate_segment_descriptions(self, segment_type):
        """Generate text descriptions for segment analysis."""
        descriptions = []
        features = self.features_dict.get(segment_type, {})

        # Add basic description
        descriptions.append(f"a {segment_type.lower()}")

        # Add specific features
        for feature_type in ['types', 'colors', 'patterns']:
            if feature_type in features:
                for value in features[feature_type]:
                    descriptions.append(f"a {value} {segment_type.lower()}")

        # Special handling for face and hair
        if segment_type == "Face" and "attributes" in features:
            for category, values in features["attributes"].items():
                for value in values:
                    descriptions.append(f"a face with {value} features")

        elif segment_type == "Hair" and "styles" in features:
            for style in features["styles"]:
                if "colors" in features:
                    for color in features["colors"]:
                        descriptions.append(f"{color} {style} hair")

        return descriptions if descriptions else ["default"]
    
    def cleanup(self):
        """清理資源"""
        if hasattr(self, '_h5file') and self._h5file is not None:
            self._h5file.close()
            self._h5file = None