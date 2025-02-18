import json
import time
import torch
import numpy as np
from PIL import Image
import cv2
import model_defs.models
import h5py
from visualizer import PersonVisualizer
import traceback
from processor.aimv2_feature_processor import AIMv2FeatureProcessor
from processor.radio_feature_processor import RADIOFeatureProcessor
import feature.feature_storage
from  feature.feature_description_processor import FeatureDescriptionProcessor
from  feature.text_feature_manager import TextFeatureGenerator


class TrackManager:
    def __init__(self, features_json_path, reset_db=True,use_previous_frame_features=True):
        # Initialize tracking information
        self.track_info = {}
        self.all_track_ids = []
        
        # Initialize feature processors
        self.feature_analyzer = AIMv2FeatureProcessor(features_json_path)

        self.model_manager = models.model_manager
        self.face_processor = RADIOFeatureProcessor()
        self.feature_description_processor = FeatureDescriptionProcessor(features_json_path)
        self.feature_storage =  feature.feature_storage.feature_storage # 使用共享管理器
        self.use_previous_frame_features = use_previous_frame_features
        self.previous_frame_features = {}
        self.active_tracks = {}
        self.base_url = "http://127.0.0.1:5001/"
        # Add text feature initialization
        self.text_features = TextFeatureGenerator.load_text_features('text_features.h5')
        
        # Store temporary feature analysis history
        self.track_features = {
            'individual': {},  
            'combined': {},    
            'style': {}       
            
        }
        
        # self.description_processor = StoreDescriptionGenerator("store-features.json")
        
        # Initialize visualizer
        self.visualizer = PersonVisualizer()
        
        # Comment out database initialization
        # try:
        #     self.track_db = TrackInfoDatabase()
        #     if reset_db:
        #         print("Resetting tracking database...")
        #         self.track_db.recreate_tables()
        #         print("Database tables have been reset.")
        #     else:
        #         print("Using existing database tables.")
        # except Exception as e:
        #     print(f"Error initializing database: {e}")
        #     raise
        
    def analyze_image_with_text_features(self, image, n=5):
        """
        Analyze image using stored text features and return top matches.
        
        Args:
            image: Input image (PIL Image, numpy array, or path)
            n: Number of top matches to return per category
            
        Returns:
            dict: Dictionary of categories with their top matches
        """
        try:
            # Process image to get features
            features = self.feature_analyzer.process_image_and_text(image=image)
            if not features or 'image_features' not in features:
                return {}
                
            # Load text features
            text_features = TextFeatureGenerator.load_text_features('text_features.h5')
            if not text_features:
                return {}
                
            # Calculate similarities for each category
            all_matches = {}
            for category, data in text_features.items():
                # Calculate similarities between image and text features
                similarities = np.dot(features['image_features'], data['features'].T)
                
                # Get top matches
                matches = []
                for idx, score in enumerate(similarities[0]):
                    if score > 0.3:  # 30% confidence threshold
                        matches.append({
                            'description': data['descriptions'][idx],
                            'confidence': float(score * 100)
                        })
                
                # Sort and get top N matches
                top_matches = sorted(matches, key=lambda x: x['confidence'], reverse=True)[:n]
                if top_matches:
                    all_matches[category] = top_matches
            
            return all_matches
            
        except Exception as e:
            print(f"Error analyzing image with text features: {e}")
            traceback.print_exc()
            return {}

        
    # In track_processing.py

    def update_track_info(self, boxes, timestamp, face_img_paths):
        try:
            # Initialize statistics if not exists
            if not hasattr(self, 'track_info_collection'):
                self.track_info_collection = {
                    'statistics': {
                        'unique_persons': set(),  # Use a set for unique persons
                        'total_tracks': 0,
                        'reappearances': 0
                    }
                }

            # Map track IDs to face image paths
            face_img_path_dict = {item['track_id']: item['face_img_path'] 
                                for item in face_img_paths if 'track_id' in item}
            current_frame_track_ids = []

            for box in boxes:
                if not isinstance(box, dict):
                    continue

                track_id = box.get('track_id')
                person_name = box.get('name', f'Unknown_{track_id}')
                
                if track_id is None:
                    continue
                    
                current_frame_track_ids.append(track_id)
                face_img_path = face_img_path_dict.get(track_id)

                # Update statistics
                self.track_info_collection['statistics']['unique_persons'].add(person_name)  # Use add() for sets
                
                if track_id not in self.track_info:
                    # Initialize new tracking target
                    self.track_info[track_id] = {
                        'appear': timestamp,
                        'disappear': None,
                        'appear_image_path': self.base_url +face_img_path,
                        'disappear_image_path': None,
                        'features': box.get('text_features', {}),
                        'recognition_type': box.get('recognition_type'),
                        'top_matches': box.get('top_matches', []),
                        'name': person_name
                    }
                    
                    if track_id not in self.all_track_ids:
                        self.all_track_ids.append(track_id)
                        self.track_info_collection['statistics']['total_tracks'] += 1
                else:
                    # Update existing tracking target
                    current_info = self.track_info[track_id]
                    
                    if current_info['disappear'] is not None:
                        # Track reappeared
                        self.track_info_collection['statistics']['reappearances'] += 1
                        current_info.update({
                            'appear': timestamp,
                            'disappear': None,
                            'appear_image_path': self.base_url + face_img_path if face_img_path else current_info['appear_image_path'],
                            'disappear_image_path': None
                        })

                    # Update features if new ones available
                    if 'text_features' in box:
                        current_info['features'] = box['text_features']
                    if 'top_matches' in box:
                        current_info['top_matches'] = box['top_matches']

            # Handle disappeared tracks
            for track_id in self.all_track_ids:
                if track_id not in current_frame_track_ids:
                    if self.track_info[track_id]['disappear'] is None:
                        self.track_info[track_id]['disappear'] = timestamp
                        if track_id in face_img_path_dict:
                            self.track_info[track_id]['disappear_image_path'] = face_img_path_dict[track_id]

            return self.track_info, self.all_track_ids

        except Exception as e:
            print(f"Error in update_track_info: {e}")
            traceback.print_exc()
            return {}, []

    def get_statistics(self):
        """Get current statistics"""
        if not hasattr(self, 'track_info_collection'):
            return {
                'unique_persons': 0,
                'total_tracks': 0,
                'reappearances': 0
            }
            
        stats = self.track_info_collection['statistics']
        return {
            'unique_persons': len(stats['unique_persons']),  # Get count of unique persons from set
            'total_tracks': stats['total_tracks'],
            'reappearances': stats['reappearances']
        }

    def search_person_features(self, image, confidence_threshold=70.0, k=5):
        """
        使用 RADIOFeatureProcessor 搜索人物特徵並返回匹配結果
        
        Args:
            image: 輸入圖像（PIL Image、numpy array 或圖像路徑）
            confidence_threshold: 置信度閾值，低於此值的匹配將被過濾（預設：70.0）
            k: 返回的最大匹配數量（預設：5）
            
        Returns:
            list: 包含匹配結果的列表，每個結果包含以下資訊：
                - name: 識別的人名
                - similarity_score: 相似度分數
                - metadata: 原始特徵資料
        """
        try:
            # 使用 RADIOFeatureProcessor 搜索特徵
            matches = self.face_processor.search_similar_features(
                image=image,
                k=k
            )
            
            if not matches:
                return []
                
            # 過濾並格式化結果
            filtered_results = []
            for match in matches:
                similarity_score = match['similarity_score']
                
                if similarity_score >= confidence_threshold:
                    result = {
                        'name': match['metadata']['identifier'],
                        'similarity_score': similarity_score,
                        'recognition_type': 'face',
                        'metadata': match['metadata']
                    }
                    filtered_results.append(result)
                    
            return sorted(filtered_results, key=lambda x: x['similarity_score'], reverse=True)
            
        except Exception as e:
            print(f"Error in search_person_features: {e}")
            import traceback
            traceback.print_exc()
            return []

    def search_person_features_batch(self, images, confidence_threshold=70.0, k=5):
        """
        批次處理多張圖像的特徵搜索
        
        Args:
            images: 圖像列表（每個元素可以是 PIL Image、numpy array 或圖像路徑）
            confidence_threshold: 置信度閾值（預設：70.0）
            k: 每張圖像返回的最大匹配數量（預設：5）
            
        Returns:
            list: 每張圖像的匹配結果列表
        """
        try:
            batch_results = []
            for img in images:
                result = self.search_person_features(
                    image=img,
                    confidence_threshold=confidence_threshold,
                    k=k
                )
                batch_results.append(result)
                
            return batch_results
            
        except Exception as e:
            print(f"Error in search_person_features_batch: {e}")
            import traceback
            traceback.print_exc()
            return []
            
    def search_or_create_person(self, body_image, face_image=None, bbox=None, identified_persons=None):
        try:
            timing_stats = {}
            total_start_time = time.time()
            
            # 處理圖像特徵
            image_start_time = time.time()
            body_pil = Image.fromarray(cv2.cvtColor(body_image, cv2.COLOR_BGR2RGB))
             
            # 使用 search_similar_features 方法
            body_matches = self.face_processor.search_similar_features(image=body_pil, k=3)
            text_matches = self.feature_analyzer.search_similar_features(image=body_pil, k=3)
            timing_stats['image_processing'] = time.time() - image_start_time
            
            # 如果找到匹配的人物
            if body_matches and body_matches[0]['similarity_score'] > 70:
                best_match = body_matches[0]
                best_text_match = text_matches[0] if text_matches else None
                result = {
                    'name': best_match['metadata']['identifier'],
                    'similarity_score': best_match['similarity_score'],
                    'similarity_text_score': best_text_match['similarity_score'] if best_text_match else 0,
                    'recognition_type': 'body',
                    'body_match': best_match,
                    'face_match': None,
                    'bbox': bbox,
                    'text_features': best_text_match.get('text_features', []) if best_text_match else [],
                    'timing_stats': timing_stats
                }
                
                # 儲存分析結果到特徵儲存器
                if self.use_previous_frame_features:
                    self.feature_storage.previous_frame_features[result['name']] = {
                        'text_features': best_match.get('text_features', []),
                        'timestamp': time.time()
                    }
                    
            else:
                # 創建新的 ID 並處理特徵
                new_track_id = f"person_{len(self.active_tracks) + 1}"
                
                # 添加新的參考特徵
                self.feature_analyzer.add_reference_features(
                    image_path=body_image,
                    feature_type="body",
                    identifier=new_track_id
                )
                
                # 取得文字特徵分析結果
                features = self.feature_analyzer.search_similar_features(image=body_pil, k=1)
                text_features = features[0].get('text_features', []) if features else []
                
                result = {
                    'name': new_track_id,
                    'similarity_score': 100.0,
                    'similarity_text_score': 0,
                    'recognition_type': 'new',
                    'body_match': None,
                    'face_match': None,
                    'bbox': bbox,
                    'text_features': text_features,
                    'timing_stats': timing_stats
                }
            
            timing_stats['total_time'] = time.time() - total_start_time
            return result

        except Exception as e:
            print(f"Error in search_or_create_person: {e}")
            traceback.print_exc()
            return None

    def _process_category_results(self, results, descriptions):
        """處理單個類別的特徵結果"""
        if results is None or 'probabilities' not in results:
            return []
            
        probs = results['probabilities']
        # 取得該類別中置信度大於 30% 的特徵
        features = []
        for idx, prob in enumerate(probs[0]):
            confidence = float(prob * 100)
            if confidence > 30:
                features.append((descriptions[idx], confidence))
        
        # 按置信度排序
        return sorted(features, key=lambda x: x[1], reverse=True)
    

    
    # def find_person(self, body_image, face_image=None, identified_persons=None):
    #     """
    #     識別人物並提取特徵描述。整合身體辨識、臉部辨識和特徵比對。

    #     Args:
    #         body_image: 身體圖像
    #         face_image: 臉部圖像(可選)
    #         identified_persons: 已識別人物字典

    #     Returns:
    #         dict: 包含識別結果的字典
    #     """
    #     try:
    #         start_time = time.time()
            
    #         # 生成並處理特徵描述
    #         all_descriptions = self.description_processor.generate_multiple_descriptions()
    #         results = self.feature_analyzer.process_image_and_text(
    #             image=body_image,
    #             text_descriptions=all_descriptions
    #         )
    #         process_image_and_text_time = time.time() - start_time
    #         print(f"process_image_and_text time: {process_image_and_text_time:.2f} seconds")
    #         print(results['text_features'])
            
    #         if results is None or 'probabilities' not in results:
    #             return None

    #         # 使用 process_image_and_text 計算的機率
    #         probs = results['probabilities']
            
    #         # 取得前5個最高機率的特徵描述
    #         top_indices = probs[0].argsort()[-5:][::-1]
    #         top_features = [
    #             (all_descriptions[idx], float(probs[0][idx] * 100))
    #             for idx in top_indices
    #         ]
    #         print(f'Top features: {top_features}')
    #         # 處理身體特徵匹配
    #         body_pil = Image.fromarray(cv2.cvtColor(body_image, cv2.COLOR_BGR2RGB))
    #         body_matches = self.feature_analyzer.search_similar_features(image=body_pil, k=3)
    #         print(f"search_similar_features time: {time.time() - start_time:.2f} seconds")
    #         match_time = time.time()
            
            
    #         if body_matches:
    #             best_body_match = body_matches[0]
    #             best_body_similarity = best_body_match['similarity_score']
    #             person_name = best_body_match['metadata'].get('identifier', 'Unknown')
                
    #             # 檢查已識別人物
    #             if identified_persons:
    #                 for track_id, person_info in identified_persons.items():
    #                     if person_info['name'] == person_name and best_body_similarity > 70:
    #                         return {
    #                             'name': person_name,
    #                             'similarity': best_body_similarity,
    #                             'recognition_type': 'body',
    #                             'body_match': best_body_match,
    #                             'face_match': None,
    #                             'top_features': top_features
    #                         }

    #         # 處理臉部辨識
    #         if face_image is not None:
    #             face_pil = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
    #             face_matches = self.face_processor.search_similar_features(image=face_pil, k=3)

    #             if face_matches:
    #                 best_face_match = face_matches[0]
    #                 best_face_similarity = best_face_match['similarity_score']
    #                 person_name = best_face_match['metadata'].get('identifier', 'Unknown')

    #                 # 跳過已識別的人物
    #                 if identified_persons and person_name in [info['name'] for info in identified_persons.values()]:
    #                     return None

    #                 if best_face_similarity > 50:
    #                     # 添加參考特徵
    #                     self.feature_analyzer.add_reference_features(
    #                         image_path=body_image,
    #                         feature_type="body",
    #                         identifier=person_name
    #                     )
    #                     # self.face_processor.add_reference_features(
    #                     #     image_path=body_image,
    #                     #     feature_type="body",
    #                     #     identifier=person_name
    #                     # )
    #                     print(f"add_reference_features time: {time.time() - match_time:.2f} seconds")
    #                     return {
    #                         'name': person_name,
    #                         'similarity': best_face_similarity,
    #                         'recognition_type': 'face',
    #                         'body_match': None,
    #                         'face_match': best_face_match,
    #                         'top_features': top_features
    #                     }

    #         return None

    #     except Exception as e:
    #         print(f"Error in find_person: {e}")
    #         traceback.print_exc()
    #         return None
    
    def find_person(self, body_image, face_image=None, identified_persons=None):
        """
        Modified find_person method that includes text feature analysis
        """
        try:
            # Start timing
            start_time = time.time()
            
            
            print(f"Text feature analysis time: {time.time() - start_time:.4f}s")
            
            # Convert body image for processing
            body_pil = Image.fromarray(cv2.cvtColor(body_image, cv2.COLOR_BGR2RGB))
            
            # Search for similar body features
            body_matches = self.feature_analyzer.search_similar_features(image=body_pil, k=1)
            print(f"Feature search time: {time.time() - start_time:.4f}s")
            
            if body_matches:
                best_body_match = body_matches[0]
                person_name = best_body_match['metadata'].get('identifier', 'Unknown')
                
                # Check if person is already identified
                if identified_persons:
                    for track_id, person_info in identified_persons.items():
                        if person_info['name'] == person_name and best_body_match['similarity_score'] > 70:
                            
                            # Process body image with text features
                            text_feature_matches = self.analyze_image_with_text_features(body_image)
                            print(f"Text feature analysis time: {time.time() - start_time:.4f}s")
                        
                            recognition_result = {
                                'name': person_name,
                                'similarity': best_body_match['similarity_score'],
                                'recognition_type': 'body',
                                'body_match': best_body_match,
                                'face_match': None,
                                'text_features': text_feature_matches  # Add text features to result
                            }
                            print(f"Recognition result: {recognition_result}")
                            
                            return recognition_result
                            
                       
            # Process face if available
            if face_image is not None:
                face_pil = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
                face_matches = self.face_processor.search_similar_features(image=face_pil, k=3)
                
                if face_matches:
                    best_face_match = face_matches[0]
                    best_face_similarity = best_face_match['similarity_score']
                    person_name = best_face_match['metadata'].get('identifier', 'Unknown')
                    
                    # Skip if already identified
                    if identified_persons and person_name in [info['name'] for info in identified_persons.values()]:
                        return None
                        
                    if best_face_similarity > 50:
                        # Add reference features
                        self.feature_analyzer.add_reference_features(
                            image_path=body_image,
                            feature_type="body",
                            identifier=person_name
                        )
                        # Process body image with text features
                        text_feature_matches = self.analyze_image_with_text_features(body_image)
                        print(f"Text feature analysis time: {time.time() - start_time:.4f}s")
                        recognition_result = {
                            'name': person_name,
                            'similarity': best_face_similarity,
                            'recognition_type': 'face',
                            'body_match': None,
                            'face_match': best_face_match,
                            'text_features': text_feature_matches  # Add text features to result
                        }
                        print(f"Recognition result: {recognition_result}")
                        
                        return recognition_result
            return None
            
        except Exception as e:
            print(f"Error in find_person: {e}")
            traceback.print_exc()
            return None

    
       

    def calculate_similarity(self, features1, features2):
        """
        Calculate similarity between two feature sets
        
        Args:
            features1: First feature set
            features2: Second feature set
            
        Returns:
            float: Similarity score (0-100)
        """
        try:
            if 'image_features' in features1 and 'image_features' in features2:
                sim = np.dot(
                    features1['image_features'].flatten(),
                    features2['image_features'].flatten()
                )
                return float(sim * 100)
            return 0
        except Exception as e:
            print(f"Error calculating similarity: {e}")
            return 0

    def analyze_and_store_features(self, track_id, image, segments=None):
        """
        Analyze and store features for a tracked person
        
        Args:
            track_id: Track ID
            image: Image to analyze
            segments: Optional segmentation masks
        """
        try:
            # Process basic features
            features = self.feature_analyzer.process_image_and_text(
                image=image,
                text_descriptions=["default"]  # Use default for initial storage
            )
            
            if features is None:
                return None
                
            # Store basic features
            metadata = {
                'track_id': track_id,
                'timestamp': time.time(),
                'has_segments': segments is not None
            }
            self.feature_analyzer.store_features(track_id, features, metadata)
            
            # Process segments if available
            if segments:
                segment_results = {}
                for segment_name, segment_data in segments.items():
                    if np.any(segment_data['mask']):
                        segment_image = Image.fromarray(segment_data['masked_image'])
                        segment_results[segment_name] = self.analyze_segment(
                            segment_image, 
                            segment_name
                        )
                        
                        
                        
                return {
                    'basic_features': features,
                    'segment_features': segment_results
                }
            
            return {'basic_features': features}
            
        except Exception as e:
            print(f"Error in analyze_and_store_features: {e}")
            traceback.print_exc()
            return None
        
    def expand_bbox(self, coords, frame_height, frame_width, padding=0):
        """
        Expands bounding box coordinates while respecting frame boundaries.
        
        Args:
            coords (tensor or array): Original coordinates [x1, y1, x2, y2]
            frame_height (int): Height of the frame
            frame_width (int): Width of the frame
            padding (int): Number of pixels to expand in each direction
            
        Returns:
            tuple: Expanded coordinates (x1, y1, x2, y2)
        """
        # Convert coordinates to integers
        x1, y1, x2, y2 = map(int, coords)
        
        # Expand coordinates with padding
        x1_expanded = max(0, x1 - padding)
        y1_expanded = max(0, y1 - padding)
        x2_expanded = min(frame_width, x2 + padding)
        y2_expanded = min(frame_height, y2 + padding)
        
        return x1_expanded, y1_expanded, x2_expanded, y2_expanded

    def analyze_segment(self, segment_image, segment_name):
        """
        Analyze a segment using AIMv2 feature processor
        
        Args:
            segment_image: PIL Image of the segment
            segment_name: Name of the segment (e.g., 'Face', 'Hair')
        """
        try:
            return self.feature_analyzer.analyze_segment(segment_image, segment_name)
        except Exception as e:
            print(f"Error analyzing segment: {e}")
            return None

    def extract_single_person_features(self, frame, frame_index, box, track_id, 
                                     identified_persons, similarity_threshold=0.6):
        """
        Extract features for a single person
        
        Args:
            frame: Current video frame
            frame_index: Frame index
            box: Detection box
            track_id: Track ID
            identified_persons: Dictionary of identified persons
            similarity_threshold: Threshold for feature matching
        """
        try:
            # Initialize and image processing
            person_data = self._prepare_person_image(frame, box, frame_index, track_id=None)
            if not person_data:
                return None

            # Process image with AIMv2
            if track_id in identified_persons:
                # Use text description if person is identified
                person_name = identified_persons[track_id]['name']
                text_descriptions = [
                    f"a photo of {person_name}",
                    f"a person named {person_name}",
                    f"an image of {person_name}"
                ]
            else:
                # Use default text for unknown persons
                text_descriptions = ["default"]

            # Extract features
            features = self.feature_analyzer.process_image_and_text(
                image=person_data['rgb_frame'],
                text_descriptions=text_descriptions
            )

            if features is None:
                return None

            # Store features for future use
            metadata = {
                'frame_index': frame_index,
                'bbox': person_data['bbox']
            }
            self.feature_analyzer.store_features(track_id, features, metadata)

            return {
                'track_id': track_id,
                'bbox': person_data['bbox'],
                'features': features
            }

        except Exception as e:
            print(f"Error extracting person features: {e}")
            traceback.print_exc()
            return None
            

    
    

    def _process_identified_person(self, person_data, track_id, identified_persons):
        """處理已識別的人物"""
        try:
            # 添加到特徵數據庫
            self.feature_analyzer.add_human_features(
                person_data['features'],
                person_data['segmented_path'],
                track_id,
                identified_persons[track_id]['name']
            )
            
            # 進行特徵分析
            return self._analyze_person_features(
                person_data,
                {
                    'track_id': track_id,
                    'name': identified_persons[track_id]['name'],
                    'distance': 1 - identified_persons[track_id]['similarity']
                }
            )
        except Exception as e:
            print(f"Error processing identified person: {e}")
            return None

    def _organize_segment_info(self, label_masks, analysis_results):
        """Helper method to organize segment information"""
        segment_info = {}
        for label_name, mask_data in label_masks.items():
            if np.any(mask_data['mask']):
                segment_results = analysis_results[label_name]
                
                segment_info[label_name] = {
                    'mask': mask_data['mask'],
                    'bbox': mask_data.get('bbox'),
                    'confidence': segment_results['confidence'],
                    'top_matches': segment_results['top_matches'][:3]
                }
        return segment_info
        
    
    def analyze_detected_people(self, boxes, frame, frame_index, identified_persons):
        """
        Analyze detected people and match with known identities
        
        Args:
            boxes: Detection boxes
            frame: Current frame
            frame_index: Frame index
            identified_persons: Dictionary of identified persons
        """
        SIMILARITY_THRESHOLD = 70.0
        analysis_results = []
        
        for box in boxes:
            if box.cls == 0 and box["id"] is not None:
                track_id = int(box["id"][0])
                
                # Extract features for current person
                person_features = self.extract_single_person_features(
                    frame, frame_index, box, track_id, identified_persons
                )
                
                if person_features:
                    # If person is already identified
                    if track_id in identified_persons:
                        person_name = identified_persons[track_id]['name']
                        # Use specific text descriptions for verification
                        features = self.feature_analyzer.process_image_and_text(
                            image=person_features['bbox_image'],
                            text_descriptions=[
                                f"a photo of {person_name}",
                                f"a person named {person_name}"
                            ]
                        )
                        if features:
                            similarities = np.dot(
                                features['image_features'], 
                                features['text_features'].T
                            )
                            max_similarity = float(similarities.max() * 100)
                            
                            if max_similarity >= SIMILARITY_THRESHOLD:
                                analysis_results.append({
                                    'track_id': track_id,
                                    'name': person_name,
                                    'similarity_score': max_similarity,
                                    'bbox': person_features['bbox']
                                })
                    else:
                        # Search in stored features
                        stored_matches = self.feature_analyzer.search_similar_features(
                            person_features['bbox_image'],
                            k=1
                        )
                        if stored_matches and stored_matches[0]['similarity_score'] >= SIMILARITY_THRESHOLD:
                            best_match = stored_matches[0]
                            analysis_results.append({
                                'track_id': track_id,
                                'name': best_match['metadata']['identifier'],
                                'similarity_score': best_match['similarity_score'],
                                'bbox': person_features['bbox']
                            })
                        
                        # Store features for future matching
                        self.analyze_and_store_features(
                            track_id,
                            person_features['bbox_image']
                        )
        
        return analysis_results



    
   
    
    
    def summarize_tracking_info(self, identified_persons, faces_per_second, socketio):
        """
        Summarize tracking information and send to frontend
        
        Args:
            identified_persons: Dictionary of identified persons
            faces_per_second: List of face counts per second
            socketio: Socket.IO instance for real-time updates
        """
        person_track_info = []
        
        # Initialize stats as an empty dictionary
        stats = {}  # {{ edit_1 }}

        for track_id, track_data in self.track_info.items():
            name = identified_persons[track_id]['name'] if track_id in identified_persons else f"Unknown {track_id}"
            
            # Get feature summary from database
            feature_summary = self.track_db.get_track_summary(track_id)
            
            # Calculate tracking duration
            appear_time = float(track_data['appear'])
            disappear_time = float(track_data['disappear']) if track_data['disappear'] is not None else appear_time
            
            track_info = {
                'track_id': track_id,
                'name': name,
                'appear': track_data['appear'],
                'disappear': track_data['disappear'],
                'appear_image_path': track_data['appear_image_path'],
                'disappear_image_path': track_data['disappear_image_path'],
                'duration': disappear_time - appear_time
            }
            
            # Add feature information
            if feature_summary:
                track_info.update({
                    'features': feature_summary['features'],
                    'trajectory': feature_summary['trajectory']
                })
            
            person_track_info.append(track_info)
            
            # Output tracking information
            disappear = track_data['disappear'] if track_data['disappear'] is not None else "None"
            print(f"Track ID {track_id} ({name}) appeared at {track_data['appear']} and disappeared at {disappear}")

        # Calculate statistics
        stats = {  # {{ edit_2 }}
            'total_tracks': len(person_track_info),
            'avg_faces_per_second': np.mean(faces_per_second) if faces_per_second else 0,
            'max_concurrent_faces': max(faces_per_second) if faces_per_second else 0,
            'total_unique_persons': len(set(info['name'] for info in person_track_info 
                                          if not info['name'].startswith('Unknown')))
        }

        final_output = {
            'tracks': person_track_info,
            'statistics': stats
        }

        # Save to file
        with open('track_info.json', 'w') as json_file:
            json.dump(final_output, json_file, indent=2) 

        # Send to frontend
        socketio.emit('track_info', {'track_info': json.dumps(final_output)})
        
        
    def analyze_full_image(self, segments, track_id=None):
        """
        Analyze all segments in an image
        
        Args:
            segments: Dictionary of image segments
            track_id: Optional track ID for feature storage
        """
        try:
            results = {}
            
            for segment_name, segment_data in segments.items():
                if np.any(segment_data['mask']):
                    # Convert segment to PIL image
                    segment_image = Image.fromarray(segment_data['masked_image'])
                    
                    # Get text descriptions for this segment
                    descriptions = self._get_segment_descriptions(segment_name)
                    
                    # Process with AIMv2
                    features = self.feature_analyzer.process_image_and_text(
                        image=segment_image,
                        text_descriptions=descriptions
                    )
                    
                    if features:
                        # Calculate similarities
                        similarities = np.dot(
                            features['image_features'], 
                            features['text_features'].T
                        )
                        
                        # Store results
                        segment_results = []
                        for idx, similarity in enumerate(similarities[0]):
                            if similarity > 0.3:  # 30% threshold
                                segment_results.append({
                                    'description': descriptions[idx],
                                    'confidence': float(similarity * 100),
                                    'category': 'visual_match'
                                })
                        
                        if segment_results:
                            results[segment_name] = {
                                'matches': segment_results,
                                'features': features
                            }
                            
                            # Store features if track_id provided
                            if track_id:
                                metadata = {
                                    'segment_name': segment_name,
                                    'timestamp': time.time()
                                }
                                self.feature_analyzer.store_features(
                                    track_id,
                                    features,
                                    metadata
                                )
            
            return results
            
        except Exception as e:
            print(f"Error in analyze_full_image: {e}")
            traceback.print_exc()
            return None

    def _get_segment_descriptions(self, segment_name):
        """Generate appropriate text descriptions for a segment"""
        if segment_name in self.feature_analyzer.features_dict:
            features = self.feature_analyzer.features_dict[segment_name]
            descriptions = [f"a {segment_name.lower()}"]
            
            # Add specific features
            for feature_type in ['types', 'colors', 'patterns']:
                if feature_type in features:
                    for value in features[feature_type]:
                        descriptions.append(f"a {value} {segment_name.lower()}")
            
            return descriptions
        
        return ["default"]


