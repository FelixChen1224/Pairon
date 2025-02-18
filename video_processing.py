import cv2
from tqdm import tqdm
from utils import video_info, get_single_image_embedding
import model_defs.models
import time
import base64
import os
import json
from db.track_info_db import TrackInfoDatabase
from PIL import Image
import numpy as np

from track_processing import TrackManager

import traceback
from processor.aimv2_feature_processor import AIMv2FeatureProcessor

from visualizer import PersonVisualizer

from datetime import datetime


class VideoProcessor:
    def __init__(self, input_path, output_video_path, socketio, features_json_path, index_path, enable_debug_frame=False):
        self.input_path = input_path
        self.output_video_path = output_video_path
        self.socketio = socketio
        self.model_manager = models.model_manager
        self.track_manager = TrackManager(features_json_path)
        self.track_db = TrackInfoDatabase()
        
        self.visualizer = PersonVisualizer()
        self.face_processor = RADIOFeatureProcessor()
        
        self.paerson_processor = AIMv2FeatureProcessor(
            features_json_path=features_json_path,
            cache_dir="face_feature_cache/aimv2"
        )
        
       # Initialize enhanced track_info_collection
        self.track_info_collection = {
            'tracks': {},
            'statistics': {
                'unique_persons': set(),
                'total_tracks': 0,
                'reappearances': 0,
                'processed_frames': 0
            }
        }
        
        # Add Track ID management
        self.next_track_id = 1
        self.active_tracks = {}
        self.track_timeout = 30000  # 30 seconds timeout
        
        # Load FAISS index
        if not self.face_processor.load_index(index_path):
            print(f"Failed to load FAISS index from {index_path}.")
    
    def process_video(self):
        """Process video and track objects"""
        video = cv2.VideoCapture(self.input_path)
        fps, width, height, total_frames, duration = video_info(video)

        frame_index = 0
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_video_path, fourcc, fps, (width, height))

        identified_persons = {}
        pending_identifications = {}
        faces_per_second = []
        all_track_ids = []
        
        try:
            with tqdm(total=total_frames, desc="Processing video") as pbar:
                while frame_index < total_frames:
                    ret, frame = video.read()
                    if not ret:
                        break

                    if frame_index % 1 == 0:
                        timestamp = f"{frame_index/fps:.2f}"
                        
                        # 处理帧并更新跟踪信息
                        results, frame = self._process_frame(
                            frame, frame_index, fps, timestamp,
                            identified_persons, pending_identifications,
                            faces_per_second, all_track_ids
                        )
                        
                        # 更新数据库
                        self._update_database(results, timestamp)
                        
                        pbar.update(1)

                    self._frame_callback(frame, pbar)
                    out.write(frame)
                    frame_index += 1

        finally:
            video.release()
            out.release()
            self.track_db.close()

        # 将输出视频转换为 MP4
        os.system(f"ffmpeg -y -i {self.output_video_path} -vcodec libx264 -f mp4 pic/output.mp4")

        # 准备和导出追踪分析
        try:
            print("\nExporting track analysis...")
            self._finalize_processing(identified_persons, faces_per_second)
            
            # 导出特征分析到单独的文件
            analysis = {
                'tracks': {},
                'statistics': self._prepare_json_serializable_data(self.track_info_collection['statistics'])
            }
            
            # 处理每个追踪对象的数据
            for track_id, track_data in self.track_info_collection['tracks'].items():
                # 计算时间统计
                timing_stats = {
                    'first_seen': track_data.get('appear'),
                    'last_seen': track_data.get('disappear'),
                    'frames_tracked': len(track_data.get('frame_detections', {}))
                }
                
                # 收集特征数据
                feature_collection = {
                    'text_features': {},
                    'recognition_scores': []
                }
                
                # 处理帧检测数据
                for frame_idx, detection in track_data.get('frame_detections', {}).items():
                    # 添加识别分数
                    if 'similarity_score' in detection:
                        feature_collection['recognition_scores'].append({
                            'frame': frame_idx,
                            'score': detection['similarity_score']
                        })
                    
                    # 收集文本特征
                    if 'text_features' in detection:
                        for category, features in detection['text_features'].items():
                            if category not in feature_collection['text_features']:
                                feature_collection['text_features'][category] = []
                            feature_collection['text_features'][category].extend(features)
                
                # 计算平均识别分数
                if feature_collection['recognition_scores']:
                    avg_score = sum(item['score'] for item in feature_collection['recognition_scores']) / \
                            len(feature_collection['recognition_scores'])
                else:
                    avg_score = 0
                
                # 组织track数据
                analysis['tracks'][track_id] = {
                    'person_info': {
                        'name': track_data.get('name'),
                        'average_recognition_score': avg_score
                    },
                    'timing': timing_stats,
                    'features': feature_collection,
                    'images': {
                        'first_appearance': track_data.get('appear_image_path'),
                        'last_appearance': track_data.get('disappear_image_path')
                    }
                }
            
            # 保存分析结果
            output_path = 'track_features_analysis.json'
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, indent=2, ensure_ascii=False)
            print(f"Feature analysis exported to {output_path}")
            
            # 发送处理完成通知
            self.socketio.emit("processing_complete", {
                "video_url": "http://127.0.0.1:5001/video/output.mp4",
                "analysis_file": output_path
            })
            
        except Exception as e:
            print(f"Error during export: {e}")
            traceback.print_exc()
        

    def _prepare_json_serializable_data(self, data):
        """將數據轉換為 JSON 可序列化的格式"""
        if isinstance(data, dict):
            return {k: self._prepare_json_serializable_data(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._prepare_json_serializable_data(item) for item in list(data)]
        elif isinstance(data, set):
            return list(data)
        elif isinstance(data, (int, float, str, bool, type(None))):
            return data
        else:
            return str(data)
    def _assign_track_id(self, box_dict, current_frame):
        """
        Assign track ID to detection based on position
        Args:
            box_dict: Dictionary containing box info with keys:
                - xyxy: Tuple of (x1, y1, x2, y2) coordinates 
                - conf: Detection confidence
                - cls: Class ID (0 for person)
            current_frame: Current frame number
        Returns:
            int: Assigned track ID
        """
        # Directly unpack the coordinates tuple
        x1, y1, x2, y2 = box_dict['xyxy']
        
        # Calculate center point of detection
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        best_track_id = None
        min_distance = float('inf')
        
        # Check existing tracks
        for track_id, track_info in list(self.active_tracks.items()):
            last_pos, last_frame = track_info
            
            # Remove inactive tracks
            if current_frame - last_frame > self.track_timeout:
                del self.active_tracks[track_id]
                continue
                
            # Calculate distance to last known position
            distance = ((center[0] - last_pos[0]) ** 2 + (center[1] - last_pos[1]) ** 2) ** 0.5
            if distance < min_distance and distance < 100:  # Distance threshold
                min_distance = distance
                best_track_id = track_id
        
        if best_track_id is None:
            # Create new track ID
            best_track_id = self.next_track_id
            self.next_track_id += 1
            
        # Update track position
        self.active_tracks[best_track_id] = (center, current_frame)
        return best_track_id




    def _process_frame_core(self, frame, frame_index, fps, timestamp, identified_persons, 
                          pending_identifications, faces_per_second, all_track_ids):
        try:
            # 執行人物與臉部偵測
            detect_time_start = time.time()
            body_results = self.model_manager.people(frame) 
            face_results = self.model_manager.face_model(frame)
            detect_time = time.time() - detect_time_start
            print(f'Detection Time: {detect_time:.4f}s')

            tracked_boxes = []
            face_img_paths = []
            detection_results = {}

            feature_time_start = time.time()

            # 處理每個人物偵測結果
            for body_box in body_results[0].boxes:
                detection_info = self._process_single_detection(
                    frame=frame,
                    frame_index=frame_index,
                    detected_box=body_box,
                    face_results=face_results,
                    identified_persons=identified_persons,
                    feature_time_start=feature_time_start
                )

                if detection_info:
                    person_name = detection_info['name']
                    
                    # 更新或添加檢測結果
                    if person_name in detection_results:
                        if detection_info['similarity_score'] > detection_results[person_name]['similarity_score']:
                            detection_results[person_name] = detection_info
                    else:
                        detection_results[person_name] = detection_info

                    # 記錄追蹤框和臉部圖片路徑
                    if detection_info.get('tracked_box'):
                        tracked_boxes.append(detection_info['tracked_box'])
                    if detection_info.get('face_path'):
                        face_img_paths.append(detection_info['face_path'])

            feature_time = time.time() - feature_time_start
            print(f'Feature Processing Time: {feature_time:.4f}s')

            tracking_time_start = time.time()
            
            # 更新追蹤資訊
            track_info, updated_track_ids = self._update_tracking(
                tracked_boxes, timestamp, face_img_paths, all_track_ids
            )

            tracking_time = time.time() - tracking_time_start
            print(f'Tracking Time: {tracking_time:.4f}s')

            render_time_start = time.time()
            
            # 繪製結果
            processed_frame = self._draw_results(frame.copy(), list(detection_results.values()))

            # 準備結果字典
            results_dict = self._prepare_results(
                tracked_boxes, 
                list(detection_results.values()), 
                face_img_paths,
                track_info, 
                all_track_ids
            )

            render_time = time.time() - render_time_start
            print(f'Rendering Time: {render_time:.4f}s')

            total_time = detect_time + feature_time + tracking_time + render_time
            print(f'Total Frame Processing Time: {total_time:.4f}s\n')

            return results_dict, processed_frame

        except Exception as e:
            print(f"Error in _process_frame_core: {e}")
            traceback.print_exc()
            return None, frame

    def _prepare_json_serializable_data(self, data):
        """將數據轉換為 JSON 可序列化的格式"""
        if isinstance(data, dict):
            return {k: self._prepare_json_serializable_data(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._prepare_json_serializable_data(item) for item in list(data)]
        elif isinstance(data, set):
            return list(data)
        elif isinstance(data, (int, float, str, bool, type(None))):
            return data
        else:
            return str(data)

    def _process_frame(self, frame, frame_index, fps, timestamp, identified_persons, 
                        pending_identifications, faces_per_second, all_track_ids):
            try:
                results_dict, processed_frame = self._process_frame_core(
                    frame, frame_index, fps, timestamp, identified_persons,
                    pending_identifications, faces_per_second, all_track_ids
                )
                
                if results_dict:
                    frame_time = float(timestamp)
                    
                    for detection in results_dict.get('valid_boxes', []):
                        track_id = str(detection['track_id'])
                        person_name = detection['name']
                        
                        # Initialize track if not exists
                        if track_id not in self.track_info_collection['tracks']:
                            self.track_info_collection['tracks'][track_id] = {
                                'name': person_name,
                                'timeline': {
                                    'first_appearance': frame_time,
                                    'last_appearance': frame_time,
                                    'total_frames': 0
                                },
                                'features': {
                                    'appearance': [],       # Visual appearance features
                                    'text_analysis': [],    # Text feature analysis results
                                    'recognition_scores': []  # Recognition confidence scores
                                },
                                'frame_data': {}  # Store per-frame feature data
                            }
                            self.track_info_collection['statistics']['total_tracks'] += 1
                            self.track_info_collection['statistics']['unique_persons'].add(person_name)
                        
                        track_info = self.track_info_collection['tracks'][track_id]
                        
                        # Update timeline
                        track_info['timeline']['last_appearance'] = frame_time
                        track_info['timeline']['total_frames'] += 1
                        
                        # Store frame-specific data
                        frame_features = {
                            'timestamp': frame_time,
                            'frame_index': frame_index,
                            'recognition': {
                                'type': detection['recognition_type'],
                                'score': detection['similarity_score']
                            }
                        }
                        
                        # Add text features if available
                        if 'text_features' in detection:
                            frame_features['text_features'] = detection['text_features']
                            track_info['features']['text_analysis'].append({
                                'frame': frame_index,
                                'features': detection['text_features']
                            })
                        
                        # Store frame data
                        track_info['frame_data'][frame_index] = frame_features
                        
                        # Update recognition scores
                        track_info['features']['recognition_scores'].append({
                            'frame': frame_index,
                            'score': detection['similarity_score']
                        })
                    
                    self.track_info_collection['statistics']['processed_frames'] += 1

                    # Prepare and emit data
                    emit_data = {
                        'frame_index': frame_index,
                        'timestamp': timestamp,
                        'results': self._prepare_json_serializable_data(results_dict),
                        # 'track_info': self._prepare_json_serializable_data(self.track_info_collection)
                    }
                    print(f"Processed frame {emit_data}")
                    self.socketio.emit('frame_processing_results', emit_data)
                
                return results_dict, processed_frame
                
            except Exception as e:
                print(f"Error in _process_frame: {e}")
                traceback.print_exc()
                return None, frame
    
    def get_track_summary(self):
        """Generate summary of tracked features and timeline"""
        summary = {}
        for track_id, track_data in self.track_info_collection['tracks'].items():
            summary[track_id] = {
                'name': track_data['name'],
                'tracking_duration': track_data['timeline']['last_appearance'] - 
                                  track_data['timeline']['first_appearance'],
                'total_frames': track_data['timeline']['total_frames'],
                'average_recognition_score': sum(score['score'] for score in 
                    track_data['features']['recognition_scores']) / 
                    len(track_data['features']['recognition_scores']),
                'feature_analysis': {
                    'total_text_features': len(track_data['features']['text_analysis'])
                }
            }
        return summary
        
    def _process_single_detection(self, frame, frame_index, detected_box, face_results, 
                            identified_persons, feature_time_start):
        """
        Process a single detection and return formatted results
        
        Args:
            frame: Current video frame
            frame_index: Index of current frame
            detected_box: Detected body bounding box
            face_results: Face detection results
            identified_persons: Dictionary of identified persons
            feature_time_start: Start time for feature processing
            
        Returns:
            dict: Detection information if successful, None otherwise
        """
        try:
            # Extract body box coordinates and confidence
            body_x1, body_y1, body_x2, body_y2 = map(int, detected_box.xyxy[0])
            conf = float(detected_box.conf[0].item())
            
            # Extract person image
            person_img = frame[body_y1:body_y2, body_x1:body_x2]
            if person_img.shape[0] * person_img.shape[1] < 1000:
                return None
                
            # Find face in body region
            face_img = None
            face_path = None
            for face_box in face_results[0].boxes:
                if face_box.conf[0].item() < 0.5:
                    continue

                fx1, fy1, fx2, fy2 = map(int, face_box.xyxy[0])
                # Check if face is within body region
                if (fx1 >= body_x1 and fx2 <= body_x2 and
                    fy1 >= body_y1 and fy2 <= body_y2):
                    face_img = frame[fy1:fy2, fx1:fx2]
                    face_filename = f"frame_{frame_index}.jpg"
                    face_path = os.path.join(self.face_processor.cache_dir, face_filename)
                    cv2.imwrite(face_path, face_img)
                    break
            
            # Assign track ID
            track_id = self._assign_track_id({
                "xyxy": (body_x1, body_y1, body_x2, body_y2),
                "conf": conf,
                "cls": 0
            }, frame_index)
            
            # Get recognition result
            recognition_result = self.track_manager.find_person(
                body_image=person_img,
                face_image=face_img,
                identified_persons=identified_persons
            )
    
          
            
            if recognition_result:
                return {
                    "name": str(recognition_result["name"]),
                    "similarity_score": float(recognition_result["similarity"]),
                    "track_id": int(track_id),
                    "recognition_type": str(recognition_result["recognition_type"]),
                    "text_features": recognition_result.get("text_features", {}),
                    "face_path": {"track_id": track_id, "face_img_path": face_path} if face_path else None
                }
                
            return None
            
        except Exception as e:
            print(f"Error processing single detection: {e}")
            traceback.print_exc()
            return None


    
    def _update_tracking(self, tracked_boxes, timestamp, face_img_paths, all_track_ids):
        track_info, updated_track_ids = self.track_manager.update_track_info(
            tracked_boxes, timestamp, face_img_paths
        )
        for track_id in updated_track_ids:
            if track_id not in all_track_ids:
                all_track_ids.append(track_id)
        return track_info, updated_track_ids

    def _draw_results(self, frame, face_results):
        if face_results:
            return self.visualizer.draw_all(frame.copy(), face_results)
        return frame.copy()
    
    
    def _process_single_detection(self, frame, frame_index, detected_box, face_results, 
                            identified_persons, feature_time_start):
        """
        Process a single detection and return formatted results with features
        """
        try:
            # Extract body box coordinates and confidence
            body_x1, body_y1, body_x2, body_y2 = map(int, detected_box.xyxy[0])
            conf = float(detected_box.conf[0].item())
            
            # Extract person image
            person_img = frame[body_y1:body_y2, body_x1:body_x2]
            if person_img.shape[0] * person_img.shape[1] < 1000:
                return None
                
            # Find face in body region and process face image
            face_img = None
            face_path = None
            for face_box in face_results[0].boxes:
                if face_box.conf[0].item() < 0.5:
                    continue

                fx1, fy1, fx2, fy2 = map(int, face_box.xyxy[0])
                if (fx1 >= body_x1 and fx2 <= body_x2 and
                    fy1 >= body_y1 and fy2 <= body_y2):
                    face_img = frame[fy1:fy2, fx1:fx2]
                    face_filename = f"frame_{frame_index}.jpg"
                    face_path = os.path.join(self.face_processor.cache_dir, face_filename)
                    cv2.imwrite(face_path, face_img)
                    break
            
            # Assign track ID
            track_id = self._assign_track_id({
                "xyxy": (body_x1, body_y1, body_x2, body_y2),
                "conf": conf,
                "cls": 0
            }, frame_index)
            
            # Get recognition result with features
            recognition_result = self.track_manager.find_person(
                body_image=person_img,
                face_image=face_img,
                identified_persons=identified_persons
            )
            
            if recognition_result:
                # Include all feature information in the result
                result = {
                    "name": str(recognition_result["name"]),
                    "similarity_score": float(recognition_result["similarity"]),
                    "bbox": (body_x1, body_y1, body_x2, body_y2),
                    "track_id": int(track_id),
                    "recognition_type": str(recognition_result["recognition_type"]),
                    "text_features": recognition_result.get("text_features", {}),
                    "top_matches": recognition_result.get("top_matches", []),
                    "confidence": conf,
                    "tracked_box": {
                        "track_id": track_id,
                        "xyxy": (body_x1, body_y1, body_x2, body_y2),
                        "confidence": conf,
                        "class": 0,
                        "text_features": recognition_result.get("text_features", {}),
                        "top_matches": recognition_result.get("top_matches", [])
                    }
                }
                
                if face_path:
                    result["face_path"] = {
                        "track_id": track_id,
                        "face_img_path": face_path
                    }
                    
                return result
                
            return None
            
        except Exception as e:
            print(f"Error processing single detection: {e}")
            traceback.print_exc()
            return None

    def _detect_face(self, person_img, frame_index):
        try:
            face_detections = self.model_manager.face_model(person_img)
            if len(face_detections[0].boxes) > 0 and face_detections[0].boxes[0].conf[0].item() >= 0.5:
                fx1, fy1, fx2, fy2 = map(int, face_detections[0].boxes[0].xyxy[0])
                if fy2 > fy1 and fx2 > fx1:
                    face_img = person_img[fy1:fy2, fx1:fx2]
                    face_filename = f"frame_{frame_index}.jpg"
                    face_path = os.path.join(self.face_processor.cache_dir, face_filename)
                    cv2.imwrite(face_path, face_img)
                    return {'face_img': face_img, 'face_path': face_path}
            return None
            
        except Exception as e:
            print(f"Error in _detect_face: {e}")
            traceback.print_exc()
            return None

    def _prepare_results(self, tracked_boxes, face_results, face_img_paths, 
                     track_info, all_track_ids):
        """
        Prepare results for frontend, removing unnecessary bbox data
        
        Args:
            tracked_boxes: List of tracked box information
            face_results: List of face detection results
            face_img_paths: List of face image paths
            track_info: Tracking information
            all_track_ids: List of all track IDs
            
        Returns:
            dict: Filtered results ready for frontend
        """
        try:
            # Create frontend-ready face results by excluding bbox
            frontend_face_results = []
            for result in face_results:
                filtered_result = {
                    "name": result["name"],
                    "similarity_score": result["similarity_score"],
                    "track_id": result["track_id"],
                    "recognition_type": result["recognition_type"],
                    "text_features": result.get("text_features", {}),
                }
                if result.get("face_path"):
                    filtered_result["face_path"] = result["face_path"]
                frontend_face_results.append(filtered_result)
            
            # Prepare final results dictionary
            results_dict = {
                'valid_boxes': frontend_face_results,
                'identified_persons': {},
                'face_img_paths': face_img_paths,
                'track_info': track_info,
                'track_ids': all_track_ids
            }
            
            return results_dict
            
        except Exception as e:
            print(f"Error in _prepare_results: {e}")
            traceback.print_exc()
            return {
                'valid_boxes': [],
                'identified_persons': {},
                'face_img_paths': [],
                'track_info': {},
                'track_ids': []
            }
    
                        
           

    def _update_database(self, results, timestamp):
        """
        Update database with frame processing results
        
        Args:
            results: Frame processing results
            timestamp: Current timestamp
        """
        if not results:
            return
            
        try:
            # Update tracking information for each valid detection
            for box in results['valid_boxes']:
                track_id = box['track_id']
                
                # Update base tracking information
                self.track_db.update_track_info(
                    track_id=track_id,
                    timestamp=timestamp,
                    is_new=track_id not in results['track_info']
                )
                
                # Update features if available
                if 'features' in box and box['features'] is not None:
                    self.track_db.update_features(
                        track_id=track_id,
                        feature_matches=box['features'].get('feature_matches', {}),
                        style_analysis=box['features'].get('style', {}),
                        top_features=box.get('top_features', [])
                    )
                    
        except Exception as e:
            print(f"Error updating database: {e}")
            traceback.print_exc()

    def _frame_callback(self, frame, pbar):
        """
        Handle frame processing callback and progress updates
        
        Args:
            frame: Current frame
            pbar: Progress bar instance
        """
        _, buffer = cv2.imencode('.jpg', frame)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Calculate remaining time
        remaining = (
            pbar.total - pbar.n
        ) / pbar.format_dict['rate'] if pbar.format_dict['rate'] and pbar.total else 0
        
        # Format time strings
        elapsed_str = pbar.format_interval(pbar.format_dict['elapsed'])
        minutes, seconds = divmod(remaining, 60)
        formatted_remaining = f"{int(minutes):02}:{int(seconds):02}"

        # Emit progress update
        self.socketio.emit(
            'processed_frame', {
                'frame': img_base64,
                'time': pbar.n,
                'total': pbar.total,
                'elapsed': elapsed_str,
                'remaining': formatted_remaining
            }
        )
    
    def _finalize_processing(self, identified_persons, faces_per_second):
        try:
            final_stats = {
                'total_unique_persons': len(self.track_info_collection['statistics']['unique_persons']),
                'avg_faces_per_second': float(np.mean(faces_per_second)) if faces_per_second else 0,
                'max_concurrent_faces': int(max(faces_per_second)) if faces_per_second else 0,
                'processing_completed': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            self.track_info_collection['statistics'].update(final_stats)

            # 保存完整的追蹤資訊
            with open('track_info.json', 'w', encoding='utf-8') as f:
                json.dump(
                    self._prepare_json_serializable_data(self.track_info_collection), 
                    f, 
                    indent=2, 
                    ensure_ascii=False
                )

            video_url = "http://127.0.0.1:5001/video/output.mp4"
            summary = {
                'video_url': video_url,
                'statistics': self._prepare_json_serializable_data(self.track_info_collection['statistics']),
                'identified_persons': {
                    str(track_id): {
                        'name': person['name'],
                        'similarity': float(person['similarity'])
                    }
                    for track_id, person in identified_persons.items()
                }
            }

            with open('processing_summary.json', 'w') as f:
                json.dump(summary, f, indent=2)

            self.socketio.emit("video_processed", {
                "video_url": video_url,
                "summary": summary,
                "track_info": self._prepare_json_serializable_data(self.track_info_collection)
            })

        except Exception as e:
            print(f"Error in _finalize_processing: {e}")
            traceback.print_exc()
            self.socketio.emit('error', {
                'status': 'error',
                'message': str(e)
            })
    
    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'track_db'):
            self.track_db.close()
    
    

    def draw_person_info(frame, person_result):
        """
        在视频帧上绘制人员信息
        """
        if 'bbox' not in person_result or 'display_info' not in person_result:
            return
        
        x1, y1, x2, y2 = map(int, person_result['bbox'])
        name = person_result['display_info']['name']
        similarity = person_result['display_info']['similarity']
        
        # 绘制边界框
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 准备显示文本
        display_text = f"{name} ({similarity})"
        
        # 计算文本大小
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(display_text, font, font_scale, thickness)
        
        # 绘制文本背景
        cv2.rectangle(frame, 
                    (x1, y1 - text_height - 10),
                    (x1 + text_width, y1),
                    (0, 255, 0),
                    -1)
        
        # 绘制文本
        cv2.putText(frame,
                    display_text,
                    (x1, y1 - 5),
                    font,
                    font_scale,
                    (0, 0, 0),
                    thickness)