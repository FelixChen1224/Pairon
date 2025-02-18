import cv2
import numpy as np

class PersonVisualizer:
    def __init__(self):
        """Initialize the visualizer with distinctive color schemes for different segments"""
        self.segment_colors = {
            # Head area - warm colors
            "Face": (255, 200, 0),      # Bright golden
            "Hair": (255, 100, 0),      # Bright orange
            "Hat": (255, 0, 0),         # Pure red
            "Sunglasses": (200, 0, 0),  # Dark red
            
            # Upper body - cool colors
            "Upper-clothes": (0, 150, 255),  # Bright blue
            "Dress": (0, 0, 255),           # Pure blue
            
            # Lower body - green tones
            "Pants": (0, 255, 0),       # Pure green
            "Skirt": (0, 200, 100),     # Blue-green
            
            # Accessories - purple tones
            "Belt": (255, 0, 255),      # Magenta
            "Bag": (180, 0, 255),       # Purple
            "Scarf": (150, 0, 200),     # Deep purple
            
            # Footwear - unique colors
            "Left-shoe": (255, 255, 0),  # Yellow
            "Right-shoe": (0, 255, 255)  # Cyan
        }
    def _normalize_similarity(self, similarity):
        """
        Normalize similarity value to percentage format
        If value is already in percentage format (>1), return as is
        If value is in decimal format (<=1), convert to percentage
        """
        if similarity > 1:  # Already in percentage format
            return similarity
        return similarity * 100

    def draw_person_box(self, frame, person, display_features=True):
        """
        Draw bounding box and ID for a person.
        
        Args:
            frame: The video frame.
            person: A dictionary containing detection results.
            display_features: Whether to display the top features.
        """
        try:
            
            # 確保必要資訊存在
            if 'bbox' not in person:
                print("Warning: Missing bbox information")
                return
                
          

      
            
            x1, y1, x2, y2 = map(int, person['bbox'])
            name = person.get('name', 'Unknown')
            similarity = person.get('similarity_score', 0)
            
            # Draw bounding box
            box_color = (0, 255, 0) if similarity >= 80 else (0, 255, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            
            # Draw label with name and similarity
            label = f"{name} ({similarity:.2f}%)"
            
            # Calculate text size for background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Draw text background
            cv2.rectangle(frame, 
                        (x1, y1 - text_height - 10),
                        (x1 + text_width, y1),
                        box_color, -1)
            
            # Draw label text
            cv2.putText(frame, label, (x1, y1 - 5),
                    font, font_scale, (0, 0, 0), thickness)
            
            # 安全處理特徵資訊
            if display_features and 'top_features' in person:
                features = person['top_features']
                if isinstance(features, list) and features:
                    self._draw_feature_list(frame, (x1, y2), features)
                    
        except Exception as e:
            print(f"Error drawing person box: {e}")
    # 在 visualizer.py 的 PersonVisualizer 類中加入
    def _draw_feature_list(self, frame, start_pos, features):
        """
        Draw feature list below person box
        
        Args:
            frame: Video frame
            start_pos: (x, y) starting position
            features: List of (description, confidence) tuples
        """
        try:
            x, y = start_pos
            y += 20  # Add some padding from the box
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            
            for desc, conf in features[:3]:  # Only show top 3 features
                # Draw background rectangle
                text = f"{desc}: {conf:.1f}%"
                (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                cv2.rectangle(frame, 
                            (x, y - text_h - 2),
                            (x + text_w, y + 2),
                            (0, 0, 0), -1)
                
                # Draw text
                cv2.putText(frame, text,
                        (x, y),
                        font, font_scale, (255, 255, 255), thickness)
                
                y += text_h + 5  # Move down for next feature
                
        except Exception as e:
            print(f"Error drawing feature list: {e}")
                
    def draw_segment_overlay(self, frame, person, segment_name, info):
        """Draw segmentation overlay with feature information"""
        try:
            x1, y1, x2, y2 = person['bbox']
            mask = info['mask']
            
            if segment_name not in self.segment_colors:
                print(f"Warning: No color defined for segment {segment_name}")
                return
                
            # Get color for this segment
            segment_color = self.segment_colors[segment_name]
            
            # Ensure mask is properly sized
            if mask.shape[:2] != (y2-y1, x2-x1):
                mask = cv2.resize(mask.astype(np.uint8), (x2-x1, y2-y1))
            
            # Create colored overlay using the mask
            overlay = np.zeros((y2-y1, x2-x1, 3), dtype=np.uint8)
            for c in range(3):
                overlay[..., c] = mask * segment_color[c]
            
            # Apply overlay with stronger opacity
            alpha = 0.5  # Increased opacity for better visibility
            mask_area = frame[y1:y2, x1:x2]
            
            # Convert mask to 3D for proper broadcasting
            mask_3d = np.stack([mask] * 3, axis=-1)
            
            # Create blended image with enhanced contrast
            blended = np.where(
                mask_3d,
                cv2.addWeighted(mask_area, 1-alpha, overlay, alpha, 0),
                mask_area
            )
            
            frame[y1:y2, x1:x2] = blended
            
            # Draw feature labels if available
            if info.get('top_matches'):
                self.draw_feature_label(frame, mask, info['top_matches'][0], 
                                      segment_name, x1, y1)
                
        except Exception as e:
            print(f"Error drawing segment overlay for {segment_name}: {e}")

    def draw_feature_label(self, frame, mask, top_match, segment_name, x_offset, y_offset):
        """Draw feature label with enhanced visibility"""
        try:
            mask_indices = np.where(mask > 0)
            if len(mask_indices[0]) > 0:
                # Calculate center position for text
                center_y = int(np.mean(mask_indices[0])) + y_offset
                center_x = int(np.mean(mask_indices[1])) + x_offset
                
                # Prepare feature text
                feature_text = f"{segment_name}: {top_match['description']}"
                confidence = top_match.get('confidence', 0)
                if confidence:
                    feature_text += f" ({confidence:.1f}%)"
                
                # Get text size for background rectangle
                font_scale = 0.7  # Increase font scale for better visibility
                font_thickness = 2  # Increase font thickness for better visibility
                text_size = cv2.getTextSize(feature_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
                
                # Draw text background with padding
                padding = 6  # Increase padding for better visibility
                bg_x1 = center_x - padding
                bg_y1 = center_y - text_size[1] - padding
                bg_x2 = center_x + text_size[0] + padding
                bg_y2 = center_y + padding
                
                # Draw outer black rectangle for better visibility
                cv2.rectangle(frame, 
                            (bg_x1 - 1, bg_y1 - 1),
                            (bg_x2 + 1, bg_y2 + 1),
                            (0, 0, 0), -1)
                
                # Draw inner white rectangle
                cv2.rectangle(frame,
                            (bg_x1, bg_y1),
                            (bg_x2, bg_y2),
                            (255, 255, 255), -1)
                
                # Draw text with increased font scale and thickness
                cv2.putText(frame, feature_text, 
                           (center_x, center_y),
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)
                
        except Exception as e:
            print(f"Error drawing feature label: {e}")
            
    def _draw_top_features(self, frame, start_pos, top_features):
        """
        Draw the top feature descriptions below the bounding box.
        
        Args:
            frame: The video frame.
            start_pos: Starting position (x, y) for the text.
            top_features: A list of tuples (description, similarity).
        """
        x, y = start_pos
        for idx, (description, similarity) in enumerate(top_features[:5]):
            text = f"{description} ({similarity:.2f}%)"
            y += 15
            cv2.putText(frame, text, (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


    def draw_all(self, frame, analyzed_people):
        """Draw all visualization elements for analyzed people"""
        try:
            if frame is None:
                print("Error: Frame is None")
                return frame
                
            if analyzed_people is None:
                print("Error: No analyzed people to draw")
                return frame
            
            for person in analyzed_people:
                # Draw main bounding box and ID
                self.draw_person_box(frame, person)
                
                # Draw segmentation overlays
                if 'segment_info' in person:
                    for segment_name, info in person['segment_info'].items():
                        self.draw_segment_overlay(frame, person, segment_name, info)
            
            return frame
            
        except Exception as e:
            print(f"Error in draw_all: {e}")
            return frame

    def draw_legend(self, frame):
        """Draw color legend for segments"""
        try:
            legend_x = 10
            legend_y = 30
            box_width = 20
            box_height = 20
            text_offset = 25
            spacing = 25
            
            for segment_name, color in self.segment_colors.items():
                # Draw color box
                cv2.rectangle(frame,
                            (legend_x, legend_y),
                            (legend_x + box_width, legend_y + box_height),
                            color, -1)
                
                # Draw text
                cv2.putText(frame, segment_name,
                           (legend_x + text_offset, legend_y + box_height - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(frame, segment_name,
                           (legend_x + text_offset, legend_y + box_height - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                
                legend_y += spacing
                
        except Exception as e:
            print(f"Error drawing legend: {e}")
    
    

    def draw_detection(self, frame, detection_result):
        """繪製檢測結果和相似度信息"""
        x1, y1, x2, y2 = detection_result['bbox']
        
        # 繪製邊界框
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        
        # 準備顯示文本
        name = detection_result['name']
        similarity = detection_result.get('similarity_score', '')
        display_text = f"{name} ({similarity})"
        
        # 繪製文本背景
        text_size = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(frame, 
                     (int(x1), int(y1) - text_size[1] - 4),
                     (int(x1) + text_size[0], int(y1)),
                     (0, 255, 0), -1)
        
        # 繪製文本
        cv2.putText(frame, display_text,
                    (int(x1), int(y1) - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)