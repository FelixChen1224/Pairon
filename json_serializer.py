class JSONSerializer:
    @staticmethod
    def serialize_numpy(obj):
        """Convert numpy types to Python native types"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    @staticmethod
    def serialize_detection(detection):
        """Serialize a single detection result"""
        return {
            "name": str(detection["name"]),
            "similarity_score": float(detection["similarity_score"]),
            "bbox": [float(x) for x in detection["bbox"]],
            "track_id": int(detection["track_id"]),
            "recognition_type": str(detection["recognition_type"]),
            "text_features": detection.get("text_features", {}),
            "confidence": float(detection["confidence"])
        }

    @staticmethod
    def serialize_frame_results(frame_results):
        """Serialize entire frame results"""
        return {
            "frame_index": int(frame_results["frame_index"]),
            "timestamp": str(frame_results["timestamp"]),
            "detections": [
                JSONSerializer.serialize_detection(d) 
                for d in frame_results["detections"]
            ],
            "timing": {
                k: float(v) 
                for k, v in frame_results["timing"].items()
            }
        }