from flask import Flask, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


from video_processing import VideoProcessor

import cv2
import model_defs.models
from flask_cors import CORS
import numpy as np


# from  openclip_feature_processor import OpenCLIPFeatureProcessor
from processor.radio_feature_processor import RADIOFeatureProcessor


app = Flask(__name__)

CORS(app, resources={
    r"/*": {
        "origins": "*",  # 允許所有來源
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": "*"  # 允許所有 headers
    }
})

socketio = SocketIO(app, 
    cors_allowed_origins="*",  # 允許所有來源
    async_mode='threading'
)


crud = None





@app.route('/video/<filename>')
def video_source(filename):
    return send_from_directory('pic', filename, as_attachment=False)

@app.route('/image/<path:filename>')
def image_source(filename):
    return send_from_directory('pic/high_res_faces', filename, as_attachment=False)

@app.route('/<path:filename>')
def serve_image(filename):
    return send_from_directory('static/images', filename)

@app.route('/upload/video', methods=['POST'])
def upload_video():
    video = request.files['video']
    
    filename = video.filename
 
    save_path = os.path.join('pic/input_video_path', filename)

    global INPUT_VIDEO_PATH
    INPUT_VIDEO_PATH = save_path
    return jsonify({
        "message": "Video uploaded",
        "filename": filename
    }), 200


INPUT_VIDEO_PATH = 'pic/cover/V8.mp4'
OUTPUT_VIDEO_PATH = 'pic/output/output_tree.mp4'
FEATURES_JSON_PATH = 'features.json'
     

@app.route('/example', methods=['POST'])
def example():
    # input_video_path = 'pic/turn3.mp4'
    input_video_path = INPUT_VIDEO_PATH
    # input_video_path = 'b2.mp4'
    output_video_path = OUTPUT_VIDEO_PATH
    # features_json_path = 'features.json'
    features_json_path = FEATURES_JSON_PATH
    
    
    # 初始化 Face Processor
    # face_processor = RADIOFeatureProcessor(features_json_path)
    face_processor = RADIOFeatureProcessor()
    
    
    data = {
        "name_images": {
            # "professor": ["pic/people/professor.png"]
             
                "Person_A": ["pic/people/Hwang2.jpg"],
                "an": ["pic/people/an4.jpg"]
                
         
        }
    }      
    
    name_images = data.get('name_images', {})
    output_face_path = 'pic/faces/'
    
    if not os.path.exists(output_face_path):
        os.makedirs(output_face_path)
        
    # # 添加参考特征
    for name, img_paths in name_images.items():
        print(f"Processing images for {name}...")
        
        for img_path in img_paths:
            success = face_processor.add_reference_features(
                image_path=img_path,
                feature_type="face",
                identifier=name
            )
                        
            if success:
                print(f"Successfully processed {img_path} for {name}")
            else:
                print(f"Failed to process {img_path} for {name}")
    
   # Save index with consistent path
    index_base_path = 'face_features/index'
    os.makedirs('face_features', exist_ok=True)
    face_processor.save_index(index_base_path)
    
    # 创建 VideoProcessor 并传入保存的索引路径
    processor = VideoProcessor(
        input_path=input_video_path,
        output_video_path=output_video_path,
        socketio=socketio,
        features_json_path=features_json_path,
        index_path=index_base_path,  # 传入索引路径
        enable_debug_frame=False
    )
    processor.process_video()
    
    return jsonify({"message": "Example video processed"}), 200

if __name__ == '__main__':
    socketio.run(app, debug=True, port=5001)