import cv2
import numpy as np
import torch
import model_defs.models
from datetime import timedelta
from deepface import DeepFace

# def get_face_embedding(face_img, output_path, img_name_prefix):
#     try:
#         face_img_copy = face_img.copy()
#         face_tensor = torch.from_numpy(face_img_copy).float().to(models.device)
#         face_tensor = face_tensor.permute(2, 0, 1)  # 調整維度順序為 (C, H, W)
#         face_tensor = face_tensor.unsqueeze(0)  # 增加一個維度

#         if face_img is not None:
#             img_embedding = model_defs.models.resnet(face_tensor)
#             return img_embedding.detach().cpu().numpy().flatten().tolist()
        
#         return None
#     except Exception as e:
#         return None
    
    
def get_face_embedding_deep(face_img, output_path, img_name_prefix):
    try:
        # face_img_copy = face_img.copy()
        # face_tensor = torch.from_numpy(face_img_copy).float().to(models.device)
        # face_tensor = face_tensor.permute(2, 0, 1)  # 調整維度順序為 (C, H, W)
        # face_tensor = face_tensor.unsqueeze(0)  # 增加一個維度

        if face_img is not None:
            img_embedding = DeepFace.represent(
                img_path = face_img,
                # enforce_detection = False,
                align= True,
                detector_backend='skip',
                model_name = 'VGG-Face'
            )
            
            return img_embedding[0]['embedding']
        return None
    except Exception as e:
        return None
    
def get_single_image_embedding(my_image,model, processor):
    image = processor(
            text = None,
            images = my_image,
            return_tensors="pt"
            )["pixel_values"].to('cuda')
    embedding = model.get_image_features(image)
    # convert the embeddings to numpy array
    embedding_as_np = embedding.cpu().detach().numpy().flatten().tolist()
    
    return embedding_as_np
        

def is_bbox_inside(big_bbox, small_bbox):
    big_x_min, big_y_min, big_x_max, big_y_max = big_bbox
    small_x_min, small_y_min, small_x_max, small_y_max = small_bbox

    return (big_x_min <= small_x_min <= big_x_max and
            big_y_min <= small_y_min <= big_y_max and
            big_x_min <= small_x_max <= big_x_max and
            big_y_min <= small_y_max <= big_y_max)

def video_info(video):
    fps = int(video.get(cv2.CAP_PROP_FPS))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    duration = total_frames / fps

    print(f"Video Information: FPS={fps}, Width={width}, Height={height}, Duration={timedelta(seconds=duration)}\n\n")

    return fps, width, height, total_frames, duration