import torch
import open_clip
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from gfpgan import GFPGANer
from transformers import AutoProcessor, AutoModel

class ModelManager:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._initialize_models()
        
    def _initialize_models(self):
        # Clear VRAM
        torch.cuda.empty_cache()
        
        # Initialize all models
        # self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.people = YOLO("model_defs/yolo11l.pt", verbose=False).to(self.device)
        self.face_model = YOLO("model_defs/yolov11m-face.pt", verbose=False).to(self.device)
        self.yolo_seg = YOLO("model_defs/yolo11m-seg.pt", verbose=False).to(self.device)
        self.seg_processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
        self.seg_model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes").to(self.device)

        self.gfpgan = GFPGANer(
            model_path='gfpgan/GFPGANv1.4.pth',
            upscale=2,
            arch='clean',
            channel_multiplier=2,
            device=self.device
        )
        print("GFPGAN model loaded successfully!")
        
        # Initialize CLIP model
        # self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
        #     'hf-hub:UCSC-VLAA/ViT-L-14-CLIPS-Recap-DataComp-1B', 
        #     # pretrained='dfn5b',
        #     # pretrained='metaclip_fullcc',
        #     device=self.device
        # )
        # self.clip_model = self.clip_model.eval()
        # # 'ViT-H-14-quickgelu'  'metaclip_fullcc' soso
        # #'hf-hub:apple/DFN5B-CLIP-ViT-H-14-384'
        # #'hf-hub:Marqo/marqo-fashionSigLIP' good
        # #hf-hub:Marqo/marqo-ecommerce-embeddings-L not good better than fashion
        # #'ViT-H-14-378-quickgelu', pretrained='dfn5b'
    
        # #'hf-hub:UCSC-VLAA/ViT-L-16-HTxt-Recap-CLIP' same as apple
        # self.tokenizer = open_clip.get_tokenizer('hf-hub:UCSC-VLAA/ViT-L-14-CLIPS-Recap-DataComp-1B')

               # Initialize AIMv2 model
           # Initialize AIMv2 model with correct configuration
        self.aimv2_processor = AutoProcessor.from_pretrained(
            "apple/aimv2-large-patch14-224-lit",
        )
        
        self.aimv2_model = AutoModel.from_pretrained(
            "apple/aimv2-large-patch14-224-lit",
            trust_remote_code=True
        ).to(self.device)


        
    def get_yolo_seg_model(self):
        return self.yolo_seg
        
    def get_seg_processor(self):
        return self.seg_processor
    
    def get_seg_model(self):
        return self.seg_model
    
    # def get_clip_model(self):
    #     return self.clip_model
        
    # def get_clip_preprocess(self):
    #     return self.clip_preprocess
        
    # def get_tokenizer(self):
    #     return self.tokenizer
    
    def get_gfpgan(self):
        return self.gfpgan
    
    
    def is_gfpgan_available(self):
        """
        Check if GFPGAN model is available.
        
        Returns:
            bool: True if GFPGAN is available, False otherwise
        """
        return self.gfpgan is not None
    
    def get_aimv2_processor(self):
        return self.aimv2_processor
    
    def get_aimv2_model(self):
        return self.aimv2_model

# 創建全局 ModelManager 實例
model_manager = ModelManager()