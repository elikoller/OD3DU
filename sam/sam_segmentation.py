import os 
import os.path as osp
from configs import sam_config
import torch
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"

checkpoint_path = sam_config.CHECKPOINT_PATH


