import dinov2.eval.segmentation_m2f.models.segmentors 
import mmcv
from mmcv.runner import load_checkpoint
import math
import itertools
from functools import partial
import urllib
import torch
import torch.nn.functional as F
import mmseg
from mmseg.apis import init_segmentor, inference_segmentor
import numpy as np
import PIL as Image
import os.path as osp
import cv2
import numpy as np
import dinov2.eval.segmentation.utils.colormaps as colormaps
import urllib
import os.path as osp
import cv2
from PIL import Image
from glob import glob
import matplotlib.pyplot as plt


# Check PyTorch CUDA availability
print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
print(torch.__version__)
# Print CUDA and cuDNN version
print(f"CUDA version: {torch.version.cuda}")
print(f"cuDNN version: {torch.backends.cudnn.version()}")
print("mmsec version", mmseg.__version__)
print("PyTorch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("Current device:", torch.cuda.current_device())
print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

#helperfunctions

def load_config_from_url(url: str) -> str:
    with urllib.request.urlopen(url) as f:
        return f.read().decode()
    
    
def render_segmentation(segmentation_logits, dataset):
    DATASET_COLORMAPS = {
    "ade20k": colormaps.ADE20K_COLORMAP,
    "voc2012": colormaps.VOC2012_COLORMAP,
}
    colormap = DATASET_COLORMAPS[dataset]
    colormap_array = np.array(colormap, dtype=np.uint8)
    segmentation_values = colormap_array[segmentation_logits + 1]
    return Image.fromarray(segmentation_values)

"""
Code cuplication with utils scan3r, Reason: the environment of 
"""

    
def load_frame_paths(self,data_dir, scan_id, skip=None):
    frame_idxs = load_frame_idxs(osp.join(data_dir, "scenes"), scan_id, skip)
    img_folder = osp.join(data_dir, "scenes", scan_id, 'sequence')
    img_paths = {}
    for frame_idx in frame_idxs:
        img_name = "frame-{}.color.jpg".format(frame_idx)
        img_path = osp.join(img_folder, img_name)
        img_paths[frame_idx] = img_path
    return img_paths

def load_frame_idxs(self,data_dir, scan_id, skip=None):
    # num_frames = len(glob(osp.join(data_dir, scan_id, 'sequence', '*.jpg')))

    # if skip is None:
    #     frame_idxs = ['{:06d}'.format(frame_idx) for frame_idx in range(0, num_frames)]
    # else:
    #     frame_idxs = ['{:06d}'.format(frame_idx) for frame_idx in range(0, num_frames, skip)]
    # return frame_idxs
    
    frames_paths = glob(osp.join(data_dir, scan_id, 'sequence', '*.jpg'))
    frame_names = [osp.basename(frame_path) for frame_path in frames_paths]
    frame_idxs = [frame_name.split('.')[0].split('-')[-1] for frame_name in frame_names]
    frame_idxs.sort()

    if skip is None:
        frame_idxs = frame_idxs
    else:
        frame_idxs = [frame_idx for frame_idx in frame_idxs[::skip]]
    return frame_idxs





#load the model
    
DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"
CONFIG_URL = f"{DINOV2_BASE_URL}/dinov2_vitg14/dinov2_vitg14_ade20k_m2f_config.py"
CHECKPOINT_URL = f"{DINOV2_BASE_URL}/dinov2_vitg14/dinov2_vitg14_ade20k_m2f.pth"

cfg_str = load_config_from_url(CONFIG_URL)
cfg = mmcv.Config.fromstring(cfg_str, file_format=".py")

model = init_segmentor(cfg)
print("--- model start evaluation ---")
load_checkpoint(model, CHECKPOINT_URL, map_location="cpu")
model.cpu()
model.eval()

print("--- model got evaluated ---")


#also let the whole thing run
torch.cuda.empty_cache()


frame_numbers = ["000008", "000007"]
scan_ids = ["02b33e01-be2b-2d54-93fb-4145a709cec5","fcf66d8a-622d-291c-8429-0e1109c6bb26"]

for frame_number in frame_numbers:
    for scan_id in scan_ids:
        img_path = osp.join("/local/home/ekoller/R3Scan/scenes", scan_id, "sequence/frame-{}.color.jpg".format(frame_number))

        img = Image.open(img_path).convert('RGB')
        img = img.transpose(Image.ROTATE_270)


        array = np.array(img)


        # #actually take the url
        # def load_image_from_url(url: str) -> Image:
        #     with urllib.request.urlopen(url) as f:
        #         return Image.open(f).convert("RGB")


        # EXAMPLE_IMAGE_URL = "https://dl.fbaipublicfiles.com/dinov2/images/example.jpg"


        # image = load_image_from_url(EXAMPLE_IMAGE_URL)
        # array = np.array(image)[:, :, ::-1] # BGR
        print("inputsize" ,array.shape)

        print("--- image got loaded ---")
        #less momry
        with torch.no_grad():
            segmentation_logits = inference_segmentor(model, array)[0]

        print("--- image gets rendered ----")
        segmented_image = render_segmentation(segmentation_logits, "ade20k")
        segmented_image_np = np.array(segmented_image)
        segmented_img = cv2.rotate(segmented_image_np, cv2.ROTATE_90_COUNTERCLOCKWISE)
        #to do assert the folder
        cv2.imwrite(osp.join("/local/home/ekoller/tmp_result", scan_id,"segmentedimg_{}.jpg".format(frame_number)), segmented_img)
        print("--- image got saved ---")

