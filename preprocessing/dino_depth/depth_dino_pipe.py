import argparse
import math
import itertools
from functools import partial
import urllib
#from models.GCVit.test import parse_args
import torch
import torch.nn.functional as F

import numpy as np
import PIL as Image
import os.path as osp
import cv2
import mmcv
from mmcv.runner import load_checkpoint
from collections import Counter
import os.path as osp
import sys
from tracemalloc import start
import cv2
import numpy as np
import math
import itertools
from functools import partial
from torchvision import transforms

import torch
import torch.nn.functional as F

from dinov2.eval.depth.models import build_depther

import urllib
from tqdm.auto import tqdm
from PIL import Image
from glob import glob
import matplotlib.pyplot as plt
ws_dir = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
print(ws_dir)
sys.path.append(ws_dir)
from utils import common


"""
this file is adapted from the demo notebook provided by facebook, found in the src folder
"""
# Check PyTorch CUDA availability
print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
print(torch.__version__)
# Print CUDA and cuDNN version
print(f"CUDA version: {torch.version.cuda}")
print(f"cuDNN version: {torch.backends.cudnn.version()}")
print("PyTorch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("Current device:", torch.cuda.current_device())
print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))






class CenterPadding(torch.nn.Module):
    def __init__(self, multiple):
        super().__init__()
        self.multiple = multiple

    def _get_pad(self, size):
        new_size = math.ceil(size / self.multiple) * self.multiple
        pad_size = new_size - size
        pad_size_left = pad_size // 2
        pad_size_right = pad_size - pad_size_left
        return pad_size_left, pad_size_right

    @torch.inference_mode()
    def forward(self, x):
        pads = list(itertools.chain.from_iterable(self._get_pad(m) for m in x.shape[:1:-1]))
        output = F.pad(x, pads)
        return output





class DinoSegmentor():
    def __init__(self, cfg, split):
        self.cfg = cfg
        #model info
        self.head_config_url = cfg.model.head_config_url
        self.head_checkpoint_url = cfg.model.head_checkpoint_url
        self.backbone_size = cfg.model.backbone_size 
        self.head_type = cfg.model.head_type
        # 3RScan data info
        ## sgaliner related cfg
        self.split = split
        self.use_predicted = cfg.sgaligner.use_predicted
        self.scan_type = cfg.sgaligner.scan_type
        ## data dir
        self.data_root_dir = cfg.data.root_dir
        scan_dirname = '' if self.scan_type == 'scan' else 'out'
        scan_dirname = osp.join(scan_dirname, 'predicted') if self.use_predicted else scan_dirname
        self.scans_dir = osp.join(cfg.data.root_dir, scan_dirname)
        self.scans_files_dir = osp.join(self.scans_dir, 'files')
        # self.mode = 'orig' if self.split == 'train' else cfg.sgaligner.val.data_mode
        # self.scans_files_dir_mode = osp.join(self.scans_files_dir, self.mode)
        self.scans_files_dir_mode = osp.join(self.scans_files_dir)
        self.scans_scenes_dir = osp.join(self.scans_dir, 'scenes')
        self.inference_step = cfg.data.inference_step


        #patch info 
        self.image_patch_w = self.cfg.data.img_encoding.patch_w
        self.image_patch_h = self.cfg.data.img_encoding.patch_h
      
       
        #scans info 
        self.rescan = cfg.data.rescan
        scan_info_file = osp.join(self.scans_files_dir, '3RScan.json')
        all_scan_data = common.load_json(scan_info_file)
        self.refscans2scans = {}
        self.scans2refscans = {}
        self.all_scans_split = []
        for scan_data in all_scan_data:
            ref_scan_id = scan_data['reference']
            self.refscans2scans[ref_scan_id] = [ref_scan_id]
            self.scans2refscans[ref_scan_id] = ref_scan_id
            for scan in scan_data['scans']:
                self.refscans2scans[ref_scan_id].append(scan['reference'])
                self.scans2refscans[scan['reference']] = ref_scan_id
                
        #take only the split file      
        self.resplit = "resplit_" if cfg.data.resplit else ""
        ref_scans_split = np.genfromtxt(osp.join(self.scans_files_dir_mode, '{}_{}scans.txt'.format(split, self.resplit)), dtype=str)
        #print("ref scan split", ref_scans_split)
        self.all_scans_split = []

        ## get all scans within the split(ref_scan + rescan)
        for ref_scan in ref_scans_split:
            #self.all_scans_split.append(ref_scan)
            # Check and add one rescan for the current reference scan
            rescans = [scan for scan in self.refscans2scans[ref_scan] if scan != ref_scan]
            if rescans:
                # Add the first rescan (or any specific rescan logic)
                self.all_scans_split.append(rescans[0])

         

        if self.rescan:
            self.scan_ids = self.all_scans_split
        else:
            self.scan_ids = ref_scans_split


        print("scan_id_length", len(self.scan_ids))




        #print("scan ids", len(self.scan_ids))
        ## images info
        self.image_paths = {}
        for scan_id in self.scan_ids:
            self.image_paths[scan_id] = self.load_frame_paths(self.scans_dir, scan_id)

        # image preprocessing 
        self.img_rotate = self.cfg.data.img_encoding.img_rotate
        self.step = self.cfg.data.img.img_step
        
        
       
      
        #output path for components
        self.out_dir_objects = '/media/ekoller/T7/Depth/DinoV2/'
        common.ensure_dir(self.out_dir_objects)
        
        self.log_file = osp.join(cfg.data.log_dir, "log_file_{}.txt".format(self.split))



    #helper functions
    #loads configs from the urs
    def load_config_from_url(self,url: str) -> str:
        with urllib.request.urlopen(url) as f:
            return f.read().decode()
        
    """
    Code cuplication with utils scan3r, Reason: the environment of this model is very lets say particular :) 
    """

    
    def load_frame_paths(self,data_dir, scan_id, skip=None):
        frame_idxs = self.load_frame_idxs(osp.join(data_dir, "scenes"), scan_id, skip)
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


    def create_depther(self,cfg, backbone_model, backbone_size, head_type):
        train_cfg = cfg.get("train_cfg")
        test_cfg = cfg.get("test_cfg")
        depther = build_depther(cfg.model, train_cfg=train_cfg, test_cfg=test_cfg)

        depther.backbone.forward = partial(
            backbone_model.get_intermediate_layers,
            n=cfg.model.backbone.out_indices,
            reshape=True,
            return_class_token=cfg.model.backbone.output_cls_token,
            norm=cfg.model.backbone.final_norm,
        )

        if hasattr(backbone_model, "patch_size"):
            depther.backbone.register_forward_pre_hook(lambda _, x: CenterPadding(backbone_model.patch_size)(x[0]))

        return depther


    def load_depth_model(self):
        # cfg_str = self.load_config_from_url(self.backbone_path)
        # cfg = mmcv.Config.fromstring(cfg_str, file_format=".py")
        print("--- load the backbone ---")
        #backbone_model = torch.hub.load(repo_or_dir="/cluster/scratch/ekoller/backbone/dinov2_vitg14_pretrain.pth", model=f"dinov2_{self.backbone_size}")
        #backbone_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=f"dinov2_{self.backbone_size}",  pretrained=True)
        backbone_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=f"dinov2_{self.backbone_size}")
        backbone_model.eval()
        backbone_model.cuda()
        print("--- load the backbone done ---")
        print("--- load the head ---")

        cfg_str = self.load_config_from_url(self.head_config_url)
        cfg = mmcv.Config.fromstring(cfg_str, file_format=".py")

        model = self.create_depther(
            cfg,
            backbone_model=backbone_model,
            backbone_size=self.backbone_size,
            head_type=self.head_type,
        )

        load_checkpoint(model, self.head_checkpoint_url, map_location="cpu")
        model.eval()
        model.cuda()
        print("--- head got loaded ---")
        return model

        
    def make_depth_transform(self):
        return transforms.Compose([
            transforms.ToTensor(),
            lambda x: 255.0 * x[:3],
            transforms.Normalize(mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375)),
        ])
    
    #for a scan do the inference for every frame // saves the entire image and a dictionary with the patches, ids, and boundingboxes
    def depth_each_scan(self, model, scan_id):
        # Load image paths and frame indices
        img_paths = self.image_paths[scan_id]
        frame_idxs_list = list(img_paths.keys())
        frame_idxs_list.sort()
        scan_result_path_color = osp.join(self.out_dir_objects,scan_id)
        common.ensure_dir(scan_result_path_color)
        


        #o over each frame index for the scan id
        for infer_step_i in range(0, len(frame_idxs_list) // self.inference_step + 1):
            start_idx = infer_step_i * self.inference_step
            end_idx = min((infer_step_i + 1) * self.inference_step, len(frame_idxs_list))
            frame_idxs_sublist = frame_idxs_list[start_idx:end_idx]

            #since we now have indices we can just access
            for frame_idx in frame_idxs_sublist:
                #get the image path
                img_path = img_paths[frame_idx]

                #get the image
                img = Image.open(img_path).convert('RGB')

                #img = img.resize((120, 50), Image.BILINEAR)

                #rotate it around if necessary
                if self.img_rotate:
                    img = img.transpose(Image.ROTATE_270)

        
                
                #make an np array
                #device = torch.device('cuda')
                #img_tensor = torch.from_numpy(np.array(img)).to(device)
                transform = self.make_depth_transform()
                scale_factor = 1
                rescaled_image = img.resize((scale_factor * img.width, scale_factor * img.height))
                transformed_image = transform(rescaled_image)
                # print("Original image size:", img.size)
                # print("Rescaled image size:", rescaled_image.size)
                # print("Transformed image shape:", transformed_image.shape)
                batch = transformed_image.unsqueeze(0).cuda() # Make a batch of one image


                #print("--- image got loaded ---")
                #less momry
                torch.cuda.empty_cache()
                with torch.inference_mode():
                    result = model.whole_inference(batch, img_meta=None, rescale=True)

            
                #safe the image as a colourful mask 
                depth_name = "frame-"+str(frame_idx)
                cpu_result = result.cpu().numpy()
                cpu_result = cpu_result.squeeze() 
                depth_image = (cpu_result - cpu_result.min()) / (cpu_result.max()- cpu_result.min()) * 65535
                depth_image = depth_image.astype(np.uint16)
                rotated_depth_image = cv2.rotate(depth_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                cv2.imwrite(osp.join(self.out_dir_objects, scan_id, depth_name) + '.pgm', rotated_depth_image)
                #Image.fromarray(rotated_depth_image).save(osp.join(self.out_dir_objects, scan_id,depth_name) + '.pgm')  # Save as .pgm


    #iterate over the scans and do the segmentation
    def depth_estim(self):
        model = self.load_depth_model()
        #import pdb; pdb.set_trace()
        #iterate over each scan which where selected in the initialization
        for scan_id in tqdm(self.scan_ids):
            with torch.no_grad():
                self.depth_each_scan(model,scan_id)
                print("scan id  is done ", scan_id)
                
            

def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess Scan3R')
    parser.add_argument('--config', type=str, default='', help='Path to the config file')
    return parser.parse_known_args()

def main():

    # get arguments
    args, _ = parse_args()
    cfg_file = args.config
    print(f"Configuration file path: {cfg_file}")

    from configs import config, update_config
    cfg = update_config(config, cfg_file, ensure_dir = False)

    #do it for the projections first
    # scan3r_gcvit_generator = Scan3rDinov2Generator(cfg, 'train', for_proj= True)
    # scan3r_gcvit_generator.register_model()
    # scan3r_gcvit_generator.generateFeatures()
    #also generate for the sam boundingboxes
    scan3r_segmentor = DinoSegmentor(cfg, 'test')
    scan3r_segmentor.depth_estim()
    # scan3r_gcvit_generator = Scan3rDinov2Generator(cfg, 'val')
    # scan3r_gcvit_generator.register_model()
    # scan3r_gcvit_generator.generateFeatures()
    # scan3r_gcvit_generator = Scan3rDinov2Generator(cfg, 'test')
    # scan3r_gcvit_generator.register_model()
    # scan3r_gcvit_generator.generateFeatures()
    
if __name__ == "__main__":
    main()
            
        
