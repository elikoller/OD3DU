# get into VLSG space for scan3r data info
from collections import Counter
import os
import os.path as osp
import sys
from tracemalloc import start
import cv2
from sklearn.utils import resample
from yaml import scan
ws_dir = osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))
print(ws_dir)
sys.path.append(ws_dir)
from utils import common, scan3r

import numpy as np
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms as T
from torchvision import transforms as tvf
import torchvision.transforms as T
import matplotlib.pyplot as plt
#import tyro
import time
import traceback
#import joblib
#import wandb
import argparse
from PIL import Image
from tqdm.auto import tqdm
from typing import Literal, Tuple, List, Union
from dataclasses import dataclass, field
# config
from configs import update_config, config
# Dino v2
from dinov2_utils import DinoV2ExtractFeatures
from dataclasses import dataclass
# @dataclass
# class LocalArgs:
#     """
#         Local arguments for the program
#     """
#     # Dino_v2 properties (parameters)
#     desc_layer: int = 31
#     desc_facet: Literal["query", "key", "value", "token"] = "value"
#     num_c: int = 32
#     # device
#     device = torch.device("cuda")
# larg = tyro.cli(LocalArgs)

"""
These features are only computed to see how close they are to the scenegraph features
"""
class Scan3rDinov2Generator():
    def __init__(self, cfg, split):
        self.cfg = cfg

       
        # 3RScan data info
        ## sgaliner related cfg
        self.split = split
        self.use_predicted = cfg.sgaligner.use_predicted
        self.sgaliner_model_name = cfg.sgaligner.model_name
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
        ## scans info

        """"""
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
        self.resplit = "resplit_" if cfg.data.resplit else ""
        ref_scans_split = np.genfromtxt(osp.join(self.scans_files_dir_mode, '{}_{}scans.txt'.format(split, self.resplit)), dtype=str)
        #print("ref scan split", ref_scans_split)
        self.all_scans_split = []
        ## get all scans within the split(ref_scan + rescan)
        for ref_scan in ref_scans_split:
            self.all_scans_split += self.refscans2scans[ref_scan]
        if self.rescan:
            self.scan_ids = self.all_scans_split
        else:
            self.scan_ids = ref_scans_split

        #print("scan ids", len(self.scan_ids))
        ## images info
        self.image_paths = {}
        for scan_id in self.scan_ids:
            self.image_paths[scan_id] = scan3r.load_frame_paths(self.scans_dir, scan_id)
        
        # model info
        self.model_name = cfg.model.name
        self.model_version = cfg.model.version
        ## 2Dbackbone
        self.num_reduce = cfg.model.backbone.num_reduce
        self.backbone_dim = cfg.model.backbone.backbone_dim
        self.img_rotate = cfg.data.img_encoding.img_rotate
        ## scene graph encoder
        self.sg_modules = cfg.sgaligner.modules
        self.sg_rel_dim = cfg.sgaligner.model.rel_dim
        self.attr_dim = cfg.sgaligner.model.attr_dim
        ## encoders
        self.patch_hidden_dims = cfg.model.patch.hidden_dims
        self.patch_encoder_dim = cfg.model.patch.encoder_dim
        self.obj_embedding_dim = cfg.model.obj.embedding_dim
        self.obj_embedding_hidden_dims = cfg.model.obj.embedding_hidden_dims
        self.obj_encoder_dim = cfg.model.obj.encoder_dim
        self.drop = cfg.model.other.drop
        
        # image preprocessing 
        self.image_w = self.cfg.data.img.w
        self.image_h = self.cfg.data.img.h
        self.image_resize_w = self.cfg.data.img_encoding.resize_w
        self.image_resize_h = self.cfg.data.img_encoding.resize_h
        self.img_rotate = self.cfg.data.img_encoding.img_rotate
        self.image_patch_w = self.cfg.data.img_encoding.patch_w
        self.image_patch_h = self.cfg.data.img_encoding.patch_h
        self.step = self.cfg.data.img.img_step
        
        ## out dir 
       
        self.proj_out_dir = osp.join(self.scans_files_dir, 'Features2D/segmented_patch', self.model_name)
        self.patch_out_dir = osp.join(self.scans_files_dir, 'Features2D/patch', self.model_name)

        common.ensure_dir(self.proj_out_dir)
        common.ensure_dir(self.patch_out_dir)
        
        self.log_file = osp.join(cfg.data.log_dir, "log_file_{}.txt".format(self.split))
        
    def register_model(self):
        
        desc_layer = 31
        desc_facet = "value"
        device = torch.device("cuda")
        self.device = torch.device("cuda")
        
        # Dinov2 extractor
        if "extractor" in globals():
            print(f"Extractor already defined, skipping")
        else:
            self.extractor = DinoV2ExtractFeatures("dinov2_vitg14", desc_layer,
                desc_facet, device=device)

        self.base_tf = tvf.Compose([
            tvf.ToTensor(),
            tvf.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
        ])
        
    def inference(self, imgs_tensor):
        feature = self.model.backbone(imgs_tensor)[-1]
        return feature
    
    

        
    def generateFeaturesEachScan(self, scan_id):
        # Initialize a dictionary to store features per frame and object ID
        patch_features = {}
        proj_patch_features = {}

        # Load image paths and frame indices
        img_paths = self.image_paths[scan_id]
        frame_idxs_list = list(img_paths.keys())
        frame_idxs_list.sort()

        #access gt already 

        gt_anno_all = scan3r.get_scan_gt_anno(self.data_root_dir, scan_id, self.image_patch_w, self.image_patch_h)

        for infer_step_i in range(0, len(frame_idxs_list) // self.inference_step + 1):
            start_idx = infer_step_i * self.inference_step
            end_idx = min((infer_step_i + 1) * self.inference_step, len(frame_idxs_list))
            frame_idxs_sublist = frame_idxs_list[start_idx:end_idx]

            for frame_idx in frame_idxs_sublist:

                img_path = img_paths[frame_idx]
                img = Image.open(img_path).convert('RGB')
                
                
                patch_size_w = int(self.image_w/self.image_patch_w)
                patch_size_h = int(self.image_h/self.image_patch_h)


                # Initialize dictionary for embeddings per object ID in the current frame
                patch_features[frame_idx] = {}
                proj_patch_features[frame_idx] = {}

                patches_matrix = [[None for _ in range(self.image_patch_w)] for _ in range(self.image_patch_h)]

                # iterate over patches within one image
                for patch_h_i in range(self.image_patch_h):
                    h_start = round(patch_h_i * patch_size_h)
                    h_end = round((patch_h_i+1) * patch_size_h)
                    for patch_w_j in range(self.image_patch_w):
                        w_start = round(patch_w_j * patch_size_w)
                        w_end = round((patch_w_j+1) * patch_size_w)
                       
                        patch_crop = img.crop((h_start, w_start, h_end, w_end))
                        #print("size of patch", patch_crop.size)
                        #define the resizing
                        h_new = int((self.image_resize_h // 14) * 14)
                        w_new = int((self.image_resize_w // 14) * 14)
                    
                        #this is the current patch
                        patch = patch_crop.resize((w_new, h_new), Image.BILINEAR)

    
                        # Optionally rotate patch if needed
                        if self.img_rotate:
                            patch = patch.transpose(Image.ROTATE_270)

                        # Convert patch to tensor and apply normalization
                        patch_pt = self.base_tf(patch)

                        # Prepare tensor for inference
                        img_tensors_list = [patch_pt]
                        imgs_tensor = torch.stack(img_tensors_list, dim=0).float().to(self.device)

                        # Perform inference to get the embedding for the patch
                        with torch.no_grad():
                            ret = self.extractor(imgs_tensor)  # [1, num_patches, desc_dim]

                        # store the matrix element into the matrix
                        patches_matrix[patch_h_i][patch_w_j] = ret[0].cpu().numpy()

                # the patch featrues are done at this step
                patch_features[frame_idx] = patches_matrix
                #print("patch features are done", frame_idx, " scene ", scan_id)
                #grpoup the patches into different object ids
                #access the gt_anno_2d for that frame_idx
                gt_anno = gt_anno_all[frame_idx]
                #check that the sizes indeed match else there might be an error when indexing
                

                #look which ids we have
                unique_ids = np.unique(gt_anno)

                #iterate through the ids
                for obj_id in unique_ids:
                    # find the indices where this id occurs in gt_anno
                    indices = np.argwhere(gt_anno == obj_id)
                    
                    # list to store patches for this id
                    patches_list = []
                    
                    # access the corresponding patches in patches_matrix
                    for idx in indices:
                        patch_h_i, patch_w_j = idx
                        patch_feature = patches_matrix[patch_h_i][patch_w_j]
                        #print("feature shape", patch_feature.shape)
                        patches_list.append(patch_feature)
                    
                    #get the mean of all the features
                    patches_array = np.array(patches_list)
                    #print("patches array shape", patches_array.shape)
                    mean_patches = np.mean(patches_array, axis=0) 
                    #print("mean patches array shape", mean_patches.shape) 

                    proj_patch_features[frame_idx][obj_id] = mean_patches

            #patche featrues is each feature per patch, proj patch features is for each object the corresponding features
            #print("everything is done for scan", scan_id)
            return patch_features, proj_patch_features

    
    
    def generateFeatures(self):
        img_num = 0
        self.feature_generation_time = 0.0
        #print("scanid in generate function", self.scan_ids)
        for scan_id in tqdm(self.scan_ids):
            with torch.no_grad():
                patch_features, proj_features = self.generateFeaturesEachScan(scan_id)
            img_num += len(patch_features)
            #save the patchwise features
            print("will save patches", scan_id)
            patch_out_file = osp.join(self.patch_out_dir, '{}.pkl'.format(scan_id))
            common.write_pkl_data(patch_features, patch_out_file)
            #save the projection features
            proj_out_file = osp.join(self.proj_out_dir, '{}.pkl'.format(scan_id))
            common.write_pkl_data(proj_features, proj_out_file)
        # log
        log_str = "Feature generation time: {:.3f}s for {} images, {:.3f}s per image\n".format(
            self.feature_generation_time, img_num, self.feature_generation_time / img_num)
        with open(self.log_file, 'a') as f:
            f.write(log_str)

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
    scan3r_gcvit_generator = Scan3rDinov2Generator(cfg, 'train')
    scan3r_gcvit_generator.register_model()
    scan3r_gcvit_generator.generateFeatures()
    # scan3r_gcvit_generator = Scan3rDinov2Generator(cfg, 'val')
    # scan3r_gcvit_generator.register_model()
    # scan3r_gcvit_generator.generateFeatures()
    # scan3r_gcvit_generator = Scan3rDinov2Generator(cfg, 'test')
    # scan3r_gcvit_generator.register_model()
    # scan3r_gcvit_generator.generateFeatures()
    
if __name__ == "__main__":
    main()