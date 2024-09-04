# get into VLSG space for scan3r data info
from collections import Counter
import os
import os.path as osp
import pickle
import sys
import h5py
from tracemalloc import start
import cv2
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.manifold import TSNE
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
    def __init__(self, cfg, split, for_proj = False, for_dino_seg = False):
        self.cfg = cfg

        #to know how to change the directory name based on the input images
        self.proj = for_proj
        self.dino = for_dino_seg
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
        # self.rescan = cfg.data.rescan
        # scan_info_file = osp.join(self.scans_files_dir, '3RScan.json')
        # all_scan_data = common.load_json(scan_info_file)
        # self.refscans2scans = {}
        # self.scans2refscans = {}
        # for scan_data in all_scan_data:
        #     ref_scan_id = scan_data['reference']
        #     self.refscans2scans[ref_scan_id] = [ref_scan_id]
        #     self.scans2refscans[ref_scan_id] = ref_scan_id
        #     if self.rescan:
        #         for scan in scan_data['scans']:
        #             self.refscans2scans[ref_scan_id].append(scan['reference'])
        #             self.scans2refscans[scan['reference']] = ref_scan_id
        # self.resplit = "resplit_" if cfg.data.resplit else ""
        # if self.rescan:
        #     ref_scans = np.genfromtxt(osp.join(self.scans_files_dir_mode, '{}_{}scans.txt'.format(split, self.resplit)), dtype=str)
        #     self.scan_ids = []
        #     for ref_scan in ref_scans:
        #         self.scan_ids += self.refscans2scans[ref_scan]
        # else:
        #     self.scan_ids = np.genfromtxt(osp.join(self.scans_files_dir_mode, '{}_{}scans.txt'.format(split, self.resplit)), dtype=str)


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


        """
        differenciate between the dinofeatures for the current scene (for_poj = True) and the ones for the rescans (for_dino_seg= True)
        """
        if self.for_proj:
            self.scan_ids = ref_scans_split #only take the reference scans
        
        if self.for_dino_seg:
            self.scan_ids = [scan for scan in self.all_scans_split if scan not in ref_scans_split]#only take the rescans



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
        if(self.proj):
            self.out_dir = osp.join("/media/ekoller/T7/Features2D_lowdim/projection", self.model_name, "patch_{}_{}".format(self.image_patch_w,self.image_patch_h))

        if(self.dino):
            self.out_dir = osp.join("/media/ekoller/T7/Features2D_lowdim/dino_segmentation", self.model_name, "patch_{}_{}".format(self.image_patch_w,self.image_patch_h))

        common.ensure_dir(self.out_dir)
        
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
    


    #accesses the bounding boxes of the segmentation computed by dino and saved in the same format as the ones for the projection will be calculated
    def bounging_boxes_for_dino_segmentation(self, data_dir,scan_id, frame_number):
        #access the precomputed boundingboxes
        # Load data from the pickle file
        # file_path = osp.join("/media/ekoller/T7/Segmentation/DinoV2/objects","{}.pkl".format(scan_id))
        # with open(file_path, 'rb') as file:
        #     object_info = pickle.load(file)

        # object_boxes = object_info[frame_number]
        # return object_boxes

        #load data from the h5 file
        file_path = osp.join("/media/ekoller/T7/Segmentation/DinoV2/objects","{}.h5".format(scan_id))
        with h5py.File(file_path, 'r') as hdf_file:
            if str(frame_number) in hdf_file:
                bounding_boxes = []
                frame_group = hdf_file[str(frame_number)]
                
                for bbox_name in frame_group.keys():
                    bbox_group = frame_group[bbox_name]
                    object_id = bbox_group.attrs['object_id']
                    bbox = bbox_group['bbox'][:]  # Read as a NumPy array
                    #create the same object as is used for the projection
                    bounding_boxes.append({
                        'object_id': object_id,
                        'bbox': bbox.tolist() # Convert NumPy array to list)
                      }  
                    )
                # Output the data for the specific frame
                return bounding_boxes
        

       

    #using the ground truth projection, create boundingboxes for each object :)
    def bounding_boxes_for_projection(self,data_dir, scan_id, frame_number):
        #access the projection

        proj_rgb= osp.join(data_dir, "files/gt_projection", "obj_id", scan_id,"frame-" + frame_number +".jpg")
        #print("proj file", proj_rgb)
        obj_mat = cv2.imread(proj_rgb, cv2.IMREAD_UNCHANGED)
        img_height, img_width= obj_mat.shape
        new_id = np.zeros_like(obj_mat)
        patch_width = int(self.image_w/self.image_patch_w)
        patch_height = int(self.image_h/self.image_patch_h)
        
        for h in range(0, img_height, patch_height):
            for w in range(0, img_width, patch_width):
                patch = obj_mat[h:h+patch_height, w:w+patch_width]
                # flatten the array to 1d
                flattened_patch = patch.flatten()
                # Find the most common value
                value_counts = Counter(flattened_patch)
                most_common_value = value_counts.most_common(1)[0][0]
                # Fill the patch with the most common color
                new_id[h:h+patch_height, w:w+patch_width] = most_common_value


        #compute the boundingboxes based on that new obj_id_mask
        bounding_boxes = []
        unique_ids = np.unique(new_id)
    
        #make the boundingboxes for the ids which got recognized
        for obj_id in unique_ids:
            # Create mask for current object ID
            mask = (new_id == obj_id)

            # Find bounding box coordinates
            rows, cols = np.nonzero(mask)
            if len(rows) > 0 and len(cols) > 0:
                min_row, max_row = np.min(rows) - (patch_height), np.max(rows) + (patch_height)
                min_col, max_col = np.min(cols) - (patch_width), np.max(cols) + (patch_width)

                # Calculate height and width
                height = max_row - min_row + 1
                width = max_col - min_col +1

                # Store bounding box information
                bounding_boxes.append({
                    'object_id': obj_id,
                    'bbox': [min_col, min_row, width, height]
                })

        return bounding_boxes
    
        
    def generateFeaturesEachScan(self, scan_id):
        # Initialize a dictionary to store features per frame and object ID
        frame_features = {}


        # Load image paths and frame indices
        img_paths = self.image_paths[scan_id]
        frame_idxs_list = list(img_paths.keys())
        frame_idxs_list.sort()

        #we need the average of each thing
        averages = []
        #we need the max-pooling
        maxes = []
        #we need the ids per point to keep track
        all_ids = []

        for infer_step_i in range(0, len(frame_idxs_list) // self.inference_step + 1):
            start_idx = infer_step_i * self.inference_step
            end_idx = min((infer_step_i + 1) * self.inference_step, len(frame_idxs_list))
            frame_idxs_sublist = frame_idxs_list[start_idx:end_idx]


            for frame_idx in frame_idxs_sublist:
                img_path = img_paths[frame_idx]
                img = Image.open(img_path).convert('RGB')
                dat_dir = self.data_root_dir

                # Load bounding boxes and patch IDs for the current frame
                #path_proj_bboxes = osp.join(self.data_root_dir, "bboxes", scan_id, "gt_projection", f"frame-{frame_idx}.npy")
                if self.proj:
                    #get the projection boundingboxes
                    bboxes = self.bounding_boxes_for_projection(dat_dir, scan_id, frame_idx)

                if self.dino:
                    #get the boundingboxes for the seggmentation with dino data
                    bboxes = self.bounging_boxes_for_dino_segmentation(dat_dir,scan_id, frame_idx)

                # if we are in the dino seg we need a differently structured dictionary
                if self.dino:
                    frame_features[frame_idx] = {}
                #we need the mean, max and ids of the frame
                frame_averages = []
                frame_maxes = []
                frame_ids = []

                for bbox in bboxes:
                    object_id = bbox["object_id"]

                    # Extract patch from the bounding box
                    min_col, min_row, width, height = bbox["bbox"]
                    x1, y1, x2, y2 = int(min_col), int(min_row),int( min_col + width),int( min_row + height)
                    h_new, w_new = (self.image_resize_h // 14) * 14, (self.image_resize_w // 14) * 14
                    patch_crop = img.crop((x1, y1, x2, y2))
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

                    # Store the embedding in the dictionary under the object ID
                    res = ret[0].cpu().numpy()
                    #frame_features[frame_idx][object_id] = res

                    """
                    prepare the features for the transformation/ fitting, keep track of the indices!!
                    """
                    #add the id in the correct place
                    all_ids.append(object_id)
                    frame_ids.append(object_id)
                    #make the average and append
                    avg = np.mean(res, axis=0)  # Max pooling across patches
                    averages.append(avg)
                    frame_averages.append(avg)

                    #make the max pooling
                    max = np.max(res, axis=0)
                    maxes.append(max)
                    frame_maxes.append(max)

                
                #this section does the projection for the dino segmentation unseen points
                if self.dino:
                    #access the different pcas
                    proj_path = osp.join("/media/ekoller/T7/Features2D/projection", self.model_name, "patch_{}_{}".format(self.image_patch_w,self.image_patch_h))
                    with open(proj_path, 'rb') as file:
                        # Load the contents of the file
                        proj_data = pickle.load(file)

                    #turn everything into a 
                    frame_ids = np.array(frame_ids)
                    frame_averages = np.array(frame_averages)
                    frame_maxes = np.array(frame_maxes)
                   
                    #access the matricies needed for the projection
                    #pca based on average and max
                    scan_pca_avg = proj_data["pca_avg"]["matrix"]
                    scan_pca_max = proj_data["pca_max"]["matrix"]
                    #mds based on average and max
                    scan_mds_avg = proj_data["mds_avg"]["matrix"]
                    scan_mds_max = proj_data["mds_max"]["matrix"]
                    #tsne based on average and max
                    scan_tsne_avg = proj_data["tsne_avg"]["matrix"]
                    scan_tsne_max = proj_data["tsne_max"]["matrix"]
                    
                    #pca reductions
                    frame_red_pca_avg = scan_pca_avg.transform(frame_averages)
                    frame_red_pca_max = scan_pca_max.transform(frame_maxes)
                    #mds reductions
                    frame_red_mds_avg = scan_mds_avg.transform(frame_averages)
                    frame_red_mds_max = scan_mds_max.transform(frame_maxes)
                    #tsne reductions
                    frame_red_tsne_avg = scan_tsne_avg.transform(frame_averages)
                    frame_red_tsne_max = scan_tsne_max.transform(frame_maxes)

                    #fill in the frame_feature dictionary for this frame, keep structure the same as for the proj for easier handling
                    frame_features[frame_idx]["obj_ids"] = frame_ids
                    #pca
                    frame_features[frame_idx]["pca_avg"] = {
                        "points": frame_red_pca_avg
                    }
                    frame_features[frame_idx]["pca_max"] = {
                        "points": frame_red_pca_max
                    }
                    #mds
                    frame_features[frame_idx]["mds_avg"] = {
                        "points": frame_red_mds_avg
                    }
                    frame_features[frame_idx]["mds_max"] = {
                        "points": frame_red_mds_max
                    }
                    #tsne
                    frame_features[frame_idx]["tsne_avg"] = {
                        "points": frame_red_tsne_avg
                    }
                    frame_features[frame_idx]["tnse_max"] = {
                        "points": frame_red_tsne_max
                    }
        



        """
        in order to save space on the disk we will only save the PCA/ MDS / t-sne model for fast access of the segdino computation
        also save the projected points since they use a lot less memory:)
        fill in this part separatly for proj
        """

        #for the current scene we need to create the projection spaces & project the points
        if self.proj:

            #turn into numpy arrays
            all_ids = np.array(all_ids)
            maxes = np.array(maxes)
            averages = np.array(averages)


            #we need the ids in every case
            frame_features["obj_ids"] = all_ids

            #pca_avg
            pca_avg = PCA(n_components=3)
            pca_avg_red = pca_avg.fit_transform(averages)
            #pca_max
            pca_max = PCA(n_components=3)
            pca_max_red = pca_max.fit_transform(maxes)
            
            #mds_avg
            mds_avg = MDS(n_components=3, random_state=42)
            mds_avg_red = mds_avg.fit_transform(averages)
            #mds_max
            mds_max = MDS(n_components=3, random_state=42)
            mds_max_red = mds_max.fit_transform(maxes)

            #tsne_avg
            tsne_avg = TSNE(n_components=3, random_state=42)
            tsne_avg_red = tsne_avg.fit_transform(averages)
            #tsne_max
            tsne_max = TSNE(n_components=3, random_state=42)
            tsne_max_red = tsne_max.fit_transform(maxes)

            #put everything into the object  
            # pca          
            frame_features["pca_avg"] = {
                "matrix": pca_avg,
                "points": pca_avg_red
            }
            frame_features["pca_max"] = {
                "matrix": pca_max,
                "points": pca_max_red
            }
            #mds
            frame_features["mds_avg"] = {
                "matrix": mds_avg,
                "points": mds_avg_red
            }
            frame_features["mds_max"] = {
                "matrix": mds_max,
                "points": mds_max_red
            }
            #tsne
            frame_features["tsne_avg"] = {
                "matrix": tsne_avg,
                "points": tsne_avg_red
            }
            frame_features["tnse_max"] = {
                "matrix": tsne_max,
                "points": tsne_max_red
            }

            

        

        return frame_features

    
    
    def generateFeatures(self):
        img_num = 0
        self.feature_generation_time = 0.0
        #print("scanid in generate function", self.scan_ids)
        for scan_id in tqdm(self.scan_ids):
            with torch.no_grad():
                imgs_features = self.generateFeaturesEachScan(scan_id)
            img_num += len(imgs_features)
            out_file = osp.join(self.out_dir, '{}.pkl'.format(scan_id))
            common.write_pkl_data(imgs_features, out_file)


            
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
    scan3r_gcvit_generator = Scan3rDinov2Generator(cfg, 'train', for_proj= True)
    scan3r_gcvit_generator.register_model()
    scan3r_gcvit_generator.generateFeatures()
    #also generate for the dino_:segmentation boundingboxes
    scan3r_gcvit_generator = Scan3rDinov2Generator(cfg, 'train', for_dino_seg = True)
    scan3r_gcvit_generator.register_model()
    scan3r_gcvit_generator.generateFeatures()

    # scan3r_gcvit_generator = Scan3rDinov2Generator(cfg, 'val', for_proj= True)
    # scan3r_gcvit_generator.register_model()
    # scan3r_gcvit_generator.generateFeatures()
    # scan3r_gcvit_generator = Scan3rDinov2Generator(cfg, 'val', for_dino_seg= True)
    # scan3r_gcvit_generator.register_model()
    # scan3r_gcvit_generator.generateFeatures()
    # scan3r_gcvit_generator = Scan3rDinov2Generator(cfg, 'test', for_proj = True)
    # scan3r_gcvit_generator.register_model()
    # scan3r_gcvit_generator.generateFeatures()
    # scan3r_gcvit_generator = Scan3rDinov2Generator(cfg, 'test', for_dino_seg = True)
    # scan3r_gcvit_generator.register_model()
    # scan3r_gcvit_generator.generateFeatures()
    
if __name__ == "__main__":
    main()