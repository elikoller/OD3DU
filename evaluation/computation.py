import argparse
import pickle

import os 
import glob
import random
from tqdm.auto import tqdm
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import plyfile
import numpy as np
import open3d as o3d
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import open3d as o3d
from sklearn.decomposition import PCA
from scipy.spatial import distance
from sklearn.metrics.pairwise import euclidean_distances

import cv2
import numpy as np

import os.path as osp
import sys
ws_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
print(ws_dir)
sys.path.append(ws_dir)
from utils import common, scan3r

"""
this currently takes only one rescan per reference scene into consideration
"""

class Evaluator():
    def __init__(self, cfg, split):
        self.cfg = cfg
        # 3RScan data info
        self.split = split
        ## data dir
        self.data_root_dir = cfg.data.root_dir
        scan_dirname = 'scan' 
        self.scans_dir = osp.join(cfg.data.root_dir, scan_dirname)
        self.scans_files_dir = osp.join(self.scans_dir, 'files')
        self.scans_files_dir_mode = osp.join(self.scans_files_dir)
        self.scans_scenes_dir = osp.join(self.scans_dir, 'scenes')
        self.inference_step = cfg.data.inference_step
        #model info
        self.model_name = cfg.model.name

        #parameters
        self.k_means = cfg.parameters.k_means
        self.ths = cfg.parameters.threshold


        


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

      
        #output path for components
        self.out_dir_objects = osp.join(self.data_root_dir, "Results" )
        common.ensure_dir(self.out_dir_objects)

     
    def load_frame_idxs(self,data_dir, scan_id, skip=None):
      
        frames_paths = glob(osp.join(data_dir, scan_id, 'sequence', '*.jpg'))
        frame_names = [osp.basename(frame_path) for frame_path in frames_paths]
        frame_idxs = [frame_name.split('.')[0].split('-')[-1] for frame_name in frame_names]
        frame_idxs.sort()

        if skip is None:
            frame_idxs = frame_idxs
        else:
            frame_idxs = [frame_idx for frame_idx in frame_idxs[::skip]]
        return frame_idxs
    
    def compute(self, mode):
        #prepare the matricies for the 4 different metrics
        all_cosine_obj = []
        all_cosine_patch = []
        all_euclid_obj = []
        all_euclid_patch = []

        #loop over each frame and append the resultsto the total matricies
        for scan_id in tqdm(self.scan_ids):
            # Load image paths and frame indices
            frame_idxs_list = self.load_frame_idxs(self.scans_scenes_dir,scan_id)
            frame_idxs_list.sort()
            #access the necessary things for the reference scene
            reference_id = scan3r.get_reference_id(self.data_root_dir, scan_id)
            reference_info_path = osp.join("/media/ekoller/T7/Features2D/projection", self.model_name, "patch_{}_{}".format(self.image_patch_w,self.image_patch_h),"{}.pkl".format(scan_id))

            with open(reference_info_path, 'rb') as file:
                ref_data = pickle.load(file)

            ref_obj_ids = np.array(ref_data["obj_ids"])
            ref_proj_points = np.array(ref_data[mode]["points"])

            #access the scan_info
            scan_info_path = self.out_dir = osp.join("/media/ekoller/T7/Features2D/dino_segmentation", self.model_name, "patch_{}_{}".format(self.image_patch_w,self.image_patch_h),"{}.pkl".format(scan_id))
            with open(scan_info_path, 'rb') as file:
                scan_data = pickle.load(file)
    

            #now the frame
            for infer_step_i in range(0, len(frame_idxs_list) // self.inference_step + 1):
                start_idx = infer_step_i * self.inference_step
                end_idx = min((infer_step_i + 1) * self.inference_step, len(frame_idxs_list))
                frame_idxs_sublist = frame_idxs_list[start_idx:end_idx]


                for frame_idx in frame_idxs_sublist:
                    #initialze the matricies we will fill
                    #get the lengths of the parameters
                    ths_len = len(self.ths)
                    k_means_len = len(self.k_means)

                    cosine_obj = np.zeros((ths_len,k_means_len))
                    cosine_patch = np.zeros((ths_len,k_means_len))
                    euclid_obj = np.zeros((ths_len,k_means_len))
                    euclid_patch = np.zeros((ths_len,k_means_len))
                    #get the projected points in the correct mode
                    frame_obj_ids = scan_data[frame_idx]["obj_ids"]
                    frame_proj_points = scan_data[frame_idx][mode]["points"]

                    #start the computations and iteraste through every combination
                    for t_idy, th in enumerate (self.th):
                        for k_idx, k in enumerate (self.k_means):
                            #compute the result for the cosine dist

                            #compute the result for the euclid dist

                    
                
               
              

                  
               
    


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
    #also generate for the dino_:segmentation boundingboxes
    Evaluator = Evaluator(cfg, 'train')
    Evaluator.compute("pca_avg")
    Evaluator.compute("pca_max")
    Evaluator.compute("mds_avg")
    Evaluator.compute("mds_max")
    Evaluator.compute("tsne_avg")
    Evaluator.compute("tsne_max")
  

    
if __name__ == "__main__":
    main()