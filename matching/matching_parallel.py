import argparse
import pickle
import plyfile
import os 
import glob
import faiss
import random
import concurrent.futures
from tqdm.auto import tqdm
import pickle
import traceback
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import plyfile
import numpy as np
import open3d as o3d
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import open3d as o3d
import h5py
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
from utils import common, scan3r, od3du_utils


class Evaluator():
    def __init__(self, cfg, split):
        #initialize the file paths for later access
        self.cfg = cfg
        # 3RScan data info
        self.split = split
        ## data dir
        self.data_root_dir = cfg.data.root_dir
        scan_dirname = 'scan' 
        self.scans_dir = osp.join(cfg.data.root_dir, scan_dirname)
        self.scans_files_dir = osp.join(self.data_root_dir, 'files')
        self.scans_files_dir_mode = osp.join(self.scans_files_dir)
        self.scans_scenes_dir = osp.join(self.data_root_dir, 'scenes')
        self.inference_step = cfg.data.inference_step
        #model info
        self.model_name = cfg.model.name

        #parameters
        self.k_means = cfg.parameters.k_means
        self.ths = cfg.parameters.threshold
        self.mode = cfg.parameters.mode


        #patch info 
        self.image_patch_w = self.cfg.data.img_encoding.patch_w
        self.image_patch_h = self.cfg.data.img_encoding.patch_h

        #img info
        self.image_width = self.cfg.data.img.w
        self.image_height = self.cfg.data.img.h
  
       
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
        
        self.all_scans_split = []

        ## get all scans within the split(ref_scan + rescan)
        for ref_scan in ref_scans_split[:]:
           
            # Check and add one rescan for the current reference scan
            rescans = [scan for scan in self.refscans2scans[ref_scan] if scan != ref_scan]
            if rescans:
                # Add the first rescan (or any specific rescan logic)
                self.all_scans_split.append(rescans[0])

         
        self.all_scans_split.sort()

        if self.rescan:
            self.scan_ids = self.all_scans_split #ouses only the rescan -> we only want to match these ones to the reference scan
        else:
            self.scan_ids = ref_scans_split


    
        #output path for components
        self.out_dir = osp.join( self.scans_files_dir, "Predicted_Matches")
        common.ensure_dir(self.out_dir)



    def compute_scan(self,scan_id):

    # Load image paths and frame indices
        frame_idxs_list = scan3r.load_frame_idxs(self.scans_scenes_dir,scan_id)
        frame_idxs_list.sort()
        #access the necessary data for the reference scene
        reference_id = scan3r.get_reference_id(self.data_root_dir, scan_id)
        reference_info_path = osp.join( self.scans_files_dir, "Features2D/projection", self.model_name, "patch_{}_{}".format(self.image_patch_w,self.image_patch_h), self.mode, "{}.h5".format(reference_id))
        ref_data = od3du_utils.read_ref_data(reference_info_path)
        
        
        #build a treee structure fo a fast access of cosine similarity keeping also the obj_ids 
        ref_obj_ids = []
        ref_vectors = []
        for obj_id, feature_list in ref_data.items():
            for feature_vector in feature_list:
                ref_vectors.append(feature_vector)
                ref_obj_ids.append(obj_id)

        #convert to numpy for faiss
        ref_obj_ids = np.array(ref_obj_ids)
        #print("reference ids", ref_obj_ids)
        ref_vectors = np.array(ref_vectors)

        #normalize vectors
        faiss.normalize_L2(ref_vectors)

        #make an faiss index
        dimension = ref_vectors.shape[1]  # vector dimension
        #use the cosine similarity (IP) indexing
        index = faiss.IndexFlatIP(dimension)
        index.add(ref_vectors)

        #initialize the result for the scene
        all_matches = {}

        #access the scan feature info to iterate over it later
        scan_info_path = osp.join(self.scans_files_dir, "Features2D/dino_segmentation", self.model_name, "patch_{}_{}".format(self.image_patch_w,self.image_patch_h), self.mode, "{}.h5".format(scan_id))
        scan_data = od3du_utils.read_scan_data(scan_info_path)

        #now the frame
        for infer_step_i in range(0, len(frame_idxs_list) // self.inference_step + 1):
            start_idx = infer_step_i * self.inference_step
            end_idx = min((infer_step_i + 1) * self.inference_step, len(frame_idxs_list))
            frame_idxs_sublist = frame_idxs_list[start_idx:end_idx]

        
            for frame_idx in frame_idxs_sublist:
                #initialze the matricies we will fill
                #get the lengths of the parameters
                all_matches[frame_idx] = []
                

                #initialize the distances
                cosine_distanc = []
                cosine_obj_ids = []

                #keep track of the order of the scan_frame_ids
                frame_obj_ids = []
                
                #iterate through the objects and get the distances of the obj in this frame
                for obj_id, feature_vector in scan_data[frame_idx].items():
                    #add the id to the frameobje ids
                    frame_obj_ids.append(obj_id)
                    # normalize the query vec
                    query_vector = np.array(feature_vector).reshape(1, -1)
                    faiss.normalize_L2(query_vector)

                    #get distance and ids for the clos
                    distances, indices = index.search(query_vector,self.k_means) #get the max of k then we already know which ones are closer :)
                    
                    # get the object ids of the closest reference vectors and the distances
                    nearest_obj_ids = [ref_obj_ids[idx] for idx in indices[0]]
                    nearest_distances = distances[0]

                    cosine_obj_ids.append(nearest_obj_ids)
                
                    cosine_distanc.append(nearest_distances)
                
                   

                cosine_majorities = od3du_utils.get_majorities(cosine_distanc, cosine_obj_ids, frame_obj_ids, self.k_means, self.ths)
                
                all_matches[frame_idx] = cosine_majorities
                          

        #save the file in the results direcrtory

        """  need to uncomment this again
        """
        result_file_path = osp.join(self.out_dir,scan_id +".h5")
        with h5py.File(result_file_path, 'w') as hdf_file:
            for frame_idx, matches in all_matches.items():
                # Create a group for each frame index
                frame_group = hdf_file.create_group(str(frame_idx))
                
                # Iterate through the matches and store frame_id and obj
                for frame_id, obj in matches.items():
                    # Store each frame_id -> obj mapping as a dataset in the frame group
                    frame_group.create_dataset(str(frame_id), data=obj)

        

    #read it out
      # Iterate through frame indices
        # with h5py.File('matches.h5', 'r') as hdf_file:
        #     loaded_matches = {}

                
                
        return True

    def compute(self):
    
        workers = 2
        
        with tqdm(total=len(self.scan_ids)) as pbar:
            for scan_id in self.scan_ids:
                # Call the compute_scan method for each scan_id
                done = self.compute_scan(scan_id)
                if done:
                    print("Added results of scan id", scan_id, "successfully")
                else:
                    print("Not successful for scan id", scan_id)
                
                pbar.update(1)

           
                

def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess Scan3R')
    parser.add_argument('--config', type=str, default='', help='Path to the config file')
    parser.add_argument('--split', type=str, default='train', help='Seed for random number generator')
    return parser.parse_known_args()

def main():

    # get arguments
    args, _ = parse_args()
    cfg_file = args.config
    split = args.split
    print(f"Configuration file path: {cfg_file}")

    from configs import config, update_config
    cfg = update_config(config, cfg_file, ensure_dir = False)

    evaluate = Evaluator(cfg, split)
    evaluate.compute()
   
    
   
  

    
if __name__ == "__main__":
    main()