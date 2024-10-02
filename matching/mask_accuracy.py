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
        self.scans_files_dir = osp.join(self.data_root_dir, 'files')
        self.scans_files_dir_mode = osp.join(self.scans_files_dir)
        self.scans_scenes_dir = osp.join(self.data_root_dir, 'scenes')
        self.inference_step = cfg.data.inference_step
        #model info
        self.model_name = cfg.model.name

        #parameters
        self.k_means = cfg.parameters.k_means
        self.ths = cfg.parameters.threshold


        


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
    
        #output path for components
        self.out_dir = osp.join(self.data_root_dir, "Results", "testset_mask_metric" )
        common.ensure_dir(self.out_dir)

     
    def load_frame_idxs(self,data_dir, scan_id, skip=None):
      
        frames_paths = glob.glob(osp.join(data_dir, scan_id, 'sequence', '*.jpg'))
        frame_names = [osp.basename(frame_path) for frame_path in frames_paths]
        frame_idxs = [frame_name.split('.')[0].split('-')[-1] for frame_name in frame_names]
        frame_idxs.sort()

        if skip is None:
            frame_idxs = frame_idxs
        else:
            frame_idxs = [frame_idx for frame_idx in frame_idxs[::skip]]
        return frame_idxs
    
    

    def reconstruct_to_image(self, patchwise_id):
        patch_width = int(self.image_width / self.image_patch_w)
        patch_height = int(self.image_height / self.image_patch_h)
        
        # Initialize an empty image with zeros (assuming same type as original)
        reconstructed_img = np.zeros((self.image_height, self.image_width), dtype=np.int32)
        
        # Loop over patches and place the patchwise_id values into the reconstructed image
        for i in range(self.image_patch_h):
            for j in range(self.image_patch_w):
                # Define the coordinates of the current patch
                h_start = i * patch_height
                w_start = j * patch_width
                h_end = h_start + patch_height
                w_end = w_start + patch_width
                
                # Assign the patchwise_id value to the corresponding patch area
                reconstructed_img[h_start:h_end, w_start:w_end] = patchwise_id[i, j]
        
        return reconstructed_img


    def compute_iou(self,mask, gt_mask):
        # intersection of masks / union
        intersection = np.logical_and(mask, gt_mask).sum()
        union = np.logical_or(mask, gt_mask).sum()
        return intersection / union if union != 0 else 0


    #returns featuer in the form of features: frame: list of {objext_id, bbox, mask} objects
    def read_segmentation_data(self,segmentation_path):
        features = {}
        with h5py.File(segmentation_path, 'r') as hdf_file:
             for frame_idx in hdf_file.keys():
                #init boxlist for curr frame
                bounding_boxes = []
                
                # get info 
                frame_group = hdf_file[frame_idx]
                
                #iterate over each boundingbox
                for bbox_key in frame_group.keys():
                    bbox_group = frame_group[bbox_key]
                    
                    #get te obj id
                    object_id = bbox_group.attrs['object_id']
                    
                    #get the boundingbox
                    bbox = bbox_group['bbox'][:]
                    
                    # get the maskt
                    mask = bbox_group['mask'][:]
                    
                    # append to list
                    bounding_boxes.append({
                        'object_id': object_id,
                        'bbox': bbox,
                        'mask': mask
                    })
                
                # stor it to the corresponding frame
               
                features[frame_idx] = bounding_boxes
        return features
 
  

    def compute_scan(self,scan_id):

    # Load image paths and frame indices
        frame_idxs_list = self.load_frame_idxs(self.scans_scenes_dir,scan_id)
        frame_idxs_list.sort()
        #access the necessary data for the reference scene
    
        reference_info_path = osp.join("/local/home/ekoller/R3Scan/files/patch_anno", "patch_anno_{}_{}".format(self.image_patch_w,self.image_patch_h),"{}.pkl".format(scan_id))
        gt_patches = scan3r.load_pkl_data(reference_info_path)
        
        #init the result for this scan_id 
        scan_mask_metric= [] 
        

        #access the segmentation of the scan_id
        segmentation_info_path = osp.join("/media/ekoller/T7/Segmentation/DinoV2/objects", scan_id + ".h5")
        segmentation_data = self.read_segmentation_data(segmentation_info_path)

    

        #now the frame
        for infer_step_i in range(0, len(frame_idxs_list) // self.inference_step + 1):
            start_idx = infer_step_i * self.inference_step
            end_idx = min((infer_step_i + 1) * self.inference_step, len(frame_idxs_list))
            frame_idxs_sublist = frame_idxs_list[start_idx:end_idx]

        
            for frame_idx in frame_idxs_sublist:
                #turn the gt into an image again
                gt_img = self.reconstruct_to_image( gt_patches[frame_idx])
                #iterate through the masks of the objec
                for boundingboxes in segmentation_data[frame_idx]:
                    #access the mask for the object
                    mask = boundingboxes['mask']
                    

                    #get the coords of the mask
                    mask_coords = np.nonzero(mask)

                    # #get the region in the gt corresponding to the mask
                    min_y, max_y = np.min(mask_coords[0]), np.max(mask_coords[0]) + 1
                    min_x, max_x = np.min(mask_coords[1]), np.max(mask_coords[1]) + 1
                    gt_region = gt_img[min_y:max_y, min_x:max_x]
                    gt_region = gt_img[mask_coords]
                
                    #get the most common value of the gt at the place of the mask
                    flattened_gt_region = gt_region.flatten()
                    value_counts = Counter(flattened_gt_region)
                    most_common_id = value_counts.most_common(1)[0][0]
                    if most_common_id != 0:
                        #turn  it into a mask 
                        gt_mask = (gt_img == most_common_id).astype(np.uint8)
                        #compute the metric of the overlapp
                        iou = self.compute_iou(mask, gt_mask)

                        scan_mask_metric.append(iou) 


        #we need the mean
        mean_val = np.mean(scan_mask_metric, axis=0)      
                        
        return mean_val

    def compute(self):
        #prepare the matricies for the 4 different metrics
        mask_metric = {}
       

        workers = 2
        
        # parallelize the computations
        with concurrent.futures.ProcessPoolExecutor(max_workers= workers) as executor:
            futures = {executor.submit(self.compute_scan, scan_id): scan_id for scan_id in self.scan_ids}
            
            # Use tqdm for progress bar, iterating as tasks are completed
            with tqdm(total=len(self.scan_ids)) as pbar:
                for future in concurrent.futures.as_completed(futures):
                    scan_id = futures[future]
                    try:
                        iou_score = future.result()
                        ref_id = scan3r.get_reference_id(self.data_root_dir, scan_id)
                        # get the result matricies
                        mask_metric[ref_id] = iou_score
                        print("added results of scan id ", scan_id, " successfully")
                    except Exception as exc:
                        print(f"Scan {scan_id} generated an exception: {exc}")
                        print("Traceback details:")
                        traceback.print_exc()
                    
                    # progressed
                    pbar.update(1)

        print("writing the file")

     


        
        #save the file in the results direcrtory
        result_file_path = osp.join(self.out_dir,  "mask_metric.pkl")
        common.write_pkl_data( mask_metric, result_file_path)
                    
                
    
              

            


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
    # evaluate = Evaluator(cfg, 'train')
    # print("start mask computation")
    # evaluate.compute()
    evaluate = Evaluator(cfg, 'test')
    print("start mask computation")
    evaluate.compute()
   

   
  

    
if __name__ == "__main__":
    main()