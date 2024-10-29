import argparse
import pickle
import plyfile
import os 
import glob
import faiss
import random
from PIL import Image
import concurrent.futures
from tqdm.auto import tqdm
import pickle
import traceback
from collections import Counter
import plyfile
import numpy as np
import open3d as o3d
from sklearn.decomposition import PCA
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import open3d as o3d
import h5py
from sklearn.decomposition import PCA
from scipy.spatial import distance

import cv2
import numpy as np

import os.path as osp
import sys

ws_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
print(ws_dir)
sys.path.append(ws_dir)
from utils import common, scan3r, od3du_utils

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
        self.voxel_size = cfg.parameters.voxel_size
        self.minimum_points = cfg.parameters.minimum_points
        self.overlap_th = cfg.parameters.overlap_threshold
        self.minimum_votes = cfg.parameters.minimum_votes


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
            self.scan_ids = self.all_scans_split
        else:
            self.scan_ids = ref_scans_split
    
        #output path for components
        self.out_dir = osp.join(self.scans_files_dir, "Predicted_Centers")
        common.ensure_dir(self.out_dir)

    

    def predict_objects_scan(self,scan_id):

    # Load image paths and frame indices
        frame_idxs_list = scan3r.load_frame_idxs(self.scans_scenes_dir,scan_id)
        frame_idxs_list.sort()
       

        #access the segmentation of the scan_id
        segmentation_info_path = osp.join(self.scans_files_dir,"Segmentation/DinoV2/objects", scan_id + ".h5")
        segmentation_data = od3du_utils.read_segmentation_data(segmentation_info_path)
        #access the matched data
        matches = od3du_utils.read_matching_data(self.scans_files_dir, scan_id)

        #prepare a dictionary for the scene containing the new object centers of the seen objects
        all_clusters = {}
        #now the frame
        for infer_step_i in range(0, len(frame_idxs_list) // self.inference_step + 1):
            start_idx = infer_step_i * self.inference_step
            end_idx = min((infer_step_i + 1) * self.inference_step, len(frame_idxs_list))
            frame_idxs_sublist = frame_idxs_list[start_idx:end_idx]

        
            for frame_idx in frame_idxs_sublist:
            
                #access the matches for this frame
                frame_matches = matches[frame_idx]
                #access the depht image
                depth_path = osp.join(self.scans_scenes_dir, scan_id, "sequence", "frame-{}.depth.pgm".format(frame_idx))
                #access the file
                pgm_file = Image.open(depth_path)
        
                #since its distances so discrete things take the nearest value not a different interpolation
                depth_mat_resized = pgm_file.resize((self.image_width, self.image_height), Image.NEAREST) 
            
                #depth is given in mm so put it into m
                depth_mat = np.array(depth_mat_resized)
                depth_mat = depth_mat * 0.001

                #transform to world coordinates in the reference frame
                world_coordinates_frame = od3du_utils.transform_to_3d(self.data_root_dir, self.scans_scenes_dir, scan_id, depth_mat, frame_idx)

                new_obj_idx = 0
                #access the segmented image
                segmented_img_path = osp.join(self.scans_files_dir,"Segmentation/DinoV2/color", scan_id, "frame-{}.jpg".format(frame_idx))
                segmented_img = cv2.imread(segmented_img_path)
                #iterate through the masks of the objec
                for boundingboxes in segmentation_data[frame_idx]:
                    #access the mask for the object (is quantized)
                    mask = boundingboxes['mask']
                    mask = mask.astype(bool)   
                    #access the patches withing the segmented image
                    masked_region = segmented_img[mask]
                    #determin the most occuring colour
                    colors_in_region = list(map(tuple, masked_region.reshape(-1, segmented_img.shape[-1])))
                    most_frequent_color = Counter(colors_in_region).most_common(1)[0][0]
                    #create a mask of the colour in the whole image
                    color_mask = np.all(segmented_img == most_frequent_color, axis=-1)
                    #we only want the mask for the region of the first region
                    result_mask = color_mask & mask


                
                    #get the dino object_id 
                    dino_id = boundingboxes["object_id"]
                    #get the matched id
                    object_id = frame_matches[str(dino_id)]

                    
                    #we only consider matched objects - the feature matches for unmatched objects is negative
                    if object_id > 0:
                        #isolate only the object pointcloud
                        obj_pcl = od3du_utils.isolate_object_coordinates(world_coordinates_frame, result_mask)
                     
                       
                        #now we need to find out if we add it to the pointcloud of the object it mapped to or not
                        if object_id not in all_clusters:
                            #there are no clusters & votes stored for this object jet
                            all_clusters[object_id] = [{'cluster': obj_pcl, 'votes': 1}]
                        #object already has pointclouds we need to see if we merge or add a new cluster
                        else:
                            #each new cluster starts unmerged
                            merged = False
                            max_overlap = 0
                            best_cluster_index = None
                            for i, cluster_data in enumerate(all_clusters[object_id]):
                                cluster = cluster_data['cluster']

                                #add to the cluster with the most overlap
                                overlap = od3du_utils.do_pcl_overlap(self.voxel_size, obj_pcl, cluster)

                                # keep track of the most overlap cluste
                                if overlap > self.overlap_th and overlap > max_overlap:
                                    max_overlap = overlap
                                    best_cluster_index = i
                                
                            if best_cluster_index is not None:
                                # Merge the point clouds with the best cluster
                                best_cluster = all_clusters[object_id][best_cluster_index]['cluster']
                                merged_points = np.vstack((obj_pcl, best_cluster))
                                
                                # Update the best cluster with the merged points
                                all_clusters[object_id][best_cluster_index]['cluster'] = merged_points
                                
                                # Increment the vote count for the best cluster
                                all_clusters[object_id][best_cluster_index]['votes'] += 1

                                # Mark as merged
                                merged = True
                            if not merged:
                                all_clusters[object_id].append({'cluster': obj_pcl, 'votes': 1})


        #now that we have the lists of clusters we need to iterate over them and choose the biggest cluster, downsample it & take the average to predict the center
        #initialize final object
        all_centers = {}
        #iterte through the objects
        for obj_id, clusters in all_clusters.items():
            #decide the most likely correct cluster based on votes first and then size
            largest_cluster_data = max(all_clusters[obj_id], key=lambda c: (c['votes'], len(c['cluster'])))
            largest_cluster = largest_cluster_data['cluster']
            #downsample to store
            cluster_point_cloud = o3d.geometry.PointCloud()
            cluster_point_cloud.points = o3d.utility.Vector3dVector(largest_cluster)
            voxel_size = 0.07  # Adjust this value based on your needs
            downsampled_point_cloud = cluster_point_cloud.voxel_down_sample(voxel_size=voxel_size)
            cl, ind = downsampled_point_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.75)
            cluster_pcl = np.asarray(cl.points)

            #get the votes
            largest_cluster_votes = largest_cluster_data["votes"]

            #decide it it is a predicted object
            if (largest_cluster_votes >= self.minimum_votes) and (len(cluster_pcl) >= self.minimum_points):
                if(obj_id < 0 ):
                    print("new object detected", obj_id)
                #create the objec center
                obj_center = np.median(cluster_pcl, axis= 0)
                #return the object for the evaluation
                all_centers[obj_id] = {
                    'center': obj_center,
                    "size": len(cluster_pcl),
                    "votes" : largest_cluster_votes,
                    "points": cluster_pcl

                }

        return all_centers
 
    
    def compute(self):
        # Use tqdm for progress bar, iterating as tasks are completed
        with tqdm(total=len(self.scan_ids)) as pbar:
                for scan_id in self.scan_ids:
                    centers = self.predict_objects_scan(scan_id)
                    result_file_path = osp.join(self.out_dir, scan_id + ".h5")
                    # common.write_pkl_data( centers, result_file_path)
                    with h5py.File(result_file_path, 'w') as h5file:
                        for obj_id, data in centers.items():
                            # Save the center, points, and votes for each object
                            obj_group = h5file.create_group(str(obj_id))
                            obj_group.create_dataset('center', data=data['center'])
                            obj_group.create_dataset('points', data=data['points'])
                            obj_group.create_dataset('votes', data=data['votes'])
                            obj_group.create_dataset('size', data=data['size'])
                    print("added results of scan id ", scan_id, " successfully")
            
                
                    # progressed
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

    #do it for the projections first
    #also generate for the dino_:segmentation boundingboxes
    evaluate = Evaluator(cfg, split)
    evaluate.compute()
   

   
  

    
if __name__ == "__main__":
    main()