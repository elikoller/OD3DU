import argparse
import pickle
import plyfile
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
from scipy.spatial import cKDTree
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


        print("scan_id_length", len(self.scan_ids))




        #print("scan ids", len(self.scan_ids))
        ## images info
        self.image_paths = {}
        for scan_id in self.scan_ids:
            self.image_paths[scan_id] = self.load_frame_paths(self.scans_dir, scan_id)

      
        #output path for components
        self.out_dir = osp.join(self.data_root_dir, "Results" )
        common.ensure_dir(self.out_dir)

     
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
    
    def normalize_points(self, points):
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        normalized_points = points / norms
        return normalized_points
    
    def get_majorities(self, distanc, obj_ids, k , th):
        #make te majority voting
        majorities = []
        for i, (dist , ids) in enumerate(zip(distanc,obj_ids)):
            closest_dist = dist[:k]
            closest_ids = ids[:k]

            #get the majority
            most_common_class, _ = Counter(closest_ids).most_common(1)[0]

            majority_distances = closest_dist[closest_ids == most_common_class]
            average_distance = np.mean(majority_distances)

            #thresthold the average majority of the distances, it it is above the threshold, give the id -1 which represents an unseen obj
            if average_distance >= th:
                #too far away
                majorities.append(-1)
            else:
                majorities.append(most_common_class)

    def generate_pixel_level(self, segmentation,frame_obj_ids,majorities):
        pixel_ids = np.zeros((self.image_height, self.image_width))

        #iterate through all the segmentation regions of the dino segmentation
        for seg_region in segmentation:
            mask_id = seg_region["object_id"]
            #find the index of mask_id in frame_obj_ids
            index = np.where(frame_obj_ids == mask_id)[0]
            #get to what the region mapped in the majorities
            matched_id = majorities[index]
            #print("matched id ", matched_id)
            mask = seg_region["mask"]
            boolean_mask = mask == 225
            pixel_ids[boolean_mask] = matched_id 
            

        return pixel_ids
    
    def quantize_to_patch_level(self, pixelwise_img):
        #get the shape of the pixelwise img
        patch_width = int(self.image_width/ self.image_patch_w)
        patch_height= int(self.image_height/self.image_patch_h)

        patchwise_id = np.zeros((self.image_patch_h,self.image_patch_w))

        for i in range(self.image_patch_h):
                for j in range(self.image_patch_w):
                    # Define the coordinates of the current patch
                    h_start = i * patch_height
                    w_start = j * patch_width
                    h_end = h_start + patch_height
                    w_end = w_start + patch_width
                    
                    # Get the current patch from the input matrix
                    patch = pixelwise_img[h_start:h_end, w_start:w_end]
                    
                    # get the most reoccuring id of the patch
                    flattened_patch = patch.flatten()
                    # Find the most common value in the patch
                    value_counts = Counter(flattened_patch)
                    most_common_id = value_counts.most_common(1)[0][0]
                    
                    # Assign the most common ID to the new matrix
                    patchwise_id[i, j] = most_common_id


        return patchwise_id


    def compute_obj_metric(self, gt_input,patch_level):

         

    
    #for a given scene get the colours of the differnt object_ids
    def get_present_obj_ids(data_dir,scan_id):
        #access the mesh file to get the colour of the ids
        mesh_file = osp.join(data_dir,"scenes", scan_id, "labels.instances.annotated.v2.ply")
        ply_data = plyfile.PlyData.read(mesh_file)
        # Extract vertex data
        vertices = ply_data['vertex']
        vertex_count = len(vertices)
        
        # Initialize dictionary to store object_id -> color mappings
        object_colors = {}
        
    # Iterate through vertices
        for i in range(vertex_count):
            vertex = vertices[i]
            object_id = vertex['objectId']
            color = (vertex['red'], vertex['green'], vertex['blue'])
            
            # Check if object_id already in dictionary, otherwise initialize a Counter
            if object_id in object_colors:
                object_colors[object_id][color] += 1
            else:
                object_colors[object_id] = Counter({color: 1})
        
        # Convert Counter to dictionary with most frequent color
        for object_id, color_counter in object_colors.items():
            most_common_color = color_counter.most_common(1)[0][0]
            object_colors[object_id] = np.array(most_common_color[::-1])
        
        return list(object_colors.keys())
    
    def compute(self, mode):
        #prepare the matricies for the 4 different metrics
        all_cosine_obj_metric = []
        all_cosine_patch_metric = []
        all_euclid_obj_metric = []
        all_euclid_patch_metric = []

        #loop over each frame and append the resultsto the total matricies
        for scan_id in tqdm(self.scan_ids):
            # Load image paths and frame indices
            frame_idxs_list = self.load_frame_idxs(self.scans_scenes_dir,scan_id)
            frame_idxs_list.sort()
            #access the necessary things for the reference scene
            reference_id = scan3r.get_reference_id(self.data_root_dir, scan_id)
            reference_info_path = osp.join("/media/ekoller/T7/Features2D/projection", self.model_name, "patch_{}_{}".format(self.image_patch_w,self.image_patch_h),"{}.pkl".format(reference_id))

            with open(reference_info_path, 'rb') as file:
                ref_data = pickle.load(file)

            ref_obj_ids = np.array(ref_data["obj_ids"])
            ref_proj_points = np.array(ref_data[mode]["points"])
            #turn them into a cKDTree to make the accesses more efficient for eucledian distances
            kdtree = cKDTree(ref_proj_points)
            #prepare the normalization for the cosine distance
            normalized_ref_proj_points = self.normalize_points(ref_proj_points)

            #access the scan feature info
            scan_info_path = osp.join("/media/ekoller/T7/Features2D/dino_segmentation", self.model_name, "patch_{}_{}".format(self.image_patch_w,self.image_patch_h),"{}.pkl".format(scan_id))
            with open(scan_info_path, 'rb') as file:
                scan_data = pickle.load(file)

            #access the segmentation of the scan_id
            segmentation_info_path = osp.join("/media/ekoller/T7files/Segmentation/DinoV2/objects", scan_id + ".pkl")
            with open(segmentation_info_path, 'rb') as file:
                segmentation_data = pickle.load(file)

            #access the colour dictionaries we will later need to see which object ids are there
            ref_colours = self.get_present_obj_ids(self.data_root_dir,scan_id)
    

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

                    cosine_obj_metric = np.zeros((ths_len,k_means_len))
                    cosine_patch_metric = np.zeros((ths_len,k_means_len))
                    euclid_obj_metric = np.zeros((ths_len,k_means_len))
                    euclid_patch_metric = np.zeros((ths_len,k_means_len))
                    #get the projected points in the correct mode
                    frame_obj_ids = scan_data[frame_idx]["obj_ids"]
                    frame_proj_points = scan_data[frame_idx][mode]["points"]

                    #compute the result for the cosine dist

                    #since for cosine the direction is important: normalize the vectors
                    normalized_frame_proj_points = self.normalize_points(frame_proj_points)
    
                    #compute all the different distances for each frame point in a distance matrix
                    cosine_dist_frame = cosine_distances(normalized_frame_proj_points, normalized_ref_proj_points)
                    cosine_distanc = []
                    cosine_obj_ids = []
                    for i, dist in enumerate(cosine_dist_frame):
                        # Find the indices of the k smallest distances
                        indices = np.argsort(dist)[:max(self.k_means)]
                        # Retrieve the corresponding object IDs and distances
                        closest_distances = dist[indices]
                        closest_ids = ref_obj_ids[indices]
                        cosine_distanc.append(closest_distances)
                        cosine_obj_ids.append(closest_ids)
                                    


                    #compute the closest k points & obj ids in terms of euclidean distance

                    euclid_distanc, euclid_idxs = kdtree.query(frame_proj_points, max(self.k_means))
                    euclid_obj_ids = ref_obj_ids[euclid_idxs]


                    #get the correct computations and iteraste through every combination
                    for t_idx, th in enumerate (self.ths):
                        for k_idx, k in enumerate (self.k_means):
                            # get the majority vote of the k closest points
                            cosine_majorities = self.get_majorities(cosine_distanc, cosine_obj_ids, k, th)
                            euclid_majorities = self.get_majorities(euclid_distanc, euclid_obj_ids)

                            #translate the matched object ids to pixellevel of the frame
                            cosine_pixel_level = self.generate_pixel_level(segmentation_data[frame_idx],frame_obj_ids,cosine_majorities)
                            euclid_pixel_level = self.generate_pixel_level(segmentation_data[frame_idx],frame_obj_ids,euclid_majorities)
                            
                            #quantize to patchlevel of to be comparable to gt
                            cosine_patch_level = self.quantize_to_patch_level(cosine_pixel_level)
                            euclid_patch_level = self.quantize_to_patch_level(euclid_pixel_level)

                            #access the ground truth patches for this frame
                            gt_input_patchwise_path =  osp.join(self.scans_files_dir,"patch_anno","patch_{}_{}".format(self.image_patch_w,self.image_patch_h),"{}.pkl".format(scan_id))

                            with open(gt_input_patchwise_path, 'rb') as file:
                                gt_input_patchwise = pickle.load(file)

                            
                            #finally compute the accuracies: based on area and object ids and fill into the matrix
                            #for cosine
                            cosine_obj_metric[t_idx][k_idx] = self.compute_obj_metric(gt_input_patchwise,cosine_patch_level)
                            cosine_patch_metric[t_idx][k_idx] = self.compute_patch_metric(gt_input_patchwise,cosine_patch_level)

                            #for euclid
                            euclid_obj_metric[t_idx][k_idx] =self.compute_obj_metric(gt_input_patchwise,euclid_patch_level)
                            euclid_obj_metric[t_idx][k_idx] = self.compute_obj_metric(gt_input_patchwise,euclid_patch_level)



                    
                #append the matrix to the other matricies
                all_cosine_obj_metric.append(cosine_obj_metric)  
                all_cosine_patch_metric.append(cosine_patch_metric)
                all_euclid_obj_metric.append(euclid_obj_metric) 
                all_euclid_patch_metric.append(euclid_patch_metric)


        #we want the result over all scenes
        mean_cosine_obj_metric = np.mean(all_cosine_obj_metric)
        mean_cosine_patch_metric = np.mean(all_cosine_patch_metric)
        mean_euclid_obj_metric = np.mean(all_euclid_obj_metric)
        mean_euclid_patch_metric = np.mean(all_euclid_patch_metric)

        #create sesult dict
        result = {"cosine_obj_metric": mean_cosine_obj_metric,
                  "cosine_patch_metric": mean_cosine_patch_metric,
                  "euclid_obj_metric": mean_euclid_obj_metric,
                  "euclid_patch_metric": mean_euclid_patch_metric
                  }
        #save the file in the results direcrtory
        result_file_path = osp.join(self.out_dir, mode + ".pkl")
        common.write_pkl_data(result_file_path, result)
                    
                
               
              

                  
               
    


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