import argparse
import pickle
import plyfile
import os 
import glob
import faiss
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
        self.scans_files_dir = osp.join(self.scans_dir, 'files')
        self.scans_files_dir_mode = osp.join(self.scans_files_dir)
        self.scans_scenes_dir = osp.join(self.scans_dir, 'scenes')
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
        self.out_dir = "/media/ekoller/T7/Matches" 
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
    

    
    def get_majorities(self, distanc, obj_ids, k , th):
        #make te majority voting
        majorities = []
        unique_new_obj = -1
        for i, (dist , ids) in enumerate(zip(distanc,obj_ids)):
            closest_dist = dist[:k]
            closest_ids = ids[:k]

            #get the majority
            most_common_class, _ = Counter(closest_ids).most_common(1)[0]

            majority_distances = closest_dist[closest_ids == most_common_class]
            average_distance = np.mean(majority_distances)

            #thresthold the average majority of the distances, it it is above the threshold, give the id -1 which represents an unseen obj
            if average_distance < th:
                #too far away
                majorities.append(unique_new_obj)
                unique_new_obj = unique_new_obj -1
            else:
                majorities.append(most_common_class)

        return majorities

 
   
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
    


    #returns an object of the form fetures     features: object_id: all feature vectors of this id
    def read_ref_data(self,ref_path):
        features = {}

        # Open the HDF5 file for reading
        with h5py.File(ref_path, 'r') as hdf_file:
            # Iterate over each object ID (which corresponds to the dataset keys)
            for object_key in hdf_file.keys():
                # Read the dataset corresponding to the object_key
                stacked_features = hdf_file[object_key][:]
                
                # Convert the string key back to the original object_id if necessary
                object_id = object_key
                
                # Store the feature list in the dictionary
                features[object_id] = [stacked_features[i] for i in range(stacked_features.shape[0])]

        return features
    
    #returns an object of the form features    features: frame_idx : obj_id : feature vector of the id
    def read_scan_data(self, scan_path):
        features = {}

        with h5py.File(scan_path, 'r') as hdf_file:
            # Iterate over each frame_idx (which corresponds to the groups in the HDF5 file)
            for frame_idx in hdf_file.keys():
                # Initialize a dictionary for each frame_idx
                features[frame_idx] = {}
                
                # Access the group corresponding to the current frame_idx
                frame_group = hdf_file[frame_idx]
                
                # Iterate over each object_id within the current frame_idx group
                for object_key in frame_group.keys():
                    # Convert object_key back to object_id if necessary
                    object_id = object_key
                
                    # Retrieve the feature vector from the dataset
                    feature_vector = frame_group[object_key][:]
                    
                    # Store the feature vector in the dictionary under the object_id
                    features[frame_idx][object_id] = feature_vector

        return features

    #returns featuer in the form of features: frame: list of {objext_id, bbox, mask} objects
    def read_segmentation_data(segmentation_path):
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

        
    def compute(self):
        
    
        #loop over each frame and append the resultsto the total matricies
        for scan_id in tqdm(self.scan_ids):
            # Load image paths and frame indices
            frame_idxs_list = self.load_frame_idxs(self.scans_scenes_dir,scan_id)
            frame_idxs_list.sort()
            #access the necessary data for the reference scene
            reference_id = scan3r.get_reference_id(self.data_root_dir, scan_id)
            reference_info_path = osp.join("/media/ekoller/T7/Features2D/projection", self.model_name, "patch_{}_{}".format(self.image_patch_w,self.image_patch_h),self.mode, "{}.h5".format(reference_id))
            ref_data = self.read_ref_data(reference_info_path)
            
            #initialze the final structure
            all_matches = {}
            
            
            #build a treee structure fo a fast access of cosine similarity keeping also the obj_ids 
            ref_obj_ids = []
            ref_vectors = []
            for obj_id, feature_list in ref_data.items():
                for feature_vector in feature_list:
                    ref_vectors.append(feature_vector)
                    ref_obj_ids.append(obj_id)

            #convert to numpy for faiss
            ref_obj_ids = np.array(ref_obj_ids)
            ref_vectors = np.array(ref_vectors)

            #normalize vectors
            faiss.normalize_L2(ref_vectors)

            #make an faiss index
            dimension = ref_vectors.shape[1]  # vector dimension
            #use the cosine similarity (IP) indexing
            index = faiss.IndexFlatIP(dimension)
            index.add(ref_vectors)



            #access the scan feature info to iterate over it later
            scan_info_path = osp.join("/media/ekoller/T7/Features2D/dino_segmentation", self.model_name, "patch_{}_{}".format(self.image_patch_w,self.image_patch_h),self.mode, "{}.h5".format(scan_id))
            scan_data = self.read_scan_data(scan_info_path)


            #access the segmentation of the scan_id
            segmentation_info_path = osp.join("/media/ekoller/T7/files/Segmentation/DinoV2/objects", scan_id + ".h5")
            segmentation_data = self.read_segmentation_data(segmentation_info_path)


            #compute the matching Id for each Id per frame
            for infer_step_i in range(0, len(frame_idxs_list) // self.inference_step + 1):
                start_idx = infer_step_i * self.inference_step
                end_idx = min((infer_step_i + 1) * self.inference_step, len(frame_idxs_list))
                frame_idxs_sublist = frame_idxs_list[start_idx:end_idx]


                for frame_idx in frame_idxs_sublist:
                    #initialize for this frame
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
                        distances, indices = index.search(query_vector,max(self.k_means)) #get the max of k then we already know which ones are closer :)

                        # get the object ids of the closest reference vectors and the distances
                        nearest_obj_ids = [ref_obj_ids[idx] for idx in indices[0]]
                        nearest_distances = distances[0]

                        cosine_obj_ids.append(nearest_obj_ids)
                        cosine_distanc.append(nearest_distances)
                    
                    #we now have the ids of the frame and the corresponding closest objects and their distances yey
                
                    # get the majority vote of the k closest points
                    cosine_majorities = self.get_majorities(cosine_distanc, cosine_obj_ids, self.k_means, self.ths)
                    #convert the matches to a usable dictionary
                    matches = {}
                    for (frame_id, obj) in enumerate (zip(frame_obj_ids, cosine_majorities)):
                        matches[frame_id] = obj

                    all_matches[frame_idx] = matches
                    
                    

                    
        
                    
        

  
        #save the file in the results direcrtory
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
            
        #     # Iterate through frame indices
        #     for frame_idx in hdf_file.keys():
        #         matches = {}
                
        #         # Access the group for each frame index
        #         frame_group = hdf_file[frame_idx]
                
        #         # Load the frame_id -> obj mappings
        #         for frame_id in frame_group.keys():
        #             obj = frame_group[frame_id][()]
        #             matches[frame_id] = obj  # Convert back to int
                
        #         loaded_matches[frame_idx] = matches
                    
                
    
              

            


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
    evaluate = Evaluator(cfg, 'train')
    evaluate.compute()
   
  

    
if __name__ == "__main__":
    main()