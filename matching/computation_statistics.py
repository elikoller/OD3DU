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
        for ref_scan in ref_scans_split[:2]:
            #self.all_scans_split.append(ref_scan)
            # Check and add one rescan for the current reference scan
            rescans = [scan for scan in self.refscans2scans[ref_scan] if scan != ref_scan]
            if rescans:
                # Add the first rescan (or any specific rescan logic)
                self.all_scans_split.append(rescans[0])

        # print("reference scans ", np.array(ref_scans_split))
        # print("rescans ", self.all_scans_split)

        if self.rescan:
            self.scan_ids = self.all_scans_split
        else:
            self.scan_ids = ref_scans_split


        print("scan_id_length", len(self.scan_ids))




    
        #output path for components
        self.out_dir = osp.join(self.data_root_dir, "Results" )
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
    

    
    def get_majorities(self, distanc, obj_ids ,frame_ids, k , th):
        #make te majority voting
        majorities = {}
        unique_new_obj = -1
        for i, (dist , ids, frame_id) in enumerate(zip(distanc,obj_ids, frame_ids)):
            closest_dist = dist[:k]
            closest_ids = ids[:k]

            #get the majority
            most_common_class, _ = Counter(closest_ids).most_common(1)[0]

            majority_distances = closest_dist[closest_ids == most_common_class]
            average_distance = np.mean(majority_distances)

            #thresthold the average majority of the distances, it it is above the threshold, give the id -1 which represents an unseen obj
            if average_distance < th:
                #too far away
                majorities[frame_id] =unique_new_obj
                unique_new_obj = unique_new_obj -1
            else:
                majorities[frame_id]= most_common_class

        return majorities

    def generate_pixel_level(self, segmentation,frame_obj_ids,majorities):
        pixel_ids = np.zeros((self.image_height, self.image_width))

        #iterate through all the segmentation regions of the dino segmentation
        for seg_region in segmentation:
            mask_id = seg_region["object_id"]
            #find the index of mask_id in frame_obj_ids
            #index = np.where(frame_obj_ids == mask_id)[0]
            #get to what the region mapped in the majorities
            matched_id = majorities[mask_id]
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

  

    def compute_patch_metric(self, gt_patches,computed_patches, new_objects):
        #make sure we dont do something dumm lol
        assert gt_patches.shape == computed_patches.shape, "Matrices must have the same shape"
        # Flatten matrices to iterate over each element
        flat_gt = gt_patches.flatten()
        flat_comp = computed_patches.flatten()
        
        #initialize the result
        percentage = []
    
        #iterate through everything

        for idx in range(len(flat_gt)):
            #since we can only compare to gt_patches skip everything that has gt_id == 0 == nothing
            if flat_gt[idx] != 0:
                #object ids are the same
                if flat_gt[idx] == flat_comp[idx]:
                    percentage.append(1)
                #the object id is actually new and was recognized as new
                elif (flat_gt[idx] in new_objects) and  (flat_comp[idx] < 0):
                    percentage.append(1)
                else:
                    percentage.append(0)

        percentage = np.array(percentage)
        mean_percentage = np.mean(percentage)
        return mean_percentage
    

    def has_frame_new_object(self, gt_patches,new_objects):
    
        #flatten the matricies
        flat_gt = gt_patches.flatten()
        present_obj = np.unique(flat_gt)
        intersection = set(present_obj) & set(new_objects)

        return intersection
       



    def compute_new_obj_metric(self, gt_patches,computed_patches, new_objects):
        #make sure we dont do something dumm lol
        assert gt_patches.shape == computed_patches.shape, "Matrices must have the same shape"

        #flatten the matricies
        flat_gt = gt_patches.flatten()
        flat_comp = computed_patches.flatten()
       
        #new objects

        new_mask_gt = (np.isin(flat_gt, new_objects)) & (flat_gt > 0)
        new_mask_comp =  flat_comp < 0
     
        new_combined = new_mask_gt & new_mask_comp
        
     
        new_filtered_comp = flat_comp[new_combined]
        
        
        #since we can not check if the ids correspond
        total_unique_new_comp = len(np.unique(new_filtered_comp))
        total_unique_new_gt = float(len(np.unique(flat_gt[new_mask_gt])))

        #it foo many new objects just set it to the gt
        if total_unique_new_comp > total_unique_new_gt:
            total_unique_new_comp = total_unique_new_gt

       
        return float(total_unique_new_comp) / total_unique_new_gt


    def compute_obj_metric(self, gt_patches,computed_patches, new_objects):
        #make sure we dont do something dumm lol
        assert gt_patches.shape == computed_patches.shape, "Matrices must have the same shape"

        #flatten the matricies
        flat_gt = gt_patches.flatten()
        flat_comp = computed_patches.flatten()
        

        #look at the not new objects
        #make a mask for not new objects and not id 0 for the gt
        mask_gt = (~np.isin(flat_gt, new_objects)) & (flat_gt > 0)
        #access the not new obj in the comp
        mask_comp = flat_comp > 0
        
        #combine the masks to get the idxs at which both are true so for both we are looking at existing objects
        combined_mask = mask_gt & mask_comp
      
        #the elements of the patches must have the same id
        same_mask = combined_mask & (flat_gt == flat_comp) 
        #check if the objects overlap at some point
        
        matched_comp = flat_comp[same_mask]

        #computed not new objects
        total_unique_comp = len((np.unique(matched_comp))) #unique ids we computed at the correct location
        total_unique_gt = float(len(np.unique(flat_gt[mask_gt]))) #how many unique objects ther would have been

        return float(total_unique_comp) /total_unique_gt
        
        
      

    
    
    #for a given scene get the colours of the differnt object_ids
    def get_present_obj_ids(self, data_dir,scan_id):
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
                features[int(object_id)] = [stacked_features[i] for i in range(stacked_features.shape[0])]

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
                    features[frame_idx][int(object_id)] = feature_vector

        return features

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


    def compute_scan(self,scan_id, mode):

    # Load image paths and frame indices
        frame_idxs_list = self.load_frame_idxs(self.scans_scenes_dir,scan_id)
        frame_idxs_list.sort()
        #access the necessary data for the reference scene
        reference_id = scan3r.get_reference_id(self.data_root_dir, scan_id)
        reference_info_path = osp.join("/media/ekoller/T7/Features2D/projection", self.model_name, "patch_{}_{}".format(self.image_patch_w,self.image_patch_h), mode, "{}.h5".format(reference_id))
        ref_data = self.read_ref_data(reference_info_path)
        
    
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
        scan_info_path = osp.join("/media/ekoller/T7/Features2D/dino_segmentation", self.model_name, "patch_{}_{}".format(self.image_patch_w,self.image_patch_h), mode, "{}.h5".format(scan_id))
        scan_data = self.read_scan_data(scan_info_path)


        #access the segmentation of the scan_id
        segmentation_info_path = osp.join("/media/ekoller/T7/Segmentation/DinoV2/objects", scan_id + ".h5")
        segmentation_data = self.read_segmentation_data(segmentation_info_path)





        #find out which objects have not been present in the reference scene ( not only frame!)
        present_obj_reference = self.get_present_obj_ids(self.data_root_dir,reference_id)
        present_obj_scan =  self.get_present_obj_ids(self.data_root_dir,scan_id)
        print("presetn obj", present_obj_reference)
        print("new obj", present_obj_scan)
        new_objects = list(set(present_obj_scan) - set(present_obj_reference))
        print("unseen obj", new_objects)

        #init the result for this scan_id
        scene_has_new_obj = False
        if len(new_objects) > 0:
            scan_cosine_new_obj_metric= []
            scene_has_new_obj= True
        scan_cosine_obj_metric= []  
        scan_cosine_patch_metric= []

     
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

                #initialize the resultmatrix for this frame
                cosine_obj_metric = np.zeros((ths_len,k_means_len))
                cosine__new_obj_metric = np.zeros((ths_len,k_means_len))
                cosine_patch_metric = np.zeros((ths_len,k_means_len))
                

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

                #access the gt for this frame
                gt_input_patchwise_path =  osp.join(self.scans_files_dir,"patch_anno","patch_anno_{}_{}".format(self.image_patch_w,self.image_patch_h),"{}.pkl".format(scan_id))

                with open(gt_input_patchwise_path, 'rb') as file:
                    gt_input_patchwise = pickle.load(file)

                gt_patches = gt_input_patchwise[frame_idx]

                #check if there are any new objects in the frame
                frame_has_new_obj = self.has_frame_new_object(gt_patches, new_objects)
                #get the correct computations and iteraste through every combination
                for t_idx, th in enumerate (self.ths):
                    for k_idx, k in enumerate (self.k_means):
                        # get the majority vote of the k closest points
                        cosine_majorities = self.get_majorities(cosine_distanc, cosine_obj_ids, frame_obj_ids, k, th)
                        # print("majorities", cosine_majorities)
                        # print("segmentation data", segmentation_data[frame_idx])
                        # print("frame obj ids", frame_obj_ids)
                        #translate the matched object ids to pixellevel of the frame
                        cosine_pixel_level = self.generate_pixel_level(segmentation_data[frame_idx],frame_obj_ids,cosine_majorities)
                        
                        #quantize to patchlevel of to be comparable to gt
                        cosine_patch_level = self.quantize_to_patch_level(cosine_pixel_level)


                        #access the ground truth patches for this frame
                        gt_input_patchwise_path =  osp.join(self.scans_files_dir,"patch_anno","patch_anno_{}_{}".format(self.image_patch_w,self.image_patch_h),"{}.pkl".format(scan_id))

                        with open(gt_input_patchwise_path, 'rb') as file:
                            gt_input_patchwise = pickle.load(file)

                        
                        #finally compute the accuracies: based on area and object ids and fill into the matrix
                        #for cosine
                        cosine_obj_metric[t_idx][k_idx] = self.compute_obj_metric(gt_patches,cosine_patch_level, new_objects)
                        
                        #only fill out if ther is a new object
                        if frame_has_new_obj:
                            cosine__new_obj_metric[t_idx][k_idx] = self.compute_new_obj_metric(gt_patches,cosine_patch_level, new_objects)

                        cosine_patch_metric[t_idx][k_idx] = self.compute_patch_metric(gt_patches,cosine_patch_level, new_objects)

                scan_cosine_obj_metric.append(cosine_obj_metric)
                if frame_has_new_obj:
                    scan_cosine_new_obj_metric.append(cosine__new_obj_metric)
                scan_cosine_patch_metric.append(cosine_patch_metric)

        

               
        if(len(scan_cosine_new_obj_metric)> 0):
             print("scanid wth new object", scan_id)      
                
                        
        return scan_cosine_obj_metric, scan_cosine_new_obj_metric, scan_cosine_patch_metric

    def compute(self, mode):
        #prepare the matricies for the 4 different metrics
        all_cosine_obj_metric = []
        all_cosine_new_obj_metric = []
        all_cosine_patch_metric = []
       
     

        workers = 3
        
        # parallelize the computations
        with concurrent.futures.ProcessPoolExecutor(max_workers= workers) as executor:
            futures = {executor.submit(self.compute_scan, scan_id, mode): scan_id for scan_id in self.scan_ids}
            
            # Use tqdm for progress bar, iterating as tasks are completed
            with tqdm(total=len(self.scan_ids)) as pbar:
                for future in concurrent.futures.as_completed(futures):
                    scan_id = futures[future]
                    try:
                        cosine_obj_metric, cosine_new_obj_metric, cosine_patch_metric = future.result()
                        
                       # get the result matricies
                        all_cosine_obj_metric.extend(cosine_obj_metric)
                        all_cosine_patch_metric.extend(cosine_patch_metric)
                        if len(cosine_new_obj_metric) > 0:
                            all_cosine_new_obj_metric.extend(cosine_new_obj_metric)
                        print("added results of scan id ", scan_id, " successfully")
                    except Exception as exc:
                        print(f"Scan {scan_id} generated an exception: {exc}")
                        print("Traceback details:")
                        traceback.print_exc()
                    
                    # progressed
                    pbar.update(1)

        # for scan_id in tqdm(self.all_scans_split, desc="Processing Scans", unit="scan"):

        #     cosine_obj_metric, cosine_new_obj_metric, cosine_patch_metric = self.compute_scan(scan_id, mode)
        #     # get the result matricies
        #     all_cosine_obj_metric.extend(cosine_obj_metric)
        #     all_cosine_patch_metric.extend(cosine_patch_metric)
        #     if len(cosine_new_obj_metric) > 0:
        #         all_cosine_new_obj_metric.extend(cosine_new_obj_metric)
        #     print("added results of scan id ", scan_id, " successfully")
       


        print("writing the file")
        #we want the result over all scenes
        mean_cosine_obj_metric = np.mean(np.array(all_cosine_obj_metric), axis= 0)
        mean_cosine_new_obj_metric = np.mean(np.array(all_cosine_new_obj_metric), axis=0)
        mean_cosine_patch_metric = np.mean(np.array(all_cosine_patch_metric), axis=0)
        
        print("obj", mean_cosine_obj_metric)
        print("new", mean_cosine_new_obj_metric)
        print("patch", mean_cosine_patch_metric)
       

        #create sesult dict
        result = {"cosine_obj_metric": mean_cosine_obj_metric,
                  "cosine_new_obj_metric" : mean_cosine_new_obj_metric,
                  "cosine_patch_metric": mean_cosine_patch_metric
                }
                  
        #save the file in the results direcrtory
        result_dir = osp.join(self.out_dir,mode)
        common.ensure_dir(result_dir)
        result_file_path = osp.join(result_dir,  "results_not_padded.pkl")
        common.write_pkl_data(result, result_file_path)
                    
                
    
              

            


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
    print("start avg computation")
    evaluate.compute("avg")
    print("start max computation")
    evaluate.compute("max")
    print("start median computation")
    evaluate.compute("median")
   
  

    
if __name__ == "__main__":
    main()