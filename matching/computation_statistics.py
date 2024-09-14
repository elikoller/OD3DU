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
        for ref_scan in ref_scans_split[:]:
            #self.all_scans_split.append(ref_scan)
            # Check and add one rescan for the current reference scan
            rescans = [scan for scan in self.refscans2scans[ref_scan] if scan != ref_scan]
            if rescans:
                # Add the first rescan (or any specific rescan logic)
                self.all_scans_split.append(rescans[0])

        # print("reference scans ", np.array(ref_scans_split))
        # print("rescans ", self.all_scans_split)
        #this way we always work with the same things
        self.all_scans_split.sort()

        if self.rescan:
            self.scan_ids = ["0cac7672-8d6f-2d13-8d84-a0418e452bb7", "fcf66d8a-622d-291c-8429-0e1109c6bb26"]#self.all_scans_split[120:]
        else:
            self.scan_ids = ref_scans_split


        print("scan_id_length", len(self.scan_ids))




    
        #output path for components
        self.out_dir = "/media/ekoller/T7/Results" #osp.join("/media/ekoller/T7", "Results")
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

  

    def calculate_iou(self,mask1, mask2):
        #intersection/ union of 2 masks -> iou score
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        if union == 0:
            return 0
        return intersection / union
    

        #iterate over the objects of the new image and make iou of the bigges gt object it represents
    def compute_iou_metric_precision(self, gt_patches, computed_patches, new_objects):
        #get the ids over which we will iterate
        img_ids = np.unique(computed_patches)
        ious = []
        
        #iterate over them
        for id in img_ids:
            #ids which are bigger than 0 are matched to the objects which were already in the scenegraph
         
            #get the coordinates of this id
            coords = np.argwhere(computed_patches == id)

            #access the coords in the gt
            gt_ids_at_coords = gt_patches[tuple(coords.T)]
            #get the max id
            gt_max_id = Counter(gt_ids_at_coords).most_common(1)[0][0]
            
            #if the gt id is 0 we have no info
            if gt_max_id != 0:
                #check if it should be a new object 
                if id < 0:
                    #if is new object but did not get detected
                    if gt_max_id not in new_objects:
                        ious.append(0)
                        continue
                    #case predicted new and gt_max also belongs to new obj
                    #create the masks for the iou computeaiton
                    #since the ids do not match filter the ids individually
                    compued_mask = (computed_patches == id)
                    gt_mask = (gt_patches == gt_max_id)

                    #compute the iou
                    iou_id = self.calculate_iou(compued_mask, gt_mask)
                    ious.append(iou_id)

                #we look at a seen object
                else:
                    #the predicted id does not match the gt so already wront
                    if id != gt_max_id:
                        ious.append(0)
                        continue
                    #the object ids overlap so find the iou
                    #create the masks for the iou computeaiton
                    compued_mask = (computed_patches == id)
                    gt_mask = (gt_patches == gt_max_id)

                    #compute the iou
                    iou_id = self.calculate_iou(compued_mask, gt_mask)
                    ious.append(iou_id)
        
        if(len(ious) == 0):
            return np.nan

        return np.nanmean(ious)


    def compute_iou_metric_recall(self, gt_patches, computed_patches, new_objects):
        gt_ids  = np.unique(gt_patches)
        ious = []
       
        #iterate over them
        for id in gt_ids:
            #only do actual objects - not 0 id
           
            if id != 0:
                #ids which are bigger than 0 are matched to the objects which were already in the scenegraph
     
                #get the coordinates of this id
                coords = np.argwhere(gt_patches == id)

                #access the coords in the gt
                computed_ids_at_coords = computed_patches[tuple(coords.T)]
                #get the max id
                computed_max_id = Counter(computed_ids_at_coords).most_common(1)[0][0]
                #check if it should be a new object
                # gt new obj 
                
                if id in new_objects:
                    #if is new object but did not get predicted as unseen
                    if computed_max_id > 0:
                      
                        ious.append(0)
                        continue

                    #create the masks for the iou computeaiton
                    gt_mask = (gt_patches == id)
                    computed_mask = (computed_patches == computed_max_id)


                    #compute the iou
                    iou_id = self.calculate_iou(gt_mask, computed_mask)
                   
                    ious.append(iou_id)
                #we are talking about a seen object
                else:
                    #not the correct predicted id
                    if id != computed_max_id:
                      
                        ious.append(0)
                        continue
                     #create the masks for the iou computeaiton
                    gt_mask = (gt_patches == id)
                    computed_mask = (computed_patches == computed_max_id)

                    #compute the iou
                    iou_id = self.calculate_iou(gt_mask, computed_mask)
                  
                    ious.append(iou_id)

        if(len(ious) == 0):
            return np.nan
        
        return np.mean(ious)




      

    
    
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
        #print(f"Process {os.getpid()} is working on scene ID: {scan_id}")
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
        # print("presetn obj", present_obj_reference)
        # print("new obj", present_obj_scan)
        new_objects = list(set(present_obj_scan) - set(present_obj_reference))
        # print("unseen obj", new_objects)

        #init the result for this scan_id
        #this bool needed for later
      
        scan_cosine_iou_metric_precision = []
        scan_cosine_iou_metric_recall = []
        scan_cosine_metric_f1 = []


     
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
                cosine_iou_metric_precision = np.zeros((ths_len,k_means_len))
                cosine_iou_metric_recall = np.zeros((ths_len,k_means_len))
                cosine_metric_f1 = np.zeros((ths_len,k_means_len))
                

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

                        
                        #finally compute the accuracies: based on area and object ids and fill into the matrix
                        precision = self.compute_iou_metric_precision(gt_patches,cosine_patch_level, new_objects)
                        recall = self.compute_iou_metric_recall(gt_patches,cosine_patch_level, new_objects)
                        cosine_iou_metric_precision[t_idx][k_idx] = precision
                        cosine_iou_metric_recall[t_idx][k_idx] = recall
                        f1 = 0 #formula 2*(precision*recall)/(precision+recall)
                        if (not np.isnan(precision)) and (not np.isnan(recall)): 
                            if (precision+ recall) == 0:
                                f1 = 0.0
                            else:
                                f1 = 2 * (precision * recall) / (precision + recall)

                            cosine_metric_f1[t_idx][k_idx] = f1
                        #we got a nan value
                        else:
                            cosine_metric_f1[t_idx][k_idx] = np.nan
           
                scan_cosine_iou_metric_precision.append(cosine_iou_metric_precision)
                scan_cosine_iou_metric_recall.append(cosine_iou_metric_recall)
                scan_cosine_metric_f1.append(cosine_metric_f1)

        

            
                        
        return  np.nanmean(scan_cosine_iou_metric_precision,axis=0), np.nanmean(scan_cosine_iou_metric_recall,axis=0), np.nanmean(scan_cosine_metric_f1,axis=0)

    def compute(self, mode):
        #prepare the matricies for the 4 different metrics
        all_cosine_iou_metric_precision = []
        all_cosine_iou_metric_recall = []
        all_cosine_iou_metric_f1 = []
       
    
        # Use tqdm for progress bar, iterating as tasks are completed
        with tqdm(total=len(self.scan_ids)) as pbar:
            for scan_id in self.scan_ids:
                print("scanid", scan_id)
                cosine_iou_metric_precision, cosine_iou_metric_recall, cosine_metric_f1 = self.compute_scan(scan_id,mode)
                
                # get the result matricies
                all_cosine_iou_metric_precision.append(cosine_iou_metric_precision)
                all_cosine_iou_metric_recall.append(cosine_iou_metric_recall)
                all_cosine_iou_metric_f1.append(cosine_metric_f1)
                print("added results of scan id ", scan_id, " successfully")
            
                
                # progressed
                pbar.update(1)

            # with tqdm(total=len(self.scan_ids)) as pbar:
            #     for future in concurrent.futures.as_completed(futures):
            #         scan_id = futures[future]
            #         try:
            #             cosine_iou_metric_precision, cosine_iou_metric_recall, cosine_metric_f1 = future.result()
                        
            #            # get the result matricies
            #             all_cosine_iou_metric_precision.append(cosine_iou_metric_precision)
            #             all_cosine_iou_metric_recall.append(cosine_iou_metric_recall)
            #             all_cosine_iou_metric_f1.append(cosine_metric_f1)
            #             print("added results of scan id ", scan_id, " successfully")
            #         except Exception as exc:
            #             print(f"Scan {scan_id} generated an exception: {exc}")
            #             print("Traceback details:")
            #             traceback.print_exc()
                    
            #         # progressed
            #         pbar.update(1)

        # new_obj = []
        # for scan_id in tqdm(self.scan_ids, desc="Processing Scans"):
        #     # print an array of scenes 
        #     reference_id = scan3r.get_reference_id(self.data_root_dir, scan_id)
        #     present_obj_reference = self.get_present_obj_ids(self.data_root_dir,reference_id)
        #     present_obj_scan =  self.get_present_obj_ids(self.data_root_dir,scan_id)
        #     new_objects = list(set(present_obj_scan) - set(present_obj_reference))
        #     if len(new_objects) > 0:
        #         new_obj.append(reference_id)


        # print("array of references with new obj", new_obj)
        


       


        print("writing the file")
        #we want the result over all scenes
        # mean_cosine_iou_metric_precision = np.mean(np.array(all_cosine_iou_metric_precision), axis=0)
        # mean_cosine_iou_metric_recall = np.mean(np.array(all_cosine_iou_metric_recall), axis=0)
        # mean_cosine_metric_f1 = np.mean(np.array(all_cosine_iou_metric_f1), axis= 0)
      
        # print("precision", mean_cosine_iou_metric_precision)
        # print("recall", mean_cosine_iou_metric_recall) 
        # print("F1 score", mean_cosine_metric_f1)     

        #create sesult dict
        result = {"cosine_iou_metric_precision": all_cosine_iou_metric_precision,
                  "cosine_iou_metric_recall": all_cosine_iou_metric_recall,
                  "cosine_mectric_f1": all_cosine_iou_metric_f1
                }
                  
        #save the file in the results direcrtory
        result_dir = osp.join(self.out_dir,mode)
        common.ensure_dir(result_dir)
        result_file_path = osp.join(result_dir,  "statistics_segmentation_randomtest.pkl")
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