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
        self.cfg = cfg
        self.split = split
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
        self.out_dir = osp.join(self.scans_files_dir, "Results", "Matching_Prediction_" + self.split)
        print(self.out_dir)
        common.ensure_dir(self.out_dir)


    def eval_scan(self,scan_id, mode):
        
        # Load image paths and frame indices
        frame_idxs_list = scan3r.load_frame_idxs(self.scans_scenes_dir,scan_id)
        frame_idxs_list.sort()
        #access the necessary data for the reference scene
        reference_id = scan3r.get_reference_id(self.data_root_dir, scan_id)
        
    

        #access the segmentation of the scan_id
        segmentation_info_path = osp.join( self.scans_files_dir, "Segmentation/DinoV2/objects", scan_id + ".h5")
        segmentation_data = od3du_utils.read_segmentation_data(segmentation_info_path)

        #access the matched ids
        predicted_ids = od3du_utils.read_matching_data(self.scans_files_dir, scan_id)


        #find out which objects have not been present in the reference scene ( not only frame!)
        present_obj_reference = scan3r.get_present_obj_ids(self.data_root_dir,reference_id)
        present_obj_scan =  scan3r.get_present_obj_ids(self.data_root_dir,scan_id)
        new_objects = list(set(present_obj_scan) - set(present_obj_reference))
       
      
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
                cosine_metric_precision = np.zeros((ths_len,k_means_len))
                cosine_metric_recall = np.zeros((ths_len,k_means_len))
                cosine_metric_f1 = np.zeros((ths_len,k_means_len))
            

                #access the gt for this frame
                gt_input_patchwise_path =  osp.join(self.scans_files_dir,"patch_anno","patch_anno_{}_{}".format(self.image_patch_w,self.image_patch_h),"{}.pkl".format(scan_id))

                with open(gt_input_patchwise_path, 'rb') as file:
                    gt_input_patchwise = pickle.load(file)

                gt_patches = gt_input_patchwise[frame_idx]
                

    
                #get the correct computations and iteraste through every combination
                for t_idx, th in enumerate (self.ths):
                    for k_idx, k in enumerate (self.k_means):
                    
                        #translate the matched object ids to pixellevel of the frame
                        cosine_pixel_level = od3du_utils.generate_pixel_level(segmentation_data[frame_idx],predicted_ids[frame_idx], self.image_height, self.image_width)
                        
                        #quantize to patchlevel of to be comparable to gt
                        computed_patches = od3du_utils.quantize_to_patch_level(cosine_pixel_level, self. image_height, self.image_width, self.image_patch_h, self.image_patch_w)

                        
                        img_ids = np.unique(computed_patches)
                        #initialize everything
                        tp, fp, fn = 0, 0, 0
                        detected_gt_ids = set()
            
                        #iterate over the predicted images
                        for id in img_ids:
                            #ids which are bigger than 0 are matched to the objects which were already in the scenegraph
                        
                            #get the coordinates of this id
                            coords = np.argwhere(computed_patches == id)

                            #access the coords in the gt
                            gt_ids_at_coords = gt_patches[tuple(coords.T)]
                            #get the max id
                            gt_max_id = Counter(gt_ids_at_coords).most_common(1)[0][0]
                            

                            if gt_max_id != 0:
                                #we look at a seen object
                                if gt_max_id not in new_objects: #tbc we only look at the present objects in the reference scan
                                    #the predicted id does not match the gt so already wront
                                    if (id != gt_max_id):
                                        #tbc
                                        if (id > 0):
                                            fp += 1
                                            continue
                                    #the the id == gt_max id
                                    else:
                                        tp += 1
                                        detected_gt_ids.add(gt_max_id)
                                #tbc added logic for new objects
                                elif gt_max_id in new_objects:
                                    if id > 0:
                                        fp += 1

                        #now we also comlete false negatives so objects which got not detected
                        gt_ids = np.unique(gt_patches)   
                        for gt_id in gt_ids:
                            if gt_id == 0:
                                continue  # Skip background
                            
                            # If this ground truth object was not detected
                            if (gt_id not in detected_gt_ids) and (gt_id not in new_objects):  #tbc the new objects part
                                fn += 1



                        # calculate precision, recall, and F1-score
                        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

                        cosine_metric_precision[t_idx][k_idx] = precision
                        cosine_metric_recall[t_idx][k_idx] = recall
                        cosine_metric_f1[t_idx][k_idx] =  f1_score


                scan_cosine_iou_metric_precision.append(cosine_metric_precision)
                scan_cosine_iou_metric_recall.append(cosine_metric_recall)
                scan_cosine_metric_f1.append(cosine_metric_f1)

        
        return  np.nanmean(scan_cosine_iou_metric_precision,axis=0), np.nanmean(scan_cosine_iou_metric_recall,axis=0), np.nanmean(scan_cosine_metric_f1,axis=0)

    
    def compute_scan(self,scan_id, mode):
        
        #print(f"Process {os.getpid()} is working on scene ID: {scan_id}")
        # Load image paths and frame indices
        frame_idxs_list = scan3r.load_frame_idxs(self.scans_scenes_dir,scan_id)
        frame_idxs_list.sort()
        #access the necessary data for the reference scene
        reference_id = scan3r.get_reference_id(self.data_root_dir, scan_id)
        reference_info_path = osp.join(self.scans_files_dir, "Features2D/projection", self.model_name, "patch_{}_{}".format(self.image_patch_w,self.image_patch_h), mode, "{}.h5".format(reference_id))
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
        ref_vectors = np.array(ref_vectors)

        #normalize vectors
        faiss.normalize_L2(ref_vectors)

        #make an faiss index
        dimension = ref_vectors.shape[1]  # vector dimension
        #use the cosine similarity (IP) indexing
        index = faiss.IndexFlatIP(dimension)
        index.add(ref_vectors)



        #access the scan feature info to iterate over it later
        scan_info_path = osp.join(self.scans_files_dir, "Features2D/dino_segmentation", self.model_name, "patch_{}_{}".format(self.image_patch_w,self.image_patch_h), mode, "{}.h5".format(scan_id))
        scan_data = od3du_utils.read_scan_data(scan_info_path)


        #access the segmentation of the scan_id
        segmentation_info_path = osp.join(self.scans_files_dir, "Segmentation/DinoV2/objects", scan_id + ".h5")
        segmentation_data = od3du_utils.read_segmentation_data(segmentation_info_path)





        #find out which objects have not been present in the reference scene ( not only frame!)
        present_obj_reference = scan3r.get_present_obj_ids(self.data_root_dir,reference_id)
        present_obj_scan =  scan3r.get_present_obj_ids(self.data_root_dir,scan_id)
        new_objects = list(set(present_obj_scan) - set(present_obj_reference))

      
        scan_cosine_metric_precision = []
        scan_cosine_metric_recall = []
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
                cosine_metric_precision = np.zeros((ths_len,k_means_len))
                cosine_metric_recall = np.zeros((ths_len,k_means_len))
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
                        cosine_majorities = od3du_utils.get_majorities(cosine_distanc, cosine_obj_ids, frame_obj_ids, k, th)

                        #translate the matched object ids to pixellevel of the frame
                        cosine_pixel_level = od3du_utils.generate_pixel_level(segmentation_data[frame_idx],cosine_majorities, self.image_height, self.image_width)
                        
                        #quantize to patchlevel of to be comparable to gt
                        computed_patches = od3du_utils.quantize_to_patch_level(cosine_pixel_level, self.image_height, self.image_width, self.image_patch_h, self.image_patch_w)

                        img_ids = np.unique(computed_patches)
                    
                        #initialize everything
                        tp, fp, fn = 0, 0, 0
                        detected_gt_ids = set()
            
                        #iterate over the predicted images
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
                                if gt_max_id not in new_objects: #tbc we only look at the present objects in the reference scan
                                    #the predicted id does not match the gt so already wront
                                    if (id != gt_max_id):
                                        #tbc
                                        if (id > 0):
                                            fp += 1
                                            continue
                                    #the the id == gt_max id
                                    else:
                                        tp += 1
                                        detected_gt_ids.add(gt_max_id)
                                #tbc added logic for new objects
                                elif gt_max_id in new_objects:
                                    if id > 0:
                                        fp += 1

                        #now we also comlete false negatives so objects which got not detected
                        gt_ids = np.unique(gt_patches)   
                        for gt_id in gt_ids:
                            if gt_id == 0:
                                continue  # Skip background
                            
                            # If this ground truth object was not detected
                            if (gt_id not in detected_gt_ids) and (gt_id not in new_objects):  #tbc the new objects part
                                fn += 1


                        # calculate precision, recall, and F1-score
                        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

                        cosine_metric_precision[t_idx][k_idx] = precision
                        cosine_metric_recall[t_idx][k_idx] = recall
                        cosine_metric_f1[t_idx][k_idx] =  f1_score

                        
                       
                scan_cosine_metric_precision.append(cosine_metric_precision)
                scan_cosine_metric_recall.append(cosine_metric_recall)
                scan_cosine_metric_f1.append(cosine_metric_f1)

        

                        
        return  np.nanmean(scan_cosine_metric_precision,axis=0), np.nanmean(scan_cosine_metric_recall,axis=0), np.nanmean(scan_cosine_metric_f1,axis=0)

    def compute(self, mode):
        #prepare the matricies for the 4 different metrics
        all_cosine_metric_precision = []
        all_cosine_metric_recall = []
        all_cosine_metric_f1 = []
       
        best_scenes = []
        best_f1scores = []
        best_precisions = []
        best_recalls = []
    
        # Use tqdm for progress bar, iterating as tasks are completed
        with tqdm(total=len(self.scan_ids)) as pbar:
            for scan_id in self.scan_ids:
                print("scanid", scan_id)
                if self.split == "train":
                    cosine_metric_precision, cosine_metric_recall, cosine_metric_f1 = self.compute_scan(scan_id,mode)
                
                if self.split == "test":
                    cosine_metric_precision, cosine_metric_recall, cosine_metric_f1 = self.eval_scan(scan_id,mode)
                # get the result matricies
                all_cosine_metric_precision.append(cosine_metric_precision)
                all_cosine_metric_recall.append(cosine_metric_recall)
                all_cosine_metric_f1.append(cosine_metric_f1)
                print("added results of scan id ", scan_id, " successfully")
            
                if self.split == "test":
                    best_scenes.append(scan_id)
                    best_f1scores.append(cosine_metric_f1[0][0])
                    best_precisions.append(cosine_metric_precision[0][0])
                    best_recalls.append(cosine_metric_recall[0][0])
                # progressed
                pbar.update(1)
        #create sesult dict
        result = {"cosine_iou_metric_precision": np.mean(all_cosine_metric_precision, axis = 0),
                  "cosine_iou_metric_recall": np.mean(all_cosine_metric_recall, axis = 0),
                  "cosine_mectric_f1": np.mean(all_cosine_metric_f1, axis = 0),
                }
                  
        #save the file in the results direcrtory
        result_dir = osp.join(self.out_dir,mode)
        common.ensure_dir(result_dir)
        result_file_path = osp.join(result_dir,  "statistic_predicted_matches.pkl")
        common.write_pkl_data(result, result_file_path)



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
    print("start avg computation")
    evaluate.compute("avg")

    # evaluate.compute("max")
    # evaluate.compute("median")
   
  

    
if __name__ == "__main__":
    main()