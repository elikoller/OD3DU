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
from sceneGraph_update import predict_objects

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
        self.minimum_votes = cfg.parameters.minimum_votes
        self.overlap_th = cfg.parameters.overlap_th
        self.voxel_size_overlap = cfg.parameters.voxel_size_overlap
        self.box_scale = cfg.parameters.box_scale
        self.minimum_points = cfg.parameters.minimum_points


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
        self.out_dir = osp.join(self.data_root_dir, "Resuls", "Center_Prediction_" + self.split)
        common.ensure_dir(self.out_dir)

    def create_smaller_boundingbox(self,large_bbox):
        
        # get the center of the large box
        min_coords = np.min(large_bbox, axis=0)
        max_coords = np.max(large_bbox, axis=0)
        center = (min_coords + max_coords) / 2
        
        # get the halfdimensions
        half_dims = (max_coords - min_coords) / 2
        
        # half dimensions of the boundingbox
        small_half_dims = half_dims * self.box_scale
        
        # define the smaller box
        smaller_bbox = np.array([
            center - small_half_dims,
            center + small_half_dims
        ])
        
        return smaller_bbox

    def is_in_boundingbox(self, center, boundingbox):
        min_coords = np.min(boundingbox, axis=0)
        max_coords = np.max(boundingbox, axis=0)

    
        is_inside = (np.all(min_coords <= center) and np.all(center <= max_coords))

        return is_inside
    
    def compute_bounding_box(self,points):
        min_point = np.min(points, axis=0)  # Minimum x, y, z
        max_point = np.max(points, axis=0)  # Maximum x, y, z
        return min_point, max_point

    def boundingbox_iou(self,boundingbox, pcl):
        
        pred_bbox = self.compute_bounding_box(pcl)
        gt_bbox = self.compute_bounding_box(boundingbox)

        #get the different sizes
        pred_size = pred_bbox[1] - pred_bbox[0]
        gt_size = gt_bbox[1] - gt_bbox[0]

        #sort them
        sorted_pred_size = np.sort(pred_size)
        sorted_gt_size = np.sort(gt_size)

        adjusted_size = np.empty_like(pred_size)
        for i in range(3):
            adjusted_size[np.argsort(pred_size)[i]] = sorted_gt_size[i]

        #adjust the predicted box around its center such that it fits the gt boundingbox in terms of dimensions, keep the predicted aspect ratio
        pred_center = (pred_bbox[0] + pred_bbox[1]) / 2
        adjusted_bbox_min = pred_center - (adjusted_size / 2)
        adjusted_bbox_max = pred_center + (adjusted_size / 2)

        min1 = np.min(boundingbox, axis=0)  
        max1 = np.max(boundingbox, axis=0)  
        min2 = adjusted_bbox_min 
        max2 = adjusted_bbox_max

        # Compute the intersection box
        intersect_min = np.maximum(min1, min2) 
        intersect_max = np.minimum(max1, max2)  

        # Compute intersection box dimensions
        intersect_dims = np.maximum(intersect_max - intersect_min, 0)  
        intersection_volume = np.prod(intersect_dims)  

        # Compute volumes of each bounding box
        volume1 = np.prod(max1 - min1)  
        volume2 = np.prod(max2 - min2)  

        # Compute union volume
        union_volume = volume1 + volume2 - intersection_volume

        # Compute IoU
        iou = intersection_volume / union_volume if union_volume > 0 else 0
        return iou 
            




    def compute_centers(self,scan_id, overlap_threshold):

    # Load image paths and frame indices
        frame_idxs_list = scan3r.load_frame_idxs(self.scans_scenes_dir,scan_id)
        frame_idxs_list.sort()
       
        
        #access the segmentation of the scan_id
        segmentation_info_path = osp.join(self.scans_files_dir, "Segmentation/DinoV2/objects", scan_id + ".h5")
        segmentation_data = od3du_utils.read_segmentation_data(segmentation_info_path)

        #access the matched data
        matches = od3du_utils.read_matching_data(self.scans_files_dir,scan_id)

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
                segmented_img_path = osp.join(self.scans_files_dir, "Segmentation/DinoV2/color", scan_id, "frame-{}.jpg".format(frame_idx))
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
                    if object_id > 0: #tbc
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
                                overlap = od3du_utils.do_pcl_overlap(self.voxel_size_overlap, obj_pcl, cluster)

                                # keep track of the most overlap cluste
                                if overlap > overlap_threshold and overlap > max_overlap:
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
 
    def compute_statistics(self,scan_id, overlap_threshold):

        #adjust the logic to compute 
        raw_predicted_centers = {}
            #compute the centers if needed
        if self.split == "train":
            raw_predicted_centers = self.compute_centers(scan_id,overlap_threshold)


        #read the data for the test set
        if self.split == "test":
            raw_predicted_centers = od3du_utils.read_predicted_data(self.scans_files_dir, scan_id)
   
        #initialize the results
        precisions = np.zeros((len(self.minimum_points), len(self.minimum_votes)))
        recalls = np.zeros((len(self.minimum_points), len(self.minimum_votes)))
        f1s = np.zeros((len(self.minimum_points), len(self.minimum_votes)))
        boundingboxes = np.zeros((len(self.minimum_points), len(self.minimum_votes)))
        avg_center_distance = np.zeros((len(self.minimum_points), len(self.minimum_votes)))
        

        #access gt pointcenters
        pklfile = osp.join(self.data_root_dir, 'files', 'orig', 'data', '{}.pkl'.format(scan_id))

        with open(pklfile, "rb") as f:
            # Load the data from the pickle file
            data = pickle.load(f)
            
        # extract object points and IDs from the pickle data
        gt_ids = data['objects_id']
        gt_centers = data["object_centers"]
        gt_boxes = data['bounding_boxes']
    
        #get the reference id
        reference_id = scan3r.get_reference_id(self.data_root_dir, scan_id)
        #find out which objects have not been present in the reference scene ( not only frame!)

        pklfile = osp.join(self.data_root_dir, 'files', 'orig', 'data', '{}.pkl'.format(reference_id))

        with open(pklfile, "rb") as f:
            # Load the data from the pickle file
            data = pickle.load(f)
            
        # extract object points and IDs from the pickle data
        ref_gt_ids = data['objects_id']
        new_objects = list(set(gt_ids) - set(ref_gt_ids))
        if len(new_objects) > 0:
            print("scan id has new objects", scan_id)


        #calculate the different metrics
        for j, min_point in enumerate(self.minimum_points):
            for i, min_vote in enumerate(self.minimum_votes):
                #only look at the predicted points
            
                predicted = {}
                #initialize 
                true_positives = 0
                false_negatives = 0
                false_positives = 0
                ious = []
                center_difference = []

                #remove points which would not pass
                for obj_id , obj_data in raw_predicted_centers.items():
                    votes = obj_data["votes"]
                    size = obj_data["size"]
                    #check if it is detectable by the parameters
                                
                    if (votes >= min_vote) and (size >= min_point):
                        predicted[obj_id] = obj_data
                        if obj_id < 0:
                            print("new object pointcloud")

            
                matched_predicted_ids = set()
                #oke now compute the true posities and false negatives
                for obj_id in gt_ids:
                    #we are not looking at a unseen object
                    #get the boundingbox for the object to calculate the threshold
                    if obj_id not in new_objects: #tbc remove this if
                        gt_center = gt_centers[obj_id]
                        boundingbox = gt_boxes[obj_id]
                        matched = False                   
                        #the object was predicted
                        if (obj_id in predicted.keys()):                        
                                #access the center
                                pred_center = predicted[obj_id]['center']
                                distance = np.linalg.norm(pred_center - gt_center)
                                center_difference.append(distance)
                                bbox_iou = self.boundingbox_iou(boundingbox, predicted[obj_id]["points"])
                                ious.append(bbox_iou)
                                matched = True
                                matched_predicted_ids.add(obj_id) 
                                if self.is_in_boundingbox(pred_center, boundingbox):
                                    #Predicted a Center and it is Within the Bounding Box (True Positive)
                                    true_positives = true_positives + 1  
                                else:
                                    #Predicted a Center but it is Outside the Bounding Box (False Positive)
                                    false_positives = false_positives + 1
                        # If no prediction matched the ground truth center, count as false negative
                        elif not matched:
                            false_negatives += 1
                    
                # Calculate false positives (predicted centers that did not match any ground truth center)
                false_positives = false_positives +  len(predicted) - len(matched_predicted_ids)
            
                # Calculate precision, recall, and false positive rate
                if true_positives + false_positives == 0:
                    precision = 0.0
                else:
                    precision = true_positives / (true_positives + false_positives)

                if true_positives + false_negatives == 0:
                    recall = 0.0
                else:
                    recall = true_positives / (true_positives + false_negatives)

                if precision + recall == 0:
                    f1_score = 0.0
                else:
                    f1_score = 2 * (precision * recall) / (precision + recall)

                if len(ious) > 0:
                    avg_iou = sum(ious) / len(ious)
                else:
                    avg_iou = 0.0

                if len(center_difference) > 0:
                    avg_center = sum(center_difference) / len(center_difference)
                else:
                    avg_center = 0.0

        

               
                precisions[j][i]= precision
                recalls[j][i] = recall
                f1s[j][i] =  f1_score
                boundingboxes[j][i] = avg_iou
                avg_center_distance[j][i] = avg_center


        return precisions,recalls,f1s, boundingboxes, avg_center_distance
          

    def compute(self, obverlap_threshold):
        #prepare the matricies for the 4 different metrics
        all_precision = []
        all_recall = []
        all_f1 = []
        all_boxes = []
        all_centers = []
        best_scenes = []
        best_f1scores = []
        best_precisions = []
        best_recalls = []
        best_distances = []

    
    # Use tqdm for progress bar, iterating as tasks are completed
        with tqdm(total=len(self.scan_ids)) as pbar:
            for scan_id in self.scan_ids:
                print("scanid", scan_id)
                precisions, recalls, f1s, bounsingboxes, avg_centers = self.compute_statistics(scan_id, obverlap_threshold)
                
                # get the result matricies
                all_precision.append(precisions)
                all_recall.append(recalls)
                all_f1.append(f1s)
                all_boxes.append(bounsingboxes)
                all_centers.append(avg_centers)


                best_scenes.append(scan_id)
                best_f1scores.append(f1s[0][0])
                best_precisions.append(precisions[0][0])
                best_recalls.append(recalls[0][0])
                best_distances.append(avg_centers[0][0])
                print("added results of scan id ", scan_id, " successfully")
                pbar.update(1)


        #create sesult dict
        result = {"precision": np.mean(all_precision, axis = 0),
                "recall": np.mean(all_recall,  axis = 0),
                "f1": np.mean(all_f1,  axis = 0),
                "iou_boxes": np.mean(all_boxes,  axis = 0),
                "mean_center_difference": np.mean(all_centers,  axis = 0),
                "s_scene": best_scenes,
                "s_precision":best_precisions, 
                "s_recall": best_recalls,
                "s_f1":best_f1scores,
                "s_distance": best_distances
                }

    

        #save the file in the results direcrtory
        result_dir = osp.join(self.out_dir, str(obverlap_threshold))
        common.ensure_dir(result_dir)
        result_file_path = osp.join(result_dir,  "statistics_predicted_centers.pkl")
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


    #do the evaluations for the train set; 
    evaluate = Evaluator(cfg, split)
    print("start mask computation")
    with tqdm(total=len(cfg.parameters.overlap_th), desc="Overall Progress") as overall_pbar:
        for threshold in cfg.parameters.overlap_th:
             evaluate.compute(threshold)
             overall_pbar.update(1)
   

   
    
if __name__ == "__main__":
    main()