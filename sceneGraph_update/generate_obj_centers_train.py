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
        self.minimum_votes = cfg.parameters.minimum_votes
        self.minimum_points: cfg.parameters.minimum_points
        self.overlap_th = cfg.parameters.overlap_th
        self.voxel_size_downsampling = cfg.parameters.voxel_size_downsampling
        self.voxel_size_overlap = cfg.parameters.voxel_size_overlap


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
        for ref_scan in ref_scans_split[:1]:
            #self.all_scans_split.append(ref_scan)
            # Check and add one rescan for the current reference scan
            rescans = [scan for scan in self.refscans2scans[ref_scan] if scan != ref_scan]
            if rescans:
                # Add the first rescan (or any specific rescan logic)
                self.all_scans_split.append(rescans[0])


        if self.rescan:
            self.scan_ids = ["fcf66d8a-622d-291c-8429-0e1109c6bb26"] #self.all_scans_split
        else:
            self.scan_ids = ref_scans_split
    
        #output path for components
        #self.out_dir = osp.join(self.data_root_dir, "Updates","depth_img")
        self.out_dir = osp.join("/media/ekoller/T7/center_statistics")
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
    

    #return an object with the structure:scan_id: frame_number: frame_obj_id: matched id
    def read_matching_data(self, scan_id):
        # get the file and iterate through everything to create an object
        matchfile = osp.join(self.data_root_dir, "Predicted_Matches", scan_id + ".h5")
        with h5py.File(matchfile, 'r') as hdf_file:
            loaded_matches = {}
            
            # Iterate through frame indices
            for frame_idx in hdf_file.keys():
                matches = {}
                
                # Access the group for each frame index
                frame_group = hdf_file[frame_idx]
                
                # Load the frame_id -> obj mappings
                for frame_id in frame_group.keys():
                    obj = frame_group[frame_id][()]
                    matches[frame_id] = int(obj)  # Convert back to int
                
                loaded_matches[frame_idx] = matches 

        return loaded_matches
    

    def pose_in_reference(self, scan_id , pose_rescan):
        #same coordinate system
        ref_id = scan3r.get_reference_id(self.data_root_dir,scan_id)
        #if we want the coords in the reference coordinate system return the boxes (based on pkl file)
        if scan_id==ref_id:
            return pose_rescan
        

        #transform the centers of rescan to ref coord
        path = osp.join(self.data_root_dir,"files", "3RScan.json")
        map_id_to_trans = scan3r.read_transform_mat(path)
        transform = map_id_to_trans[scan_id]
        transform= transform.reshape(4,4)

        #transform the pose

        return  transform.transpose() * pose_rescan

    def transform_to_3d(self, scan_id, depth_map, frame_number):
        
        #access the extrinsic/pose of the camera
        pose_rescan = scan3r.load_pose(self.scans_scenes_dir, scan_id, frame_number)
        pose_in_ref = self.pose_in_reference( scan_id, pose_rescan)
        
        camera_info = scan3r.load_intrinsics(self.scans_scenes_dir, scan_id)
        intrinsics = camera_info['intrinsic_mat']
        img_width = int(camera_info['width'])
        img_height = int(camera_info['height'])

        """
        do the computations based on following formula 

        """
        #from 2d to camera coordinates xc = (u-cx)*z / fx,   yc = (v-cy)*z/ fy    , zc= z 
        #create a mesh grid 
        u, v = np.meshgrid(np.arange(img_width), np.arange(img_height))

        #also access the intrinsic values which are stored the following way
        # intrinsic_mat = np.array([[intrinsic_fx, 0, intrinsic_cx],
        #                                     [0, intrinsic_fy, intrinsic_cy],
        #                                     [0, 0, 1]])

        fx = intrinsics[0, 0]  # Focal length in x direction
        fy = intrinsics[1, 1]  # Focal length in y direction
        cx = intrinsics[0, 2]  # Principal point x
        cy = intrinsics[1, 2]  # Principal point y
        #flatten everything for computations
        u_flat = u.flatten()
        v_flat = v.flatten()
        depth_flat = depth_map.flatten()

        #apply the formula from above
        x_c = (u_flat - cx) * depth_flat / fx
        y_c = (v_flat - cy) * depth_flat / fy
        z_c = depth_flat

        #turn the camera coordinates into homogeneous coordinates
        camera_coords_homog  = np.vstack((x_c, y_c, z_c, np.ones_like(x_c)))  

        #apply the extrinsic matrix
        world_coords_homog = pose_in_ref @ camera_coords_homog
        #normalize
        world_coords_homog /= world_coords_homog[3, :]  

        world_coords = world_coords_homog[:3,:]
        world_coords_T = world_coords.T
    

        return world_coords_T
    
    def isolate_object_coordinates(self,world_coordinates, mask):
        #make sure it is an array
        mask = np.array(mask)
        #flatten & turn into boolean mask
        mask = mask.flatten()
        mask = mask.astype(bool)
        #get the part belonging to the object
        obj_coordinates = world_coordinates[mask]

        return obj_coordinates

    def voxel_grid_to_coordinates(self,voxel_grid):
        """Extract voxel coordinates from a VoxelGrid object."""
        voxels = voxel_grid.get_voxels()
        voxel_coords = np.array([voxel.grid_index for voxel in voxels])
        return voxel_coords


    def compare_voxel_grids(self, voxel_grid1, voxel_grid2):
        """Compare two voxel grids to see how much they overlap."""
        coords1 = self.voxel_grid_to_coordinates(voxel_grid1)
        coords2 = self.voxel_grid_to_coordinates(voxel_grid2)
        
        # Convert to sets of tuples for intersection
        voxels1_set = set(map(tuple, coords1))
        voxels2_set = set(map(tuple, coords2))
        
        # Compute intersection
        intersection = voxels1_set.intersection(voxels2_set)
        union = voxels1_set.union(voxels2_set)
        
        similarity = len(intersection) / len(union) if len(union) > 0 else 0
        return similarity


    def do_pcl_overlap(self,obj_pcl, cluster):
        #create a voxel grid
        #turn into pointclouds
        obj_point_cloud = o3d.geometry.PointCloud()
        obj_point_cloud.points = o3d.utility.Vector3dVector(obj_pcl)
        voxel_grid1 = o3d.geometry.VoxelGrid.create_from_point_cloud(obj_point_cloud, self.voxel_size_overlap)

        cluster_point_cloud = o3d.geometry.PointCloud()
        cluster_point_cloud.points = o3d.utility.Vector3dVector(cluster)
        voxel_grid2 = o3d.geometry.VoxelGrid.create_from_point_cloud(cluster_point_cloud, self.voxel_size_overlap)
    
        
        """Compare two voxel grids to see how much they overlap."""
        return self.compare_voxel_grids(voxel_grid1, voxel_grid2)
        


    def compute_centers(self,scan_id):

    # Load image paths and frame indices
        frame_idxs_list = self.load_frame_idxs(self.scans_scenes_dir,scan_id)
        frame_idxs_list.sort()
       
        
        #access the segmentation of the scan_id
        segmentation_info_path = osp.join("/media/ekoller/T7/Segmentation/DinoV2/objects", scan_id + ".h5")
        segmentation_data = self.read_segmentation_data(segmentation_info_path)

        #access the matched data
        matches = self.read_matching_data(scan_id)

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
                depth_mat_resized = pgm_file.resize((self.image_height,self.image_width), Image.NEAREST) 
            
                #depth is given in mm so put it into m
                depth_mat = np.array(depth_mat_resized)
                depth_mat = depth_mat * 0.001

                #transform to world coordinates in the reference frame
                world_coordinates_frame = self.transform_to_3d(scan_id, depth_mat, frame_idx)

                new_obj_idx = 0
               
                #iterate through the masks of the objec
                for boundingboxes in segmentation_data[frame_idx]:
                    #access the mask for the object
                    mask = boundingboxes['mask']
                
                    #get the dino object_id 
                    dino_id = boundingboxes["object_id"]
                    #print("frame ", frame_idx, " dino_id ", dino_id)
                    #get the matched id
                    #print("matches frame ", frame_matches)
                    object_id = frame_matches[str(dino_id)]
                    

                    #isolate only the object pointcloud
                    obj_pcl = self.isolate_object_coordinates(world_coordinates_frame, mask)
                    #not a new object so regular precedure
                    if object_id > 0:
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
                                overlap = self.do_pcl_overlap(obj_pcl, cluster)

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
                    #new object
                    else:
                        #get the negative keys
                        negative_keys = [object_id for object_id in all_clusters.keys() if object_id < 0]
                        
                        #no negative keys yet
                        if len(negative_keys) == 0:
                            new_obj_idx = new_obj_idx - 1
                            all_clusters[new_obj_idx] = [{'cluster': obj_pcl, 'votes': 1}]
                        #since we don't know the correspondance of the points we just add id to the new cluster with the most points
                        else:
                            max_overlap = 0
                            best_cluster_index = None
                            best_object_id = None

                            # iterate over every cluster to get the one with the most overlap
                            for neg_key in negative_keys:
                                for i, cluster_data in enumerate(all_clusters[neg_key]):
                                    cluster = cluster_data['cluster']
                                    overlap = self.do_pcl_overlap(obj_pcl, cluster)

                                    # Track the cluster with the highest overlap
                                    if overlap > self.overlap_th and overlap > max_overlap:
                                        max_overlap = overlap
                                        best_cluster_index = i
                                        best_object_id = neg_key

                            # we found a best cluster so merge it
                            if best_object_id is not None and best_cluster_index is not None:
                                best_cluster = all_clusters[best_object_id][best_cluster_index]['cluster']
                                merged_points = np.vstack((obj_pcl, best_cluster))

                                # Update the best cluster with the merged points
                                all_clusters[best_object_id][best_cluster_index]['cluster'] = merged_points

                                # increment the vote
                                all_clusters[best_object_id][best_cluster_index]['votes'] += 1
                            else:
                                # did not find a good cluster create a new one
                                new_obj_idx = new_obj_idx -1
                                all_clusters[new_obj_idx] = [{'cluster': obj_pcl, 'votes': 1}]





        #now that we have the lists of clusters we need to iterate over them and choose the biggest cluster, downsample it & take the average to predict the center
        #initialize final object
        all_centers = {}
        #iterte through the objects
        for obj_id, clusters in enumerate(all_clusters):
            #get the cluster with the most points aka largest 
            print(clusters)
            #decide the most likely correct cluster based on votes first and then size
            largest_cluster_data = max(all_clusters[object_id], key=lambda c: (c['votes'], len(c['cluster'])))
            largest_cluster = largest_cluster_data['cluster']
            largest_cluster_votes = largest_cluster_data["votes"]
            #create the objec center
            obj_center = np.mean(largest_cluster, axis= 0)

            
        
            
            #return the object for the evaluation
            all_centers[obj_id] = {
                'center': obj_center,
                "points": len(largest_cluster),
                "votes" : largest_cluster_votes
            }
            #print(all_centers[0])

            return all_centers
 


    

        
    def compute_statistics(self,scan_id):
        #compute the centers
        predicted_centers = self.compute_centers(scan_id)
        #initialize the results
        precisions = np.zeros((self.minimum_points, self.minimum_points))
        recalls = np.zeros((self.minimum_points, self.minimum_points))
        f1s = np.zeros((self.minimum_points, self.minimum_points))
        false_positive_rates = np.zeros((self.minimum_points, self.minimum_points))

        #access gt pointcenters
        pklfile = osp.join(self.data_root_dir, 'files', 'orig', 'data', '{}.pkl'.format(scan_id))

        with open(pklfile, "rb") as f:
            # Load the data from the pickle file
            data = pickle.load(f)
            
        # extract object points and IDs from the pickle data
        gt_ids = data['objects_id']
        gt_centers = data["object_centers"]
    
    
        #calculate the different metrics
        for min_vote in enumerate(self.minimum_votes):
            for num_points in enumerate(self.minimum_points):
                #filter the points based on the minvote and numpoints
                predicted = {}
                #initialize 
                true_positives = 0
                false_negatives = 0
                false_positives = 0

                #remove points which would not pass
                for obj_id , obj_data in predicted_centers.items():
                    votes = obj_data["votes"]
                    num_points = obj_data["points"]
                    #check if it is detectable by the parameters
                                
                    if any(votes >= min_vote) and any(num_points >= num_points):
                        predicted[obj_id] = obj_data

                matched_predicted_ids = set()
                """ 
                add logiv for newly seeno objects
                """
                #oke now compute the true posities and false negatives
                for obj_id in gt_ids:
                    if obj_id in gt_centers:
                        gt_center = gt_centers[obj_id]
                        matched = False

                        for pred_id, pred_data in predicted.items():
                            pred_center = pred_data['center']
                            distance = np.linalg.norm(pred_center - gt_center)

                            if distance <= 0.2:
                                true_positives += 1
                                matched = True
                                matched_predicted_ids.add(pred_id)
                                break  # Exit the loop once a match is found

                        # If no prediction matched the ground truth center, count as false negative
                        if not matched:
                            false_negatives += 1

                # Calculate false positives (predicted centers that did not match any ground truth center)
                false_positives = len(predicted) - len(matched_predicted_ids)

                # Total number of objects (ground truth) for computing the FPR
                total_objects = len(gt_ids)

                # Calculate precision, recall, and false positive rate
                if true_positives + false_positives == 0:
                    precision = 0.0
                else:
                    precision = true_positives / (true_positives + false_positives)

                if true_positives + false_negatives == 0:
                    recall = 0.0
                else:
                    recall = true_positives / (true_positives + false_negatives)

                if total_objects == 0:
                    false_positive_rate = 0.0
                else:
                    false_positive_rate = false_positives / (true_positives + false_positives + false_negatives)

                
                if precision + recall == 0:
                    f1_score = 0.0
                else:
                    f1_score = 2 * (precision * recall) / (precision + recall)
               
                precisions[min_vote][num_points]= precision
                recalls[min_vote][num_points] = recall
                f1s[min_vote][num_points] =  f1_score

        return precisions,recalls,f1s
          

    def compute(self):
        workers = 2
        
        # parallelize the computations
        with concurrent.futures.ProcessPoolExecutor(max_workers= workers) as executor:
            futures = {executor.submit(self.compute_scan, scan_id): scan_id for scan_id in self.scan_ids}
            
            # Use tqdm for progress bar, iterating as tasks are completed
            with tqdm(total=len(self.scan_ids)) as pbar:
                for future in concurrent.futures.as_completed(futures):
                    scan_id = futures[future]
                    try:
                        centers = future.result()

                        result_file_path = osp.join(self.out_dir, scan_id + ".pkl")
                        # common.write_pkl_data( centers, result_file_path)
                        with h5py.File(result_file_path, 'w') as h5file:
                            for obj_id, data in centers.items():
                                # Save the center, points, and votes for each object
                                obj_group = h5file.create_group(str(obj_id))
                                obj_group.create_dataset('center', data=data['center'])
                                obj_group.create_dataset('points', data=data['points'])
                                obj_group.create_dataset('votes', data=data['votes'])
                        print("added results of scan id ", scan_id, " successfully")
                    except Exception as exc:
                        print(f"Scan {scan_id} generated an exception: {exc}")
                        print("Traceback details:")
                        traceback.print_exc()
                    
                    # progressed
                    pbar.update(1)

        print("Done")
        
        
       
            
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
    print("start mask computation")
    evaluate.compute()
   

   
  

    
if __name__ == "__main__":
    main()