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
        self.voxel_size = cfg.parameters.voxel_size
        self.minimum_points: cfg.parameters.minimum_points


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
            self.scan_ids = self.all_scans_split
        else:
            self.scan_ids = ref_scans_split
    
        #output path for components
        #self.out_dir = osp.join(self.data_root_dir, "Updates","depth_img")
        self.out_dir = osp.join("/media/ekoller/T7/Predicted_Centers")
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
        voxel_grid1 = o3d.geometry.VoxelGrid.create_from_point_cloud(obj_point_cloud, self.voxel_size)

        cluster_point_cloud = o3d.geometry.PointCloud()
        cluster_point_cloud.points = o3d.utility.Vector3dVector(cluster)
        voxel_grid2 = o3d.geometry.VoxelGrid.create_from_point_cloud(cluster_point_cloud, self.voxel_size)
    
        
        """Compare two voxel grids to see how much they overlap."""
        return self.compare_voxel_grids(voxel_grid1, voxel_grid2)
        


    def compute_scan(self,scan_id):

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

                    #now we need to find out if we add it to the pointcloud of the object it mapped to or not
                    if object_id not in all_clusters:
                        #there are no clusters stored for this object jet
                        all_clusters[object_id] = []
                        #add the pointcloud directly
                        all_clusters[object_id].append(obj_pcl)
                    #object already has pointclouds we need to see if we merge or add a new cluster
                    else:
                        #each new cluster starts unmerged
                        merged = False
                        #iterate through current clusters and see if we merge it
                        for i, cluster in enumerate(all_clusters[object_id]):
                            #look if it overlapt with a cluster
                            if self.do_pcl_overlap(obj_pcl, cluster) > 0.3:
                                #merge it
                                merged_points  = np.vstack((obj_pcl,cluster))
                                all_clusters[object_id][i] = merged_points
                                #got merged
                                merged = True
                        #not merged yet so create new cluster for this obje       
                        if not merged:
                            all_clusters[object_id].append(obj_pcl)


        #now that we have the lists of clusters we need to iterate over them and choose the biggest cluster, downsample it & take the average to predict the center
        #initialize final object
        all_centers = {}
        #iterte through the objects
        for obj_id, clusters in enumerate(all_clusters):
            #get the cluster with the most points aka largest 
            print(clusters)
            largest_cluster = max(clusters, key=lambda c:len(c))
            #create pointcloud and downsample it
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(largest_cluster)
            
            # Downsample the point cloud using voxel grid filtering
            downsampled_pcd = point_cloud.voxel_down_sample(self.voxel_size)
            
            # Convert downsampled point cloud back to a numpy array
            downsampled_points = np.asarray(downsampled_pcd.points)
            obj_center = np.mean(downsampled_points, axis=0)
            
            #save everithing to be able to visualize it later :)

            all_centers[obj_id] = {
                'center': obj_center,
                'points': downsampled_points
            }


            return all_centers
    """ Todo: for the new objects we cann not use the mapped ids since they are rando for each frame: keep a list which does the following_
    """

    

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
                        common.write_pkl_data( centers, result_file_path)
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