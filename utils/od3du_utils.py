 
import h5py
import os.path as osp
import numpy as np
import plyfile
import sys
import open3d as o3d
from collections import Counter
ws_dir = osp.dirname(osp.abspath(__file__))
print(ws_dir)
sys.path.append(ws_dir)
from utils import scan3r



"""  functions for accessing data
"""
 #return an object with the structure:scan_id: frame_number: frame_obj_id: matched id
def read_matching_data(scans_files_dir, scan_id):
    # get the file and iterate through everything to create an object
    matchfile = osp.join(scans_files_dir, "Predicted_Matches", scan_id + ".h5")
    with h5py.File(matchfile, 'r') as hdf_file:
        loaded_matches = {}
        
        # Iterate through frame indices
        for frame_idx in hdf_file.keys():
            matches = {}
            
            # Access the group for each frame index
            frame_group = hdf_file[frame_idx]
            
            # Load the frame_id -> obj mappings
            for frame_id in frame_group.keys():
                obj = int(frame_group[frame_id][()])
                matches[int(frame_id)] = int(obj)  # Convert back to int
            
            loaded_matches[frame_idx] = matches 

    return loaded_matches
    

#reads the file for the predicted centers
def read_predicted_data( scans_files_dir, scan_id):
    all_centers = {}
    filename = osp.join(scans_files_dir, "Predicted_Centers", scan_id + ".h5")
    with h5py.File(filename, 'r') as h5file:
        for obj_id in h5file.keys():
            obj_group = h5file[obj_id]
            center = np.array(obj_group['center'])
            points = np.array(obj_group['points'])
            votes = int(obj_group['votes'][()])
            size = int(obj_group["size"][()])
            
            # Add it back to the dictionary
            all_centers[int(obj_id)] = {
                'center': center,
                'points': points,
                'votes': votes,
                "size": size
            }
    return all_centers




 # returns featuer in the form of features: frame: list of {objext_id, bbox, mask} objects
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


 #returns an object for the reference scan of the form fetures     features: object_id: all feature vectors of this id
def read_ref_data(ref_path):
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

#returns an object for the features of the rescan  of the form features    features: frame_idx : obj_id : feature vector of the id
def read_scan_data(scan_path):
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


"""
functionc for matching Prediction
"""
#get the majorities for the knn
def get_majorities(distanc, obj_ids ,frame_ids, k , th):
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
            majorities[frame_id] = unique_new_obj
            unique_new_obj = unique_new_obj -1
            

        else:
            majorities[frame_id]= most_common_class

    return majorities

#use the mask information to generte an image on pixel level
def generate_pixel_level(segmentation,majorities, image_height, image_width):
    pixel_ids = np.zeros((image_height, image_width))
    #print(majorities.keys())
    #iterate through all the segmentation regions of the dino segmentation
    for seg_region in segmentation:
        mask_id = seg_region["object_id"]
        #get to what the region mapped in the majorities
        matched_id = majorities[int(mask_id)]
        #print("matched id ", matched_id)
        mask = seg_region["mask"]
        boolean_mask = mask == 225
        pixel_ids[boolean_mask] = matched_id 
        

    return pixel_ids

#quantize an image into patches
def quantize_to_patch_level(pixelwise_img, image_height, image_width, image_patch_h, image_patch_w):
    #get the shape of the pixelwise img
    patch_width = int(image_width/ image_patch_w)
    patch_height= int(image_height/ image_patch_h)

    patchwise_id = np.zeros((image_patch_h,image_patch_w))

    for i in range(image_patch_h):
            for j in range(image_patch_w):
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


#for a given scene get the colours of the differnt object_ids
def get_id_colours(data_dir,scan_id):
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
        object_colors[object_id] = np.array(most_common_color)
    
    return object_colors















"""
functionc for Center Prediction
"""
def transform_to_3d(data_root_dir, scans_scenes_dir, scan_id, depth_map, frame_number):
        
    #access the extrinsic/pose of the camera
    pose_rescan = scan3r.load_pose(scans_scenes_dir, scan_id, frame_number)
    pose_in_ref = scan3r.pose_in_reference(data_root_dir, scan_id, pose_rescan)
    
    camera_info = scan3r.load_intrinsics(scans_scenes_dir, scan_id)
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


def isolate_object_coordinates(world_coordinates, mask):
    #make sure it is an array
    mask = np.array(mask)
    #flatten & turn into boolean mask
    mask = mask.flatten()
    mask = mask.astype(bool)
    #get the part belonging to the object
    obj_coordinates = world_coordinates[mask]

    return obj_coordinates



def voxel_grid_to_coordinates(voxel_grid):
    """Extract voxel coordinates from a VoxelGrid object."""
    voxels = voxel_grid.get_voxels()
    voxel_coords = np.array([voxel.grid_index for voxel in voxels])
    return voxel_coords


def compare_voxel_grids(voxel_grid1, voxel_grid2):
        """Compare two voxel grids to see how much they overlap."""
        coords1 = voxel_grid_to_coordinates(voxel_grid1)
        coords2 = voxel_grid_to_coordinates(voxel_grid2)
        
        # Convert to sets of tuples for intersection
        voxels1_set = set(map(tuple, coords1))
        voxels2_set = set(map(tuple, coords2))
        
        # Compute intersection
        intersection = voxels1_set.intersection(voxels2_set)
        union = voxels1_set.union(voxels2_set)
        
        similarity = len(intersection) / len(union) if len(union) > 0 else 0
        return similarity


def do_pcl_overlap(voxel_size, obj_pcl, cluster):
    #create a voxel grid
    #turn into pointclouds
    obj_point_cloud = o3d.geometry.PointCloud()
    obj_point_cloud.points = o3d.utility.Vector3dVector(obj_pcl)
    voxel_grid1 = o3d.geometry.VoxelGrid.create_from_point_cloud(obj_point_cloud, voxel_size)

    cluster_point_cloud = o3d.geometry.PointCloud()
    cluster_point_cloud.points = o3d.utility.Vector3dVector(cluster)
    voxel_grid2 = o3d.geometry.VoxelGrid.create_from_point_cloud(cluster_point_cloud, voxel_size)

    
    """Compare two voxel grids to see how much they overlap."""
    return compare_voxel_grids(voxel_grid1, voxel_grid2)
        