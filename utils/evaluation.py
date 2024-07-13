import os 
import os.path as osp
import numpy as np 
import cv2
import open3d as o3d
import json
import pickle
import matplotlib.pyplot as plt
import glob
from plyfile import PlyData

import sys
sys.path.append('.')
vlsg_dir = osp.dirname(osp.dirname(__file__))
sys.path.append(vlsg_dir)
from utils import scan3r






#this file contains functions which are working with the evaluation of the changes a lot is based on the projection onto the current mesh ( referrence mesh)
#from the new scene


#the function which actually does the intersection part
def segmentResult(scene, intrinsics, extrinsics, width, height,
                      mesh_triangles, num_triangles, colors, obj_ids, global_ids):
        
        #create the rays to shoot in the scene
        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
            intrinsic_matrix = intrinsics.astype(np.float64),
            extrinsic_matrix = extrinsics.astype(np.float64),
            width_px = width, height_px = height
        )
        
       
        #find the intersections of the rays
        ans = scene.cast_rays(rays)
        hit_triangles_ids = ans['primitive_ids'].numpy()
        hit_triangles_ids_valid_masks = (hit_triangles_ids<num_triangles)
        hit_triangles_ids_valid = hit_triangles_ids[hit_triangles_ids_valid_masks]
        hit_triangles_valid = mesh_triangles[hit_triangles_ids_valid]
        hit_points_ids_valid = hit_triangles_valid[:,0]
        
        #create resulting maps initialized to 0
        color_map = np.zeros((height,width,3), dtype=np.uint8)
        obj_id_map = np.zeros((height,width), dtype=np.uint8)
        global_id_map = np.zeros((height,width), dtype=np.uint8)

        #go through the array and add the corresponding values
        color_map[hit_triangles_ids_valid_masks] = colors[hit_points_ids_valid]
        obj_id_map[hit_triangles_ids_valid_masks] = obj_ids[hit_points_ids_valid]
        global_id_map[hit_triangles_ids_valid_masks] = global_ids[hit_points_ids_valid]
        
        return color_map, obj_id_map, global_id_map



#given a current scene and a new scene: the reference scene is in the reference coordinate system, in this we will project the current camera frame & perform
#raytracing based on the new camera pose onto the current mesh
#frame number is for the new_scan_id
#returns numpy arrays of object_id, color, global_id
def project_new_pose_in_curr_mesh(data_dir, scenes_dir,curr_scan_id, new_scan_id, frame_number):

    """
    Read/prepare the data of the current mesh (mesh, colours, objid, globalid)
    """
    #get the current coordinate system the reference coordinate system
    mesh_file = osp.join(scenes_dir, curr_scan_id, "labels.instances.align.annotated.v2.ply") #exists for bot reference and rescan
        


    # load labels based on current mesh "status quo"
    plydata_npy = np.load(osp.join(scenes_dir, curr_scan_id, "data.npy"), allow_pickle=True)
      
    obj_labels = plydata_npy['objectId']
    global_labels = plydata_npy['globalId']
    

    # load curr mesh and scene
    mesh = o3d.io.read_triangle_mesh(mesh_file)
    mesh_triangles = np.asarray(mesh.triangles)
    #also load the colour
    colors = np.asarray(mesh.vertex_colors)*255.0
    colors = colors.round()
    num_triangles = mesh_triangles.shape[0]
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))


    """
    Read/prepare the data of the new mesh (pose, intrinsics)
    """
    # get img info and camera intrinsics of the new img
    camera_info = scan3r.load_intrinsics(scenes_dir, new_scan_id)
    intrinsics = camera_info['intrinsic_mat']
    img_width = int(camera_info['width'])
    img_height = int(camera_info['height'])

    # get frame_indexes and pose of the new
    frame_idxs = frame_number
    poses = scan3r.load_pose(scenes_dir, new_scan_id, frame_number)


    """
    Do the intersection
    """

    # project 3D model using rays
    obj_id_imgs = {}
    global_id_imgs = {}
    color_imgs = {}
    for idx in range(0, len(poses), 1):
        #frame_idx = frame_idxs[idx] #adjusted for single use in this jupiter notebook
        frame_idx = frame_idxs
        #img_pose = poses[idx] #adjusted for single use in this jupiter notebook
        img_pose = poses
        extrinsic = np.linalg.inv(img_pose)

        if scan3r.is_rescan(data_dir, new_scan_id):
            #need to adjust the extrinsic matrix to the reference coordinate system
            #get the path to the matricies of each scan_id for transformation of rescan to reference
            path = osp.join(data_dir,"files", "3RScan.json")
            ref2_rescan_all_id = scan3r.read_transform_mat(path)
            ref2rescan = ref2_rescan_all_id[new_scan_id] #2
            rescan2ref = np.linalg.inv(ref2rescan) #3
            

            #multiply such that we get transformation of camera to reference scan
            cam2ref =     extrinsic * rescan2ref.transpose()
    
            #convert
            extrinsic = np.asarray(cam2ref)

        #get the mappings for the image
        color_map, obj_id_map, global_id_map = segmentResult(
            scene, intrinsics, extrinsic, img_width, img_height, 
            mesh_triangles, num_triangles, colors, obj_labels, global_labels
        )
        obj_id_imgs[frame_idx] = obj_id_map
        global_id_imgs[frame_idx] = global_id_map
        color_imgs[frame_idx] = color_map

   
        
    #create directories (here tmp to try out & save 
    # save_scan_color_dir = osp.join(data_dir, "proj", curr_scan_id, "color")
    # save_scan_obj_dir = osp.join(data_dir, "proj", curr_scan_id, "obj_id")
    # save_scan_global_dir = osp.join(data_dir, "proj", curr_scan_id, "global_id")

    # #make sure the directories exist
    # for dir_path in [save_scan_color_dir, save_scan_obj_dir, save_scan_global_dir]:
    #     try:
    #         os.makedirs(dir_path, exist_ok=True)
            
    #     except Exception as e:
    #         print(f"Failed to create directory {dir_path}: {e}")

    
    for frame_idx in obj_id_imgs:
        obj_id_img = obj_id_imgs[frame_idx]
        global_id_img = global_id_imgs[frame_idx]
        color_img = color_imgs[frame_idx]

        
        # img_name = "proj_pose_"+ str(new_scan_id)+"_"+str(frame_idx)+".jpg"
        # obj_id_img_file = osp.join(save_scan_obj_dir, img_name)
        # global_img_file = osp.join(save_scan_global_dir, img_name)
        # color_img_file = osp.join(save_scan_color_dir, img_name)
        # success_obj = cv2.imwrite(obj_id_img_file, obj_id_img)
        # success_global = cv2.imwrite(global_img_file, global_id_img)
        # success_color = cv2.imwrite(color_img_file, color_img)

        #also vusually double check if the images are correct
        # cv2.imshow("object_id", obj_id_img)
        # cv2.imshow("global_id", global_id_img)
        # cv2.imshow("color", color_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()   

    #return the color matrix and the obj id matrix
    return obj_id_img, color_img

    
    