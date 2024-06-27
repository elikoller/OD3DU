import os 
import os.path as osp
import numpy as np 
import cv2
import open3d as o3d
import json
import pickle
import sys


ws_dir = '/local/home/ekoller/BT'
print(ws_dir)
sys.path.append(ws_dir)
from utils import scan3r,visualisation


#the function which actually does the intersection part

#this code returns the colour segmentation of the gt projection on pixelwise level
def load_gt_color_annos(data_dir, scan_id, skip=None):
    anno_imgs = {}
    frame_idxs = scan3r.load_frame_idxs(osp.join(data_dir, "scenes"), scan_id, skip)
    anno_folder = osp.join(data_dir, "files", 'gt_projection/color', scan_id)
    for frame_idx in frame_idxs:
        anno_color_file = osp.join(anno_folder, "frame-{}.jpg".format(frame_idx))
        anno_color = cv2.imread(anno_color_file, cv2.IMREAD_UNCHANGED)
        anno_imgs[frame_idx] = anno_color
    return anno_imgs

#this code returns the object_id segmentation of the gt projection on pixelwise level
def load_gt_obj_id_annos(data_dir, scan_id, skip=None):
    anno_imgs = {}
    frame_idxs = scan3r.load_frame_idxs(osp.join(data_dir, "scenes"), scan_id, skip)
    anno_folder = osp.join(data_dir, "files", 'gt_projection/obj_id', scan_id)
    for frame_idx in frame_idxs:
        anno_obj_file = osp.join(anno_folder, "frame-{}.jpg".format(frame_idx))
        anno_obj = cv2.imread(anno_obj_file, cv2.IMREAD_UNCHANGED)
        anno_imgs[frame_idx] = anno_obj
    return anno_imgs


#this code returns the colour segmentation of the gt projection on pixelwise level
def load_gt_gloabl_id_annos(data_dir, scan_id, skip=None):
    anno_imgs = {}
    frame_idxs = scan3r.load_frame_idxs(osp.join(data_dir, "scenes"), scan_id, skip)
    anno_folder = osp.join(data_dir, "files", 'gt_projection/global_id', scan_id)
    for frame_idx in frame_idxs:
        anno_global_file = osp.join(anno_folder, "frame-{}.jpg".format(frame_idx))
        anno_global = cv2.imread(anno_global_file, cv2.IMREAD_UNCHANGED)
        anno_imgs[frame_idx] = anno_global
    return anno_imgs


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
def project_mesh_diff(data_dir, scenes_dir,curr_scan_id, new_scan_id, frame_number):

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
    save_scan_color_dir = osp.join(data_dir, "proj", curr_scan_id, "color")
    save_scan_obj_dir = osp.join(data_dir, "proj", curr_scan_id, "obj_id")
    save_scan_global_dir = osp.join(data_dir, "proj", curr_scan_id, "global_id")

    #make sure the directories exist
    for dir_path in [save_scan_color_dir, save_scan_obj_dir, save_scan_global_dir]:
        try:
            os.makedirs(dir_path, exist_ok=True)
            
        except Exception as e:
            print(f"Failed to create directory {dir_path}: {e}")

    
    for frame_idx in obj_id_imgs:
        obj_id_img = obj_id_imgs[frame_idx]
        global_id_img = global_id_imgs[frame_idx]
        color_img = color_imgs[frame_idx]

        
        img_name = "projection_pose_in_"+ str(new_scan_id)+"_"+str(frame_idx)+".jpg"
        obj_id_img_file = osp.join(save_scan_obj_dir, img_name)
        global_img_file = osp.join(save_scan_global_dir, img_name)
        color_img_file = osp.join(save_scan_color_dir, img_name)
        success_obj = cv2.imwrite(obj_id_img_file, obj_id_img)
        success_global = cv2.imwrite(global_img_file, global_id_img)
        success_color = cv2.imwrite(color_img_file, color_img)
    
    return



    #this function gives the transformation on a pixelwise level between the current mesh and the new mesh
#this method goes over the projection and the semantically segmented image provided as input from (new_scan_id)
#it stores the mask following the logic described above
def get_transformation_mask(data_dir, scenes_dir,curr_scan_id, new_scan_id, frame_number):

    #the ground truth of the new scan id must exist already
    #project_mesh_gt_generation(data_dir,scenes_dir, new_scan_id, frame_number)
    


    #get the information of the rgb image of new_scan_id based on the ground truth (this code is for after the ground truth has beeen generated)
    gt_colors = load_gt_color_annos(data_dir,new_scan_id)
    gt_obj_ids = load_gt_obj_id_annos(data_dir, new_scan_id)
    gt_global_ids = load_gt_gloabl_id_annos(data_dir,new_scan_id)

    #get the infor for the specific frame number
    gt_color = gt_colors[frame_number]
    gt_obj_id = gt_obj_ids[frame_number]
    gt_global_id= gt_global_ids[frame_number]

    # scan_color_file = osp.join(data_dir, "gt_tmp", new_scan_id, "color","gt_frame-{}.jpg".format(frame_number))
    # scan_obj_file = osp.join(data_dir, "gt_tmp", new_scan_id, "obj_id","gt_frame-{}.jpg".format(frame_number))
    # scan_global_file = osp.join(data_dir, "gt_tmp", new_scan_id, "global_id","gt_frame-{}.jpg".format(frame_number))

    # gt_color = cv2.imread(scan_color_file, cv2.IMREAD_UNCHANGED)
    # gt_obj_id = cv2.imread(scan_obj_file, cv2.IMREAD_UNCHANGED)
    # gt_global_id= cv2.imread(scan_global_file, cv2.IMREAD_UNCHANGED)


    """
    until here the dimensions and entries are as excpected, however not something is wrong with the way the matrix is saved and filled
    """
    #get the rayintersection of the mesh based on the new computations
    #compute the intersections and maps
    project_mesh_diff(data_dir, scenes_dir,curr_scan_id, new_scan_id, frame_number)
    color_path = osp.join(data_dir,"proj",curr_scan_id, "color","projection_pose_in_{}_{}.jpg".format(new_scan_id,frame_number))
    obj_path = osp.join(data_dir,"proj",curr_scan_id, "obj_id","projection_pose_in_{}_{}.jpg".format(new_scan_id,frame_number))
    gloabl_path = osp.join(data_dir,"proj",curr_scan_id, "global_id","projection_pose_in_{}_{}.jpg".format(new_scan_id,frame_number))

    #access the files
    pj_color = cv2.imread(color_path, cv2.IMREAD_UNCHANGED)
    pj_obj_id = cv2.imread(obj_path, cv2.IMREAD_UNCHANGED)
    pj_global_id = cv2.imread(gloabl_path, cv2.IMREAD_UNCHANGED)

   

    img_height, img_width, channels = pj_color.shape
    #iterate through the whole image 
    curr_res_color = np.full_like(gt_color, 0)
    curr_res_obj_id = np.full((img_height,img_width,1), 0)
    curr_res_global_id = np.full((img_height,img_width,1), 0)
   

    new_res_color = np.full_like(gt_color, 0)
    new_res_obj_id = np.full((img_height,img_width,1), 0)
    new_res_global_id = np.full((img_height,img_width,1), 0)

    #make the mask: originally black: wherever there is a difference take the value of the new_scan_id gt

    

   
    for i in range(img_height):
        for j in range(img_width):
            if (pj_obj_id[i][j] != gt_obj_id[i][j]):
                #the new values in the modified area
                new_res_color[i][j] = gt_color[i][j]
                new_res_obj_id[i][j] = gt_obj_id[i][j]
                new_res_global_id[i][j] = gt_global_id[i][j]

                #the values which are impacted in curr 
                curr_res_color[i][j] = pj_color[i][j]
                curr_res_obj_id[i][j] = pj_obj_id[i][j]
                curr_res_global_id[i][j] = pj_global_id[i][j]


    #also save the files as needed
    #create directories (here tmp to try out & save 
    save_mask_color_dir = osp.join(data_dir, "mask", curr_scan_id, "color")
    save_mask_obj_dir = osp.join(data_dir, "mask", curr_scan_id, "obj_id")
    save_mask_global_dir = osp.join(data_dir, "mask", curr_scan_id, "global_id")

    #make sure the directories exist
    for dir_path in [save_mask_color_dir, save_mask_obj_dir, save_mask_global_dir]:
        try:
            os.makedirs(dir_path, exist_ok=True)
            
        except Exception as e:
            print(f"Failed to create directory {dir_path}: {e}")

    #save the files in the corresponding directories
        
    
    img_name = "new_val_mask_pose_in_"+ str(new_scan_id)+"_"+str(frame_number)+".jpg"
    obj_id_img_file = osp.join(save_mask_obj_dir, img_name)
    global_img_file = osp.join(save_mask_global_dir, img_name)
    color_img_file = osp.join(save_mask_color_dir, img_name)
    cv2.imwrite(obj_id_img_file, new_res_obj_id)
    cv2.imwrite(global_img_file, new_res_global_id)
    cv2.imwrite(color_img_file, new_res_color)


    img_name = "curr_val_mask_pose_in_"+ str(new_scan_id)+"_"+str(frame_number)+".jpg"
    obj_id_img_file = osp.join(save_mask_obj_dir, img_name)
    global_img_file = osp.join(save_mask_global_dir, img_name)
    color_img_file = osp.join(save_mask_color_dir, img_name)
    cv2.imwrite(obj_id_img_file, curr_res_obj_id)
    cv2.imwrite(global_img_file, curr_res_global_id)
    cv2.imwrite(color_img_file, curr_res_color)

    return 




   
            
   







   


 

    
    



