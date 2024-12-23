import os
import os.path as osp
import numpy as np

import torch
import torch.utils.data as data
import argparse
import cv2
import open3d as o3d
# import open3d.visualization.rendering as rendering
from tqdm import tqdm
import sys
ws_dir = osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))
print(ws_dir)
sys.path.append(ws_dir)

from utils import common, scan3r


class Scan3RIMGProjector():
    def __init__(self, cfg, split):

        #initialize the paths to access the data 
        self.split = split
        self.resplit = cfg.data.resplit
        self.use_rescan = cfg.data.rescan
        self.data_root_dir = cfg.data.root_dir
        
        scan_dirname = ''
        self.scans_dir = osp.join(self.data_root_dir, scan_dirname)
        self.scans_scenes_dir = osp.join(self.scans_dir, 'scenes')
        self.scans_files_dir = osp.join(self.scans_dir, 'files')
        
        self.scenes_config_file = osp.join(self.scans_dir, 'files', '3RScan.json')
        self.scenes_configs = common.load_json(self.scenes_config_file)
        self.objs_config_file = osp.join(self.scans_dir, 'files', 'objects.json')
        self.objs_configs = common.load_json(self.objs_config_file)
        self.scan_ids = []
        
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
        ref_scans_split = np.genfromtxt(osp.join(self.scans_files_dir, '{}_{}scans.txt'.format(split, self.resplit)), dtype=str)
        #print("ref scan split", ref_scans_split)
        self.all_scans_split = []

        ## get all scans within the split(ref_scan + rescan)
        for ref_scan in ref_scans_split[:]:
            #self.all_scans_split.append(ref_scan)
            # Check and add one rescan for the current reference scan
            rescans = [scan for scan in self.refscans2scans[ref_scan] if scan != ref_scan]
            self.all_scans_split.append(ref_scan)
            if rescans:
                # Add the first rescan (or any specific rescan logic)
                self.all_scans_split.append(rescans[0])


        self.scan_ids = self.all_scans_split
  
        
        # get save dir 
        self.save_dir = osp.join(self.scans_dir, 'files', 'gt_projection')
        self.save_color_dir = osp.join(self.save_dir, 'color')
        self.save_obj_dir = osp.join(self.save_dir, 'obj_id')
        #self.save_global_dir = osp.join(self.save_dir, 'global_id')
        common.ensure_dir(self.save_dir)
        common.ensure_dir(self.save_color_dir)
        common.ensure_dir(self.save_obj_dir)
        #common.ensure_dir(self.save_global_dir)
      
    def __len__(self):
        return len(self.scan_ids)


    def project(self, scan_idx, step = 1):
        # get related files
        scan_id = self.scan_ids[scan_idx]
        #get the mesh
        mesh_file = osp.join(self.scans_scenes_dir, scan_id, "labels.instances.annotated.v2.ply")
        
        
        # get img info and camera intrinsics needed for projection
        camera_info = scan3r.load_intrinsics(self.scans_scenes_dir, scan_id)
        intrinsics = camera_info['intrinsic_mat']
        img_width = int(camera_info['width'])
        img_height = int(camera_info['height'])
        
        # load labels
        plydata_npy = np.load(osp.join(self.scans_scenes_dir, scan_id, "data.npy"))
        obj_labels = plydata_npy['objectId']
        global_labels = plydata_npy['globalId']
    
       # load mesh and scene
        mesh = o3d.io.read_triangle_mesh(mesh_file)
        mesh_triangles = np.asarray(mesh.triangles)
        #also load the colour
        colors = np.asarray(mesh.vertex_colors)*255.0
        colors = colors.round()
        num_triangles = mesh_triangles.shape[0]
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        
        # get frame_indexes
        frame_idxs = scan3r.load_frame_idxs(self.scans_scenes_dir, scan_id)
        poses = scan3r.load_all_poses(self.scans_scenes_dir, scan_id, frame_idxs)
        
        # project 3D model based on raycasting on a pin hole model
        obj_id_imgs = {}
        #global_id_imgs = {}
        color_imgs = {}
        for idx in range(0, len(poses), step):
            frame_idx = frame_idxs[idx]
            img_pose = poses[idx]
            img_pose_inv = np.linalg.inv(img_pose)
            #do the raycasting to obtain the colour and object id
            color_map, obj_id_map = self.segmentResult(
                scene, intrinsics, img_pose_inv, img_width, img_height, 
                mesh_triangles, num_triangles, colors, obj_labels
            ) #global_labels
            obj_id_imgs[frame_idx] = obj_id_map
            #global_id_imgs[frame_idx] = global_id_map
            color_imgs[frame_idx] = color_map
            
        # save 
        save_scan_color_dir = osp.join(self.save_color_dir, scan_id)
        save_scan_obj_dir = osp.join(self.save_obj_dir, scan_id)
        #save_scan_global_dir = osp.join(self.save_global_dir, scan_id)
        common.ensure_dir(save_scan_color_dir)
        common.ensure_dir(save_scan_obj_dir)
        #common.ensure_dir(save_scan_global_dir)
        
        for frame_idx in obj_id_imgs:
            obj_id_img = obj_id_imgs[frame_idx]
            color_img = color_imgs[frame_idx]
            #global_id_img = global_id_imgs[frame_idx]
            
            img_name = "frame-"+str(frame_idx)+".jpg"
            obj_id_img_file = osp.join(save_scan_obj_dir, img_name)
            color_img_file = osp.join(save_scan_color_dir, img_name)
            #global_id_img_file = osp.join(save_scan_global_dir, img_name)
            
            cv2.imwrite(obj_id_img_file, obj_id_img)
            cv2.imwrite(color_img_file, color_img)
            #cv2.imwrite(global_id_img_file, global_id_img)
            
    
    def segmentResult(self, scene, intrinsics, extrinsics, width, height,
                      mesh_triangles, num_triangles, colors, obj_ids): #global_ids
        #initialize the raycasting with the camera parameters
        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
            intrinsic_matrix = intrinsics.astype(np.float64),
            extrinsic_matrix = extrinsics.astype(np.float64),
            width_px = width, height_px = height
        )
        #initialize the rays and get the intersections
        ans = scene.cast_rays(rays)
        hit_triangles_ids = ans['primitive_ids'].numpy()
        hit_triangles_ids_valid_masks = (hit_triangles_ids<num_triangles)
        hit_triangles_ids_valid = hit_triangles_ids[hit_triangles_ids_valid_masks]
        hit_triangles_valid = mesh_triangles[hit_triangles_ids_valid]
        hit_points_ids_valid = hit_triangles_valid[:,0]
        
        color_map = np.zeros((height,width,3), dtype=np.uint8)
        obj_id_map = np.zeros((height,width), dtype=np.uint8)
       

        color_map[hit_triangles_ids_valid_masks] = colors[hit_points_ids_valid]
        obj_id_map[hit_triangles_ids_valid_masks] = obj_ids[hit_points_ids_valid]
        return color_map, obj_id_map 
    


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess Scan3R')
    parser.add_argument('--config', type=str, default='', help='Path to the config file')
    parser.add_argument('--split', type=str, default='train', help='Seed for random number generator')
    return parser.parse_known_args()


if __name__ == '__main__':

    args, _ = parse_args()
    cfg_file = args.config
    split = args.split
    print(f"Configuration file path: {cfg_file}")

    from configs import config, update_config
    cfg = update_config(config, cfg_file, ensure_dir = False)
       
    scan3r_img_projector = Scan3RIMGProjector(cfg, split=split)
    step=1
    for idx in tqdm(range(len(scan3r_img_projector.scan_ids))):
        scan3r_img_projector.project(idx, step=step)