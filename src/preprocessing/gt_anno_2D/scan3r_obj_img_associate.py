import os
import os.path as osp
import numpy as np

import torch
import torch.utils.data as data
import argparse
import cv2
from tqdm import tqdm
import sys
ws_dir = osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))
print(ws_dir)
sys.path.append(ws_dir)

from configs import config, update_config
from utils import common, scan3r

# associate image patch and obj id

class Scan3ROBJAssociator():
    def __init__(self, cfg, split):
        #preparation of the used paths and scan infor to be able to access the needed scans
        self.cfg = cfg
        self.split = split
        self.resplit = cfg.data.resplit
        self.use_rescan = self.cfg.data.rescan
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
        
        self.step = self.cfg.data.img.img_step
        
   
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
                
        #take our resplit file      
        self.resplit = "resplit_" if cfg.data.resplit else ""
        ref_scans_split = np.genfromtxt(osp.join(self.scans_files_dir, '{}_{}scans.txt'.format(split, self.resplit)), dtype=str)
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
  
        
        # 2D object id annotation directory
        self.obj_2D_anno_dir = osp.join(self.scans_dir, 'files', 'gt_projection', 'obj_id')
        
        # 2D image pObjectEmbeddingGeneratoratch annotation
        self.image_w = self.cfg.data.img.w
        self.image_h = self.cfg.data.img.h
        self.image_patch_w = self.cfg.data.img_encoding.patch_w
        self.image_patch_h = self.cfg.data.img_encoding.patch_h
        self.patch_w_size = self.image_w / self.image_patch_w
        self.patch_h_size = self.image_h / self.image_patch_h
        self.patch_anno_folder_name = "patch_anno_{}_{}".format(self.image_patch_w, self.image_patch_h)
        self.anno_out_dir = osp.join(self.scans_dir, 'files', 'patch_anno', self.patch_anno_folder_name)
        common.ensure_dir(self.anno_out_dir)
        
      
    def __len__(self):
        return len(self.anchor_data)

    def annotate(self, scan_idx, step, th=0.2):
        # get related files
        scan_id = self.scan_ids[scan_idx]
        # get frame annotations
        frame_idxs = scan3r.load_frame_idxs(self.scans_scenes_dir, scan_id, step)
        gt_2D_obj_anno_imgs = scan3r.load_gt_2D_anno(self.data_root_dir, scan_id, step)
        
        patch_annos_scan = {}
        # iterate over images
        for frame_idx in frame_idxs:
            gt_2D_obj_anno_img = gt_2D_obj_anno_imgs[frame_idx]
            patch_annos = np.zeros((self.image_patch_h, self.image_patch_w), dtype=np.uint8)
            # iterate over patches within one image and assign the majority id of the projection to that patch
            for patch_h_i in range(self.image_patch_h):
                h_start = round(patch_h_i * self.patch_h_size)
                h_end = round((patch_h_i+1) * self.patch_h_size)
                for patch_w_j in range(self.image_patch_w):
                    w_start = round(patch_w_j * self.patch_w_size)
                    w_end = round((patch_w_j+1) * self.patch_w_size)
                    patch_size = (w_end - w_start) * (h_end - h_start)
                    patch_anno = gt_2D_obj_anno_img[h_start:h_end, w_start:w_end]
                    obj_ids, counts = np.unique(patch_anno.reshape(-1), return_counts=True)
                    max_idx = np.argmax(counts)
                    max_count = counts[max_idx]
                    if(max_count > th*patch_size):
                        patch_annos[patch_h_i,patch_w_j] = obj_ids[max_idx]
            #save the frame of patch annotations
            patch_annos_scan[frame_idx] = patch_annos   
                    
        return patch_annos_scan
    
    def annotate_scans(self):
        self.patch_annos_scans = {}
        for scan_idx in tqdm(range(len(self.scan_ids))):
            patch_annos_scan = self.annotate(scan_idx, self.step)
            self.patch_annos_scans[scan_idx] = patch_annos_scan
            
        # save file
        for scan_idx in tqdm(range(len(self.scan_ids))):
            scan_id = self.scan_ids[scan_idx]
            patch_anno_file = osp.join(self.anno_out_dir, scan_id+".pkl")
            common.write_pkl_data(self.patch_annos_scans[scan_idx], patch_anno_file)
        
def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess Scan3R')
    parser.add_argument('--config', type=str, default='', help='Path to the config file')
    parser.add_argument('--split', type=str, default='train', help='Seed for random number generator')
    return parser.parse_known_args()
        
if __name__ == '__main__':
    # get arguments
    args, _ = parse_args()
    cfg_file = args.config
    split = args.split
    print(f"Configuration file path: {cfg_file}")

    cfg = update_config(config, cfg_file, ensure_dir = False)
   
    scan3r_img_projector = Scan3ROBJAssociator(cfg, split=split)
    scan3r_img_projector.annotate_scans()