import argparse
import dinov2.eval.segmentation_m2f.models.segmentors 
import mmcv
import h5py
from mmcv.runner import load_checkpoint
import math
import itertools
from functools import partial
import urllib
#from models.GCVit.test import parse_args
import torch
import torch.nn.functional as F
import mmseg
from mmseg.apis import init_segmentor, inference_segmentor
import numpy as np
import PIL as Image
import os.path as osp
import cv2
from collections import Counter
from tracemalloc import start
import cv2
import numpy as np
import dinov2.eval.segmentation.utils.colormaps as colormaps
import urllib
from tqdm.auto import tqdm
from PIL import Image
from glob import glob
import matplotlib.pyplot as plt


import os.path as osp
import sys
ws_dir = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
print(ws_dir)
sys.path.append(ws_dir)
from utils import common
from configs import config, update_config


"""
this file is adapted from the demo notebook provided by facebook, found in the src folder
"""

class DinoSegmentor():
    def __init__(self, cfg, split):
        #get the paths needed to access the data and prepare the filepaths
        self.cfg = cfg
        #model info
        self.config_path = cfg.model.config_url
        self.checkpoint_path = cfg.model.checkpoint_url
        # 3RScan data info
        ## sgaliner related cfg
        self.split = split
        self.use_predicted = cfg.sgaligner.use_predicted
        self.scan_type = cfg.sgaligner.scan_type
        ## data dir
        self.data_root_dir = cfg.data.root_dir
        scan_dirname = '' if self.scan_type == 'scan' else 'out'
        scan_dirname = osp.join(scan_dirname, 'predicted') if self.use_predicted else scan_dirname
        self.scans_dir = osp.join(cfg.data.root_dir, scan_dirname)
        self.scans_files_dir = osp.join(self.scans_dir, 'files')
        self.scans_files_dir_mode = osp.join(self.scans_files_dir)
        self.scans_scenes_dir = osp.join(self.scans_dir, 'scenes')
        self.inference_step = cfg.data.inference_step


        #patch info 
        self.image_patch_w = self.cfg.data.img_encoding.patch_w
        self.image_patch_h = self.cfg.data.img_encoding.patch_h
      
       
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
        for ref_scan in ref_scans_split[229:]:
            #self.all_scans_split.append(ref_scan)
            # Check and add one rescan for the current reference scan
            rescans = [scan for scan in self.refscans2scans[ref_scan] if scan != ref_scan]
            if rescans:
                # Add the first rescan
                self.all_scans_split.append(rescans[0])

        #choos the scans
        if self.rescan:
            self.scan_ids = self.all_scans_split #we only use the rescans so the unseen rbg sequence
        else:
            self.scan_ids = ref_scans_split


        # images info
        self.image_paths = {}
        for scan_id in self.scan_ids:
            self.image_paths[scan_id] = self.load_frame_paths(self.scans_dir, scan_id)

        # image preprocessing 
        self.img_rotate = self.cfg.data.img_encoding.img_rotate
        self.step = self.cfg.data.img.img_step
        
        
        #outputpath for total images
        self.out_dir_color = osp.join(self.scans_files_dir,'Segmentation/DinoV2/color')
        #output path for components
        self.out_dir_objects = osp.join(self.scans_files_dir, "Segmentation/DinoV2/objects")


        common.ensure_dir(self.out_dir_color)
        common.ensure_dir(self.out_dir_objects)
        

    #helper functions
    #loads configs from the urs
    def load_config_from_url(self,url: str) -> str:
        with urllib.request.urlopen(url) as f:
            return f.read().decode()
        
    """
    Code Duplication: this file is run in a docker environment where the dependencies dont align with the ones used in the rest of the project
    hence we dublicate some code from the utils section
    """

    
    #this returns the image patshs for every frame_id
    def load_frame_paths(self,data_dir, scan_id, skip=None):
        frame_idxs = self.load_frame_idxs(osp.join(data_dir, "scenes"), scan_id, skip)
        img_folder = osp.join(data_dir, "scenes", scan_id, 'sequence')
        img_paths = {}
        for frame_idx in frame_idxs:
            img_name = "frame-{}.color.jpg".format(frame_idx)
            img_path = osp.join(img_folder, img_name)
            img_paths[frame_idx] = img_path
        return img_paths
    

    #return the frame idxs 
    def load_frame_idxs(self,data_dir, scan_id, skip=None):
        frames_paths = glob(osp.join(data_dir, scan_id, 'sequence', '*.jpg'))
        frame_names = [osp.basename(frame_path) for frame_path in frames_paths]
        frame_idxs = [frame_name.split('.')[0].split('-')[-1] for frame_name in frame_names]
        frame_idxs.sort()

        if skip is None:
            frame_idxs = frame_idxs
        else:
            frame_idxs = [frame_idx for frame_idx in frame_idxs[::skip]]
        return frame_idxs




    #this function helps to render the segmented image such that we get the different colours 
    def render_segmentation(self,segmentation_logits, dataset):
        DATASET_COLORMAPS = {
        "ade20k": colormaps.ADE20K_COLORMAP,
        "voc2012": colormaps.VOC2012_COLORMAP,
    }
        colormap = DATASET_COLORMAPS[dataset]
        colormap_array = np.array(colormap, dtype=np.uint8)
        segmentation_values = colormap_array[segmentation_logits + 1]
        return Image.fromarray(segmentation_values)
    
    #load the model
    def load_pretrained_model(self):

        cfg_str = self.load_config_from_url(self.config_path)
        cfg = mmcv.Config.fromstring(cfg_str, file_format=".py")

        model = init_segmentor(cfg)
        print("--- model start evaluation ---")
        load_checkpoint(model, self.checkpoint_path, map_location="cpu")
        model.cuda()
        model.eval()
        print("--- model got evaluated ---")

        return model

        

    
    #for a scan do the inference for every frame // saves the entire image and a dictionary with the patches (quantized instance mask), ids, and boundingboxes
    def segment_each_scan(self, model, scan_id):
        # Load image paths and frame indices
        img_paths = self.image_paths[scan_id]
        frame_idxs_list = list(img_paths.keys())
        frame_idxs_list.sort()
        scan_result_path_color = osp.join(self.out_dir_color,scan_id)

    
        #initialize a dict for the objects
        info = {}
        

        common.ensure_dir(scan_result_path_color)
        

        #o over each frame index for the scan id
        for infer_step_i in range(0, len(frame_idxs_list) // self.inference_step + 1):
            start_idx = infer_step_i * self.inference_step
            end_idx = min((infer_step_i + 1) * self.inference_step, len(frame_idxs_list))
            frame_idxs_sublist = frame_idxs_list[start_idx:end_idx]

            #since we now have indices we can just access
            for frame_idx in frame_idxs_sublist:
                #get the image path
                img_path = img_paths[frame_idx]

                #get the image
                img = Image.open(img_path).convert('RGB')

                #img = img.resize((120, 50), Image.BILINEAR)

                #rotate it around if necessary
                if self.img_rotate:
                    img = img.transpose(Image.ROTATE_270)

                #make an np array
                array = np.array(img)


                #less memory
                torch.cuda.empty_cache()
                with torch.no_grad():
                    segmentation_logits = inference_segmentor(model, array)[0]

                #render the inferred image
                segmented_img = self.render_segmentation(segmentation_logits, "ade20k")
                segmented_img = np.array(segmented_img)
                #rotate the image such that the dimensions align with the other dimensions
                segmented_img = cv2.rotate(segmented_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
               
                
                #safe the image as a colourful mask 
                img_name = "frame-"+str(frame_idx)+".jpg"
                file_path = osp.join(scan_result_path_color,img_name)
                cv2.imwrite(file_path,segmented_img)

                """
                create objects based on components and save the infor in a dict: we dont want the masks on class level but instead on instance level
                we use connected components to get the individual instances
                """
                
                #from the just computed segmentation get the shape
                img_height, img_width, _ = segmented_img.shape  

                # Initialize the new image
                patch_annos = np.zeros_like(segmented_img)

                # Calculate patch dimensions
                patch_width = img_width // self.image_patch_w
                patch_height = img_height // self.image_patch_h
                #since the ref scene is also given on patch basis - speed up the neighbouhod by bringing it from pixel level to patch level
                # Process each patch
                for h in range(0, img_height, patch_height):
                    for w in range(0, img_width, patch_width):
                        # Extract the patch
                        patch = segmented_img[h:h+patch_height, w:w+patch_width]

                        # Reshape patch to a list of colors
                        reshaped_patch = patch.reshape(-1, 3)

                        # Find the most common color
                        reshaped_patch_tuple = [tuple(color) for color in reshaped_patch]
                        value_counts = Counter(reshaped_patch_tuple)
                        most_common_value = value_counts.most_common(1)[0][0]

                        # Fill the patch with the most common color
                        patch_annos[h:h+patch_height, w:w+patch_width] = most_common_value
                        

                #create the dictionary we will retun later analogously to the projection boundingboxes
                #compute the boundingboxes based on that new obj_id_mask
                bounding_boxes = []
                segment_id = 0


                #look how many colours there are in the img and make the components withing one colour e.g. 2 blue chairs in an image is one blue mask -> after we have 2 separate masks
                unique_colors = np.unique(patch_annos.reshape(-1, 3), axis=0)
                for color in unique_colors:
                    # get the mask per colour
                    mask = cv2.inRange(patch_annos, color, color)
                    # get the individual components within that mask (connedted regions) using 8 neighboudhood connectvity
                    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

                                    
                    #visualize the rectangles
                    for i in range(1, num_labels):  # Start from 1 to skip the background
                        #assign the component a new id
                        segment_id += 1
                        min_col, min_row, width, height, _  = stats[i]
                    
                        # Extract the mask for the current component
                        component_mask = (labels == i).astype(np.uint8) * 225
                        #Store bounding box information
                        bounding_boxes.append({
                            'object_id': segment_id,
                            'bbox': [min_col, min_row, width, height],
                            'mask': component_mask
                            })
                        

                
                info[frame_idx]= bounding_boxes
               

               
              
            
 
        # save file
        objects_file = osp.join(self.out_dir_objects, scan_id+".h5")
        with h5py.File(objects_file, 'w') as hdf_file:
            for frame_idx, bboxes in info.items():
                # Create a group for each frame index
                frame_group = hdf_file.create_group(str(frame_idx))
                
                for i, bbox in enumerate(bboxes):
                    # Create a subgroup for each bounding box
                    bbox_group = frame_group.create_group(f'bbox_{i}')
                    
                    # Store object_id as an attribute
                    bbox_group.attrs['object_id'] = bbox['object_id']
                    
                    # Store bbox as a dataset
                    bbox_group.create_dataset('bbox', data=np.array(bbox['bbox']))
                    
                    # Store mask as a dataset (assuming component_mask is a numpy array)
                    bbox_group.create_dataset('mask', data=bbox['mask'])
        




    #iterate over the scans and do the segmentation
    def segmentation(self):
        model = self.load_pretrained_model()
        #iterate over each scan which where selected in the initialization
        for scan_id in tqdm(self.scan_ids):
            with torch.no_grad():
                self.segment_each_scan(model,scan_id)
                print("scan id  is done ", scan_id)
                
            

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


    cfg = update_config(config, cfg_file, ensure_dir = False)

 
    scan3r_segmentor = DinoSegmentor(cfg, split)
    scan3r_segmentor.segmentation()
   
    
if __name__ == "__main__":
    main()
            
        
