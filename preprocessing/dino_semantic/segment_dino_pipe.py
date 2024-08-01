import argparse
import dinov2.eval.segmentation_m2f.models.segmentors 
import mmcv
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
import os.path as osp
import sys
from tracemalloc import start
import cv2
import numpy as np
import dinov2.eval.segmentation.utils.colormaps as colormaps
import urllib
from tqdm.auto import tqdm
from PIL import Image
from glob import glob
import matplotlib.pyplot as plt
ws_dir = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
print(ws_dir)
sys.path.append(ws_dir)
from utils import common

"""
this file is adapted from the demo notebook provided by facebook, found in the src folder
"""
# Check PyTorch CUDA availability
# print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
# print(torch.__version__)
# # Print CUDA and cuDNN version
# print(f"CUDA version: {torch.version.cuda}")
# print(f"cuDNN version: {torch.backends.cudnn.version()}")
# print("mmsec version", mmseg.__version__)
# print("PyTorch version:", torch.__version__)
# print("CUDA version:", torch.version.cuda)
# print("CUDA available:", torch.cuda.is_available())
# print("Current device:", torch.cuda.current_device())
# print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

class DinoSegmentor():
    def __init__(self, cfg, split):
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
        # self.mode = 'orig' if self.split == 'train' else cfg.sgaligner.val.data_mode
        # self.scans_files_dir_mode = osp.join(self.scans_files_dir, self.mode)
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
        self.resplit = "resplit_" if cfg.data.resplit else ""
        ref_scans_split = np.genfromtxt(osp.join(self.scans_files_dir_mode, '{}_{}scans.txt'.format(split, self.resplit)), dtype=str)
        #print("ref scan split", ref_scans_split)
        self.all_scans_split = []
        ## get all scans within the split(ref_scan + rescan)
        for ref_scan in ref_scans_split:
            self.all_scans_split += self.refscans2scans[ref_scan]
        if self.rescan:
            self.scan_ids = self.all_scans_split
        else:
            self.scan_ids = ref_scans_split

        #print("scan ids", len(self.scan_ids))
        ## images info
        self.image_paths = {}
        for scan_id in self.scan_ids:
            self.image_paths[scan_id] = self.load_frame_paths(self.scans_dir, scan_id)

        # image preprocessing 
        self.img_rotate = self.cfg.data.img_encoding.img_rotate
        self.step = self.cfg.data.img.img_step
        
        
        #outputpath for total images
        self.out_dir_color = osp.join(self.scans_files_dir, 'Segmentation/Dinov2/color')
        #output path for components
        self.out_dir_objects = osp.join(self.scans_files_dir, 'Segmentation/Dinov2/objects')


        common.ensure_dir(self.out_dir_color)
        common.ensure_dir(self.out_dir_objects)
        
        self.log_file = osp.join(cfg.data.log_dir, "log_file_{}.txt".format(self.split))



    #helper functions
    #loads configs from the urs
    def load_config_from_url(self,url: str) -> str:
        with urllib.request.urlopen(url) as f:
            return f.read().decode()
        
    """
    Code cuplication with utils scan3r, Reason: the environment of 
    """

    
    def load_frame_paths(self,data_dir, scan_id, skip=None):
        frame_idxs = self.load_frame_idxs(osp.join(data_dir, "scenes"), scan_id, skip)
        img_folder = osp.join(data_dir, "scenes", scan_id, 'sequence')
        img_paths = {}
        for frame_idx in frame_idxs:
            img_name = "frame-{}.color.jpg".format(frame_idx)
            img_path = osp.join(img_folder, img_name)
            img_paths[frame_idx] = img_path
        return img_paths
    
    def load_frame_idxs(self,data_dir, scan_id, skip=None):
        # num_frames = len(glob(osp.join(data_dir, scan_id, 'sequence', '*.jpg')))

        # if skip is None:
        #     frame_idxs = ['{:06d}'.format(frame_idx) for frame_idx in range(0, num_frames)]
        # else:
        #     frame_idxs = ['{:06d}'.format(frame_idx) for frame_idx in range(0, num_frames, skip)]
        # return frame_idxs
        
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
    
        # DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"
        # CONFIG_URL = f"{DINOV2_BASE_URL}/dinov2_vitg14/dinov2_vitg14_ade20k_m2f_config.py"
        # CHECKPOINT_URL = f"{DINOV2_BASE_URL}/dinov2_vitg14/dinov2_vitg14_ade20k_m2f.pth"

        cfg_str = self.load_config_from_url(self.config_path)
        cfg = mmcv.Config.fromstring(cfg_str, file_format=".py")

        model = init_segmentor(cfg)
        print("--- model start evaluation ---")
        load_checkpoint(model, self.checkpoint_path, map_location="cpu")
        model.cpu()
        model.eval()
        print("--- model got evaluated ---")

        return model

        

    
    #for a scan do the inference for every frame // saves the entire image and a dictionary with the patches, ids, and boundingboxes
    def segment_each_scan(self, model, scan_id):
        # Load image paths and frame indices
        img_paths = self.image_paths[scan_id]
        frame_idxs_list = list(img_paths.keys())
        frame_idxs_list.sort()
        scan_result_path_color = osp.join(self.out_dir_color,scan_id)

        #initialize a dict for the objects
        info = {}
        scan_result_path_objects =  osp.join(self.out_dir_objects,scan_id)

        common.ensure_dir(scan_result_path_color)
        common.ensure_dir(scan_result_path_objects)

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


                #rotate it around if necessary
                if self.img_rotate:
                    img = img.transpose(Image.ROTATE_270)

                #make an np array
                array = np.array(img)
                print("image shape", array.shape)

                #print("inputsize" ,array.shape)

                #print("--- image got loaded ---")
                #less momry
                torch.cuda.empty_cache()
                with torch.no_grad():
                    segmentation_logits = inference_segmentor(model, array)[0]

                #print("--- image gets rendered ----")
                segmented_img = self.render_segmentation(segmentation_logits, "ade20k")
                segmented_img = np.array(segmented_img)
                #rotate the image such that the dimensions align with the other dimensions
                segmented_img = cv2.rotate(segmented_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                #to do assert the folder
                #cv2.imwrite("/local/home/ekoller/tmp_result/segmentedimg_r3_rotated.jpg", segmented_image_np)
                #print("--- image got saved ---")
                
                #safe the image as a colourful mask 
                img_name = "frame-"+str(frame_idx)+".jpg"
                file_path = osp.join(scan_result_path_color,img_name)
                cv2.imwrite(file_path,segmented_img)

                #also create a dictionary for the  by creating a object_based view 
                 #accesses the bounding boxes of the segmentation computed by dino and saved in the same format as the ones for the projection will be calculated
    
                """
                create objects based on components and save the infor in a dict
                """

                img_height, img_width, _ = segmented_img.shape  # Assuming the image is in RGB format

                # init the quantized image ofd dimenstion patch_wxpatch_h
                patch_annos = np.zeros((self.image_patch_h, self.image_patch_w, 3), dtype=np.uint8)

                # get the size of the patches
                patch_width = img_width // self.image_patch_w
                patch_height = img_height // self.image_patch_h

                # iterate through each patch
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

                        #get the index for patch annos
                        patch_h_idx = h // patch_height
                        patch_w_idx = w // patch_width
                        # Fill the patch with the most common color
                        patch_annos[patch_h_idx, patch_w_idx] = most_common_value

               
                #look how many colours there are in the originally sized segmetned image
                unique_colors = np.unique(patch_annos.reshape(-1, 3), axis=0)
                
                #create the dictionary we will retun later analogously to the projection boundingboxes
                #compute the boundingboxes based on that new obj_id_mask
                bounding_boxes = []
                segment_id = 0
            
                #go throught each colour and get the compoentnts
                for color in unique_colors:
                    # get the mask per colour
                    mask = cv2.inRange(patch_annos, color, color)
                    # get the individual components within that mask (connedted regions) using 8 neighboudhood connectvity
                    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
                    #visualize the rectangles
                    for i in range(1, num_labels):  # Start from 1 to skip the background
                        #assign the component a new id
                        segment_id += 1
                        min_col, min_row, width, height, _ = stats[i]
                        #get the mask of the componenet
                        component_mask = (labels == i).astype(np.uint8) *225

                        # Store bounding box information
                        bounding_boxes.append({
                            'object_id': segment_id,
                            'bbox': [min_col*self.image_patch_w, min_row*self.image_patch_h, width*self.image_patch_w, height*self.image_patch_h],
                            'mask': component_mask
                            })

               
            
                #save everything in a dictionary
                info[frame_idx]= bounding_boxes
            
        #save the info 
        # save file
        objects_file = osp.join(self.out_dir_objects, scan_id+".pkl")
        common.write_pkl_data(info, objects_file)




    #iterate over the scans and do the segmentation
    def segmentation(self):
        model = self.load_pretrained_model()
        #iterate over each scan which where selected in the initialization
        for scan_id in tqdm(self.scan_ids):
            with torch.no_grad():
                self.segment_each_scan(model,scan_id)

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
    # scan3r_gcvit_generator = Scan3rDinov2Generator(cfg, 'train', for_proj= True)
    # scan3r_gcvit_generator.register_model()
    # scan3r_gcvit_generator.generateFeatures()
    #also generate for the sam boundingboxes
    scan3r_segmentor = DinoSegmentor(cfg, 'train')
    scan3r_segmentor.segmentation()
    # scan3r_gcvit_generator = Scan3rDinov2Generator(cfg, 'val')
    # scan3r_gcvit_generator.register_model()
    # scan3r_gcvit_generator.generateFeatures()
    # scan3r_gcvit_generator = Scan3rDinov2Generator(cfg, 'test')
    # scan3r_gcvit_generator.register_model()
    # scan3r_gcvit_generator.generateFeatures()
    
if __name__ == "__main__":
    main()
            
        