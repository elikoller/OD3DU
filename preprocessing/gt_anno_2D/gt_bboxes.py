import argparse
import json
import os 
import glob
import random
import pickle
import cv2
import numpy as np
import os.path as osp
from collections import Counter
import sys
from tqdm import tqdm

ws_dir = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
print("ws_dir ", ws_dir)
sys.path.append(ws_dir)

from BT_config import overall_configuration

#get the patches from the config
patch_height = overall_configuration.patch_height
patch_width = overall_configuration.patch_width


#this codesegment takes in a semantic segmentation of the projection with sam and translates it into the object ids
def bounding_boxes_for_projection(data_dir, scan_id, file_name):
    #access the projection

    proj_rgb= osp.join(data_dir, "files/gt_projection", "obj_id", scan_id,file_name +".jpg")
    #print("proj file", proj_rgb)
    obj_mat = cv2.imread(proj_rgb, cv2.IMREAD_UNCHANGED)
    img_height, img_width= obj_mat.shape
    new_id = np.zeros_like(obj_mat)
    
    for h in range(0, img_height, patch_height):
        for w in range(0, img_width, patch_width):
            patch = obj_mat[h:h+patch_height, w:w+patch_width]
             # flatten the array to 1d
            flattened_patch = patch.flatten()
             # Find the most common value
            value_counts = Counter(flattened_patch)
            most_common_value = value_counts.most_common(1)[0][0]
            # Fill the patch with the most common color
            new_id[h:h+patch_height, w:w+patch_width] = most_common_value


    #compute the boundingboxes based on that new obj_id_mask
    bounding_boxes = []
    unique_ids = np.unique(new_id)
  
  #make the boundingboxes for the ids which got recognized
    for obj_id in unique_ids:
        # Create mask for current object ID
        mask = (new_id == obj_id)

        # Find bounding box coordinates
        rows, cols = np.nonzero(mask)
        if len(rows) > 0 and len(cols) > 0:
            min_row, max_row = np.min(rows) - (patch_height), np.max(rows) + (patch_height)
            min_col, max_col = np.min(cols) - (patch_width), np.max(cols) + (patch_width)

            # Calculate height and width
            height = max_row - min_row + 1
            width = max_col - min_col +1

            # Store bounding box information
            bounding_boxes.append({
                'object_id': obj_id,
                'bbox': [min_col, min_row, width, height]
            })

    return bounding_boxes




#wil be called by the rescan input image, saves boundingboxes at sam_data in the folder of R3Scan
def get_all_projection_bboxes_scene(data_dir, scan_id):

    #go to the folder of the scene
    folder_path = osp.join(data_dir, "scenes", scan_id, "sequence")
    #print("folderpath", folder_path)
    file_pattern = 'frame-*.color.jpg'


    #search for all the colour files
    file_list = glob.glob(os.path.join(folder_path, file_pattern))
    #print("file list", file_list)

    #create a directory in which the boxes get saved
    output_path = osp.join(data_dir, "bboxes", scan_id, "gt_projection")
    if not osp.exists(output_path):
        os.makedirs(output_path)

    #print("filelist for scan id ",scan_id , " has length", len(file_list))
    for file in file_list:
        #get the name of the file currently accessed
        filename = os.path.basename(file)
        pattern_part = filename.split('.')[0] #basically the frame-xxxxx  part

        filename = pattern_part + ".npy" 
        #make a new file to save the information of the boundingboxes

        frame_boxes = bounding_boxes_for_projection(data_dir,scan_id, pattern_part)
        #print("filename ", filename)
        box_file_path = osp.join(output_path, filename)
        success_obj = np.save(box_file_path, frame_boxes)
    return


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', dest='split', default='train', type=str, help='split to run subscan generation on')


    args = parser.parse_args()
    return parser, args
        

if __name__ == '__main__':
    #get the split from the bash input
    _, args= parse_args()
    split = args.split

    #access the data
    data_dir = overall_configuration.data_dir
    json_path = osp.join(data_dir,"files","3RScan.json")
    
    scan_ids = []
    #open the json file with all the scans
    with open(json_path, 'r') as file:
        data = json.load(file)
  
     #get all the 
    for scan_data in data:
        if scan_data["type"]  == split:
            #get the reference
            scan_ids.append(scan_data["reference"])
            #get all the rescans
            for scan in scan_data['scans']:
                scan_ids.append(scan['reference'] )
    

    #comput the boundingboxes for all the sequences of the reference scans
    print("---Start: Computation of the boundingboxes for the projections---")
    for scan_id in tqdm(scan_ids, desc='Computing bounding boxes'):
        get_all_projection_bboxes_scene(data_dir, scan_id)

    print("---Finish: Computation of the boundingboxes for the projections---")
                       



    
