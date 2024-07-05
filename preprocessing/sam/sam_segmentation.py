import torch
import os 
import glob
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
import os.path as osp
import sys
from collections import Counter
ws_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
print("ws_dir ", ws_dir)
sys.path.append(ws_dir)
from sam import sam_config
import argparse
from tqdm import tqdm

#initialization for cuda stuff
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")
DEVICE = torch.device('cuda:0' if cuda_available else 'cpu')
print(f"Using device: {DEVICE}")

#initialization of needed variables
MODEL_TYPE = sam_config.MODEL_TYPE
CHECKPOINT_PATH = sam_config.CHECKPOINT_PATH



#generates the mask using sam
def generate_sam_data(path_to_img):
    #print("checkpointpath", CHECKPOINT_PATH)

    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)


    mask_generator = SamAutomaticMaskGenerator(sam)

    # Give the path of your image
    IMAGE_PATH= osp.join(path_to_img)
    # Read the image from the path
    image= cv2.imread(IMAGE_PATH)
    # Convert to RGB format
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Generate segmentation mask
    try:
        # Clear GPU cache
        torch.cuda.empty_cache()

        # Generate segmentation mask
        output_mask = mask_generator.generate(image_rgb)
        #print(output_mask)
        return output_mask

    except RuntimeError as e:
        if 'out of memory' in str(e):
            print('CUDA out of memory. Consider reducing batch size or model complexity.')
            torch.cuda.empty_cache()
        else:
            raise e
        
    


#returns for every semantic regtion of thet frame x_min, y_min, width, height and the mask on pitelwise level
def get_sam_boundingboxes_and_mask_frame(path_to_img):

    data_dict = generate_sam_data(path_to_img)
    #print("sam ressult", data_dict)

    bounding_boxes = []
     #iterate through evey segmentation field to get the result
    unique_obj_id = 0
    for val in data_dict:
        unique_obj_id = unique_obj_id + 1
        bbox = val['bbox']
        mask = val['segmentation']
        #choose this restoring because the boundingboxes of the projection are stored the same -> better for the code
        x_min, y_min, width, height = bbox
        bounding_boxes.append({
                    'object_id': unique_obj_id,
                    'bbox': [x_min, y_min, width, height],
                    "mask": mask
                })
        
    return bounding_boxes


#wil be called by the rescan input image, saves boundingboxes at sam_data in the folder of R3Scan
def get_all_sam_semantic_boxes_and_mask_scene(data_dir, scan_id):

    #go to the folder of the scene
    folder_path = osp.join(data_dir, "scenes", scan_id, "sequence")
    #print("folderpath", folder_path)
    file_pattern = 'frame-*.color.jpg'


    #search for all the colour files
    file_list = glob.glob(os.path.join(folder_path, file_pattern))
    #print("file list", file_list)

    #create a directory in which the boxes get saved
    output_path = osp.join(data_dir, "files/sam_data", scan_id)
    if not osp.exists(output_path):
        os.makedirs(output_path)

    for file in file_list:
        frame_boxes = get_sam_boundingboxes_and_mask_frame(file)

        #get the name of the file currently accessed
        filename = os.path.basename(file)
        pattern_part = filename.split('.')[0]

        filename = pattern_part + ".npy"
        #make a new file to save the information of the boundingboxes
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
    data_dir = sam_config.data_dir
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
    print("---Start: Computation of the boundingboxes for the input rgb-images using sam---")
    for rescan_id in tqdm(scan_ids, desc='Computing bounding boxes'):
        get_all_sam_semantic_boxes_and_mask_scene(data_dir, rescan_id)

    print("---Finish: Computation of the boundingboxes for the input rgb-images using sam---")


    
   
    
 