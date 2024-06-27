
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import cv2
import numpy as np
import os 
import os.path as osp
import matplotlib.pyplot as plt
import glob
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_l"
CHECKPOINT_PATH='/local/home/ekoller/sam_data/sam_vit_l_0b3195.pth'

cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")
DEVICE = torch.device('cuda:0' if cuda_available else 'cpu')
print(f"Using device: {DEVICE}")


#use sam kernel for this part!!!


def generate_sam_data(path_to_img):

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
        print(output_mask)

    except RuntimeError as e:
        if 'out of memory' in str(e):
            print('CUDA out of memory. Consider reducing batch size or model complexity.')
            torch.cuda.empty_cache()
        else:
            raise e
        
    return output_mask

#returns for every semantic regtion of thet frame x_min,x_max, y_min, y_max, height, width
def get_sam_boundingboxes_frame(path_to_img):

    data_dict = generate_sam_data(path_to_img)

    bboxes = []
     #iterate through evey segmentation field to get the result
    for val in data_dict:
        bbox = val['bbox']
        x_min, y_min, width, height = bbox
        x_max = x_min + width
        y_max = y_min + height
        bboxes.append(np.array(x_min,x_max, y_min, y_max, height, width))


    return bboxes



def get_all_sam_semantic_boxes_scene(data_dir, scan_id):

    #go to the folder of the scene
    folder_path = osp.join(data_dir, "scenes", scan_id, "sequence")
    file_pattern = 'frame-*-colour.jpg'

    #search for all the colour files
    file_list = glob.glob(os.path.join(folder_path, file_pattern))

    #create a directory in which the boxes get saved
    output_path = osp.join(data_dir, "tmp", scan_id, "boundingboxes")
    try:
        os.makedirs(output_path, exist_ok=True)
    except Exception as e:
        print(f"Failed to create directory {output_path}: {e}")

    for file in file_list:
        frame_boxes = get_sam_boundingboxes_frame(file)

        #get the name of the file currently accessed
        filename = os.path.basename(file)
        pattern_part = filename.split('.')[0]

        filename = pattern_part + ".npy"
        #make a new file to save the information of the boundingboxes
        box_file_path = osp.join(output_path, filename)
        success_obj = np.save(box_file_path, frame_boxes)



    
if __name__ == "__main__":   

    #try it out for one scan
    data_dir ='/local/home/ekoller/R3Scan'
    scenes_dir = '/local/home/ekoller/R3Scan/scenes'
    scan_id= "38770c95-86d7-27b8-8717-3485b411ddc7"

    get_all_sam_semantic_boxes_scene(data_dir,scan_id)





    

