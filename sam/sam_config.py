#this file stores thee needed parameters for the sam section
from BT_config import overall_configuration
CHECKPOINT_PATH= "/local/home/ekoller/sam_data/sam_vit_b_01ec64.pth" #modify to the path to the sam_vit data
MODEL_TYPE = "vit_b"
patch_height= overall_configuration.patch_height
patch_width= overall_configuration.patch_width #how big the patches are to cancle the noise of the projection
data_dir = overall_configuration.data_dir 
scenes_dir = overall_configuration.scenes_dir
