import argparse
import urllib
#from models.GCVit.test import parse_args
import torch
import numpy as np
import PIL as Image
import os.path as osp
import sys
from tracemalloc import start
import cv2
import numpy as np
# Add the repository path to sys.path
repo_path = osp.abspath(osp.join(osp.dirname(__file__), 'Depth-Anything-V2/metric_depth'))
sys.path.append(repo_path)
from depth_anything_v2.dpt import DepthAnythingV2


from tqdm.auto import tqdm
from glob import glob
ws_dir = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
print(ws_dir)
sys.path.append(ws_dir)
from utils import common


"""
this file is adapted from the demo notebook provided by facebook, found in the src folder
"""
# Check PyTorch CUDA availability
print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
print(torch.__version__)
# Print CUDA and cuDNN version
print(f"CUDA version: {torch.version.cuda}")
print(f"cuDNN version: {torch.backends.cudnn.version()}")
print("PyTorch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("Current device:", torch.cuda.current_device())
print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))




class DinoSegmentor():
    def __init__(self, cfg, split):
        self.cfg = cfg
        self.split = split
        #model info
        self.encoder = cfg.models.encoder
        self.features = cfg.models.features
        self.out_channels = cfg.models.out_channels
        self.max_depth = cfg.models.max_depth
        self.data_set = cfg.models.dataset
        # 3RScan data info

        ## data dir
        self.data_root_dir = cfg.data.root_dir
        self.scan_type = cfg.sgaligner.scan_type
        scan_dirname = '' if self.scan_type == 'scan' else 'out'
        self.use_predicted = cfg.sgaligner.use_predicted
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
                
        #take only the split file      
        self.resplit = "resplit_" if cfg.data.resplit else ""
        ref_scans_split = np.genfromtxt(osp.join(self.scans_files_dir_mode, '{}_{}scans.txt'.format(split, self.resplit)), dtype=str)
        #print("ref scan split", ref_scans_split)
        self.all_scans_split = []

        ## get all scans within the split(ref_scan + rescan)
        for ref_scan in ref_scans_split:
            #self.all_scans_split.append(ref_scan)
            # Check and add one rescan for the current reference scan
            rescans = [scan for scan in self.refscans2scans[ref_scan] if scan != ref_scan]
            if rescans:
                # Add the first rescan (or any specific rescan logic)
                self.all_scans_split.append(rescans[0])

         

        if self.rescan:
            self.scan_ids = self.all_scans_split[:1]
        else:
            self.scan_ids = ref_scans_split


        print("scan_id_length", len(self.scan_ids))




        #print("scan ids", len(self.scan_ids))
        ## images info
        self.image_paths = {}
        for scan_id in self.scan_ids:
            self.image_paths[scan_id] = self.load_frame_paths(self.scans_dir, scan_id)

        # image preprocessing 
        self.img_rotate = self.cfg.data.img_encoding.img_rotate
        self.step = self.cfg.data.img.img_step
        
        
       
      
        #output path for components
        self.out_dir_objects = '/media/ekoller/T7/Depth_Anything/'
        common.ensure_dir(self.out_dir_objects)
        
        self.log_file = osp.join(cfg.data.log_dir, "log_file_{}.txt".format(self.split))



    #helper functions
    #loads configs from the urs
    def load_config_from_url(self,url: str) -> str:
        with urllib.request.urlopen(url) as f:
            return f.read().decode()
        

    
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




    def load_depth_model(self):
        
        print("--- load the model ----")
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
        }
        # model = DepthAnythingV2(encoder=self.encoder, features=self.features, out_channels=self.out_channels, max_depth=self.max_depth)
        model = DepthAnythingV2(**{**model_configs[self.encoder], 'max_depth': self.max_depth})
        model.load_state_dict(torch.load(f'/media/ekoller/T7/depth_anything_v2_metric_{self.data_set}_{self.encoder}.pth', map_location='cpu'))
        model.cuda()
        model.eval()
        print("--- load model done ---")
       
        return model

        

    
    #for a scan do the inference for every frame // saves the entire image and a dictionary with the patches, ids, and boundingboxes
    def depth_each_scan(self, model, scan_id):
        # Load image paths and frame indices
        img_paths = self.image_paths[scan_id]
        frame_idxs_list = list(img_paths.keys())
        frame_idxs_list.sort()
        scan_result_path_color = osp.join(self.out_dir_objects,scan_id)
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
                #img = Image.open(img_path).convert('RGB')
                img_bgr = cv2.imread(img_path)
                img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (224,172), interpolation=cv2.INTER_CUBIC)
                #rotate it around if necessary
                if self.img_rotate:
                    #img = img.transpose(Image.ROTATE_270)
                    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)



                #do the magic
                img_np = np.array(img)

                #safe the image as a colourful mask 
                depth_name = "frame-"+str(frame_idx)
                result = model.infer_image(img_np) #result is in m numpy
                rotated_depth_image = cv2.rotate(result, cv2.ROTATE_90_COUNTERCLOCKWISE)
                scaled_depth_image = rotated_depth_image * 1000
                scaled_depth_image = np.clip(scaled_depth_image, 0, 65535).astype(np.uint16)

                #cv2.imwrite(osp.join(self.out_dir_objects, scan_id), depth_name + '.pgm', scaled_depth_image)
                depth_file_path = osp.join(self.out_dir_objects, scan_id, f"{depth_name}.pgm")
                cv2.imwrite(depth_file_path, scaled_depth_image)
                #Image.fromarray(rotated_depth_image).save(osp.join(self.out_dir_objects, scan_id,depth_name) + '.pgm')  # Save as .pgm


    #iterate over the scans and do the segmentation
    def depth_estim(self):
        model = self.load_depth_model()
        #import pdb; pdb.set_trace()
        #iterate over each scan which where selected in the initialization
        for scan_id in tqdm(self.scan_ids):
            with torch.no_grad():
                self.depth_each_scan(model,scan_id)
                print("scan id  is done ", scan_id)
                
            

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
    scan3r_segmentor = DinoSegmentor(cfg, 'test')
    scan3r_segmentor.depth_estim()
    # scan3r_gcvit_generator = Scan3rDinov2Generator(cfg, 'val')
    # scan3r_gcvit_generator.register_model()
    # scan3r_gcvit_generator.generateFeatures()
    # scan3r_gcvit_generator = Scan3rDinov2Generator(cfg, 'test')
    # scan3r_gcvit_generator.register_model()
    # scan3r_gcvit_generator.generateFeatures()
    
if __name__ == "__main__":
    main()
            
        
