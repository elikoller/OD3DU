export OD3DU_SPACE='/local/home/ekoller/BT'
export Scan3R_ROOT_DIR='/local/home/ekoller/R3Scan'
export CONDA_BIN='/local/home/ekoller/anaconda3/bin'

# activate conda env
# conda activate VLSG
source $CONDA_BIN/activate OD3DU

cd $OD3DU_SPACE
# project 3D object annotations to 2D query images

# aggretate pixel-wise annotations to patch-wise annotations
python ./preprocessing/gt_anno_2D/scan3r_obj_projector.py \
    --config ./preprocessing/gt_anno_2D/gt_anno.yaml \
    --split train
python ./preprocessing/gt_anno_2D/scan3r_obj_img_associate.py \
    --config ./preprocessing/gt_anno_2D/gt_anno.yaml \
    --split train


python ./preprocessing/gt_anno_2D/scan3r_obj_projector.py \
    --config ./preprocessing/gt_anno_2D/gt_anno.yaml \
    --split test
python ./preprocessing/gt_anno_2D/scan3r_obj_img_associate.py \
    --config ./preprocessing/gt_anno_2D/gt_anno.yaml \
    --split test

