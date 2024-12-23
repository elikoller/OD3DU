export OD3DU_SPACE='/local/home/ekoller/BT'
export Scan3R_ROOT_DIR='/local/home/ekoller/R3Scan'
export CONDA_BIN='/local/home/ekoller/anaconda3/bin'

# activate conda env
source $CONDA_BIN/activate OD3DU

cd $OD3DU_SPACE

python ./src/preprocessing/gt_anno_2D/scan3r_obj_projector.py \
    --config ./src/preprocessing/gt_anno_2D/gt_anno.yaml \
    --split train
python ./src/preprocessing/gt_anno_2D/scan3r_obj_img_associate.py \
    --config ./src/preprocessing/gt_anno_2D/gt_anno.yaml \
    --split train


python ./src/preprocessing/gt_anno_2D/scan3r_obj_projector.py \
    --config ./src/preprocessing/gt_anno_2D/gt_anno.yaml \
    --split test
python ./src/preprocessing/gt_anno_2D/scan3r_obj_img_associate.py \
    --config ./src/preprocessing/gt_anno_2D/gt_anno.yaml \
    --split test

