export VLSG_SPACE='/local/home/ekoller/BT'
export Scan3R_ROOT_DIR='/local/home/ekoller/R3Scan'
export CONDA_BIN='/local/home/ekoller/anaconda3/bin'


source $CONDA_BIN/activate BT

cd $VLSG_SPACE

# generate patch-level features with Dinov2
python ./preprocessing/dino_depth/depth_dino_pipe.py \
    --config ./preprocessing/dino_depth/dinov2_depth.yaml
