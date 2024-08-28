export VLSG_SPACE='/local/home/ekoller/BT'
export Scan3R_ROOT_DIR='/local/home/ekoller/R3Scan'
export CONDA_BIN='/local/home/ekoller/anaconda3/bin'

# activate conda env
# conda activate VLSG   new_dino_seg
source $CONDA_BIN/activate new_dino_seg

cd $VLSG_SPACE

# generate patch-level features with Dinov2
python ./preprocessing/dino_semantic/segment_dino_pipe.py \
    --config ./preprocessing/dino_semantic/dinov2_segmentor.yaml
