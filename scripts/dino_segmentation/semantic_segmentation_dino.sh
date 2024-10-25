export OD3DU_SPACE='/local/home/ekoller/BT'
export Scan3R_ROOT_DIR='/local/home/ekoller/R3Scan'
export CONDA_BIN='/local/home/ekoller/anaconda3/bin'


#make sure youexcecute this within a container

cd $OD3DU_SPACE

# generate patch-level features with Dinov2
python ./preprocessing/dino_semantic/segment_dino_pipe.py \
    --config ./preprocessing/dino_semantic/dinov2_segmentor.yaml \
    --split train

python ./preprocessing/dino_semantic/segment_dino_pipe.py \
    --config ./preprocessing/dino_semantic/dinov2_segmentor.yaml \
    --split test
    