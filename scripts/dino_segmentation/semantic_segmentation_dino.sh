export OD3DU_SPACE='/local/home/ekoller/BT'
export Scan3R_ROOT_DIR='/local/home/ekoller/R3Scan'
export CONDA_BIN='/local/home/ekoller/anaconda3/bin'


#make sure youexcecute this within a container
source $CONDA_BIN/activate OD3DU
cd $OD3DU_SPACE

# generate patch-level features with Dinov2
python ./src/dino_semantic_segmentation_rescan/segment_dino_pipe.py \
    --config ./src/dino_semantic_segmentation_rescan/dinov2_segmentor.yaml \
    --split train

python ./src/dino_semantic_segmentation_rescan/segment_dino_pipe.py \
    --config ./src/dino_semantic_segmentation_rescan/dinov2_segmentor.yaml \
    --split test
    