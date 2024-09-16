export VLSG_SPACE='/local/home/ekoller/BT'
export Scan3R_ROOT_DIR='/local/home/ekoller/R3Scan'
export CONDA_BIN='/local/home/ekoller/anaconda3/bin'

# activate conda env
# conda activate VLSG   new_dino_seg
source $CONDA_BIN/activate BT

cd $VLSG_SPACE

# generate patch-level features with Dinov2
python ./sceneGraph_update/predict_obj_centers_statistics.py \
    --config ./sceneGraph_update/centers_prediction_train.yaml