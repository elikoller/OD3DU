export OD3DU_SPACE='/local/home/ekoller/BT'
export Scan3R_ROOT_DIR='/local/home/ekoller/R3Scan'
export CONDA_BIN='/local/home/ekoller/anaconda3/bin'

# activate conda env
# conda activate VLSG   new_dino_seg
source $CONDA_BIN/activate OD3DU

cd $OD3DU_SPACE

# generate patch-level features with Dinov2
python ./src/center_prediction_3D/predict_objects.py \
    --config ./src/center_prediction_3D/object_predictor.yaml \
    --split test