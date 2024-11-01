export OD3DU_SPACE='/local/home/ekoller/BT'
export Scan3R_ROOT_DIR='/local/home/ekoller/R3Scan'
export CONDA_BIN='/local/home/ekoller/anaconda3/bin'



# activate conda env
source $CONDA_BIN/activate OD3DU

cd $OD3DU_SPACE
python ./src/preprocessing/scan3r/preprocess_scan3r.py \
    --config ./src/preprocessing/scan3r/preprocess_scan3r.yaml \
    --split train
python ./src/preprocessing/scan3r/preprocess_scan3r.py \
    --config ./src/preprocessing/scan3r/preprocess_scan3r.yaml \
    --split test




