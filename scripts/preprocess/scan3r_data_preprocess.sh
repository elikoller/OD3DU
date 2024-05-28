export VLSG_SPACE='/local/home/ekoller/BT'
export Scan3R_ROOT_DIR='/local/home/ekoller/R3Scan'
export CONDA_BIN='/local/home/ekoller/anaconda3/bin'



# activate conda env
# conda activate VLSG
source $CONDA_BIN/activate BT

cd $VLSG_SPACE
python ./preprocessing/scan3r/preprocess_scan3r.py \
    --config ./preprocessing/scan3r/preprocess_scan3r.yaml \
    --split train
python ./preprocessing/scan3r/preprocess_scan3r.py \
    --config ./preprocessing/scan3r/preprocess_scan3r.yaml \
    --split val
python ./preprocessing/scan3r/preprocess_scan3r.py \
    --config ./preprocessing/scan3r/preprocess_scan3r.yaml \
    --split test