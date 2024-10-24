export VLSG_SPACE='/local/home/ekoller/BT'
export Scan3R_ROOT_DIR='/local/home/ekoller/R3Scan'
export CONDA_BIN='/local/home/ekoller/anaconda3/bin'

# activate conda env
# conda activate VLSG   new_dino_seg
source $CONDA_BIN/activate OD3DU

cd $VLSG_SPACE

# generate patch-level features with Dinov2
python ./matching/computation_lowdim.py \
    --config ./matching/evaluator.yaml
