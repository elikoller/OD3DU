export VLSG_SPACE='/local/home/ekoller/BT'
export Scan3R_ROOT_DIR='/local/home/ekoller/R3Scan'
export CONDA_BIN='/local/home/ekoller/anaconda3/bin'



# activate conda env
# conda activate VLSG
source $CONDA_BIN/activate sam

cd $VLSG_SPACE
python ./preprocessing/sam/sam_segmentation.py \
    --split train
# python ./preprocessing/sam/sam_segmentation.py \
#     --split val
# python ./preprocessing/sam/sam_segmentation.py \
#     --split test
   