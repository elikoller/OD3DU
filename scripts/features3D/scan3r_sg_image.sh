export VLSG_SPACE='/local/home/ekoller/BT'
export Scan3R_ROOT_DIR='/local/home/ekoller/R3Scan'
export CONDA_BIN='/local/home/ekoller/anaconda3/bin'

# activate conda env
# conda activate VLSG
source $CONDA_BIN/activate BT

cd $VLSG_SPACE

# generate patch-level features with Dinov2
python ./preprocessing/sg_features/obj_visual_embeddings/Dinov2/gen_obj_visual_embeddings.py \
    --config ./preprocessing/sg_features/obj_visual_embeddings/Dinov2/obj_visual_embeddings.yaml \
    --split train

python ./preprocessing/sg_features/obj_visual_embeddings/Dinov2/gen_obj_visual_embeddings.py \
    --config ./preprocessing/sg_features/obj_visual_embeddings/Dinov2/obj_visual_embeddings.yaml \
    --split val

python ./preprocessing/sg_features/obj_visual_embeddings/Dinov2/gen_obj_visual_embeddings.py \
    --config ./preprocessing/sg_features/obj_visual_embeddings/Dinov2/obj_visual_embeddings.yaml \
    --split test

