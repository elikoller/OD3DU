export OD3DU_SPACE='/local/home/ekoller/BT'
export Scan3R_ROOT_DIR='/local/home/ekoller/R3Scan'
export CONDA_BIN='/local/home/ekoller/anaconda3/bin'

# activate conda env
source $CONDA_BIN/activate OD3DU

cd $OD3DU_SPACE

python ./src/object_features/DinoV2/scan3r_dinov2_generator.py \
    --config ./src/object_features/DinoV2/scan3r_dinov2_generator.yaml\
    --split train \
    --for_proj



python ./src/object_features/DinoV2/scan3r_dinov2_generator.py \
    --config ./src/object_features/DinoV2/scan3r_dinov2_generator.yaml\
    --split train \
    --for_dino_seg



python ./src/object_features/DinoV2/scan3r_dinov2_generator.py \
    --config ./src/object_features/DinoV2/scan3r_dinov2_generator.yaml\
    --split test\
    --for_proj

python ./src/object_features/DinoV2/scan3r_dinov2_generator.py \
    --config ./src/object_features/DinoV2/scan3r_dinov2_generator.yaml\
    --split test\
    --for_dino_seg

