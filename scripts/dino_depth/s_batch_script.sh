#!/bin/bash

#make an eth connection
module load  eth_proxy

#open a singularity shell with gpu
singularity exec --nv \
    --bind /cluster/home/ekoller:/cluster/home/ekoller \
    --bind /cluster/scratch/ekoller:/cluster/scratch/ekoller \
    dinov2manyproblem_lastest.sif  \
    bash /cluster/home/ekoller/BT_euler/scripts/dino_depth/depth_estim_dino.sh
