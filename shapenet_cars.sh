#bin/bash

#ln -s /home/dangeo314/Documents/class/ece285dgm/eg3d/dataset_preprocessing/shapenet_cars datasets
#ln -s /media/dangeo314/05da5c10-00e4-4cb4-a9fc-e95e52cc04ed/teff_logs/ training-runs
export CUDA_HOME=/usr/local/cuda/
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
#export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=1800
# export NCCL_P2P_DISABLE=1

python train.py  --outdir=training-runs --cfg=shapenet --data=datasets/cars_128_20k.zip \
  --gpus=2 --batch=32 --gamma=0.3 --gen_pose_cond=false --dis_pose_cond=True --dis_cam_weight=2 \
  --cbase 32768 --dataset_resolution=128    --dis_linear_pose=False \
  --gpc_reg_prob=0.5 \
  --flip_to_dis=true --flip_to_disd=true  --gpc_reg_fade_kimg=1000 \
  --neural_rendering_resolution_final 64 \
  --dino_data=datasets/in_the_wild/shapenetcars_dinov1_stride4_pca16_nomask_5k.zip --dino_channals 3\
  --bg_modeling_2d=true --seed 10086  --temperature=100.0 \
  --use_intrinsic_label=True --maskbody=False \
  --v_start=0 --v_end=1 --v_discrete_num=18 --uniform_sphere_sampling=True \
  --h_discrete_num=36 --h_mean=3.1415926 --flip_type=flip_both_shapenet \
  --dis_cam_dim=2 --lambda_cvg_fg=100 \
  --temperature_init=10.0 --temperature_start_kimg=1500 --temperature_end_kimg=2500 \
  --shapenet_multipeak=False \
  --create_label_fov=69.1882