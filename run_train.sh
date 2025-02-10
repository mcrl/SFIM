#!/bin/bash

# killall python3; mpirun -np 4 -H b03:4 -x MASTER_ADDR=b03 ./run_train.sh

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=4

# Change for multinode config
set -a
: ${MASTER_ADDR=b00}
: ${MASTER_PORT=4157}
: ${OMP_NUM_THREADS=1}
: ${DISTRIBUTED_BACKEND=nccl}
: ${PATCH_SIZE=512}
: ${OVERLAP=0}
: ${BATCH_SIZE=2}
set +a

WORLD_RANK=${OMPI_COMM_WORLD_RANK}
LOCAL_RANK=${OMPI_COMM_WORLD_LOCAL_RANK}
LOCAL_SIZE=${OMPI_COMM_WORLD_LOCAL_SIZE}

source ${HOME}/miniconda3/etc/profile.d/conda.sh
conda activate udc

python3 train.py \
    --arch SFIM \
    --batch_size ${BATCH_SIZE} --patch_size $PATCH_SIZE --patch_size_val $PATCH_SIZE \
    --mse_scale 1.0 --fft_scale_abs 1.0 --fft_scale_angle 1.0 --fft_scale 0.0 --ssim_scale 1.0 \
    --data_name UDC_SIT --data_dir UDC_SIT --data_format 0 \
    --embed_dim 48 --num_FFTblock 6 --ffn_expansion_factor 3 \
    --nepoch 1000 --save_epoch 200 --ckpt 200 --val_epoch 1 \
    --save_name 'SIT_phase1' --lr_initial 0.0002 \
    --log_dir ./logs/
