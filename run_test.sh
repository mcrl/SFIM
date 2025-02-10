#!/bin/bash

# killall python3; mpirun -np 4 -H b03:4 -x MASTER_ADDR=b03 ./run_test.sh

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=4

# Change for multinode config
set -a
: ${MASTER_ADDR=localhost}
: ${MASTER_PORT=4155}
: ${OMP_NUM_THREADS=1}
: ${DISTRIBUTED_BACKEND=nccl}
: ${PATCH_SIZE=768}
: ${OVERLAP=0}
: ${BATCH_SIZE=1}
: ${MODEL_PATH=/shared/s1/lab08/udc_logs/phase2_768_run1/models/model_best.pth}
set +a

WORLD_RANK=${OMPI_COMM_WORLD_RANK}
LOCAL_RANK=${OMPI_COMM_WORLD_LOCAL_RANK}
LOCAL_SIZE=${OMPI_COMM_WORLD_LOCAL_SIZE}

source ${HOME}/miniconda3/etc/profile.d/conda.sh
conda activate udc

python3 test.py \
    --arch SFIM \
    --batch_size ${BATCH_SIZE} --patch_size $PATCH_SIZE --patch_size_val $PATCH_SIZE \
    --mse_scale 1.0 --fft_scale_abs 1.0 --fft_scale_angle 1.0 --ssim_scale 1.0 \
    --data_name UDC-SIT --data_dir UDC-SIT --data_format 0 \
    --embed_dim 48 --num_FFTblock 6 --ffn_expansion_factor 3 \
    --save_name 'SIT_test' --precision "fp32" \
    --resume_from ./logs/SIT_phase1_run1/models/model_best.pth \
    --restored_img_dir ./restored_images/SIT_phase1
