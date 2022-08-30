# Copyright (c) 2010-2022, InterDigital
# All rights reserved. 

# See LICENSE under the root folder.


PY_NAME="${HOME_DIR}/experiments/train.py"

# Main configurations
HETERO="False"
NET_CONFIG="${HOME_DIR}/config/net_config/voxelcontext_entropycoder_9.yaml"
OPTIM_CONFIG="${HOME_DIR}/config/optim_config/optim_ce_vcn.yaml"
TRAIN_DATA_CONFIG="${HOME_DIR}/config/data_config/ford_oct_voxel_context_minibatch_12.yaml train_cfg"
VAL_DATA_CONFIG="${HOME_DIR}/config/data_config/ford_oct_voxel_context_minibatch_12.yaml val_cfg"
DDP=False

# Logging settings
PRINT_FREQ="20"
PC_WRITE_FREQ="-1"
TF_SUMMARY="True"
SAVE_CHECKPOINT_FREQ="1"
SAVE_CHECKPOINT_MAX="10"
VAL_FREQ="-1"
VAL_PRINT_FREQ="20"
LOG_FILE=$(date); LOG_FILE=log_${LOG_FILE//' '/$'_'}.txt
LOG_FILE_ONLY="False"
