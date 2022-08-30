# Copyright (c) 2010-2022, InterDigital
# All rights reserved. 

# See LICENSE under the root folder.


PY_NAME="${HOME_DIR}/experiments/bench.py"

# Main configurations
HETERO="False"
CODEC_CONFIG="${HOME_DIR}/config/codec_config/ford_voxel_context_oct_trunc_12bits.yaml"
CHECKPOINTS="${HOME_DIR}/results/train_ford_vcn/epoch_newest.pth"
CHECKPOINT_NET_CONFIG="False"
NET_CONFIG="${HOME_DIR}/config/net_config/voxelcontext_entropycoder_9.yaml"
INPUT="${HOME_DIR}/datasets/ford/Ford_02_q_1mm ${HOME_DIR}/datasets/ford/Ford_03_q_1mm"
# INPUT="${HOME_DIR}/datasets/ford/ford_02_q1mm/Ford_02_vox1mm-0100.ply ${HOME_DIR}/datasets/ford/ford_02_q1mm/Ford_02_vox1mm-0101.ply ${HOME_DIR}/datasets/ford/ford_02_q1mm/Ford_02_vox1mm-1599.ply ${HOME_DIR}/datasets/ford/ford_03_q1mm/Ford_03_vox1mm-0200.ply ${HOME_DIR}/datasets/ford/ford_03_q1mm/Ford_03_vox1mm-0201.ply ${HOME_DIR}/datasets/ford/ford_03_q1mm/Ford_03_vox1mm-1699.ply"

COMPUTE_D2="True"
MPEG_REPORT="mpeg_report.csv"
MPEG_REPORT_SEQUENCE="True" # view the input point clouds as sequences
WRITE_PREFIX="oec_"
PRINT_FREQ="1"
PC_WRITE_FREQ="-1"
TF_SUMMARY="False"
REMOVE_COMPRESSED_FILES="True"
PEAK_VALUE="30000"
BIT_DEPTH="18"
SLICE="0"
LEVEL_MAX="11"
LOG_FILE=$(date); LOG_FILE=log_${LOG_FILE//' '/$'_'}.txt
LOG_FILE_ONLY="False"
MPEG_REPORT="mpeg_report.csv"
MPEG_REPORT_SEQUENCE="True"