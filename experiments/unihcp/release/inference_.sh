#!/usr/bin/env bash
set -u

################
gpu="2"
TASK="parsing"
GINFO_INDEX="1"   # task index config cherrypick (if necessary)
################### 
################### ||| additional params usually not used
job_name='coslr1e3_104k_b4324g88_h256_I2k_1_10_001_2I_fairscale_m256'  #${4-debug}
iter='newest'  #${5-newest}
PRETRAIN_JOB_NAME=${job_name} #${6-${job_name}}
CONFIG="${job_name}.yaml"
TEST_CONFIG="vd_${TASK}_test.yaml"
TEST_MODEL="checkpoints/${PRETRAIN_JOB_NAME}/ckpt_task${GINFO_INDEX}_iter_${iter}.pth.tar"
################

full_job_name=${job_name}_test_${TASK}
echo 'start job:' ${job_name} ' config:' ${CONFIG} ' test_config:' ${TEST_CONFIG}
echo "TEST_MODEL: ${TEST_MODEL}"
echo

ROOT=../../../
python -W ignore -u ${ROOT}/inference.py \
    --expname ${full_job_name} \
    --config ${CONFIG} \
    --test_config ${TEST_CONFIG} \
    --spec_ginfo_index ${GINFO_INDEX} \
    --load-path=${TEST_MODEL} 

    # --tcp_port $PORT \