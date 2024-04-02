gpu=$1
dataset_name=$2

LOG_FILE="logs/${dataset_name}.log"
CUDA_VISIBLE_DEVICES=${gpu} python -u inference_for_dataset.py \
    --gpu 0 --dataset_dir "/media/hdd2/data/${dataset_name}" \
    2>&1 | tee ${LOG_FILE}

    # --gpu 0 --dataset_dir "ip_data/test_data/sample_hp_ssense_dataset" \