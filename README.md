# UniHCP: A Unified Model for Human-Centric Perceptions

## Changes to the Original Repo:
Follow and extract code from [batch_test.sh](experiments/unihcp/release/batch_test.sh) and [test.py](test.py) to create 
simple-to-use non-slurm-based distributed single-gpu inference code for **human parsing** task, 
in the format of
1. jupyter notebook: check [inference.ipynb](inference.ipynb) for details
2. python script (UniHCP-CIHP only): see [inference_for_dataset.py](inference_for_dataset.py) and check [run_inference_for_dataset.sh](run_inference_for_dataset.sh) for 
example command to run.


# Original Documentation:

## Preparation

1. Install all required dependencies in requirements.txt.
2. Replace all `path...to...` in the .yaml configuration files to the absolute path
to corresponding dataset locations.
3. Place MAE pretrained weight <a href="https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth">mae_pretrain_vit_base.pth</a> under `core\models\backbones\pretrain_weights` folder.

*Only slurm-based distributed training & single-gpu testing is implemented in this repo.

## Experiments

All experiment configurations files and launch scripts are located in `experiments/unihcp/release` folder.

To perform full multi-task training for UniHCP, replace `<your partition>` in `train.sh` launch script and run:

```bash
sh train.sh 88 coslr1e3_104k_b4324g88_h256_I2k_1_10_001_2I_fairscale_m256
```

To perform evaluations, keep the test_info_list assignments corresponding to the tests you want to perform
, replace `<your partition>`, then run :

```bash
sh batch_test.sh  1 coslr1e3_104k_b4324g88_h256_I2k_1_10_001_2I_fairscale_m256
```

Note that in this case, the program would look for checkpoints located at `experiments/unihcp/release/checkpoints/coslr1e3_104k_b4324g88_h256_I2k_1_10_001_2I_fairscale_m256`


# Pretrained Models
Please send the signed <a href="https://drive.google.com/file/d/1O4Z7d5b1w0Vh4T8jvQ1tj_WzX12KWnT9/view?usp=share_link">agreement</a> to `mail@yuanzheng.ci` to get the download link.
