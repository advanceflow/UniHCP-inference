"""
Non-Slurm-based single-gpu version inference script to run UniHCP-CIHP

label definition:
{
    0:'background',
    1:'hat',
    2:'hair',
    3:'glove',
    4:'sunglasses',
    5:'upperclothes',
    6:'dress',
    7:'coat',
    8:'socks',
    9:'pants',
    10:'torsoSkin',
    11:'scarf',
    12:'skirt',
    13:'face',
    14:'leftArm',
    15:'rightArm',
    16:'leftLeg',
    17:'rightLeg',
    18:'leftShoe',
    19:'rightShoe'
}

"""
import os
import re
import yaml
loader = yaml.SafeLoader
loader.add_implicit_resolver(
    u'tag:yaml.org,2002:float',
    re.compile(u'''^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$''', re.X),
    list(u'-+0123456789.'))
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from argparse import Namespace

from core.config import Config 
from core.testers import tester_entry 
# https://stackoverflow.com/questions/56395914/how-to-fix-error-while-loading-shared-libraries-libpython3-6m-so-1-0-cannot-o
# https://stackoverflow.com/questions/71759248/importerror-cannot-import-name-builder-from-google-protobuf-internal
from core import distributed_utils as dist
from core.distributed_utils import dist_init
from core.testers import tester_entry
from core.data.datasets.images.parsing_dataset import Human3M6ParsingDataset
import core.data.transforms.parsing_transforms as T

parser = argparse.ArgumentParser(description='Non-slurm-based inference script to run pre-trained model')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--dataset_dir', type=str, default="ip_data/test_data/sample_hp_ssense_dataset")

UNIHCP_OUTPUT_SAVENAME = "uniHCP.png"
UNIHCP_OUTPUT2OURS_SAVENAME = "uniHCP_ours.png"

merged_cls_label = ['background', 'outerwear', 'allbody', 'tops', 'bottoms', 'face', 'shoe'] + \
                    ["torsoSkin","leftArm", "rightArm", "leftLeg", "rightLeg"] 
hp2uni_seg_map = { # translate UniHCP-CIHP format to our format
    1: merged_cls_label.index("face"),
    2: merged_cls_label.index("face"),
    4: merged_cls_label.index("face"),
    5: merged_cls_label.index("tops"),
    6: merged_cls_label.index("allbody"),
    7: merged_cls_label.index("outerwear"),
    9: merged_cls_label.index("bottoms"),
    10: merged_cls_label.index("torsoSkin"),
    12: merged_cls_label.index("bottoms"),
    13: merged_cls_label.index("face"),
    14: merged_cls_label.index("leftArm"),
    15: merged_cls_label.index("rightArm"),
    16: merged_cls_label.index("leftLeg"),
    17: merged_cls_label.index("rightLeg"),
    18: merged_cls_label.index("shoe"),
    19: merged_cls_label.index("shoe"),
}

def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return np.array(palette).reshape(-1,3)

class ProductInferenceParsingDataset(Human3M6ParsingDataset):
    def __init__(self,
             data_paths,
             dataset='train',
             is_train=True,
             cfg=None,
             **kwargs):
        """
        Dataset for human parsing
        Args:
            root_dir ([str]): where dataset
            dataset: train / val
            cfg: yaml format config
        """
        self.cfg = cfg
        self.dataset = dataset
        self.is_train = is_train

        self.product_list = data_paths

        self.images = self.product_list
        self.ignore_label = cfg.ignore_value
        self.num = len(self.images)
        self.num_classes = len(self.cfg.label_list)  # - 1
        assert self.num_classes == self.cfg.num_classes, f"num of class mismatch, len(label_list)={self.num_classes}, num_classes:{self.cfg.num_classes}"

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.original_label = np.array(self.cfg.label_list)

        self.augs = T.compose([T.resize_image_eval(cfg.eval_crop_size),
                              T.transpose()])
        print(f"-- Loading {dataset} dataset of {len(data_paths)} images")
    
    def __len__(self):
        return len(self.product_list)
    
    def __getitem__(self, index):
        dataset_dict = {}
        product_path = self.product_list[index]
        dataset_dict["product_path"] = product_path
        img_path = f"{product_path}/ou.png"
        
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        parsing_seg_gt = np.zeros_like(image)

        self._record_image_size(dataset_dict, image)

        image, parsing_seg_gt = self.augs(image, parsing_seg_gt)
        image = torch.as_tensor(np.ascontiguousarray(image))
        dataset_dict["image"] = image
        return dataset_dict
    
def load_args():
    TASK="par_cihp_lpe"
    GINFO_INDEX="4"
    # save_name = "UniHCP-CIHP"

    job_name='coslr1e3_104k_b4324g88_h256_I2k_1_10_001_2I_fairscale_m256'  #${4-debug}
    CONFIG_BASE = "experiments/unihcp/release"
    PRETRAIN_JOB_NAME=job_name #${6-${job_name}}
    CONFIG=f"{CONFIG_BASE}/{job_name}.yaml"
    TEST_CONFIG=f"{CONFIG_BASE}/vd_{TASK}_test.yaml"
    TEST_MODEL=f"checkpoints/{PRETRAIN_JOB_NAME}/ckpt_task{GINFO_INDEX}_iter_newest.pth.tar"

    assert os.path.isfile(CONFIG)
    assert os.path.isfile(TEST_CONFIG)
    assert os.path.isfile(TEST_MODEL)

    args = Namespace(
        expname=f"test_{TASK}",
        config=CONFIG,
        test_config=TEST_CONFIG,
        spec_ginfo_index=int(GINFO_INDEX),
        load_path=TEST_MODEL,
        load_single=False,
        auto_resume=None,
        recover=False,
        ignore=[]
    )
    return args


def main():
    args = parser.parse_args()
    dist_init(gpu=args.gpu)
    dataset_dir = args.dataset_dir

    # load config
    args = load_args() # real default argument for inference
    print("-- Load config")
    C_train = Config(args.config, spec_ginfo_index=args.spec_ginfo_index)
    print("----------------------")

    # load test config
    print("-- Load test config")
    with open(args.test_config) as f:
        test_config = yaml.load(f, Loader=loader)
    assert len(test_config['tasks']) == 1, "One task/dataset"
    C_test = Config(args.test_config, spec_ginfo_index=0)
    print("----------------------")
    if args.expname is not None:
        C_train.config['expname'] = args.expname
    
    # load manager
    product_ids = [os.path.join(dataset_dir, product_id) for product_id in os.listdir(dataset_dir) ]
    S = tester_entry(C_train, C_test)
    S.config.dataset.kwargs.ginfo = S.ginfo
    S.dataset = ProductInferenceParsingDataset(data_paths=product_ids, **S.config.dataset["kwargs"])
    dist.barrier()

    # load model
    S.create_model()
    print("-- Load checkpoint")
    S.load_args = args
    S.load(args);

    # load palette for debugging
    palette_merged = get_palette(len(merged_cls_label))
    palette_hp = get_palette(20) # CIHP

    # inference
    S.create_dataloader()
    S.model.eval()
    with torch.no_grad():
        for idx, S.tmp.input in tqdm(enumerate(S.test_loader), total=len(S.test_loader)):
            S.prepare_data()
            outputs = S.model(S.tmp.input_var, idx)
            torch.cuda.synchronize()

            if idx % 100 == 0:
                print(f"Index {idx}, process: {S.tmp.input_var['product_path'][0]}")
            
            for i, output in enumerate(outputs):
                par_pred = output["sem_seg"]
                output = par_pred.argmax(dim=0).cpu()
                pred = np.array(output, dtype=np.int)
                product_path = S.tmp.input_var["product_path"][i]
                save_path = os.path.join(product_path, UNIHCP_OUTPUT_SAVENAME)
                cv2.imwrite(save_path, pred)

                # save UniHCP-CIHP converted to our output format
                hp2uni_img = np.zeros_like(pred)
                for hp_cls, uni_seg_cls in hp2uni_seg_map.items():
                    hp2uni_img[ (pred==hp_cls) ] = uni_seg_cls
                save_path = os.path.join(product_path, UNIHCP_OUTPUT2OURS_SAVENAME)
                cv2.imwrite(save_path, hp2uni_img)
                # print("Save to", save_path)

                # debug
                if False:
                    save_path = os.path.join(product_path, "_"+UNIHCP_OUTPUT_SAVENAME)
                    cv2.imwrite(save_path, cv2.cvtColor(palette_hp[pred].astype(np.uint8), cv2.COLOR_RGB2BGR))
                    save_path = os.path.join(product_path, "_"+UNIHCP_OUTPUT2OURS_SAVENAME)
                    cv2.imwrite(save_path, cv2.cvtColor(palette_merged[hp2uni_img].astype(np.uint8), cv2.COLOR_RGB2BGR))


if __name__ == '__main__':
    main()
