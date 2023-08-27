import os
import os.path as osp
import argparse
from common.utils.dir_utils import make_folder
import albumentations as A


class Config:
    # directories
    cur_dir = osp.dirname(os.path.abspath(__file__))

    root_dir = cur_dir
    output_dir = osp.join(root_dir, 'output')
    model_dir = osp.join(output_dir, 'model_dump')
    vis_dir = osp.join(output_dir, 'vis')

    batch_size = 16
    num_thread = 2
    CROP_IMG_HEIGHT = 224
    CROP_IMG_WIDTH = 224

    lr = 2e-3
    num_epochs = 140
    wd = 1e-6

    # augmentation
    data_transforms = {
        "train": A.Compose([
            A.CenterCrop(CROP_IMG_HEIGHT, CROP_IMG_WIDTH),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.05, rotate_limit=10, p=0.5),
            A.OneOf([
                A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
                A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=1.0),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0)
            ], p=0.25),
            A.CoarseDropout(max_holes=8, max_height=CROP_IMG_HEIGHT // 20, max_width=CROP_IMG_WIDTH // 20,
                            min_holes=5, fill_value=0, mask_fill_value=0, p=0.5),
        ], p=1.0),

        "valid": A.Compose([
            A.CenterCrop(CROP_IMG_HEIGHT, CROP_IMG_WIDTH),
        ], p=1.0)
    }

cfg = Config()

# Make folder
make_folder(cfg.output_dir)
make_folder(cfg.model_dir)
make_folder(cfg.vis_dir)
