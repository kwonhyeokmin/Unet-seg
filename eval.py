import numpy as np

from config import cfg
from collections import defaultdict
from dataset.dataset import CTDataset
from dataset.uwmgi import UWMGIDataset
from torch.utils.data import DataLoader
import torch
from models.unet import UNet
from tqdm import tqdm
import argparse


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint for evaluate')
    args = parser.parse_args()
    return args


def iou_coef(y_true, y_pred, thr=0.5, dim=(2,3), epsilon=0.001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred>thr).to(torch.float32)
    inter = (y_true*y_pred).sum(dim=dim)
    union = (y_true + y_pred - y_true * y_pred).sum(dim=dim)
    iou = ((inter+epsilon)/(union+epsilon))
    return iou


if __name__ == '__main__':
    args = make_args()

    # Select device (gpu | cpu)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device('cpu')
    score = defaultdict(list)

    # ********************
    # 1. Load datasets
    # ********************
    dataset = UWMGIDataset()
    dataset_loader = CTDataset(dataset, is_train=True, transforms=cfg.data_transforms['train'])
    data_generator = DataLoader(dataset=dataset_loader, batch_size=cfg.batch_size, shuffle=True,
                                num_workers=cfg.num_thread, pin_memory=True)
    # ********************
    # 2. Load model
    # ********************
    # load model
    model = UNet(in_channels=3, n_classes=3, n_channels=48).to(device)
    model.load_state_dict(torch.load(args.checkpoint))
    model.eval()

    # ****************
    # 3. Evaluate
    # ****************
    pbar = tqdm(enumerate(data_generator), total=len(data_generator), desc='Eval ')
    for step, (images, masks) in pbar:
        images = images.to(device, dtype=torch.float)
        masks = masks.to(device, dtype=torch.float)

        y_pred = model(images)

        # calculate iou
        for iou_thrs in [0.50, 0.75, 0.95]:
            iou = iou_coef(masks, y_pred)
            iou_score = (iou > iou_thrs).to(torch.float32).mean(dim=(1,0)).cpu().detach().numpy()
            score[f'iou:{iou_thrs:.2f}'].append(float(iou_score))

    # ****************
    # 3. Print result
    # ****************
    for k, v in score.items():
        print(k, ': ', np.mean(v))
