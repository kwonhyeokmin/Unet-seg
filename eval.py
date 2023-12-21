import cv2
import numpy as np

from config import cfg
from collections import defaultdict
from dataset.dataset import CTDataset
from dataset.uwmgi import UWMGIDataset
from dataset.nia import NIADataset
from torch.utils.data import DataLoader
import torch
from models.unet import UNet
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import copy



def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint for evaluate')
    parser.add_argument('--dataset', required=True, help='Dataset name', choices=['uwmgi', 'nia'])
    parser.add_argument('--data_tag',
                        help='data tag (AXL, COR, SAG)',
                        required=True)
    args = parser.parse_args()
    return args


def iou_coef(y_true, y_pred, thr=0.8, dim=(2,3), epsilon=0.001):
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
    tag = args.data_tag
    score = defaultdict(list)

    # ********************
    # 1. Load datasets
    # ********************
    if args.dataset == 'uwgi':
        test_dataset = UWMGIDataset()
    elif args.dataset == 'nia':
        test_dataset = NIADataset(data_split='test', tag=tag)
    else:
        raise ValueError(f'Dataset name {args.dataset} is not supported yet.')
    dataset_loader = CTDataset(test_dataset, transforms=cfg.data_transforms['test'])
    data_generator = DataLoader(dataset=dataset_loader, batch_size=cfg.batch_size, shuffle=False,
                                num_workers=cfg.num_thread, pin_memory=True)
    # ********************
    # 2. Load model
    # ********************
    # load model
    model = UNet(in_channels=3, n_classes=len(test_dataset.cat_name), n_channels=48).to(device)
    model.load_state_dict(torch.load(args.checkpoint))
    model.eval()

    # ****************
    # 3. Evaluate
    # ****************
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(test_dataset.cat_name) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    pbar = tqdm(enumerate(data_generator), total=len(data_generator), desc='Eval ')
    for step, (images, masks, images_bgr) in pbar:
        images = images.to(device, dtype=torch.float)
        masks = masks.to(device, dtype=torch.float)
        B = images.shape[0]

        y_pred = model(images)

        # calculate iou
        for iou_thrs in [0.50, 0.75, 0.95]:
            iou = iou_coef(masks, y_pred)
            iou_score = (iou > iou_thrs).to(torch.float32).mean(dim=(1,0)).cpu().detach().numpy()
            score[f'iou:{iou_thrs:.2f}'].append(float(iou_score))

        # ****************
        # 4. Visualization (with first batch)
        # ****************
        def apply_mask(image, mask, color, alpha=0.5):
            """Apply the given mask to the image.
            """
            for c in range(3):
                image[:, :, c] = np.where(mask == 1,
                                          image[:, :, c] *
                                          (1 - alpha) + alpha * color[c],
                                          image[:, :, c])
            return image

        if step % 10:
            for b in range(B):
                gt_img = copy.deepcopy(images_bgr[b]).cpu().numpy()
                vis_gt_msk = masks[b].detach().cpu().numpy()
                vis_gt_result = (vis_gt_msk > 0.9).astype(np.float32) * 1.0
                for cat, gt_mask in enumerate(vis_gt_result):
                    gt_img = apply_mask(gt_img, gt_mask, color=colors[cat])
                cv2.imwrite(f'{cfg.vis_dir}/{step}_{b}_gt.jpg', gt_img)

                pred_img = copy.deepcopy(images_bgr[b]).cpu().numpy()
                vis_pred_msk = y_pred[b].detach().cpu().numpy()
                vis_pred_result = (vis_pred_msk > 0.95).astype(np.float32) * 1.0
                for cat, pred_mask in enumerate(vis_pred_result):
                    pred_img = apply_mask(pred_img, pred_mask, color=colors[cat])
                cv2.imwrite(f'{cfg.vis_dir}/{step}_{b}_pred.jpg', pred_img)
    # ****************
    # 3. Print result
    # ****************
    for k, v in score.items():
        print(k, ': ', np.mean(v))
