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
from datetime import datetime
from sklearn.metrics import precision_recall_curve

def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint for evaluate')
    parser.add_argument('--dataset', required=True, help='Dataset name', choices=['uwmgi', 'nia'])
    parser.add_argument('--data_tag',
                        help='data tag (AXL, COR, SAG)',
                        required=True)
    parser.add_argument('--is_vis',
                        help='True/False of visualization. If you store option, it means true',
                        action='store_true')
    args = parser.parse_args()
    return args


def iou_coef(y_true, y_pred, thr=0.8, dim=(2,3), epsilon=0.001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred>thr).to(torch.float32)
    inter = (y_true*y_pred).sum(dim=dim)
    union = (y_true + y_pred - y_true * y_pred).sum(dim=dim)
    iou = ((inter+epsilon)/(union+epsilon))
    return iou


def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c],
                                  image[:, :, c])
    return image


if __name__ == '__main__':
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print('Start evaluation. ', dt_string)

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
    n_cls = len(test_dataset.cat_name)
    model = UNet(in_channels=3, n_classes=n_cls, n_channels=48).to(device)
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()

    # ****************
    # 3. Evaluate
    # ****************
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, n_cls + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    cnt = 0
    stp, sfp, recall, precision = 0, 0, 0, 0
    score_dict, correct_dict = dict(), dict()
    for cls in test_dataset.cat_name:
        score_dict[cls] = []
        correct_dict[cls] = []

    s = ('%12s' + '%40s' + '%12s' * 8) % ('No', 'DataID', 'TP', 'TN', 'FP', 'FN', 'sTP', 'sFP', 'P', 'R')
    print()
    print(s)

    pbar = tqdm(enumerate(data_generator), total=len(data_generator))
    for step, (images, masks, images_bgr, data_ids) in pbar:
        images = images.to(device, dtype=torch.float)
        masks = masks.to(device, dtype=torch.float)
        B, C, H, W = images.shape
        with torch.no_grad():
            y_pred = model(images)

        iou_thrs = 0.50
        _score = iou_coef(masks, y_pred).cpu().numpy()

        # calculate mAP with class
        for b in range(B):
            for cls, io in enumerate(_score[b] >= iou_thrs):
                score[test_dataset.cat_name[cls]].append(io)

        for b in range(B):
            tp = _score[b] >= iou_thrs
            tn = 0.0
            fp = 0.0
            fn = _score[b] < iou_thrs
            _precision = tp / (tp + fp + 0.001)
            _recall = tp / (tp + fn + 0.001)
            for i in range(n_cls):
                correct_dict[test_dataset.cat_name[i]].extend(tp.tolist())
                score_dict[test_dataset.cat_name[i]].extend(_score[b].tolist())

            sfp += fp
            stp += sum(tp)
            precision = round(precision + sum(_precision), 2)
            recall = round(recall + sum(_recall), 2)
            pbar.set_description(
                 ('%12s' + '%40s' + '%12s' * 8) % (cnt, data_ids[b], sum(tp), tn, fp, sum(fn), stp, sfp, precision, recall))
            cnt += 1

        # ****************
        # 4. Visualization (with first batch)
        # ****************
        if args.is_vis:
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
    score['All'] = np.array(list(score.values())).mean()
    s2 = ('%15s' * 2) % ('Class', 'mAP@.5')
    print()
    print(s2)
    for k, v in score.items():
        print(('%15s' * 2) % (k, np.round(np.mean(v), 2)))
    score.pop('All')
    # save graph
    fig, ax = plt.subplots()
    for k, v in score.items():
        precision, recall, thresholds = precision_recall_curve(correct_dict[k], score_dict[k])
        # create precision recall curve
        ax.plot(recall, precision, label=k)

    # add axis labels to plot
    ax.set_title('Precision-Recall Curve')
    ax.set_ylabel('Precision')
    ax.set_xlabel('Recall')
    ax.legend()
    ax.axis([0,1,0,1])

    # display plot
    plt.savefig(f'../data/PR_curve_{tag}_seg.png')

    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print('End evaluation. ', dt_string)
