from config import cfg
import torch
from dataset.dataset import CTDataset
from dataset.uwmgi import UWMGIDataset
from dataset.nia import NIADataset
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from models.unet import UNet
from tqdm import tqdm
import torch.optim as optim
from torch.optim import lr_scheduler
import os.path as osp
import argparse
from config import update_config
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
from eval import iou_coef
from collections import defaultdict
import numpy as np
import torch.distributed as dist


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=False, help='Path to pretrained checkpoint')
    parser.add_argument('--dataset', required=True, help='Dataset name', choices=['uwmgi', 'nia'])
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('--gpus',
                        help='gpu ids for use',
                        default='0',
                        type=str)
    parser.add_argument('--use_wandb',
                        help='use wandb',
                        action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = make_args()
    update_config(args.cfg)

    # Select device (gpu | cpu)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    use_wandb = args.use_wandb
    n_gpus = len(args.gpus.split(','))

    if use_wandb:
        wandb.init(project=cfg.hyp.PROJECT_NAME,
                   name=f'{cfg.hyp.OPTIMIZER.TYPE}_lr{cfg.hyp.OPTIMIZER.LR}{args.dataset}')
        wandb.config.update({
            'batch_size': cfg.batch_size,
            'num_workers': cfg.num_thread,
            'optimizer': cfg.hyp.OPTIMIZER.TYPE,
            'learning_rate': cfg.hyp.OPTIMIZER.LR,
            'weight_decay': cfg.hyp.OPTIMIZER.WD,
            'dataset': args.dataset
        })
    # ********************
    # 1. Load datasets
    # ********************
    if args.dataset == 'uwgi':
        train_dataset = UWMGIDataset()
        val_dataset = UWMGIDataset()
    elif args.dataset == 'nia':
        train_dataset = NIADataset(data_split='train')
        val_dataset = NIADataset(data_split='val')
    else:
        raise ValueError(f'Dataset name {args.dataset} is not supported yet.')

    train_loader = CTDataset(train_dataset, transforms=cfg.data_transforms['train'])
    # sampler = torch.utils.data.distributed.DistributedSampler(dataset=train_loader, shuffle=True)
    train_generator = DataLoader(dataset=train_loader,
                                 batch_size=int(cfg.batch_size / n_gpus),
                                 num_workers=int(cfg.num_thread / n_gpus),
                                 pin_memory=True,
                                 # sampler=sampler
                                 )
    val_loader = CTDataset(val_dataset, transforms=cfg.data_transforms['train'])
    # sampler = torch.utils.data.distributed.DistributedSampler(val_loader, shuffle=False)
    val_generator = DataLoader(dataset=val_loader,
                               batch_size=int(cfg.batch_size / n_gpus),
                               num_workers=int(cfg.num_thread / n_gpus),
                               pin_memory=True,
                               # sampler=sampler
                               )
    # ****************
    # 2. Setting Loss function
    # ****************
    BCELoss = smp.losses.SoftBCEWithLogitsLoss()
    TverskyLoss = smp.losses.TverskyLoss(mode='multilabel', log_loss=False)

    # ****************
    # 3. Training
    # ****************
    # load model
    model = UNet(in_channels=3, n_classes=len(train_dataset.cat_name), n_channels=48).to(device)
    if n_gpus > 1:
        model = DDP(model)
    optimizer = optim.Adam(model.parameters(),
                           lr=float(cfg.hyp.OPTIMIZER.LR),
                           weight_decay=float(cfg.hyp.OPTIMIZER.WD))

    for epoch in range(int(cfg.hyp.TRAINING.EPOCHS)):
        pbar = tqdm(enumerate(train_generator), total=len(train_generator), desc=f'Train - epoch: {epoch}')
        tracking_loss = {
            'Loss': torch.tensor(0.).float(),
            'BCELoss': torch.tensor(0.).float(),
            'TverskyLoss': torch.tensor(0.).float(),

            'ValLoss': torch.tensor(0.).float(),
            'ValBCELoss': torch.tensor(0.).float(),
            'ValTverskyLoss': torch.tensor(0.).float()
        }
        avg_loss = 0.0
        val_avg_loss = 0.0
        score = defaultdict(list)

        for step, (images, masks, _) in pbar:
            B = images.shape[0]
            images = images.to(device, dtype=torch.float)
            masks = masks.to(device, dtype=torch.float)

            y_pred = model(images)
            bce_loss = BCELoss(y_pred, masks)
            tversky_loss = TverskyLoss(y_pred, masks)
            losses = 0.5 * bce_loss + 0.5 * tversky_loss

            # Gradient update
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            # tracking loss
            tracking_loss['Loss'] += losses.detach().item() / B / int(cfg.hyp.TRAINING.EPOCHS)
            tracking_loss['BCELoss'] += bce_loss.detach().item() / B / int(cfg.hyp.TRAINING.EPOCHS)
            tracking_loss['TverskyLoss'] += tversky_loss.detach().item() / B / int(cfg.hyp.TRAINING.EPOCHS)

            _losses = float(losses.detach().cpu().numpy())
            pbar.set_description(
                f'Epoch {epoch + 1}/{cfg.hyp.TRAINING.EPOCHS} Train Loss - {format(_losses, ".04f")}')

            avg_loss += _losses

        pbar.set_description(f'Epoch {epoch+1}/{cfg.hyp.TRAINING.EPOCHS} Train Loss - {format(avg_loss/B, ".04f")}')

        # validation
        vpbar = tqdm(enumerate(val_generator), total=len(val_generator), desc=f'Val - epoch: {epoch}')
        for step, (images, masks, _) in vpbar:
            B = images.shape[0]
            images = images.to(device, dtype=torch.float)
            masks = masks.to(device, dtype=torch.float)
            with torch.no_grad():
                y_pred = model(images)
                val_bce_loss = BCELoss(y_pred, masks)
                val_tversky_loss = TverskyLoss(y_pred, masks)
                val_losses = 0.5 * val_bce_loss + 0.5 * val_tversky_loss

            # tracking loss
            tracking_loss['ValLoss'] += val_losses.detach().item() / B / int(cfg.hyp.TRAINING.EPOCHS)
            tracking_loss['ValBCELoss'] += val_bce_loss.detach().item() / B / int(cfg.hyp.TRAINING.EPOCHS)
            tracking_loss['ValTverskyLoss'] += val_tversky_loss.detach().item() / B / int(cfg.hyp.TRAINING.EPOCHS)

            # calculate iou
            for iou_thrs in [.50, .95]:
                iou = iou_coef(masks, y_pred)
                iou_score = (iou > iou_thrs).to(torch.float32).mean(dim=(1,0)).cpu().detach().numpy()
                score[f'mAP@{format(iou_thrs,".2f")}'].append(float(iou_score))

            score_str = ''
            for k, v in score.items():
                score_str += f'{k}:{format(np.mean(v),".2f")} '
            vpbar.set_description(f'Validation IoU - {score_str}')

        if use_wandb:
            wandb.log(tracking_loss)

        # model save
        if (epoch+1) % 10 == 0:
            file_path = osp.join(cfg.model_dir, f'snapshot_{epoch}.pt')
            torch.save(model.state_dict(), file_path)

    # model save
    file_path = osp.join(cfg.model_dir, f'snapshot_{cfg.hyp.TRAINING.EPOCHS}.pt')
    torch.save(model.state_dict(), file_path)
    print('End training')
