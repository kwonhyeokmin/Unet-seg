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


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=False, help='Path to pretrained checkpoint')
    parser.add_argument('--dataset', required=True, help='Dataset name', choices=['uwmgi', 'nia'])
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
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
        dataset = UWMGIDataset()
    elif args.dataset == 'nia':
        dataset = NIADataset()
    else:
        raise ValueError(f'Dataset name {args.dataset} is not supported yet.')
    dataset_loader = CTDataset(dataset, is_test=False, transforms=cfg.data_transforms['train'])
    data_generator = DataLoader(dataset=dataset_loader, batch_size=cfg.batch_size, shuffle=True,
                                num_workers=cfg.num_thread, pin_memory=True)

    # ****************
    # 2. Setting Loss function
    # ****************
    BCELoss = smp.losses.SoftBCEWithLogitsLoss()
    TverskyLoss = smp.losses.TverskyLoss(mode='multilabel', log_loss=False)

    # ****************
    # 3. Training
    # ****************
    # load model
    model = UNet(in_channels=3, n_classes=len(dataset.cat_name), n_channels=48).to(device)

    optimizer = optim.Adam(model.parameters(),
                           lr=float(cfg.hyp.OPTIMIZER.LR),
                           weight_decay=float(cfg.hyp.OPTIMIZER.WD))

    for epoch in range(int(cfg.hyp.TRAINING.EPOCHS)):
        pbar = tqdm(enumerate(data_generator), total=len(data_generator), desc=f'Train - epoch: {epoch}')
        tracking_loss = {
            'Loss': torch.tensor(0.).float(),
            'BCELoss': torch.tensor(0.).float(),
            'TverskyLoss': torch.tensor(0.).float()
        }
        avg_loss = 0.0
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
            pbar.set_description('[epoch %d] Train - Loss %.04f' % (epoch, _losses))
            avg_loss += _losses

        pbar.set_description('[epoch %d] Train Loss - %.04f' % (epoch, avg_loss / B))
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
